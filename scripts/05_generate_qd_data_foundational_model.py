"""Generate temporally paired query decomposition data aligned with latex.txt.

This script creates a benchmark where the same high-level question is meaningful
before and after 2022-01-01, but the retrieval strategy should differ because
the world changed. The final SFT train/test set uses only the pre-cutoff
decomposition, while preserving the post-cutoff decomposition and changed facts
for downstream update/evaluation work.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.fnspid import get_text_column
from sot.data.temporal_qd import (
    build_temporal_topic_pairs,
    build_temporal_training_example,
    decomposition_contrast_score,
    generate_temporal_decomposition,
    generate_temporal_decomposition_api,
    generate_temporal_question,
    generate_temporal_question_api,
    load_json,
    save_json,
    score_decomposition_recall,
    split_train_test,
)
from sot.data.triple_extract import load_triples
from sot.models.base import load_model
from sot.retrieval.encoder import Encoder
from sot.retrieval.index import load_index
from sot.utils.config import load_config


def _build_openai_api_func(model_name: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI teacher requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --teacher-provider=openai")

    client = OpenAI(api_key=api_key)

    def call_api(prompt: str, text_format: dict | None = None) -> str:
        kwargs = {
            "model": model_name,
            "input": prompt,
        }
        if text_format is not None:
            kwargs["text"] = {"format": text_format}

        response = client.responses.create(**kwargs)
        return response.output_text

    return call_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-provider", choices=["local", "openai"], default="local")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--min-articles-per-side", type=int, default=5)
    parser.add_argument("--bundle-size", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.7)
    parser.add_argument("--min-contrast", type=float, default=0.35)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=[], help="OmegaConf dot-list overrides")
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    faiss_cfg = OmegaConf.load("configs/retrieval/faiss.yaml")

    data_root = Path(cfg.paths.data_root)
    output_dir = Path(args.output_dir or cfg.paths.qd_temporal_data_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_pairs_path = output_dir / "topic_pairs.json"
    paired_examples_path = output_dir / "paired_examples.json"
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    metadata_path = output_dir / "metadata.json"

    pre_path = data_root / "fnspid" / "processed" / "pre_cutoff.parquet"
    post_path = data_root / "fnspid" / "processed" / "post_cutoff.parquet"
    triples_path = data_root / "fnspid" / "triples" / "filtered_triples.json"
    index_path = data_root / "fnspid" / "index" / "corpus.faiss"
    doc_ids_path = data_root / "fnspid" / "index" / "doc_ids.npy"

    for path in [pre_path, post_path, triples_path, index_path, doc_ids_path]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    teacher_model_name = args.teacher_model or (
        "gpt-5-mini" if args.teacher_provider == "openai" else cfg.model.name
    )
    metadata = {
        "teacher_provider": args.teacher_provider,
        "teacher_model": teacher_model_name,
        "cutoff_date": fnspid_cfg.cutoff_date,
        "min_articles_per_side": args.min_articles_per_side,
        "bundle_size": args.bundle_size,
        "min_recall": args.min_recall,
        "min_contrast": args.min_contrast,
        "max_pairs": args.max_pairs,
        "seed": cfg.seed,
    }

    print("Loading temporal corpora...")
    pre_df = pd.read_parquet(pre_path)
    post_df = pd.read_parquet(post_path)
    text_col = get_text_column(pre_df, fnspid_cfg.text_columns)
    print(f"Pre-cutoff articles: {len(pre_df)}")
    print(f"Post-cutoff articles: {len(post_df)}")
    print(f"Using text column: {text_col}")

    if topic_pairs_path.exists() and not args.force:
        print(f"Loading cached topic pairs from {topic_pairs_path}")
        topic_pairs = load_json(topic_pairs_path)
    else:
        print("\nBuilding topic-centered temporal pairs...")
        post_triples = load_triples(str(triples_path))
        topic_pairs = build_temporal_topic_pairs(
            pre_df=pre_df,
            post_df=post_df,
            ticker_column=fnspid_cfg.ticker_column,
            text_column=text_col,
            post_triples=post_triples,
            date_column=fnspid_cfg.date_column,
            min_articles_per_side=args.min_articles_per_side,
            bundle_size=args.bundle_size,
            max_pairs=args.max_pairs,
            seed=cfg.seed,
        )
        save_json(topic_pairs, topic_pairs_path)
        print(f"Topic pairs: {len(topic_pairs)} -> {topic_pairs_path}")

    if paired_examples_path.exists() and train_path.exists() and test_path.exists() and not args.force:
        print(f"Loading cached paired examples from {paired_examples_path}")
        paired_examples = load_json(paired_examples_path)
    else:
        print("\nLoading retrieval encoder and FAISS index...")
        encoder = Encoder(faiss_cfg.encoder)
        faiss_index = load_index(index_path)
        doc_ids = np.load(doc_ids_path).tolist()

        model = None
        tokenizer = None
        api_func = None
        if args.teacher_provider == "local":
            print(f"\nLoading local teacher model: {teacher_model_name}")
            model, tokenizer = load_model(teacher_model_name, cfg.model.dtype)
        else:
            print(f"\nConfiguring OpenAI teacher: {teacher_model_name}")
            api_func = _build_openai_api_func(teacher_model_name)

        paired_examples = []
        for i, pair in enumerate(topic_pairs, start=1):
            print(f"\n[{i}/{len(topic_pairs)}] {pair['entity']}")

            if args.teacher_provider == "local":
                question = generate_temporal_question(pair, model, tokenizer)
            else:
                question = generate_temporal_question_api(pair, api_func)
            if not question:
                print("  Skipping: question generation failed")
                continue

            if args.teacher_provider == "local":
                pre_decomp = generate_temporal_decomposition(pair, question, "pre", model, tokenizer)
                post_decomp = generate_temporal_decomposition(pair, question, "post", model, tokenizer)
            else:
                pre_decomp = generate_temporal_decomposition_api(pair, question, "pre", api_func)
                post_decomp = generate_temporal_decomposition_api(pair, question, "post", api_func)

            if not pre_decomp or not post_decomp:
                print("  Skipping: missing decomposition")
                continue

            pre_gold_articles = [a["doc_id"] for a in pair["pre_articles"]]
            post_gold_articles = [a["doc_id"] for a in pair["post_articles"]]

            pre_recall = score_decomposition_recall(
                pre_decomp, encoder, faiss_index, doc_ids, pre_gold_articles, nprobe=faiss_cfg.nprobe
            )
            post_recall = score_decomposition_recall(
                post_decomp,
                encoder,
                faiss_index,
                doc_ids,
                post_gold_articles,
                nprobe=faiss_cfg.nprobe,
            )
            contrast_score = decomposition_contrast_score(pre_decomp, post_decomp)

            if pre_recall < args.min_recall:
                print(f"  Skipping: pre recall {pre_recall:.3f} < {args.min_recall}")
                continue
            if post_recall < args.min_recall:
                print(f"  Skipping: post recall {post_recall:.3f} < {args.min_recall}")
                continue
            if contrast_score < args.min_contrast:
                print(f"  Skipping: contrast {contrast_score:.3f} < {args.min_contrast}")
                continue

            paired_examples.append(
                {
                    "topic_id": pair["topic_id"],
                    "entity": pair["entity"],
                    "question": question,
                    "pre_articles": pair["pre_articles"],
                    "post_articles": pair["post_articles"],
                    "changed_facts": pair["changed_facts"],
                    "pre_decomposition": pre_decomp,
                    "post_decomposition": post_decomp,
                    "pre_gold_articles": pre_gold_articles,
                    "post_gold_articles": post_gold_articles,
                    "pre_recall": pre_recall,
                    "post_recall": post_recall,
                    "contrast_score": contrast_score,
                }
            )
            print(
                f"  Kept: pre_recall={pre_recall:.3f}, "
                f"post_recall={post_recall:.3f}, contrast={contrast_score:.3f}"
            )

        save_json(paired_examples, paired_examples_path)
        print(f"\nPaired examples: {len(paired_examples)} -> {paired_examples_path}")

    print("\nExporting train/test splits from the pre-cutoff side...")
    paired_train, paired_test = split_train_test(paired_examples, test_ratio=args.test_ratio, seed=cfg.seed)

    train_data = [build_temporal_training_example(item) for item in paired_train]
    test_data = [build_temporal_training_example(item) for item in paired_test]

    save_json(train_data, train_path)
    save_json(test_data, test_path)
    metadata["n_topic_pairs"] = len(topic_pairs)
    metadata["n_paired_examples"] = len(paired_examples)
    metadata["n_train"] = len(train_data)
    metadata["n_test"] = len(test_data)
    save_json(metadata, metadata_path)

    print(f"Train set: {len(train_data)} -> {train_path}")
    print(f"Test set:  {len(test_data)} -> {test_path}")
    print(f"Metadata: {metadata_path}")
    print("\nDone. Temporal QD data ready.")


if __name__ == "__main__":
    main()
