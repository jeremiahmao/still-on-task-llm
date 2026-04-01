"""Generate query decomposition training data using a teacher LLM.

Pipeline:
1. Generate complex financial questions from groups of pre-cutoff articles
2. Generate candidate sub-query decompositions for each question
3. Filter decompositions by Recall@10 against the FAISS index
4. Split into train/test sets

All steps checkpoint progress and resume on restart.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.decomp_filter import (
    filter_decompositions,
    format_qd_example,
    save_qd_dataset,
    split_train_test,
)
from sot.data.decomp_gen import generate_decompositions, save_decompositions
from sot.data.fnspid import get_text_column
from sot.data.query_gen import generate_questions, save_questions
from sot.models.base import load_model
from sot.retrieval.encoder import Encoder
from sot.retrieval.index import load_index
from sot.utils.config import load_config


def main():
    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    faiss_cfg = OmegaConf.load("configs/retrieval/faiss.yaml")

    data_root = Path(cfg.paths.data_root)
    corpus_path = data_root / "fnspid" / "processed" / "pre_cutoff.parquet"
    index_path = data_root / "fnspid" / "index" / "corpus.faiss"
    doc_ids_path = data_root / "fnspid" / "index" / "doc_ids.npy"

    for p in [corpus_path, index_path, doc_ids_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    # Load corpus
    print("Loading pre-cutoff corpus...")
    df = pd.read_parquet(corpus_path)
    text_col = get_text_column(df, fnspid_cfg.text_columns)
    articles = df.to_dict("records")
    print(f"Corpus: {len(articles)} articles")

    qd_dir = data_root / "qd"
    qd_dir.mkdir(parents=True, exist_ok=True)

    questions_path = qd_dir / "raw_questions.json"
    questions_ckpt = qd_dir / ".questions_checkpoint.json"
    decomps_path = qd_dir / "raw_decompositions.json"
    decomps_ckpt = qd_dir / ".decomps_checkpoint.json"

    # Load teacher model (needed for steps 1 and 2 if outputs don't exist)
    model = None
    tokenizer = None
    need_model = not questions_path.exists() or not decomps_path.exists()
    if need_model:
        print(f"\nLoading teacher model: {cfg.model.name}")
        model, tokenizer = load_model(cfg.model.name, cfg.model.dtype)

    # Step 1: Generate questions (resume-aware)
    if questions_path.exists():
        from sot.data.query_gen import load_questions

        print(f"Loading existing questions from {questions_path}")
        questions = load_questions(questions_path)
    else:
        print("\nStep 1: Generating questions...")
        questions = generate_questions(
            articles,
            model,
            tokenizer,
            text_column=text_col,
            n_questions=5000,
            articles_per_question=3,
            seed=cfg.seed,
            checkpoint_path=str(questions_ckpt),
            checkpoint_every=200,
        )
        save_questions(questions, questions_path)

        # Clean up checkpoint
        if questions_ckpt.exists():
            questions_ckpt.unlink()
    print(f"Questions: {len(questions)}")

    # Step 2: Generate decompositions (resume-aware)
    if decomps_path.exists():
        from sot.data.decomp_gen import load_decompositions

        print(f"Loading existing decompositions from {decomps_path}")
        decomps = load_decompositions(decomps_path)
    else:
        print("\nStep 2: Generating decompositions...")
        decomps = generate_decompositions(
            questions,
            model,
            tokenizer,
            n_candidates=5,
            checkpoint_path=str(decomps_ckpt),
            checkpoint_every=200,
        )
        save_decompositions(decomps, decomps_path)

        # Clean up checkpoint
        if decomps_ckpt.exists():
            decomps_ckpt.unlink()
    print(f"Questions with decompositions: {len(decomps)}")

    # Free GPU memory from teacher model
    if model is not None:
        del model
        import torch

        torch.cuda.empty_cache()

    # Step 3: Filter by Recall@10
    print("\nStep 3: Filtering by Recall@10...")
    encoder = Encoder(faiss_cfg.encoder)
    faiss_index = load_index(index_path)
    doc_ids = np.load(doc_ids_path).tolist()

    filtered = filter_decompositions(
        decomps,
        encoder,
        faiss_index,
        doc_ids,
        min_recall=0.7,
        k=10,
    )
    print(f"After Recall@10 filter: {len(filtered)} questions")

    # Step 4: Format and split
    print("\nStep 4: Train/test split...")
    train_data, test_data = split_train_test(filtered, test_ratio=0.2, seed=cfg.seed)

    train_formatted = [format_qd_example(item) for item in train_data]
    test_formatted = [format_qd_example(item) for item in test_data]

    save_qd_dataset(train_formatted, qd_dir / "train.json")
    save_qd_dataset(test_formatted, qd_dir / "test.json")

    print(f"Train set: {len(train_formatted)} examples -> {qd_dir / 'train.json'}")
    print(f"Test set:  {len(test_formatted)} examples -> {qd_dir / 'test.json'}")
    print("\nDone. QD training data ready.")


if __name__ == "__main__":
    main()
