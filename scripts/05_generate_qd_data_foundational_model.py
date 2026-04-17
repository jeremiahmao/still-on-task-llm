"""Generate temporally paired query decomposition data aligned with latex.txt.

This script creates a benchmark where the same high-level question is meaningful
before and after 2022-01-01, but the retrieval strategy should differ because
the world changed. The final SFT train/test set uses only the pre-cutoff
decomposition, while preserving the post-cutoff decomposition and changed facts
for downstream update/evaluation work.
"""

import asyncio
import argparse
import json
import os
import random
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
    generate_temporal_decomposition_api_sync,
    generate_temporal_question,
    generate_temporal_question_api,
    generate_temporal_question_api_sync,
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


def _build_openai_api_func(
    model_name: str,
    max_retries: int = 6,
    base_retry_seconds: float = 10.0,
):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI teacher requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --teacher-provider=openai")

    client = AsyncOpenAI(api_key=api_key)

    async def call_api(prompt: str, text_format: dict | None = None) -> str:
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "input": prompt,
                }
                if text_format is not None:
                    kwargs["text"] = {"format": text_format}

                response = await client.responses.create(**kwargs)
                return response.output_text
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = "rate limit" in message or "429" in message
                if not is_rate_limit or attempt >= max_retries:
                    raise
                delay = base_retry_seconds * (2**attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)

    return call_api


def _build_gemini_api_func(
    model_name: str,
    max_retries: int = 6,
    base_retry_seconds: float = 10.0,
    tpm_limit: int = 1_000_000,
    rpm_limit: int = 1000,
):
    """Build an async API function targeting Gemini via its OpenAI-compatible endpoint."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "Gemini teacher requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is required when --teacher-provider=gemini"
        )

    client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )

    from sot.utils.rate_limit import AsyncRateLimiter, estimate_tokens

    limiter = AsyncRateLimiter(tpm_limit, rpm_limit)

    async def call_api(prompt: str, text_format: dict | None = None) -> str:
        est = estimate_tokens(prompt) + 300
        await limiter.acquire(est)
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if text_format is not None:
                    kwargs["response_format"] = {"type": "json_object"}

                response = await client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as exc:
                message = str(exc).lower()
                is_transient = (
                    "rate limit" in message
                    or "429" in message
                    or "503" in message
                    or "502" in message
                    or "queue" in message
                    or "timeout" in message
                    or "overloaded" in message
                )
                if not is_transient or attempt >= max_retries:
                    raise
                delay = base_retry_seconds * (2**attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)

    return call_api


def _build_cerebras_api_func(
    model_name: str,
    max_retries: int = 6,
    base_retry_seconds: float = 10.0,
    tpm_limit: int = 500_000,
    rpm_limit: int = 500,
):
    """Build an async API function targeting the Cerebras Inference API."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "Cerebras teacher requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is required when --teacher-provider=cerebras")

    client = AsyncOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )

    from sot.utils.rate_limit import AsyncRateLimiter, estimate_tokens

    limiter = AsyncRateLimiter(tpm_limit, rpm_limit)

    async def call_api(prompt: str, text_format: dict | None = None) -> str:
        est = estimate_tokens(prompt) + 300
        await limiter.acquire(est)
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if text_format is not None:
                    kwargs["response_format"] = {"type": "json_object"}

                response = await client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = "rate limit" in message or "429" in message
                if not is_rate_limit or attempt >= max_retries:
                    raise
                delay = base_retry_seconds * (2**attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)

    return call_api


def _build_cerebras_api_func_sync(
    model_name: str,
    max_retries: int = 6,
    base_retry_seconds: float = 10.0,
    tpm_limit: int = 500_000,
    rpm_limit: int = 500,
):
    """Build a sync API function targeting the Cerebras Inference API."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Cerebras teacher requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is required when --teacher-provider=cerebras")

    client = OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )

    from sot.utils.rate_limit import SyncRateLimiter, estimate_tokens

    limiter = SyncRateLimiter(tpm_limit, rpm_limit)

    def call_api(prompt: str, text_format: dict | None = None) -> str:
        est = estimate_tokens(prompt) + 300
        limiter.acquire(est)
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if text_format is not None:
                    kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = "rate limit" in message or "429" in message
                if not is_rate_limit or attempt >= max_retries:
                    raise
                delay = base_retry_seconds * (2**attempt) + random.uniform(0, 0.25)
                import time

                time.sleep(delay)

    return call_api


def _build_openai_api_func_sync(
    model_name: str,
    max_retries: int = 6,
    base_retry_seconds: float = 10.0,
):
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
        for attempt in range(max_retries + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "input": prompt,
                }
                if text_format is not None:
                    kwargs["text"] = {"format": text_format}

                response = client.responses.create(**kwargs)
                return response.output_text
            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = "rate limit" in message or "429" in message
                if not is_rate_limit or attempt >= max_retries:
                    raise
                delay = base_retry_seconds * (2**attempt) + random.uniform(0, 0.25)
                import time

                time.sleep(delay)

    return call_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug corpora, triples, and FAISS index from configs/data/fnspid.yaml.",
    )
    parser.add_argument("--teacher-provider", choices=["local", "openai", "cerebras", "gemini"], default="local")
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--min-articles-per-side", type=int, default=2)
    parser.add_argument("--bundle-size", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.65)
    parser.add_argument("--min-contrast", type=float, default=0.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--base-retry-seconds", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--tpm-limit",
        type=int,
        default=500_000,
        help="Tokens-per-minute rate limit for Cerebras API (default: 500000).",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=500,
        help="Requests-per-minute rate limit for Cerebras API (default: 500).",
    )
    parser.add_argument("--overrides", nargs="*", default=[], help="OmegaConf dot-list overrides")
    args = parser.parse_args()

    cfg = load_config(overrides=args.overrides)
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    faiss_cfg_path = (
        "configs/retrieval/faiss_debug.yaml" if args.debug else "configs/retrieval/faiss.yaml"
    )
    faiss_cfg = OmegaConf.load(faiss_cfg_path)

    data_root = Path(cfg.paths.data_root)
    suffix = fnspid_cfg.debug.output_suffix if args.debug else ""
    default_output_dir = Path(cfg.paths.qd_temporal_data_root)
    if suffix:
        default_output_dir = default_output_dir / suffix.lstrip("_")
    output_dir = Path(args.output_dir or default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_pairs_path = output_dir / "topic_pairs.json"
    paired_examples_path = output_dir / "paired_examples.json"
    teacher_cache_path = output_dir / "teacher_cache.jsonl"
    train_path = output_dir / "train.json"
    test_path = output_dir / "test.json"
    metadata_path = output_dir / "metadata.json"

    pre_path = data_root / "fnspid" / "processed" / f"pre_cutoff{suffix}.parquet"
    post_path = data_root / "fnspid" / "processed" / f"post_cutoff{suffix}.parquet"
    triples_path = data_root / "fnspid" / "triples" / f"filtered_triples{suffix}.json"
    index_path = data_root / "fnspid" / "index" / f"corpus{suffix}.faiss"
    doc_ids_path = data_root / "fnspid" / "index" / f"doc_ids{suffix}.npy"
    embeddings_path = data_root / "fnspid" / "index" / f"embeddings{suffix}.npy"

    for path in [pre_path, post_path, triples_path, index_path, doc_ids_path]:
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    _default_teacher_models = {
        "openai": "gpt-5-mini",
        "cerebras": "qwen-3-235b-a22b-instruct-2507",
        "gemini": "gemini-3.1-flash-lite-preview",
    }
    teacher_model_name = args.teacher_model or _default_teacher_models.get(
        args.teacher_provider, cfg.model.name
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
        "debug": args.debug,
        "concurrency": args.concurrency,
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

    # Skip re-running ONLY if no force, final outputs exist, AND no teacher cache
    # (teacher cache presence means user may want to re-filter at a new threshold).
    should_skip = (
        paired_examples_path.exists()
        and train_path.exists()
        and test_path.exists()
        and not args.force
    )
    if should_skip:
        print(f"Loading cached paired examples from {paired_examples_path}")
        paired_examples = load_json(paired_examples_path)
    else:
        print("\nLoading retrieval encoder and FAISS indices (pre + post)...")
        encoder = Encoder(faiss_cfg.encoder)
        faiss_index_pre = load_index(index_path)
        # FAISS index contains chunk embeddings. chunk_to_article.npy maps
        # each chunk index -> its source article index.
        chunk_to_article_pre_path = data_root / "fnspid" / "index" / f"chunk_to_article{suffix}.npy"
        if chunk_to_article_pre_path.exists():
            doc_ids_pre = np.load(chunk_to_article_pre_path).tolist()
        else:
            doc_ids_pre = np.load(doc_ids_path).tolist()

        # Load post-cutoff index + mapping for scoring post decompositions.
        post_index_path = data_root / "fnspid" / "index" / f"corpus_post{suffix}.faiss"
        post_chunk_path = data_root / "fnspid" / "index" / f"chunk_to_article_post{suffix}.npy"
        if post_index_path.exists() and post_chunk_path.exists():
            faiss_index_post = load_index(post_index_path)
            doc_ids_post = np.load(post_chunk_path).tolist()
        else:
            print(f"WARN: post-cutoff index not found at {post_index_path}; post_recall will be 0.")
            faiss_index_post = None
            doc_ids_post = []

        corpus_embeddings = np.load(embeddings_path) if args.debug and embeddings_path.exists() else None

        # Back-compat aliases (earlier code in this file references these names).
        faiss_index = faiss_index_pre
        doc_ids = doc_ids_pre

        model = None
        tokenizer = None
        api_func = None
        api_func_sync = None
        if args.teacher_provider == "local":
            print(f"\nLoading local teacher model: {teacher_model_name}")
            model, tokenizer = load_model(teacher_model_name, cfg.model.dtype)
        elif args.teacher_provider == "cerebras":
            print(f"\nConfiguring Cerebras teacher: {teacher_model_name}")
            print(f"  TPM limit: {args.tpm_limit:,}, RPM limit: {args.rpm_limit:,}")
            if args.concurrency <= 1:
                print("Using synchronous Cerebras teacher path")
                api_func_sync = _build_cerebras_api_func_sync(
                    teacher_model_name,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                    tpm_limit=args.tpm_limit,
                    rpm_limit=args.rpm_limit,
                )
            else:
                api_func = _build_cerebras_api_func(
                    teacher_model_name,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                    tpm_limit=args.tpm_limit,
                    rpm_limit=args.rpm_limit,
                )
        elif args.teacher_provider == "gemini":
            print(f"\nConfiguring Gemini teacher: {teacher_model_name}")
            print(f"  TPM limit: {args.tpm_limit:,}, RPM limit: {args.rpm_limit:,}")
            api_func = _build_gemini_api_func(
                teacher_model_name,
                max_retries=args.max_retries,
                base_retry_seconds=args.base_retry_seconds,
                tpm_limit=args.tpm_limit,
                rpm_limit=args.rpm_limit,
            )
        else:
            print(f"\nConfiguring OpenAI teacher: {teacher_model_name}")
            if args.concurrency <= 1:
                print("Using synchronous OpenAI teacher path")
                api_func_sync = _build_openai_api_func_sync(
                    teacher_model_name,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                )
            else:
                api_func = _build_openai_api_func(
                    teacher_model_name,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                )

        paired_examples = []
        if args.teacher_provider == "local":
            for i, pair in enumerate(topic_pairs, start=1):
                print(f"\n[{i}/{len(topic_pairs)}] {pair['entity']}")

                question = generate_temporal_question(pair, model, tokenizer)
                if not question:
                    print("  Skipping: question generation failed")
                    continue

                pre_decomp = generate_temporal_decomposition(pair, question, "pre", model, tokenizer)
                post_decomp = generate_temporal_decomposition(pair, question, "post", model, tokenizer)

                if not pre_decomp or not post_decomp:
                    print("  Skipping: missing decomposition")
                    continue

                pre_gold_articles = [a["doc_id"] for a in pair["pre_articles"]]
                post_gold_articles = [a["doc_id"] for a in pair["post_articles"]]

                pre_recall = score_decomposition_recall(
                    pre_decomp,
                    encoder,
                    faiss_index_pre,
                    doc_ids_pre,
                    pre_gold_articles,
                    corpus_embeddings=corpus_embeddings,
                    nprobe=faiss_cfg.nprobe,
                )
                post_recall = (
                    score_decomposition_recall(
                        post_decomp,
                        encoder,
                        faiss_index_post,
                        doc_ids_post,
                        post_gold_articles,
                        corpus_embeddings=None,
                        nprobe=faiss_cfg.nprobe,
                    )
                    if faiss_index_post is not None
                    else 0.0
                )
                contrast_score = decomposition_contrast_score(pre_decomp, post_decomp)

                if not args.debug and pre_recall < args.min_recall:
                    print(f"  Skipping: pre recall {pre_recall:.3f} < {args.min_recall}")
                    continue
                if not args.debug and post_recall < args.min_recall:
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
        elif args.concurrency <= 1:
            for i, pair in enumerate(topic_pairs, start=1):
                print(f"\n[{i}/{len(topic_pairs)}] {pair['entity']}")

                print("  Generating question...")
                question = generate_temporal_question_api_sync(pair, api_func_sync)
                if not question:
                    print("  Skipping: question generation failed")
                    continue
                print(f"  Question: {question}")

                print("  Generating pre decomposition...")
                pre_decomp = generate_temporal_decomposition_api_sync(
                    pair, question, "pre", api_func_sync
                )
                print("  Generating post decomposition...")
                post_decomp = generate_temporal_decomposition_api_sync(
                    pair, question, "post", api_func_sync
                )
                if not pre_decomp or not post_decomp:
                    print("  Skipping: missing decomposition")
                    continue

                pre_gold_articles = [a["doc_id"] for a in pair["pre_articles"]]
                post_gold_articles = [a["doc_id"] for a in pair["post_articles"]]

                print("  Scoring retrieval recall...")
                pre_recall = score_decomposition_recall(
                    pre_decomp,
                    encoder,
                    faiss_index_pre,
                    doc_ids_pre,
                    pre_gold_articles,
                    corpus_embeddings=corpus_embeddings,
                    nprobe=faiss_cfg.nprobe,
                )
                post_recall = (
                    score_decomposition_recall(
                        post_decomp,
                        encoder,
                        faiss_index_post,
                        doc_ids_post,
                        post_gold_articles,
                        corpus_embeddings=None,
                        nprobe=faiss_cfg.nprobe,
                    )
                    if faiss_index_post is not None
                    else 0.0
                )
                contrast_score = decomposition_contrast_score(pre_decomp, post_decomp)

                if not args.debug and pre_recall < args.min_recall:
                    print(f"  Skipping: pre recall {pre_recall:.3f} < {args.min_recall}")
                    continue
                if not args.debug and post_recall < args.min_recall:
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
        else:
            semaphore = asyncio.Semaphore(max(args.concurrency, 1))
            from tqdm import tqdm as tqdm_sync
            from collections import Counter
            drop_reasons = Counter()

            # Load prior teacher outputs from cache (by topic_id).
            # Cache holds RAW outputs (question + both decomps) before any filtering,
            # so threshold changes can be reapplied without new API calls.
            cached_outputs: dict[str, dict] = {}
            if teacher_cache_path.exists():
                with teacher_cache_path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            if rec.get("topic_id"):
                                cached_outputs[rec["topic_id"]] = rec
                        except json.JSONDecodeError:
                            continue
                print(f"Loaded {len(cached_outputs)} cached teacher outputs from {teacher_cache_path}")

            cache_lock = asyncio.Lock()

            async def fetch_or_cache(pair: dict) -> dict | None:
                """Return {topic_id, question, pre_decomp, post_decomp} using cache when available."""
                topic_id = pair["topic_id"]
                if topic_id in cached_outputs:
                    return cached_outputs[topic_id]

                async with semaphore:
                    try:
                        question = await generate_temporal_question_api(pair, api_func)
                    except Exception as exc:
                        drop_reasons[f"question_error:{type(exc).__name__}"] += 1
                        return None
                    if not question:
                        drop_reasons["question_empty"] += 1
                        return None

                    try:
                        pre_decomp = await generate_temporal_decomposition_api(pair, question, "pre", api_func)
                        post_decomp = await generate_temporal_decomposition_api(pair, question, "post", api_func)
                    except Exception as exc:
                        drop_reasons[f"decomp_error:{type(exc).__name__}"] += 1
                        return None

                record = {
                    "topic_id": topic_id,
                    "entity": pair["entity"],
                    "question": question,
                    "pre_decomp": pre_decomp or [],
                    "post_decomp": post_decomp or [],
                }
                async with cache_lock:
                    cached_outputs[topic_id] = record
                    with teacher_cache_path.open("a") as f:
                        f.write(json.dumps(record) + "\n")
                return record

            def score_and_filter(pair: dict, record: dict) -> dict | None:
                """Apply recall + contrast filter on top of a cached teacher record."""
                question = record.get("question")
                pre_decomp = record.get("pre_decomp") or []
                post_decomp = record.get("post_decomp") or []
                if not question:
                    drop_reasons["question_empty"] += 1
                    return None
                if not pre_decomp:
                    drop_reasons["pre_decomp_none"] += 1
                    return None
                if not post_decomp:
                    drop_reasons["post_decomp_none"] += 1
                    return None

                pre_gold_articles = [a["doc_id"] for a in pair["pre_articles"]]
                post_gold_articles = [a["doc_id"] for a in pair["post_articles"]]

                pre_recall = score_decomposition_recall(
                    pre_decomp,
                    encoder,
                    faiss_index_pre,
                    doc_ids_pre,
                    pre_gold_articles,
                    corpus_embeddings=corpus_embeddings,
                    nprobe=faiss_cfg.nprobe,
                )
                if faiss_index_post is not None:
                    post_recall = score_decomposition_recall(
                        post_decomp,
                        encoder,
                        faiss_index_post,
                        doc_ids_post,
                        post_gold_articles,
                        corpus_embeddings=None,
                        nprobe=faiss_cfg.nprobe,
                    )
                else:
                    post_recall = 0.0
                contrast_score = decomposition_contrast_score(pre_decomp, post_decomp)

                if (not args.debug) and (pre_recall < args.min_recall or post_recall < args.min_recall):
                    drop_reasons[f"low_recall(pre={pre_recall:.2f},post={post_recall:.2f})"] += 1
                    return None
                if contrast_score < args.min_contrast:
                    drop_reasons["low_contrast"] += 1
                    return None

                drop_reasons["KEPT"] += 1
                return {
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

            async def process_all_pairs():
                # Phase 1: fetch (or reuse cached) teacher outputs for every pair.
                tasks = [
                    asyncio.create_task(fetch_or_cache(pair))
                    for pair in topic_pairs
                ]
                records: list[dict | None] = []
                pbar = tqdm_sync(total=len(tasks), desc="Teacher LLM (Q + decomp)")
                for task in asyncio.as_completed(tasks):
                    records.append(await task)
                    pbar.update(1)
                pbar.close()
                # Phase 2: local filter (recall + contrast). Fast, no API calls.
                kept = []
                record_by_topic = {r["topic_id"]: r for r in records if r and r.get("topic_id")}
                for pair in topic_pairs:
                    rec = record_by_topic.get(pair["topic_id"])
                    if rec is None:
                        continue
                    kept_item = score_and_filter(pair, rec)
                    if kept_item is not None:
                        kept.append(kept_item)
                return kept

            paired_examples = asyncio.run(process_all_pairs())
            print("\nDrop reason counts:")
            for reason, count in drop_reasons.most_common():
                print(f"  {reason}: {count}")

        save_json(paired_examples, paired_examples_path)
        print(f"\nPaired examples: {len(paired_examples)} -> {paired_examples_path}")

    print("\nExporting train/test splits from the pre-cutoff side...")
    if args.debug:
        paired_train, paired_test = paired_examples, []
    else:
        paired_train, paired_test = split_train_test(paired_examples, test_ratio=args.test_ratio, seed=cfg.seed)

    train_data = [build_temporal_training_example(item) for item in paired_train]
    test_data = [build_temporal_training_example(item) for item in paired_test]

    # Post-cutoff test set: same questions, gold = post-2022 articles.
    # Used to measure whether updated models retrieve post-cutoff docs better.
    post_test_data = [
        {
            "question": item["question"],
            "gold_articles": item["post_gold_articles"],
            "changed_facts": item["changed_facts"],
            "topic_id": item["topic_id"],
            "entity": item["entity"],
        }
        for item in paired_examples
    ]
    post_test_path = output_dir / "post_test.json"

    save_json(train_data, train_path)
    save_json(test_data, test_path)
    save_json(post_test_data, post_test_path)
    metadata["n_topic_pairs"] = len(topic_pairs)
    metadata["n_paired_examples"] = len(paired_examples)
    metadata["n_train"] = len(train_data)
    metadata["n_test"] = len(test_data)
    metadata["n_post_test"] = len(post_test_data)
    save_json(metadata, metadata_path)

    print(f"Train set: {len(train_data)} -> {train_path}")
    print(f"Test set:  {len(test_data)} -> {test_path}")
    print(f"Post-cutoff test set: {len(post_test_data)} -> {post_test_path}")
    print(f"Metadata: {metadata_path}")
    print("\nDone. Temporal QD data ready.")


if __name__ == "__main__":
    main()
