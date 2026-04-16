"""Extract fact triples from post-cutoff FNSPID articles, filter, and sample at scales."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.fnspid import get_text_column, load_fnspid
from sot.data.triple_extract import (
    extract_triples_api,
    extract_triples_api_async,
    extract_triples_batch,
    load_progress_jsonl,
    load_triples,
    save_triples,
)
from sot.data.triple_filter import (
    filter_cross_doc_agreement,
    sample_at_scales,
    save_scaled_triples,
)
from sot.models.base import load_model
from sot.utils.config import load_config


def _build_openai_api_func(model_name: str):
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI provider requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --provider=openai")

    client = AsyncOpenAI(api_key=api_key)

    async def call_api(prompt: str) -> str:
        response = await client.responses.create(
            model=model_name,
            input=prompt,
        )
        return response.output_text

    return call_api


def _build_gemini_api_func(model_name: str, tpm_limit: int = 1_000_000, rpm_limit: int = 1000):
    """Build an async API function targeting Gemini via its OpenAI-compatible endpoint."""
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "Gemini provider requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is required when --provider=gemini"
        )

    client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )

    from sot.utils.rate_limit import AsyncRateLimiter, estimate_tokens

    limiter = AsyncRateLimiter(tpm_limit, rpm_limit)

    async def call_api(prompt: str) -> str:
        est = estimate_tokens(prompt) + 200
        await limiter.acquire(est)
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    return call_api


def _build_cerebras_api_func(model_name: str, tpm_limit: int = 500_000, rpm_limit: int = 500):
    """Build an async API function targeting the Cerebras Inference API.

    Cerebras exposes an OpenAI-compatible chat completions endpoint at
    https://api.cerebras.ai/v1, so we reuse the openai SDK with a custom base_url.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "Cerebras provider requested but the 'openai' package is not installed."
        ) from exc

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise RuntimeError("CEREBRAS_API_KEY is required when --provider=cerebras")

    client = AsyncOpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )

    from sot.utils.rate_limit import AsyncRateLimiter, estimate_tokens

    limiter = AsyncRateLimiter(tpm_limit, rpm_limit)

    async def call_api(prompt: str) -> str:
        est = estimate_tokens(prompt) + 200  # prompt + expected output (triples JSON is small)
        await limiter.acquire(est)
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    return call_api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug ticker filtering from configs/data/fnspid.yaml.",
    )
    parser.add_argument("--provider", choices=["local", "openai", "cerebras", "gemini"], default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Concurrent OpenAI requests when --provider=openai.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Max retries per article on transient OpenAI rate limits.",
    )
    parser.add_argument(
        "--base-retry-seconds",
        type=float,
        default=10.0,
        help="Base sleep time for exponential backoff on OpenAI rate limits.",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=[],
        help="Optional ticker whitelist, e.g. --tickers AAPL MSFT NVDA",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output files, e.g. '_debug'.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Flush per-article extraction progress every N completed articles.",
    )
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
    args = parser.parse_args()

    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    triples_cfg = OmegaConf.load("configs/data/triples.yaml")

    data_root = Path(cfg.paths.data_root)
    debug_suffix = fnspid_cfg.debug.output_suffix if args.debug else ""
    post_path = data_root / "fnspid" / "processed" / f"post_cutoff{debug_suffix}.parquet"
    pre_path = data_root / "fnspid" / "processed" / f"pre_cutoff{debug_suffix}.parquet"

    if not post_path.exists():
        print(f"ERROR: {post_path} not found. Run 02_build_corpus.py first.")
        sys.exit(1)

    raw_dir = data_root / "fnspid" / "triples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.output_suffix
    if args.debug and not suffix:
        suffix = fnspid_cfg.debug.output_suffix
    if not suffix and args.tickers:
        suffix = "_" + "_".join(t.lower() for t in args.tickers)
    raw_triples_path = raw_dir / f"raw_triples{suffix}.json"
    filtered_triples_path = raw_dir / f"filtered_triples{suffix}.json"
    progress_path = raw_dir / f"raw_triples{suffix}.jsonl"

    ticker_filter = {ticker.upper() for ticker in args.tickers}
    if args.debug:
        ticker_filter = {ticker.upper() for ticker in fnspid_cfg.debug.tickers}
    _default_models = {
        "openai": "gpt-5-mini",
        "cerebras": "qwen-3-235b-a22b-instruct-2507",
        "gemini": "gemini-3.1-flash-lite-preview",
    }
    model_name = args.model or _default_models.get(args.provider, cfg.model.name)

    # Skip extraction if raw triples already exist
    if raw_triples_path.exists():
        print(f"Raw triples already exist at {raw_triples_path}, skipping extraction.")
        raw_triples = load_triples(str(raw_triples_path))
    else:
        print("Loading post-cutoff articles...")
        post_df = load_fnspid(post_path)
        text_col = get_text_column(post_df, fnspid_cfg.text_columns)
        if ticker_filter:
            ticker_col = fnspid_cfg.ticker_column
            post_df = post_df[post_df[ticker_col].astype(str).str.upper().isin(ticker_filter)].copy()
            print(f"Filtering post-cutoff corpus to tickers: {sorted(ticker_filter)}")
        print(f"Post-cutoff articles: {len(post_df)}, text column: {text_col}")

        articles = post_df.to_dict("records")

        print(f"\nExtracting triples from {len(articles)} articles...")
        if args.provider == "local":
            print(f"\nLoading extraction model: {model_name}")
            model, tokenizer = load_model(model_name, cfg.model.dtype)
            raw_triples = extract_triples_batch(
                articles,
                model,
                tokenizer,
                text_column=text_col,
                id_column=None,
                batch_size=1,
                progress_path=str(progress_path),
                save_every=args.save_every,
            )
        elif args.provider == "cerebras":
            print(f"\nConfiguring Cerebras extractor: {model_name}")
            print(f"  TPM limit: {args.tpm_limit:,}, RPM limit: {args.rpm_limit:,}")
            api_func = _build_cerebras_api_func(model_name, tpm_limit=args.tpm_limit, rpm_limit=args.rpm_limit)
            print(f"Using async Cerebras extraction with concurrency={args.concurrency}")
            raw_triples = asyncio.run(
                extract_triples_api_async(
                    articles,
                    text_column=text_col,
                    id_column=None,
                    api_func_async=api_func,
                    concurrency=args.concurrency,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                    progress_path=str(progress_path),
                    save_every=args.save_every,
                )
            )
        elif args.provider == "gemini":
            print(f"\nConfiguring Gemini extractor: {model_name}")
            print(f"  TPM limit: {args.tpm_limit:,}, RPM limit: {args.rpm_limit:,}")
            api_func = _build_gemini_api_func(model_name, tpm_limit=args.tpm_limit, rpm_limit=args.rpm_limit)
            print(f"Using async Gemini extraction with concurrency={args.concurrency}")
            raw_triples = asyncio.run(
                extract_triples_api_async(
                    articles,
                    text_column=text_col,
                    id_column=None,
                    api_func_async=api_func,
                    concurrency=args.concurrency,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                    progress_path=str(progress_path),
                    save_every=args.save_every,
                )
            )
        else:
            print(f"\nConfiguring OpenAI extractor: {model_name}")
            api_func = _build_openai_api_func(model_name)
            print(f"Using async OpenAI extraction with concurrency={args.concurrency}")
            raw_triples = asyncio.run(
                extract_triples_api_async(
                    articles,
                    text_column=text_col,
                    id_column=None,
                    api_func_async=api_func,
                    concurrency=args.concurrency,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                    progress_path=str(progress_path),
                    save_every=args.save_every,
                )
            )
        print(f"Raw triples extracted: {len(raw_triples)}")

        save_triples(raw_triples, str(raw_triples_path))
        if progress_path.exists():
            processed_articles, _ = load_progress_jsonl(str(progress_path))
            print(
                f"Saved incremental progress log: {progress_path} ({len(processed_articles)} articles)"
            )

        if args.provider == "local":
            del model
            import torch

            torch.cuda.empty_cache()

    # Filter by cross-document agreement (1 in debug = keep all)
    min_agree = 1 if args.debug else triples_cfg.get("min_cross_doc_agreement", 2)
    print(f"\nFiltering by cross-doc agreement (min={min_agree})...")
    agreed = filter_cross_doc_agreement(raw_triples, min_agreement=min_agree)
    print(f"After cross-doc filter: {len(agreed)} triples")

    # Skip entity-name filtering for now. The extractor emits company names like
    # "Nvidia" while the corpus entity list is ticker-based ("NVDA"), so exact
    # matching drops otherwise usable debug triples.
    filtered = agreed
    print(f"Skipping entity filter. Keeping {len(filtered)} triples after agreement filter.")

    save_triples(filtered, str(filtered_triples_path))

    # Sample at target scales
    if args.debug:
        scales = triples_cfg.get("debug_scales", [50])
    else:
        scales = triples_cfg.get("scales", [1000, 3000])
    print(f"\nSampling at scales: {scales}")
    scaled = sample_at_scales(filtered, scales, seed=cfg.seed)
    save_scaled_triples(
        {scale: triples for scale, triples in scaled.items()},
        raw_dir if not suffix else raw_dir / suffix.lstrip("_"),
    )

    for scale, triples in scaled.items():
        print(f"  Scale {scale}: {len(triples)} triples saved")

    print("\nDone. Triple extraction complete.")


if __name__ == "__main__":
    main()
