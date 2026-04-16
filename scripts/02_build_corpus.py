"""Temporal split and subsample FNSPID corpus."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.fnspid import get_text_column, load_fnspid, subsample_stratified, temporal_split
from sot.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug ticker filtering from configs/data/fnspid.yaml.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Optional fast-path: load only the first N rows of the raw CSV before splitting.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional suffix for output parquet names, e.g. '_sample'.",
    )
    args = parser.parse_args()

    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")

    data_root = Path(cfg.paths.data_root)

    # Choose which raw CSV to consume for corpus building. This can point at
    # the full download or a filtered local derivative such as the 2019+ file.
    input_news_file = fnspid_cfg.get("input_news_file", fnspid_cfg.news_file)
    raw_path = data_root / "fnspid" / "raw" / input_news_file
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run 01_download_data.py first.")
        sys.exit(1)

    print(f"Loading FNSPID from {raw_path}...")
    if args.sample_rows:
        print(f"Using sample mode: first {args.sample_rows} rows")
        df = load_fnspid(raw_path).head(args.sample_rows)
    else:
        df = load_fnspid(raw_path)
    print(f"Total articles: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    suffix = args.output_suffix
    if args.debug:
        ticker_col = fnspid_cfg.ticker_column
        debug_tickers = {ticker.upper() for ticker in fnspid_cfg.debug.tickers}
        print(f"Applying debug ticker filter: {sorted(debug_tickers)}")
        df = df[df[ticker_col].astype(str).str.upper().isin(debug_tickers)].copy()
        suffix = suffix or fnspid_cfg.debug.output_suffix
        print(f"After debug filter: {len(df)}")

    # Verify text column exists
    text_col = get_text_column(df, fnspid_cfg.text_columns)
    print(f"Using text column: {text_col}")

    # Temporal split
    print(f"\nSplitting at cutoff: {fnspid_cfg.cutoff_date}")
    pre, post = temporal_split(df, fnspid_cfg.date_column, fnspid_cfg.cutoff_date)
    print(f"Pre-cutoff: {len(pre)} articles")
    print(f"Post-cutoff: {len(post)} articles")

    # Subsample pre-cutoff
    n = fnspid_cfg.subsample_pre_cutoff
    if n and n < len(pre):
        print(f"\nSubsampling pre-cutoff to {n} (stratified by ticker + year)...")
        pre = subsample_stratified(pre, fnspid_cfg.ticker_column, n, seed=cfg.seed)
        print(f"Subsampled: {len(pre)} articles")

    # Subsample post-cutoff
    n_post = fnspid_cfg.get("subsample_post_cutoff", None)
    if n_post and n_post < len(post):
        print(f"\nSubsampling post-cutoff to {n_post} (stratified by ticker + year)...")
        post = subsample_stratified(post, fnspid_cfg.ticker_column, n_post, seed=cfg.seed)
        print(f"Subsampled: {len(post)} articles")

    # Save
    corpus_dir = data_root / "fnspid" / "processed"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    pre_path = corpus_dir / f"pre_cutoff{suffix}.parquet"
    post_path = corpus_dir / f"post_cutoff{suffix}.parquet"

    pre.to_parquet(pre_path, index=False)
    post.to_parquet(post_path, index=False)

    print(f"\nSaved pre-cutoff corpus: {pre_path} ({len(pre)} articles)")
    print(f"Saved post-cutoff articles: {post_path} ({len(post)} articles)")


if __name__ == "__main__":
    main()
