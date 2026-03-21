"""Temporal split and subsample FNSPID corpus."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.fnspid import load_fnspid, temporal_split, subsample_stratified, get_text_column
from sot.utils.config import load_config


def main():
    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")

    data_root = Path(cfg.paths.data_root)
    raw_path = data_root / "fnspid" / "raw" / "fnspid.parquet"

    print("Loading FNSPID...")
    df = load_fnspid(raw_path)
    print(f"Total articles: {len(df)}")
    print(f"Columns: {list(df.columns)}")

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

    # Save
    corpus_dir = data_root / "fnspid" / "processed"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    pre_path = corpus_dir / "pre_cutoff.parquet"
    post_path = corpus_dir / "post_cutoff.parquet"

    pre.to_parquet(pre_path, index=False)
    post.to_parquet(post_path, index=False)

    print(f"\nSaved pre-cutoff corpus: {pre_path} ({len(pre)} articles)")
    print(f"Saved post-cutoff articles: {post_path} ({len(post)} articles)")


if __name__ == "__main__":
    main()
