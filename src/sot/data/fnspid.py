"""FNSPID dataset: download, temporal split, and subsample."""

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig


def download_fnspid(cfg: DictConfig, data_root: str | Path) -> Path:
    """Download FNSPID news CSV from HuggingFace.

    Returns path to the downloaded CSV file.
    """
    data_root = Path(data_root)
    raw_dir = data_root / "fnspid" / "raw"

    news_file = cfg.get("news_file", "Stock_news/nasdaq_exteral_data.csv")
    csv_path = raw_dir / "Stock_news" / Path(news_file).name

    if csv_path.exists():
        return csv_path

    hf_hub_download(
        cfg.source,
        news_file,
        repo_type="dataset",
        local_dir=str(raw_dir),
    )
    return csv_path


def load_fnspid(path: str | Path) -> pd.DataFrame:
    """Load FNSPID CSV into a DataFrame."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    cutoff_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split FNSPID into pre-cutoff and post-cutoff DataFrames."""
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce", utc=True)
    df = df.dropna(subset=[date_column])

    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    pre = df[df[date_column] < cutoff].copy()
    post = df[df[date_column] >= cutoff].copy()
    return pre, post


def subsample_stratified(
    df: pd.DataFrame,
    ticker_column: str,
    n: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample articles, stratified by ticker to preserve entity coverage.

    Also stratifies by year to preserve temporal distribution.
    """
    if len(df) <= n:
        return df

    df = df.copy()
    df["_year"] = df.iloc[:, 0]  # Will be overridden below
    # Use the date column (first datetime column found).
    # Must handle both naive datetime64 and timezone-aware datetime64[ns, UTC]
    # (temporal_split converts to UTC-aware, which select_dtypes(include=["datetime64"]) misses).
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if dt_cols:
        df["_year"] = df[dt_cols[0]].dt.year

    # Group by ticker + year, sample proportionally
    df["_group"] = df[ticker_column].astype(str) + "_" + df["_year"].astype(str)
    sampled = df.groupby("_group", group_keys=False).apply(
        lambda g: (
            g.sample(
                n=max(1, int(len(g) / len(df) * n)),
                random_state=seed,
                replace=False,
            )
            if len(g) > 0
            else g
        )
    )

    # Trim to exact n if oversampled
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)

    return sampled.drop(columns=["_year", "_group"]).reset_index(drop=True)


def get_text_column(df: pd.DataFrame, text_columns: list[str]) -> str:
    """Find the best available text column from a priority list."""
    for col in text_columns:
        if col in df.columns:
            return col
    raise ValueError(f"None of {text_columns} found in DataFrame columns: {list(df.columns)}")
