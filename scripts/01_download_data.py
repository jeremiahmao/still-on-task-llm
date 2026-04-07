"""Download FNSPID and optionally FinQA datasets."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from omegaconf import OmegaConf

from sot.data.finqa import download_finqa
from sot.data.fnspid import download_fnspid, load_fnspid
from sot.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-finqa",
        action="store_true",
        help="Download only FNSPID. Useful for lightweight pipeline smoke tests.",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(cfg.paths.data_root)

    print("=== Downloading FNSPID ===")
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    fnspid_path = download_fnspid(fnspid_cfg, data_root)
    print(f"FNSPID saved to: {fnspid_path}")

    # Quick schema check
    df = (
        load_fnspid(fnspid_path)
        if fnspid_path.stat().st_size < 1e9
        else pd.read_csv(fnspid_path, nrows=5, low_memory=False)
    )
    print(f"Columns: {list(df.columns)}")
    print(f"Rows: {len(df) if len(df) > 5 else 'sampled 5'}")

    if args.skip_finqa:
        print("\nSkipping FinQA download (--skip-finqa).")
    else:
        print("\n=== Downloading FinQA ===")
        finqa_dir = download_finqa(data_root)
        print(f"FinQA dataset at: {finqa_dir}")


if __name__ == "__main__":
    main()
