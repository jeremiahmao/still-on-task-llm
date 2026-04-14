"""Build stratified locality test facts from extracted triples."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from omegaconf import OmegaConf

from sot.data.triple_extract import load_triples
from sot.data.triple_render import load_templates_from_config
from sot.eval.locality import prepare_locality_facts
from sot.utils.config import load_config


def build_sector_map(df: pd.DataFrame, ticker_col: str) -> dict[str, str]:
    """Build a simple ticker -> sector mapping.

    Uses the first character of the ticker as a rough proxy for sector grouping
    when no GICS data is available. In production, you'd use a proper mapping.
    """
    tickers = df[ticker_col].dropna().unique()
    # Simple heuristic: group by first letter as a proxy
    # Replace this with a proper GICS mapping if available
    return {str(t): f"sector_{str(t)[0].upper()}" for t in tickers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Use debug suffix from fnspid.yaml")
    args = parser.parse_args()

    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    triples_cfg = OmegaConf.load("configs/data/triples.yaml")

    data_root = Path(cfg.paths.data_root)
    triples_dir = data_root / "fnspid" / "triples"

    suffix = fnspid_cfg.debug.output_suffix if args.debug else ""

    # Load all filtered triples (superset)
    filtered_path = triples_dir / f"filtered_triples{suffix}.json"
    if not filtered_path.exists():
        print(f"ERROR: {filtered_path} not found. Run 04_extract_triples.py first.")
        sys.exit(1)

    all_triples = load_triples(str(filtered_path))
    print(f"All filtered triples: {len(all_triples)}")

    # Load the edited triples at 1K scale (locality is relative to what was edited)
    scaled_dir = triples_dir / suffix.lstrip("_") if suffix else triples_dir

    # Find the best available scaled triples file
    if args.debug:
        candidate_scales = triples_cfg.get("debug_scales", [50])
    else:
        candidate_scales = triples_cfg.get("scales", [200, 1000, 3000])

    edited_path = None
    for s in candidate_scales:
        candidate = scaled_dir / f"triples_{s}.json"
        if candidate.exists():
            edited_path = candidate
            break

    if edited_path is None:
        print(f"ERROR: No scaled triples found in {scaled_dir}. Run 04_extract_triples.py first.")
        sys.exit(1)

    edited_triples = load_triples(str(edited_path))
    print(f"Edited triples ({len(edited_triples)}): {edited_path.name}")

    # Build sector map from pre-cutoff corpus
    pre_path = data_root / "fnspid" / "processed" / f"pre_cutoff{suffix}.parquet"
    pre_df = pd.read_parquet(pre_path)
    sector_map = build_sector_map(pre_df, fnspid_cfg.ticker_column)
    print(f"Sector map: {len(sector_map)} tickers")

    # Load templates
    templates = load_templates_from_config(triples_cfg)

    # Build locality facts
    print("\nBuilding stratified locality facts...")
    locality_facts = prepare_locality_facts(
        all_triples,
        edited_triples,
        sector_map,
        templates or None,
    )

    # Count by stratum
    from collections import Counter

    strata = Counter(f["stratum"] for f in locality_facts)
    for stratum, count in sorted(strata.items()):
        print(f"  {stratum}: {count}")
    print(f"  Total: {len(locality_facts)}")

    # Save
    output_path = data_root / "fnspid" / f"locality_facts{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(locality_facts, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
