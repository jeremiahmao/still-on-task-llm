"""Split filtered triples into disjoint per-round batches for sequential editing.

For sequential editing we need N non-overlapping subsets of the same size so
that method comparisons see fresh knowledge each round. Output:
  data/fnspid/triples/sequential/round_1.json
  data/fnspid/triples/sequential/round_2.json
  ...
Each round has `--per-round` unique triples (default 1000).
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-round", type=int, default=1000)
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None, help="Override cfg.seed.")
    parser.add_argument("--output-subdir", default="sequential")
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(cfg.paths.data_root)
    seed = args.seed if args.seed is not None else cfg.seed
    rng = random.Random(seed)

    filtered_path = data_root / "fnspid" / "triples" / "filtered_triples.json"
    if not filtered_path.exists():
        print(f"ERROR: {filtered_path} not found. Run 04_extract_triples.py first.")
        sys.exit(1)

    with open(filtered_path) as f:
        triples = json.load(f)
    print(f"Loaded {len(triples)} filtered triples")

    needed = args.per_round * args.n_rounds
    if needed > len(triples):
        print(
            f"ERROR: need {needed} unique triples for {args.n_rounds} rounds of "
            f"{args.per_round}, but only have {len(triples)}."
        )
        sys.exit(1)

    rng.shuffle(triples)
    out_dir = data_root / "fnspid" / "triples" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in range(1, args.n_rounds + 1):
        start = (r - 1) * args.per_round
        subset = triples[start : start + args.per_round]
        out_path = out_dir / f"round_{r}.json"
        with open(out_path, "w") as f:
            json.dump(subset, f, indent=2)
        print(f"  Round {r}: {len(subset)} triples -> {out_path}")

    meta = {
        "per_round": args.per_round,
        "n_rounds": args.n_rounds,
        "seed": seed,
        "source": str(filtered_path),
        "total_triples_available": len(triples),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
