"""Compute per-method durability statistics from the sequential trajectory.

Reads `final_results/phase3_sequential_trajectory.csv` and emits
`final_results/phase3_durability.csv` with, per method:

  - rounds_with_data            number of rounds with non-null metrics
  - preservation_mean_1to10     mean preservation across all rounds (durability)
  - preservation_auc_trap       trapezoidal AUC normalised to [0, 9]
  - preservation_final_round    round-10 value for comparison
  - absorption_mean_1to10       mean absorption across rounds (growth trajectory)
  - absorption_auc_trap         trapezoidal AUC
  - absorption_final_round      round-10 value
  - locality_mean_1to10         mean locality across rounds
  - locality_auc_trap           trapezoidal AUC
  - locality_final_round        round-10 value

Run: python scripts/21_compute_durability.py
"""

import csv
from collections import defaultdict
from pathlib import Path


TRAJECTORY_CSV = Path("final_results/phase3_sequential_trajectory.csv")
OUTPUT_CSV = Path("final_results/phase3_durability.csv")
METHODS = [
    "naive_sft",
    "kl_reg_sft",
    "copr",
    "copr_gold_injection",
    "copr_gold_injection_anchored",
    "copr_anchored",
]


def _to_float(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def trapezoidal_auc(xs: list[int], ys: list[float]) -> float:
    """Trapezoidal AUC normalised by the x-span (i.e., mean value estimate)."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    pts.sort()
    if len(pts) < 2:
        return 0.0 if not pts else pts[0][1]
    total = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        total += 0.5 * (y1 + y2) * (x2 - x1)
    return total / (pts[-1][0] - pts[0][0])


def main() -> None:
    if not TRAJECTORY_CSV.exists():
        raise SystemExit(f"missing {TRAJECTORY_CSV}; run scripts/20_snapshot_results.py first")

    rows_by_method: dict[str, list[dict]] = defaultdict(list)
    with open(TRAJECTORY_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_method[row["method"]].append(row)

    out_rows = []
    for m in METHODS:
        rounds = rows_by_method.get(m, [])
        if not rounds:
            continue
        rs = sorted({int(r["round"]) for r in rounds})
        pres = [_to_float(r.get("preservation_mean")) for r in sorted(rounds, key=lambda x: int(x["round"]))]
        abs_f1 = [_to_float(r.get("abs_mean_f1")) for r in sorted(rounds, key=lambda x: int(x["round"]))]
        loc = [_to_float(r.get("loc_overall_f1")) for r in sorted(rounds, key=lambda x: int(x["round"]))]

        def _mean(xs):
            v = [x for x in xs if x is not None]
            return sum(v) / len(v) if v else None

        def _final(xs):
            non_null = [x for x in xs if x is not None]
            return non_null[-1] if non_null else None

        out_rows.append(
            {
                "method": m,
                "rounds_with_data": sum(1 for x in pres if x is not None),
                "preservation_mean_1to10": _mean(pres),
                "preservation_auc_trap": trapezoidal_auc(rs, pres),
                "preservation_final_round": _final(pres),
                "absorption_mean_1to10": _mean(abs_f1),
                "absorption_auc_trap": trapezoidal_auc(rs, abs_f1),
                "absorption_final_round": _final(abs_f1),
                "locality_mean_1to10": _mean(loc),
                "locality_auc_trap": trapezoidal_auc(rs, loc),
                "locality_final_round": _final(loc),
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {OUTPUT_CSV} ({len(out_rows)} methods)")

    print()
    print(f"{'method':<34}{'pres_mean':>12}{'pres_final':>12}{'abs_mean':>12}{'abs_final':>12}{'loc_mean':>12}{'loc_final':>12}")
    for r in out_rows:
        print(
            f"{r['method']:<34}"
            f"{r['preservation_mean_1to10']:>12.3f}"
            f"{r['preservation_final_round']:>12.3f}"
            f"{r['absorption_mean_1to10']:>12.3f}"
            f"{r['absorption_final_round']:>12.3f}"
            f"{r['locality_mean_1to10']:>12.3f}"
            f"{r['locality_final_round']:>12.3f}"
        )


if __name__ == "__main__":
    main()
