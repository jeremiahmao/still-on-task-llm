"""Generate result tables from experiment outputs."""

import json
from pathlib import Path

import pandas as pd


def collect_results(output_root: str = "outputs") -> pd.DataFrame:
    """Collect all eval_results.json files into a single DataFrame."""
    output_root = Path(output_root)
    rows = []

    for run_dir in sorted(output_root.iterdir()):
        if not run_dir.is_dir():
            continue

        eval_path = run_dir / "eval_results.json"
        meta_path = run_dir / "metadata.json"

        if not eval_path.exists():
            continue

        with open(eval_path) as f:
            eval_results = json.load(f)

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        row = {
            "run_id": run_dir.name,
            "method": meta.get("method", ""),
            "task": meta.get("task", ""),
            "scale": meta.get("scale", 0),
            "gpu_hours": meta.get("gpu_hours", 0),
            "peak_memory_gb": meta.get("peak_memory_gb", 0),
        }

        # Task preservation
        pres = eval_results.get("task_preservation", {})
        row["recall_at_10"] = pres.get("mean", None)

        # Knowledge absorption
        absorb = eval_results.get("knowledge_absorption", {})
        row["fact_exact_match"] = absorb.get("exact_match", None)
        row["fact_mean_f1"] = absorb.get("mean_f1", None)

        # Generic forgetting
        forget = eval_results.get("generic_forgetting", {})
        row["finqa_accuracy"] = forget.get("execution_accuracy", None)

        # Locality
        loc = eval_results.get("locality", {})
        for stratum in ["same_entity", "same_sector", "other_sector", "overall"]:
            s = loc.get(stratum, {})
            row[f"locality_{stratum}"] = s.get("accuracy", None)

        rows.append(row)

    return pd.DataFrame(rows)


def generate_phase1_table(df: pd.DataFrame) -> str:
    """Generate Phase 1 results table (query decomposition)."""
    qd = df[df["task"] == "qd"].copy()
    if qd.empty:
        return "No Phase 1 results found."

    pivot = qd.pivot_table(
        index="method",
        columns="scale",
        values=["recall_at_10", "fact_exact_match", "gpu_hours"],
        aggfunc="first",
    )

    return pivot.to_markdown()


def main():
    df = collect_results()

    if df.empty:
        print("No results found in outputs/")
        return

    print("=== All Results ===")
    print(df.to_markdown(index=False))

    print("\n=== Phase 1 Table ===")
    print(generate_phase1_table(df))

    # Save
    df.to_csv("outputs/all_results.csv", index=False)
    print("\nSaved to outputs/all_results.csv")


if __name__ == "__main__":
    main()
