"""Snapshot every eval_results.json / trajectory.json / lora_analysis.json we
have on disk into clean CSVs under final_results/. Read-only over outputs/.

Emits:
  final_results/phase1_batch_scale1000.csv       batch regime, 1K edits
  final_results/phase2_batch_scale3000.csv       batch regime, 3K edits
  final_results/phase3_sequential_trajectory.csv per-round metrics per method
  final_results/phase3_sequential_final.csv      round 10 row per method
  final_results/phase4_compositional.csv         compositional + temporal_contrast
  final_results/phase6_lora_deltas.csv           mechanistic per-method summary
  final_results/phase6_lora_subspace_overlap.csv pairwise cosine overlap
  final_results/baseline_no_update.csv           task-tuned model, no updates

Run: python scripts/20_snapshot_results.py
"""

import csv
import json
import re
from pathlib import Path


OUTPUTS = Path("outputs")
FINAL = Path("final_results")
FINAL.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _flatten_eval(results: dict) -> dict:
    """Pull the fields we care about out of a single eval_results.json."""
    row: dict = {}
    tp = results.get("task_preservation") or {}
    if tp:
        row["preservation_mean"] = tp.get("mean")
        row["preservation_std"] = tp.get("std")
        row["preservation_n"] = tp.get("n_queries")
    ka = results.get("knowledge_absorption") or {}
    if ka:
        row["abs_exact_match"] = ka.get("exact_match")
        row["abs_mean_f1"] = ka.get("mean_f1")
        row["abs_contains"] = ka.get("contains")
        row["abs_fact_mean_f1"] = ka.get("fact_mean_f1")
        row["abs_fact_worst_f1"] = ka.get("fact_worst_f1")
        row["abs_contains_any_phrasing"] = ka.get("contains_any_phrasing")
        row["abs_contains_all_phrasings"] = ka.get("contains_all_phrasings")
        row["abs_n_facts"] = ka.get("n_facts")
        row["abs_n_probes"] = ka.get("n_probes")
    loc = results.get("locality") or {}
    for stratum in ("same_entity", "other_sector", "same_sector", "overall"):
        s = loc.get(stratum) or {}
        if isinstance(s, dict):
            row[f"loc_{stratum}_f1"] = s.get("f1")
            row[f"loc_{stratum}_acc"] = s.get("accuracy")
            row[f"loc_{stratum}_n"] = s.get("n")
    comp = results.get("compositional") or {}
    if comp:
        row["comp_exact_match"] = comp.get("exact_match")
        row["comp_contains_final_answer"] = comp.get("contains_final_answer")
        row["comp_contains_bridging_entity"] = comp.get("contains_bridging_entity")
        row["comp_token_f1"] = comp.get("token_f1")
        row["comp_n_probes"] = comp.get("n_probes")
    tc = results.get("temporal_contrast") or {}
    if tc:
        row["tc_pre_alignment_f1"] = tc.get("pre_alignment_f1")
        row["tc_post_alignment_f1"] = tc.get("post_alignment_f1")
        row["tc_shift_score"] = tc.get("shift_score")
        row["tc_n_probes"] = tc.get("n_probes")
        row["tc_n_skipped"] = tc.get("n_skipped")
    post = results.get("post_task_preservation") or {}
    if post:
        row["post_preservation_mean"] = post.get("mean")
        row["post_preservation_std"] = post.get("std")
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  skip {path.name}: no rows")
        return
    # Union of keys across all rows, preserving insertion order seen.
    fieldnames: list[str] = []
    seen: set = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path} ({len(rows)} rows)")


METHODS = [
    "naive_sft",
    "kl_reg_sft",
    "copr",
    "copr_gold_injection",
    "copr_gold_injection_anchored",
    "copr_anchored",
]


def snapshot_batch_scale(scale: int) -> list[dict]:
    rows = []
    for m in METHODS:
        run_dir = OUTPUTS / f"{m}_qd_scale{scale}"
        results = _load_json(run_dir / "eval_results.json")
        meta = _load_json(run_dir / "metadata.json") or {}
        if results is None:
            continue
        row = {"method": m, "scale": scale}
        row.update(_flatten_eval(results))
        row["gpu_hours"] = meta.get("gpu_hours")
        row["peak_memory_gb"] = meta.get("peak_memory_gb")
        row["elapsed_seconds"] = meta.get("elapsed_seconds")
        rows.append(row)
    return rows


def snapshot_seq_trajectory() -> list[dict]:
    rows = []
    round_re = re.compile(r"_round_(\d+)_qd_scale")
    for m in METHODS:
        for round_dir in sorted(OUTPUTS.glob(f"seq_{m}_round_*_qd_scale*")):
            match = round_re.search(round_dir.name)
            if not match:
                continue
            k = int(match.group(1))
            scale_match = re.search(r"scale(\d+)", round_dir.name)
            scale = int(scale_match.group(1)) if scale_match else None
            results = _load_json(round_dir / "eval_results.json")
            if results is None:
                continue
            row = {
                "method": m,
                "round": k,
                "scale": scale,
                "run_id": round_dir.name,
            }
            row.update(_flatten_eval(results))
            rows.append(row)
    rows.sort(key=lambda r: (r["method"], r["round"]))
    return rows


def snapshot_seq_final() -> list[dict]:
    """Round 10 (or latest available round) per method."""
    traj = snapshot_seq_trajectory()
    latest: dict[str, dict] = {}
    for r in traj:
        if r["method"] not in latest or r["round"] > latest[r["method"]]["round"]:
            latest[r["method"]] = r
    # Return in METHODS order for a stable paper table.
    return [latest[m] for m in METHODS if m in latest]


def snapshot_phase4() -> list[dict]:
    """Same as seq_final but projects only the Phase 4 metric columns."""
    rows = []
    for m in METHODS:
        for round_dir in sorted(OUTPUTS.glob(f"seq_{m}_round_*_qd_scale*")):
            round_match = re.search(r"_round_(\d+)_", round_dir.name)
            if not round_match or int(round_match.group(1)) != 10:
                continue
            results = _load_json(round_dir / "eval_results.json")
            if results is None:
                continue
            if not (results.get("compositional") or results.get("temporal_contrast")):
                continue
            row = {"method": m, "round": 10, "run_id": round_dir.name}
            eval_row = _flatten_eval(results)
            for k, v in eval_row.items():
                if k.startswith(("comp_", "tc_")):
                    row[k] = v
            rows.append(row)

    # Also grab the no-update baseline if it has compositional
    baseline_results = _load_json(OUTPUTS / "no_update_qd" / "eval_results.json")
    if baseline_results is not None and baseline_results.get("compositional"):
        row = {"method": "no_update_baseline", "round": None, "run_id": "no_update_qd"}
        eval_row = _flatten_eval(baseline_results)
        for k, v in eval_row.items():
            if k.startswith(("comp_", "tc_")):
                row[k] = v
        rows.append(row)
    return rows


def snapshot_baseline() -> list[dict]:
    run_dir = OUTPUTS / "no_update_qd"
    results = _load_json(run_dir / "eval_results.json")
    if results is None:
        return []
    row = {"method": "no_update_baseline", "run_id": run_dir.name}
    row.update(_flatten_eval(results))
    return [row]


def snapshot_phase6() -> tuple[list[dict], list[dict]]:
    path = OUTPUTS / "mechanistic" / "lora_analysis.json"
    data = _load_json(path)
    if data is None:
        return [], []
    per_method_rows = []
    for name, s in (data.get("per_method") or {}).items():
        row = {
            "method": name,
            "adapter_dir": s.get("adapter_dir"),
            "lora_scaling": s.get("lora_scaling"),
            "n_pairs": s.get("n_pairs"),
            "total_fro_norm": s.get("total_fro_norm"),
            "mean_stable_rank": s.get("mean_stable_rank"),
            "mean_effective_rank": s.get("mean_effective_rank"),
        }
        for module, v in (s.get("per_module_mean_fro") or {}).items():
            row[f"module_{module}_mean_fro"] = v
        per_method_rows.append(row)

    overlap_rows = []
    for pair, v in (data.get("pairwise_subspace_overlap") or {}).items():
        methods = pair.split("__vs__")
        overlap_rows.append(
            {
                "method_a": methods[0] if len(methods) > 0 else pair,
                "method_b": methods[1] if len(methods) > 1 else "",
                "n_shared_sites": v.get("n_shared_sites"),
                "mean_cos_first_angle": v.get("mean_cos_first_angle"),
                "mean_cos_all_angles": v.get("mean_cos_all_angles"),
            }
        )
    return per_method_rows, overlap_rows


def main() -> None:
    print(f"Writing CSVs to {FINAL.resolve()}")

    _write_csv(FINAL / "phase1_batch_scale1000.csv", snapshot_batch_scale(1000))
    _write_csv(FINAL / "phase2_batch_scale3000.csv", snapshot_batch_scale(3000))
    _write_csv(FINAL / "phase3_sequential_trajectory.csv", snapshot_seq_trajectory())
    _write_csv(FINAL / "phase3_sequential_final.csv", snapshot_seq_final())
    _write_csv(FINAL / "phase4_compositional.csv", snapshot_phase4())
    _write_csv(FINAL / "baseline_no_update.csv", snapshot_baseline())

    per_method, overlap = snapshot_phase6()
    _write_csv(FINAL / "phase6_lora_deltas.csv", per_method)
    _write_csv(FINAL / "phase6_lora_subspace_overlap.csv", overlap)

    print("\nDone. Contents:")
    for p in sorted(FINAL.glob("*.csv")):
        with open(p) as f:
            n = sum(1 for _ in f) - 1
        print(f"  {p.name}: {n} data rows")


if __name__ == "__main__":
    main()
