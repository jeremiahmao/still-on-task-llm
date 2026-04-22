"""Snapshot every artifact we have on disk — metrics, configs, metadata, data
inventory — into clean CSVs + a README under final_results/. Read-only over
outputs/ and data/.

Emits (final_results/):
  README.md                                Per-file description + method index.
  methods_index.csv                        Static table of methods + their configs + short description.
  data_inventory.csv                       Triple counts, QD splits, FAISS corpus sizes.
  run_metadata.csv                         Per-run: method, scale/round, triples_path, base_model, starting_checkpoint, gpu_hours, elapsed.
  run_configs.csv                          Per-run: flattened OmegaConf (method hyperparams).
  phase1_batch_scale1000.csv               Batch regime, 1K edits.
  phase2_batch_scale3000.csv               Batch regime, 3K edits.
  phase3_sequential_trajectory.csv         Per-round metrics per method.
  phase3_sequential_final.csv              Final-round row per method.
  phase4_compositional.csv                 Compositional + temporal_contrast.
  phase6_lora_deltas.csv                   Mechanistic per-method summary.
  phase6_lora_subspace_overlap.csv         Pairwise principal-angle cosines.
  baseline_no_update.csv                   Task-tuned model, no updates.

Run: python scripts/20_snapshot_results.py
"""

import csv
import json
import re
from pathlib import Path


OUTPUTS = Path("outputs")
DATA = Path("data")
CONFIGS = Path("configs")
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


def _load_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def _flatten(d: dict, prefix: str = "", out: dict | None = None) -> dict:
    """Flatten nested dict keys with dot-notation for CSV columns."""
    out = {} if out is None else out
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            _flatten(v, key, out)
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


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


METHOD_DESCRIPTIONS = {
    "naive_sft": "SFT on fact QA pairs; no task anchor, no replay.",
    "kl_reg_sft": "SFT + KL divergence vs pre-update model on task-replay batches.",
    "copr": "Paper-faithful COPR: MSE fit on K=8 ranked candidates + gold-NLL SFT anchor + KL replay.",
    "copr_gold_injection": "Our variant: COPR with gold answer injected into the candidate set; anchor OFF.",
    "copr_gold_injection_anchored": "Our variant: COPR with gold injection AND SFT anchor ON.",
    "copr_anchored": "Our variant: COPR with task-replay-normalized reference mixed into log pi_ref (per-candidate).",
    "no_update_baseline": "Task-tuned model (qd_sft/final), LoRA merged, no knowledge updates applied.",
}

CONFIG_FILE_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "copr": "configs/update/copr.yaml",
    "copr_gold_injection": "configs/update/copr_gold_injection.yaml",
    "copr_gold_injection_anchored": "configs/update/copr_gold_injection_anchored.yaml",
    "copr_anchored": "configs/update/copr_anchored.yaml",
}


def snapshot_methods_index() -> list[dict]:
    rows = []
    for m in METHODS:
        cfg_path = Path(CONFIG_FILE_MAP[m])
        cfg = _load_yaml(cfg_path) or {}
        row: dict = {
            "method": m,
            "description": METHOD_DESCRIPTIONS.get(m, ""),
            "config_file": str(cfg_path),
        }
        row.update({f"cfg.{k}": v for k, v in _flatten(cfg).items()})
        rows.append(row)
    # Add the baseline as a pseudo-method row.
    rows.append(
        {
            "method": "no_update_baseline",
            "description": METHOD_DESCRIPTIONS["no_update_baseline"],
            "config_file": "",
        }
    )
    return rows


def _count_json_array(path: Path) -> int | None:
    d = _load_json(path)
    if isinstance(d, list):
        return len(d)
    return None


def snapshot_data_inventory() -> list[dict]:
    rows: list[dict] = []

    def add(kind: str, path: Path, count=None, notes: str = ""):
        rows.append(
            {
                "kind": kind,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
                "count": count,
                "notes": notes,
            }
        )

    # Filtered triples (source for batch scales)
    filt = DATA / "fnspid" / "triples" / "filtered_triples.json"
    add("triples_filtered", filt, _count_json_array(filt), "all filtered fact triples")

    # Batch-scale triple files
    for s in (50, 200, 1000, 3000):
        p = DATA / "fnspid" / "triples" / f"triples_{s}.json"
        add(f"triples_scale{s}", p, _count_json_array(p), f"batch-scale {s} sample")

    # Sequential round triples
    seq_dir = DATA / "fnspid" / "triples" / "sequential"
    for p in sorted(seq_dir.glob("round_*.json")):
        add(f"triples_sequential_{p.stem}", p, _count_json_array(p), "disjoint per-round batch")
    seq_meta = seq_dir / "metadata.json"
    if seq_meta.exists():
        meta = _load_json(seq_meta) or {}
        rows.append(
            {
                "kind": "sequential_metadata",
                "path": str(seq_meta),
                "exists": True,
                "size_bytes": seq_meta.stat().st_size,
                "count": None,
                "notes": (
                    f"per_round={meta.get('per_round')} n_rounds={meta.get('n_rounds')} "
                    f"seed={meta.get('seed')} total_available={meta.get('total_triples_available')}"
                ),
            }
        )

    # QD temporal train/test + paired
    for name in ("train.json", "test.json", "post_test.json", "paired_examples.json"):
        p = DATA / "qd_temporal" / name
        add(f"qd_temporal_{name}", p, _count_json_array(p))

    # Compositional probes
    p = DATA / "fnspid" / "compositional" / "probes.json"
    add("compositional_probes", p, _count_json_array(p), "2-hop probes for Phase 4")

    # Locality facts
    p = DATA / "fnspid" / "locality_facts.json"
    add("locality_facts", p, _count_json_array(p), "unrelated facts for locality eval")

    # FAISS index binaries
    for name in ("corpus.faiss", "corpus_post.faiss", "doc_ids.npy", "chunk_to_article.npy"):
        p = DATA / "fnspid" / "index" / name
        add(f"faiss_{name}", p)

    return rows


def _iter_all_run_dirs():
    """Yield every output dir that looks like a run dir (has eval_results OR metadata)."""
    for run_dir in OUTPUTS.iterdir():
        if not run_dir.is_dir():
            continue
        if run_dir.name in ("mechanistic", "sequential", "_logs"):
            continue
        if (run_dir / "eval_results.json").exists() or (run_dir / "metadata.json").exists() or (run_dir / "config.yaml").exists():
            yield run_dir


def _classify_run_dir(run_dir: Path) -> dict:
    """Pull phase / method / scale / round out of a directory name."""
    name = run_dir.name
    info = {"run_id": name}
    m = re.match(r"^seq_(?P<method>[a-z_]+?)_round_(?P<round>\d+)_qd_scale(?P<scale>\d+)$", name)
    if m:
        info["phase"] = "phase3_sequential"
        info["method"] = m.group("method")
        info["round"] = int(m.group("round"))
        info["scale"] = int(m.group("scale"))
        return info
    m = re.match(r"^(?P<method>[a-z_]+?)_qd_scale(?P<scale>\d+)$", name)
    if m:
        scale = int(m.group("scale"))
        info["phase"] = "phase2_batch" if scale == 3000 else ("phase1_batch" if scale == 1000 else "batch_other")
        info["method"] = m.group("method")
        info["scale"] = scale
        return info
    if name == "no_update_qd":
        info["phase"] = "baseline"
        info["method"] = "no_update_baseline"
        return info
    info["phase"] = "other"
    return info


def snapshot_run_metadata() -> list[dict]:
    rows = []
    for run_dir in _iter_all_run_dirs():
        meta = _load_json(run_dir / "metadata.json") or {}
        info = _classify_run_dir(run_dir)
        row = {**info, **meta}
        rows.append(row)
    rows.sort(key=lambda r: (r.get("phase", ""), r.get("method", ""), r.get("scale", 0), r.get("round", 0)))
    return rows


def snapshot_run_configs() -> list[dict]:
    rows = []
    for run_dir in _iter_all_run_dirs():
        cfg = _load_yaml(run_dir / "config.yaml") or {}
        if not cfg:
            continue
        info = _classify_run_dir(run_dir)
        row = {**info}
        row.update({f"cfg.{k}": v for k, v in _flatten(cfg).items()})
        rows.append(row)
    rows.sort(key=lambda r: (r.get("phase", ""), r.get("method", ""), r.get("scale", 0), r.get("round", 0)))
    return rows


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


README_TEMPLATE = """# Results Snapshot

Generated by `scripts/20_snapshot_results.py`. Every CSV in this folder is
derived from the raw JSON artifacts in `outputs/` and `data/` at the time the
script was run. Re-run the script any time to refresh the snapshot.

## CSVs

| File | What's in it |
|---|---|
| `methods_index.csv` | Static table: every method's short description + flattened config hyperparameters. |
| `data_inventory.csv` | Triple counts, QD data splits, compositional probe counts, FAISS index file sizes. |
| `run_metadata.csv` | One row per run directory under `outputs/` — method, scale, round, base_model, starting_checkpoint, gpu_hours, elapsed. |
| `run_configs.csv` | One row per run with its flattened `config.yaml` (reproducibility). |
| `phase1_batch_scale1000.csv` | Batch regime at 1K edits — preservation, absorption, locality per method. |
| `phase2_batch_scale3000.csv` | Batch regime at 3K edits. |
| `phase3_sequential_trajectory.csv` | Per-round metrics for all 6 methods across 10 rounds of sequential editing. |
| `phase3_sequential_final.csv` | Latest round per method (final-round table for the paper). |
| `phase4_compositional.csv` | Compositional (2-hop) + temporal-contrast metrics per round-10 checkpoint + baseline. |
| `phase6_lora_deltas.csv` | Mechanistic probe — Frobenius norm, stable/effective rank, per-module magnitudes. |
| `phase6_lora_subspace_overlap.csv` | Pairwise principal-angle cosines between methods' input subspaces. |
| `baseline_no_update.csv` | Pre-update task-tuned model evaluated on every metric. |

## Methods in scope

{methods_list}

## Pipeline overview

1. **Phase 0** — Data: FNSPID corpus -> filtered fact triples -> QD temporal splits -> FAISS retrieval index.
2. **Phase 1** — Batch edits at 1K triples; 5 methods + no-update baseline.
3. **Phase 2** — Batch edits at 3K triples; same method set.
4. **Phase 3** — Sequential edits (10 rounds x 200 triples) with chained checkpoints; 6 methods.
5. **Phase 4** — Rich eval (compositional 2-hop + temporal contrast) on round-10 checkpoints.
6. **Phase 5** — `copr_anchored` novel method (appears in Phase 3; deferred in Phases 1/2 except where noted).
7. **Phase 6** — LoRA weight-delta analysis across methods.

## Base model

- Student: Qwen3-4B-Instruct-2507 (bf16).
- Task-tuning: LoRA r=32 / alpha=64 on query-decomposition SFT (pre-cutoff split).
- Update-time LoRA: r=16 / alpha=32 applied on top of the merged task-tuned model.

## Regenerate

```
python scripts/20_snapshot_results.py
```

Read-only over `outputs/` and `data/`. Safe to re-run any time.
"""


def write_readme() -> None:
    methods_lines = []
    for m in METHODS + ["no_update_baseline"]:
        methods_lines.append(f"- **{m}**: {METHOD_DESCRIPTIONS.get(m, '')}")
    (FINAL / "README.md").write_text(
        README_TEMPLATE.format(methods_list="\n".join(methods_lines))
    )
    print(f"  wrote {FINAL / 'README.md'}")


def main() -> None:
    print(f"Writing CSVs to {FINAL.resolve()}")

    _write_csv(FINAL / "methods_index.csv", snapshot_methods_index())
    _write_csv(FINAL / "data_inventory.csv", snapshot_data_inventory())
    _write_csv(FINAL / "run_metadata.csv", snapshot_run_metadata())
    _write_csv(FINAL / "run_configs.csv", snapshot_run_configs())

    _write_csv(FINAL / "phase1_batch_scale1000.csv", snapshot_batch_scale(1000))
    _write_csv(FINAL / "phase2_batch_scale3000.csv", snapshot_batch_scale(3000))
    _write_csv(FINAL / "phase3_sequential_trajectory.csv", snapshot_seq_trajectory())
    _write_csv(FINAL / "phase3_sequential_final.csv", snapshot_seq_final())
    _write_csv(FINAL / "phase4_compositional.csv", snapshot_phase4())
    _write_csv(FINAL / "baseline_no_update.csv", snapshot_baseline())

    per_method, overlap = snapshot_phase6()
    _write_csv(FINAL / "phase6_lora_deltas.csv", per_method)
    _write_csv(FINAL / "phase6_lora_subspace_overlap.csv", overlap)

    write_readme()

    print("\nDone. Contents:")
    for p in sorted(FINAL.iterdir()):
        if p.suffix == ".csv":
            with open(p) as f:
                n = sum(1 for _ in f) - 1
            print(f"  {p.name}: {n} data rows")
        else:
            print(f"  {p.name}")


if __name__ == "__main__":
    main()
