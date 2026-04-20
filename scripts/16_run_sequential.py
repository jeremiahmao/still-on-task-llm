"""Phase 3: Sequential editing over N rounds with chained checkpoints.

For each configured method we run N rounds of (update -> evaluate). Each round
injects a disjoint batch of PER_ROUND triples, starting from the previous
round's merged output. This is the regime where COPR's continual-learning
motivation should matter.

Preconditions:
  - scripts/15_prepare_sequential_triples.py has written
    data/fnspid/triples/sequential/round_{1..N}.json
  - checkpoints/qd_sft/final exists (round 1 starting point)

Outputs per method:
  outputs/seq_{method}_round_{k}_qd_scale{PER_ROUND}/            model + eval
  outputs/sequential/{method}/trajectory.json                    per-round metrics
  outputs/sequential/summary.json                                cross-method summary
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

METHODS = [
    "naive_sft",
    "kl_reg_sft",
    "copr",
    "copr_gold_injection",
    "copr_gold_injection_anchored",
    "copr_anchored",
]

CONFIG_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "copr": "configs/update/copr.yaml",
    "copr_gold_injection": "configs/update/copr_gold_injection.yaml",
    "copr_gold_injection_anchored": "configs/update/copr_gold_injection_anchored.yaml",
    "copr_anchored": "configs/update/copr_anchored.yaml",
}

N_ROUNDS = 10
PER_ROUND = 200
TASK = "qd"
LOCALITY_SUBSAMPLE = 2000


def _run(cmd: list[str]) -> int:
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def _extract_round_metrics(eval_path: Path) -> dict:
    """Flatten eval_results.json into a single trajectory row."""
    if not eval_path.exists():
        return {}
    with open(eval_path) as f:
        results = json.load(f)
    row = {}
    tp = results.get("task_preservation") or {}
    if tp:
        row["preservation_recall_at_10"] = tp.get("mean")
        row["preservation_std"] = tp.get("std")
    ka = results.get("knowledge_absorption") or {}
    if ka:
        row["abs_exact_match"] = ka.get("exact_match")
        row["abs_mean_f1"] = ka.get("mean_f1")
        row["abs_contains"] = ka.get("contains")
        row["abs_fact_worst_f1"] = ka.get("fact_worst_f1")
    loc = results.get("locality") or {}
    if loc:
        # locality is nested per-stratum with its own 'overall' or per-stratum f1
        for stratum, stats in loc.items():
            if isinstance(stats, dict) and "f1" in stats:
                row[f"loc_{stratum}_f1"] = stats["f1"]
            if isinstance(stats, dict) and "accuracy" in stats:
                row[f"loc_{stratum}_acc"] = stats["accuracy"]
    return row


def run_method(
    method: str,
    n_rounds: int,
    per_round: int,
    debug: bool,
    output_root: Path,
    data_root: Path,
) -> list[dict]:
    """Run all N rounds for one method, returning a list of trajectory rows."""
    config = CONFIG_MAP[method]
    trajectory: list[dict] = []

    # Always read from the "sequential" subdir; in debug mode the user is
    # expected to have re-run scripts/15_prepare_sequential_triples.py with
    # the smaller --per-round / --n-rounds values before invoking this script.
    round_dir = data_root / "fnspid" / "triples" / "sequential"

    starting_checkpoint = (
        "checkpoints/qd_sft_debug/final" if debug else "checkpoints/qd_sft/final"
    )
    if not Path(starting_checkpoint).exists():
        print(f"ERROR: missing starting checkpoint {starting_checkpoint}.")
        print("  Run scripts/07_task_tune_qd.py first.")
        sys.exit(1)

    prev_model_dir: Path | None = None  # set after round 1

    for k in range(1, n_rounds + 1):
        round_triples = round_dir / f"round_{k}.json"
        if not round_triples.exists():
            print(f"ERROR: missing {round_triples}. Run scripts/15_prepare_sequential_triples.py.")
            sys.exit(1)

        run_name = f"seq_{method}_round_{k}"
        run_id = f"{run_name}_{TASK}_scale{per_round}"
        run_dir = output_root / run_id

        # Resume support: if this round already produced eval_results.json,
        # record the existing metrics and skip re-running. Model/ may have been
        # deleted for intermediate rounds to save disk; we only need it on disk
        # for the LAST completed round (to seed the next round's --base-model).
        model_dir = run_dir / "model"
        eval_path = run_dir / "eval_results.json"
        if eval_path.exists():
            print(f"[resume] {method} round {k}: reusing {run_dir}")
            row = {"round": k, "run_id": run_id, "resumed": True}
            row.update(_extract_round_metrics(eval_path))
            trajectory.append(row)
            if model_dir.exists():
                prev_model_dir = model_dir
            # If this round's model was pruned for disk, keep prev_model_dir
            # pointing at the most recent model that DOES exist (set in a prior
            # iteration).
            continue

        update_cmd = [
            sys.executable,
            "scripts/09_run_update.py",
            "--method", method,
            "--run-name", run_name,
            "--scale", str(per_round),
            "--task", TASK,
            "--config", config,
            "--triples-path", str(round_triples),
        ]
        if k == 1:
            update_cmd += ["--starting-checkpoint", starting_checkpoint]
        else:
            assert prev_model_dir is not None
            update_cmd += ["--base-model", str(prev_model_dir)]

        rc = _run(update_cmd)
        if rc != 0:
            print(f"FAILED: {method} round {k}")
            trajectory.append({"round": k, "status": "update_failed"})
            return trajectory

        eval_cmd = [
            sys.executable,
            "scripts/10_evaluate.py",
            "--model_path", str(run_dir),
            "--task", TASK,
            "--metrics", "preservation,absorption,locality",
            "--locality-subsample", str(LOCALITY_SUBSAMPLE),
        ]
        _run(eval_cmd)

        row = {"round": k, "run_id": run_id}
        row.update(_extract_round_metrics(run_dir / "eval_results.json"))
        trajectory.append(row)
        prev_model_dir = run_dir / "model"

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--n-rounds", type=int, default=None,
        help=f"Override N_ROUNDS (default {N_ROUNDS}).",
    )
    parser.add_argument(
        "--per-round", type=int, default=None,
        help=f"Override PER_ROUND (default {PER_ROUND}).",
    )
    parser.add_argument(
        "--methods", default=None,
        help="Comma-separated subset of methods (defaults to all).",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated GPU ids (e.g. '0,1,2,3'). When set, methods are "
        "run in parallel across GPUs: each method's full chain binds to one "
        "GPU via CUDA_VISIBLE_DEVICES. Default: sequential on default device.",
    )
    args = parser.parse_args()

    n_rounds = args.n_rounds or (3 if args.debug else N_ROUNDS)
    per_round = args.per_round or (50 if args.debug else PER_ROUND)
    methods = args.methods.split(",") if args.methods else METHODS

    output_root = Path("outputs")
    data_root = Path(os.environ.get("DATA_ROOT", "data"))

    summary: dict = {
        "n_rounds": n_rounds,
        "per_round": per_round,
        "methods": {},
    }

    if args.gpus:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
        _run_methods_parallel(
            methods, gpus, n_rounds, per_round, args.debug, output_root
        )
    else:
        for method in methods:
            print(f"\n{'=' * 60}\n{method}: {n_rounds} rounds x {per_round} edits\n{'=' * 60}")
            trajectory = run_method(method, n_rounds, per_round, args.debug, output_root, data_root)
            out_dir = output_root / "sequential" / method
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "trajectory.json", "w") as f:
                json.dump(trajectory, f, indent=2)
            summary["methods"][method] = trajectory

    # Collect each method's trajectory.json (written either by this process or
    # by child processes in the --gpus path) into a final summary.
    for method in methods:
        traj_path = output_root / "sequential" / method / "trajectory.json"
        if traj_path.exists():
            with open(traj_path) as f:
                summary["methods"][method] = json.load(f)

    summary_dir = output_root / "sequential"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_dir / 'summary.json'}")


def _run_methods_parallel(
    methods: list[str],
    gpus: list[str],
    n_rounds: int,
    per_round: int,
    debug: bool,
    output_root: Path,
) -> None:
    """Schedule methods across GPUs. Each method runs as a child process of
    this same script (--methods <one> without --gpus), bound to one GPU via
    CUDA_VISIBLE_DEVICES. Child processes write their own trajectory.json;
    the parent aggregates them afterward.
    """
    log_dir = output_root / "sequential" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    queue = list(methods)
    active: dict[str, tuple[subprocess.Popen, str, object]] = {}  # gpu -> (proc, method, log_fh)

    def _launch(gpu: str, method: str) -> None:
        cmd = [
            sys.executable,
            __file__,
            "--methods", method,
            "--n-rounds", str(n_rounds),
            "--per-round", str(per_round),
        ]
        if debug:
            cmd.append("--debug")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path = log_dir / f"{method}_gpu{gpu}.log"
        fh = open(log_path, "w")
        print(f"[gpu {gpu}] launching {method} -> {log_path}")
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        active[gpu] = (proc, method, fh)

    # Fill idle GPUs and drain.
    while queue or active:
        for gpu in gpus:
            if gpu not in active and queue:
                _launch(gpu, queue.pop(0))

        time.sleep(5)
        finished: list[str] = []
        for gpu, (proc, method, fh) in active.items():
            rc = proc.poll()
            if rc is not None:
                tag = "OK" if rc == 0 else f"FAIL rc={rc}"
                print(f"[gpu {gpu}] {method} {tag}")
                fh.close()
                finished.append(gpu)
        for gpu in finished:
            del active[gpu]


if __name__ == "__main__":
    main()
