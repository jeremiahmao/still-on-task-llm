"""Phase 1: Core comparison on query decomposition at 1K edits.

Minimum viable result for the paper: 4 update methods + no-update baseline.
See FINAL_PLAN.md for the full execution plan.
"""

import subprocess
import sys
from pathlib import Path

METHODS = ["naive_sft", "kl_reg_sft", "alphaedit", "copr"]
SCALE = 1000
TASK = "qd"

CONFIG_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "alphaedit": "configs/update/alphaedit.yaml",
    "copr": "configs/update/copr.yaml",
}


def run_no_update_baseline():
    """Evaluate the task-tuned model with no knowledge injection."""
    print(f"\n{'=' * 60}")
    print("Evaluating no-update baseline...")
    print(f"{'=' * 60}")

    checkpoint_dir = "checkpoints/qd_sft/final"
    if not Path(checkpoint_dir).exists():
        print(f"WARNING: No checkpoint at {checkpoint_dir}. Skipping baseline.")
        return False

    run_dir = Path("outputs/no_update_qd")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_link = run_dir / "model"
    if not model_link.exists():
        model_link.symlink_to(Path(checkpoint_dir).resolve())

    eval_cmd = [
        sys.executable,
        "scripts/10_evaluate.py",
        "--model_path",
        str(run_dir),
        "--task",
        TASK,
        "--metrics",
        "preservation,locality",
    ]
    result = subprocess.run(eval_cmd)
    return result.returncode == 0


def main():
    total = len(METHODS)
    completed = 0
    failed = []

    # No-update baseline first
    if not run_no_update_baseline():
        failed.append("no_update")

    # Run each method at 1K edits
    for method in METHODS:
        completed += 1
        print(f"\n{'=' * 60}")
        print(f"[{completed}/{total}] {method} @ scale={SCALE}")
        print(f"{'=' * 60}")

        config = CONFIG_MAP[method]
        cmd = [
            sys.executable,
            "scripts/09_run_update.py",
            "--method",
            method,
            "--scale",
            str(SCALE),
            "--task",
            TASK,
            "--config",
            config,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {method} @ scale={SCALE}")
            failed.append(method)
            continue

        # Run evaluation
        run_id = f"{method}_{TASK}_scale{SCALE}"
        model_path = f"outputs/{run_id}"

        eval_cmd = [
            sys.executable,
            "scripts/10_evaluate.py",
            "--model_path",
            model_path,
            "--task",
            TASK,
            "--metrics",
            "preservation,absorption,locality",
        ]
        subprocess.run(eval_cmd)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Phase 1 complete. {completed - len(failed)}/{total} methods succeeded.")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print("Results in outputs/*/eval_results.json")
    print("\nNext: run scripts/12_run_phase2.py for 3K scaling.")


if __name__ == "__main__":
    main()
