"""Phase 1: Core comparison on query decomposition.

Minimum viable experiment: 3 update methods + no-update baseline at 1K edits.
Expand to more scales (200, 3K) and methods (mixed_replay) once pipeline is proven.
"""

import subprocess
import sys
from pathlib import Path

# Core methods only — naive_sft is just kl_reg with lambda=0, mixed_replay deferred
METHODS = ["kl_reg_sft", "alphaedit", "copr"]
SCALES = [1000]  # Start with midpoint; expand to [200, 1000, 3000] after validation
TASK = "qd"

CONFIG_MAP = {
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "alphaedit": "configs/update/alphaedit.yaml",
    "copr": "configs/update/copr.yaml",
}


def run_no_update_baseline():
    """Evaluate the task-tuned model with no knowledge injection."""
    print(f"\n{'=' * 60}")
    print("Evaluating no-update baseline...")
    print(f"{'=' * 60}")

    # The no-update baseline evaluates the task-tuned checkpoint directly
    checkpoint_dir = "checkpoints/qd_sft/final"
    if not Path(checkpoint_dir).exists():
        print(f"WARNING: No checkpoint at {checkpoint_dir}. Skipping baseline.")
        return

    # Copy checkpoint to outputs for consistent evaluation
    run_dir = Path("outputs/no_update_qd_scale0")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Symlink model dir if not already there
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
    subprocess.run(eval_cmd)


def main():
    total = len(METHODS) * len(SCALES)
    completed = 0

    # No-update baseline first
    run_no_update_baseline()

    # Run each method
    for method in METHODS:
        for scale in SCALES:
            completed += 1
            print(f"\n{'=' * 60}")
            print(f"[{completed}/{total}] {method} @ scale={scale}")
            print(f"{'=' * 60}")

            config = CONFIG_MAP.get(method)
            cmd = [
                sys.executable,
                "scripts/09_run_update.py",
                "--method",
                method,
                "--scale",
                str(scale),
                "--task",
                TASK,
            ]
            if config:
                cmd.extend(["--config", config])

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAILED: {method} @ scale={scale}")
                continue

            # Run evaluation
            run_id = f"{method}_{TASK}_scale{scale}"
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

    print(f"\nPhase 1 complete. {completed} experiments + baseline run.")
    print("Results in outputs/*/eval_results.json")
    print("\nIf results are promising, expand with:")
    print("  --scales 200,3000")
    print("  --methods mixed_replay")


if __name__ == "__main__":
    main()
