"""Phase 1: Run all methods on query decomposition at all edit scales."""

import subprocess
import sys
from pathlib import Path

METHODS = ["naive_sft", "kl_reg_sft", "mixed_replay", "alphaedit", "copr"]
SCALES = [200, 1000, 3000]
TASK = "qd"

CONFIG_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "mixed_replay": "configs/update/mixed_replay.yaml",
    "alphaedit": "configs/update/alphaedit.yaml",
    "copr": "configs/update/copr.yaml",
}


def main():
    total = len(METHODS) * len(SCALES)
    completed = 0

    for method in METHODS:
        for scale in SCALES:
            completed += 1
            print(f"\n{'='*60}")
            print(f"[{completed}/{total}] {method} @ scale={scale}")
            print(f"{'='*60}")

            config = CONFIG_MAP.get(method)
            cmd = [
                sys.executable, "scripts/09_run_update.py",
                "--method", method,
                "--scale", str(scale),
                "--task", TASK,
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
                sys.executable, "scripts/10_evaluate.py",
                "--model_path", model_path,
                "--task", TASK,
                "--metrics", "preservation,absorption,locality",
            ]
            subprocess.run(eval_cmd)

    # Also evaluate the no-update baseline
    print(f"\n{'='*60}")
    print("Evaluating no-update baseline...")
    print(f"{'='*60}")
    # The no-update baseline is just the task-tuned model evaluated directly
    # (handled separately since there's no update step)

    print(f"\nPhase 1 complete. {completed} experiments run.")
    print("Results in outputs/*/eval_results.json")


if __name__ == "__main__":
    main()
