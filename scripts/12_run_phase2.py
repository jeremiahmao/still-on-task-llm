"""Phase 2: Scaling stress test at 3K edits.

Run after Phase 1 validates the pipeline at 1K edits.
See FINAL_PLAN.md for the full execution plan.
"""

import subprocess
import sys

METHODS = ["naive_sft", "kl_reg_sft", "copr"]
SCALE = 3000
TASK = "qd"

CONFIG_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "copr": "configs/update/copr.yaml",
}


def main():
    print(f"=== Phase 2: QD at {SCALE} edits ===")
    total = len(METHODS)
    completed = 0
    failed = []

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

        run_id = f"{method}_{TASK}_scale{SCALE}"
        eval_cmd = [
            sys.executable,
            "scripts/10_evaluate.py",
            "--model_path",
            f"outputs/{run_id}",
            "--task",
            TASK,
            "--metrics",
            "preservation,absorption,locality",
        ]
        subprocess.run(eval_cmd)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Phase 2 complete. {completed - len(failed)}/{total} methods succeeded.")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print("\nNext (time permitting): run scripts/13_run_phase3.py for FinQA control.")


if __name__ == "__main__":
    main()
