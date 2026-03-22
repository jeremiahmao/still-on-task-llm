"""Phase 2: Expand to multiple scales + FinQA forgetting control.

Run after Phase 1 validates the pipeline at 1K edits.
"""

import subprocess
import sys

# Expand core methods to all scales
METHODS = ["kl_reg_sft", "alphaedit", "copr"]
SCALES = [200, 3000]  # 1K already done in Phase 1
TASK = "qd"

CONFIG_MAP = {
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "alphaedit": "configs/update/alphaedit.yaml",
    "copr": "configs/update/copr.yaml",
}


def main():
    # Part A: Expand QD to remaining scales
    print("=== Phase 2A: QD at 200 and 3K scales ===")
    total = len(METHODS) * len(SCALES)
    completed = 0

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

            run_id = f"{method}_{TASK}_scale{scale}"
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

    # Part B: FinQA forgetting control at 1K for top methods
    print("\n=== Phase 2B: FinQA forgetting control ===")
    for method in METHODS:
        print(f"\n{'=' * 60}")
        print(f"FinQA: {method} @ scale=1000")
        print(f"{'=' * 60}")

        config = CONFIG_MAP.get(method)
        cmd = [
            sys.executable,
            "scripts/09_run_update.py",
            "--method",
            method,
            "--scale",
            "1000",
            "--task",
            "finqa",
        ]
        if config:
            cmd.extend(["--config", config])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {method} @ FinQA")
            continue

        eval_cmd = [
            sys.executable,
            "scripts/10_evaluate.py",
            "--model_path",
            f"outputs/{method}_finqa_scale1000",
            "--task",
            "finqa",
            "--metrics",
            "forgetting,absorption",
        ]
        subprocess.run(eval_cmd)

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
