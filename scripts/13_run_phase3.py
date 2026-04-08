"""Phase 3: FinQA generic-forgetting control (time permitting).

Run top methods from Phase 1 on a FinQA-tuned model at 1K edits.
Expected outcome: no degradation (knowledge updates shouldn't affect table arithmetic).
See FINAL_PLAN.md for the full execution plan.
"""

import subprocess
import sys

# Run the methods that matter most for the paper's story
METHODS = ["kl_reg_sft", "copr"]
SCALE = 1000

CONFIG_MAP = {
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "copr": "configs/update/copr.yaml",
}


def main():
    print("=== Phase 3: FinQA forgetting control ===")

    # No-update baseline on FinQA
    print(f"\n{'=' * 60}")
    print("Evaluating FinQA no-update baseline...")
    print(f"{'=' * 60}")
    eval_cmd = [
        sys.executable,
        "scripts/10_evaluate.py",
        "--model_path",
        "checkpoints/finqa_sft/final",
        "--task",
        "finqa",
        "--metrics",
        "forgetting",
    ]
    subprocess.run(eval_cmd)

    # Run each method
    for i, method in enumerate(METHODS):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(METHODS)}] FinQA: {method} @ scale={SCALE}")
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
            "finqa",
            "--config",
            config,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {method} @ FinQA")
            continue

        eval_cmd = [
            sys.executable,
            "scripts/10_evaluate.py",
            "--model_path",
            f"outputs/{method}_finqa_scale{SCALE}",
            "--task",
            "finqa",
            "--metrics",
            "forgetting,absorption",
        ]
        subprocess.run(eval_cmd)

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
