"""Phase 2: Scaling stress test at 3K edits.

Run after Phase 1 validates the pipeline at 1K edits.
See FINAL_PLAN.md for the full execution plan.
"""

import os
import subprocess
import sys


def _torchrun(script, *args):
    nproc = int(os.environ.get("NPROC_PER_NODE", "1"))
    if nproc > 1:
        return ["torchrun", f"--nproc_per_node={nproc}", script] + list(args)
    return [sys.executable, script] + list(args)

METHODS = ["naive_sft", "kl_reg_sft", "copr", "copr_v2"]
SCALE = 3000
DEBUG_SCALE = 50
TASK = "qd"

CONFIG_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "copr": "configs/update/copr.yaml",
    "copr_v2": "configs/update/copr_v2.yaml",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Use debug triples subdirectory")
    args = parser.parse_args()
    debug_flag = ["--debug"] if args.debug else []
    scale = DEBUG_SCALE if args.debug else SCALE

    print(f"=== Phase 2: QD at {scale} edits ===")
    total = len(METHODS)
    completed = 0
    failed = []

    for method in METHODS:
        completed += 1
        print(f"\n{'=' * 60}")
        print(f"[{completed}/{total}] {method} @ scale={scale}")
        print(f"{'=' * 60}")

        config = CONFIG_MAP[method]
        cmd = [
            sys.executable,
            "scripts/09_run_update.py",
            "--method", method,
            "--scale", str(scale),
            "--task", TASK,
            "--config", config,
            *debug_flag,
        ]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {method} @ scale={scale}")
            failed.append(method)
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
            "--locality-subsample",
            "2000",
            *debug_flag,
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
