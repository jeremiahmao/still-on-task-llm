"""Phase 2: Run top methods from Phase 1 on FinQA (generic-forgetting control)."""

import json
import subprocess
import sys
from pathlib import Path


def get_top_methods(n: int = 4) -> list[str]:
    """Select top N methods from Phase 1 by task preservation score."""
    results_dir = Path("outputs")
    scores = []

    for run_dir in results_dir.iterdir():
        if not run_dir.is_dir() or "qd" not in run_dir.name:
            continue
        eval_path = run_dir / "eval_results.json"
        if not eval_path.exists():
            continue
        with open(eval_path) as f:
            results = json.load(f)
        preservation = results.get("task_preservation", {}).get("mean", 0)
        method = run_dir.name.split("_qd_")[0]
        scores.append((method, preservation))

    # Sort by preservation (descending), deduplicate methods
    scores.sort(key=lambda x: x[1], reverse=True)
    seen = set()
    top = []
    for method, score in scores:
        if method not in seen:
            seen.add(method)
            top.append(method)
        if len(top) >= n:
            break

    return top


SCALES = [200, 1000, 3000]
TASK = "finqa"

CONFIG_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "mixed_replay": "configs/update/mixed_replay.yaml",
    "alphaedit": "configs/update/alphaedit.yaml",
    "copr": "configs/update/copr.yaml",
}


def main():
    top_methods = get_top_methods(4)
    print(f"Top methods from Phase 1: {top_methods}")

    total = len(top_methods) * len(SCALES)
    completed = 0

    for method in top_methods:
        for scale in SCALES:
            completed += 1
            print(f"\n{'='*60}")
            print(f"[{completed}/{total}] {method} @ scale={scale} (FinQA)")
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

            run_id = f"{method}_{TASK}_scale{scale}"
            eval_cmd = [
                sys.executable, "scripts/10_evaluate.py",
                "--model_path", f"outputs/{run_id}",
                "--task", TASK,
                "--metrics", "forgetting,absorption",
            ]
            subprocess.run(eval_cmd)

    print(f"\nPhase 2 complete. {completed} experiments run.")


if __name__ == "__main__":
    main()
