"""Phase 1: Core comparison on query decomposition at 1K edits.

Minimum viable result for the paper: 4 update methods + no-update baseline.
See FINAL_PLAN.md for the full execution plan.
"""

import os
import subprocess
import sys
from pathlib import Path


def _torchrun(script, *args):
    nproc = int(os.environ.get("NPROC_PER_NODE", "1"))
    if nproc > 1:
        return ["torchrun", f"--nproc_per_node={nproc}", script] + list(args)
    return [sys.executable, script] + list(args)

METHODS = ["naive_sft", "kl_reg_sft", "copr"]
SCALE = 1000
DEBUG_SCALE = 50
TASK = "qd"

CONFIG_MAP = {
    "naive_sft": "configs/update/naive_sft.yaml",
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "copr": "configs/update/copr.yaml",
}


def run_no_update_baseline(debug: bool = False):
    """Evaluate the task-tuned model with no knowledge injection."""
    print(f"\n{'=' * 60}")
    print("Evaluating no-update baseline...")
    print(f"{'=' * 60}")

    checkpoint_dir = "checkpoints/qd_sft/final"
    if not Path(checkpoint_dir).exists() and debug:
        checkpoint_dir = "checkpoints/qd_sft_debug/final"
    if not Path(checkpoint_dir).exists():
        print(f"WARNING: No checkpoint at {checkpoint_dir}. Skipping baseline.")
        return False

    run_dir = Path("outputs/no_update_qd")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = run_dir / "model"
    if not model_dir.exists():
        # Checkpoint only has LoRA adapter files — merge into a full model
        # so 10_evaluate.py can load it with AutoModelForCausalLM.from_pretrained().
        sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))
        from sot.models.base import load_model
        from sot.models.lora import load_lora, merge_lora
        from sot.utils.config import load_config

        cfg = load_config()
        print("  Loading base model + LoRA adapter for baseline merge...")
        model, tokenizer = load_model(cfg.model.name, cfg.model.dtype, device_map="auto")
        model = load_lora(model, checkpoint_dir)
        model = merge_lora(model)
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        del model
        import torch; torch.cuda.empty_cache()
        print(f"  Saved merged baseline to {model_dir}")

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
    if debug:
        eval_cmd.append("--debug")
    result = subprocess.run(eval_cmd)
    return result.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Use debug triples subdirectory")
    args = parser.parse_args()
    debug_flag = ["--debug"] if args.debug else []
    scale = DEBUG_SCALE if args.debug else SCALE

    total = len(METHODS)
    completed = 0
    failed = []

    # No-update baseline first
    if not run_no_update_baseline(debug=args.debug):
        failed.append("no_update")

    # Run each method at 1K edits
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
            *debug_flag,
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
