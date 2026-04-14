"""Phase 3: FinQA generic-forgetting control (time permitting).

Run top methods from Phase 1 on a FinQA-tuned model at 1K edits.
Expected outcome: no degradation (knowledge updates shouldn't affect table arithmetic).
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

# Run the methods that matter most for the paper's story
METHODS = ["kl_reg_sft", "copr"]
SCALE = 1000

CONFIG_MAP = {
    "kl_reg_sft": "configs/update/kl_reg_sft.yaml",
    "copr": "configs/update/copr.yaml",
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Use debug triples subdirectory")
    args = parser.parse_args()
    debug_flag = ["--debug"] if args.debug else []

    print("=== Phase 3: FinQA forgetting control ===")

    # No-update baseline on FinQA
    print(f"\n{'=' * 60}")
    print("Evaluating FinQA no-update baseline...")
    print(f"{'=' * 60}")
    checkpoint_dir = "checkpoints/finqa_sft/final"
    if not Path(checkpoint_dir).exists() and args.debug:
        checkpoint_dir = "checkpoints/finqa_sft_debug/final"
    baseline_dir = Path("outputs/no_update_finqa")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    model_dir = baseline_dir / "model"
    if not model_dir.exists():
        # Checkpoint only has LoRA adapter files — merge into a full model
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
        str(baseline_dir),
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
        cmd = _torchrun(
            "scripts/09_run_update.py",
            "--method", method,
            "--scale", str(SCALE),
            "--task", "finqa",
            "--config", config,
            *debug_flag,
        )

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
