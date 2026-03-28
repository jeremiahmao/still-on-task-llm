"""Apply a single knowledge update method to a task-tuned model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.models.base import load_model
from sot.models.lora import load_lora, merge_lora
from sot.update.alphaedit import AlphaEditUpdate
from sot.update.copr import COPRUpdate
from sot.update.kl_reg_sft import KLRegSFTUpdate
from sot.update.naive_sft import NaiveSFTUpdate
from sot.utils.config import load_config, save_config
from sot.utils.gpu import track_compute
from sot.utils.logging import save_metadata
from sot.utils.seed import seed_everything

METHODS = {
    "naive_sft": NaiveSFTUpdate,
    "kl_reg_sft": KLRegSFTUpdate,
    "alphaedit": AlphaEditUpdate,
    "copr": COPRUpdate,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--scale", type=int, required=True, choices=[1000, 3000])
    parser.add_argument("--task", default="qd", choices=["qd", "finqa"])
    parser.add_argument("--config", default=None, help="Method-specific config YAML")
    parser.add_argument("--overrides", nargs="*", default=[], help="OmegaConf dot-list overrides")
    args = parser.parse_args()

    base_cfg = load_config()
    method_cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    if args.overrides:
        method_cfg = OmegaConf.merge(method_cfg, OmegaConf.from_dotlist(args.overrides))

    seed_everything(base_cfg.seed)

    data_root = Path(base_cfg.paths.data_root)
    output_root = Path(base_cfg.paths.output_root)

    # Load fact triples at the specified scale
    triples_path = data_root / "fnspid" / "triples" / f"triples_{args.scale}.json"
    with open(triples_path) as f:
        fact_qa_pairs = json.load(f)
    print(f"Loaded {len(fact_qa_pairs)} fact QA pairs at scale {args.scale}")

    # Load task data for replay (if needed)
    task_data = None
    if args.task == "qd":
        qd_train_path = data_root / "qd" / "train.json"
        if qd_train_path.exists():
            with open(qd_train_path) as f:
                task_data = json.load(f)

    # Load the task-tuned model (merged LoRA)
    print("Loading task-tuned model...")
    model, tokenizer = load_model(base_cfg.model.name, base_cfg.model.dtype)
    checkpoint_path = Path(base_cfg.paths.checkpoint_root) / f"{args.task}_sft" / "final"
    if checkpoint_path.exists():
        model = load_lora(model, checkpoint_path)
        model = merge_lora(model)
        print(f"Loaded and merged LoRA from {checkpoint_path}")

    # Apply update method
    method = METHODS[args.method]()
    print(f"\nApplying {method.name} at scale {args.scale}...")

    with track_compute() as stats:
        updated_model = method.apply(model, tokenizer, fact_qa_pairs, task_data, method_cfg)

    # Save
    run_id = f"{args.method}_{args.task}_scale{args.scale}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    updated_model.save_pretrained(str(run_dir / "model"))
    tokenizer.save_pretrained(str(run_dir / "model"))

    save_metadata(
        {
            "method": args.method,
            "task": args.task,
            "scale": args.scale,
            "gpu_hours": stats.gpu_hours,
            "peak_memory_gb": stats.peak_memory_gb,
            "elapsed_seconds": stats.elapsed_seconds,
            "seed": base_cfg.seed,
        },
        run_dir / "metadata.json",
    )

    save_config(OmegaConf.merge(base_cfg, method_cfg), run_dir / "config.yaml")

    print(f"\nDone. Results saved to {run_dir}")
    print(f"  GPU-hours: {stats.gpu_hours:.2f}")
    print(f"  Peak memory: {stats.peak_memory_gb:.1f} GB")


if __name__ == "__main__":
    main()
