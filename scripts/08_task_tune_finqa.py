"""Task-tune Qwen2.5-3B on FinQA via LoRA SFT (for Phase 3 forgetting control).

Prerequisite: FinQA dataset at data/finqa/dataset/ (from 01_download_data.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.finqa import prepare_finqa_dataset
from sot.models.base import load_model
from sot.models.lora import apply_lora, get_lora_config, save_lora
from sot.training.sft import run_sft
from sot.utils.config import load_config
from sot.utils.seed import seed_everything


def main():
    cfg = load_config()
    train_cfg = OmegaConf.load("configs/training/finqa_sft.yaml")
    finqa_cfg = OmegaConf.load("configs/data/finqa.yaml")
    seed_everything(cfg.seed)

    data_root = Path(cfg.paths.data_root)
    checkpoint_root = Path(cfg.paths.checkpoint_root)
    dataset_dir = data_root / "finqa" / "dataset"

    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} not found. Run 01_download_data.py first.")
        sys.exit(1)

    # Load and format FinQA data
    print("Loading FinQA training data...")
    train_data = prepare_finqa_dataset(dataset_dir, finqa_cfg.system_prompt, split="train")
    eval_data = prepare_finqa_dataset(dataset_dir, finqa_cfg.system_prompt, split="dev")
    print(f"FinQA train: {len(train_data)}, dev: {len(eval_data)}")

    # Load base model
    print(f"\nLoading {cfg.model.name}...")
    model, tokenizer = load_model(cfg.model.name, cfg.model.dtype, device_map=None)

    # Apply LoRA
    lora_cfg = train_cfg.get("lora", {})
    lora_config = get_lora_config(
        r=lora_cfg.get("r", 32),
        alpha=lora_cfg.get("alpha", 64),
        target_modules=list(lora_cfg.get("target_modules", None)),
        dropout=lora_cfg.get("dropout", 0.05),
    )
    model = apply_lora(model, lora_config)
    model.print_trainable_parameters()

    # Train
    output_dir = checkpoint_root / "finqa_sft"
    print(f"\nTraining FinQA SFT -> {output_dir}")
    trainer = run_sft(
        model,
        tokenizer,
        train_data,
        eval_data=eval_data,
        cfg=train_cfg,
        output_dir=str(output_dir),
    )

    # Save final LoRA adapter
    save_lora(trainer.model, str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nDone. FinQA checkpoint saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
