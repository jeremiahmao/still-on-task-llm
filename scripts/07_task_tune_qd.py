"""Task-tune Qwen2.5-3B on query decomposition via LoRA SFT.

This produces the baseline checkpoint that all update methods start from.
Prerequisite: QD training data at data/qd/train.json (from 05_generate_qd_data.py).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.models.base import load_model
from sot.models.lora import apply_lora, get_lora_config, save_lora
from sot.training.sft import run_sft
from sot.utils.config import load_config
from sot.utils.seed import seed_everything


def main():
    cfg = load_config()
    train_cfg = OmegaConf.load("configs/training/query_decomp_sft.yaml")
    seed_everything(cfg.seed)

    data_root = Path(cfg.paths.data_root)
    checkpoint_root = Path(cfg.paths.checkpoint_root)

    # Load QD training data
    train_path = data_root / "qd" / "train.json"
    test_path = data_root / "qd" / "test.json"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run 05_generate_qd_data.py first.")
        sys.exit(1)

    with open(train_path) as f:
        train_data = json.load(f)
    print(f"QD training examples: {len(train_data)}")

    eval_data = None
    if test_path.exists():
        with open(test_path) as f:
            eval_data = json.load(f)
        print(f"QD eval examples: {len(eval_data)}")

    # Load base model
    print(f"\nLoading {cfg.model.name}...")
    model, tokenizer = load_model(cfg.model.name, cfg.model.dtype)

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
    output_dir = checkpoint_root / "qd_sft"
    print(f"\nTraining QD SFT -> {output_dir}")
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
    print(f"\nDone. QD checkpoint saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
