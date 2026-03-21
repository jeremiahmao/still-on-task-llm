"""DPO trainer for optional query decomposition refinement (Phase 3)."""

from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, TrainingArguments
from trl import DPOTrainer


def run_dpo(
    model,
    ref_model,
    tokenizer: AutoTokenizer,
    train_data: list[dict],
    cfg: DictConfig | None = None,
    output_dir: str | Path = "./checkpoints/dpo",
) -> DPOTrainer:
    """Run DPO training for query decomposition refinement.

    Args:
        model: Model to train (should be SFT checkpoint).
        ref_model: Frozen reference model (same as model before DPO).
        tokenizer: Tokenizer.
        train_data: List of dicts with 'prompt', 'chosen', 'rejected' keys.
            'chosen' = decomposition with higher Recall@10.
            'rejected' = decomposition with lower Recall@10.
        cfg: DPO training config.
        output_dir: Where to save checkpoints.

    Returns:
        The trainer after training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    beta = 0.1
    lr = 5e-5
    epochs = 1
    batch_size = 4
    grad_accum = 4

    if cfg is not None:
        t = cfg.get("training", {})
        beta = t.get("beta", beta)
        lr = t.get("lr", lr)
        epochs = t.get("epochs", epochs)
        batch_size = t.get("batch_size", batch_size)
        grad_accum = t.get("gradient_accumulation_steps", grad_accum)

    train_dataset = Dataset.from_list(train_data)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        beta=beta,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    return trainer
