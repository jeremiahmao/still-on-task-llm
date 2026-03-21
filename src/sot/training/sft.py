"""SFT trainer wrapping trl.SFTTrainer with project defaults."""

from pathlib import Path

from datasets import Dataset
from omegaconf import DictConfig
from peft import PeftModel
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer

from sot.utils.seed import seed_everything


def run_sft(
    model,
    tokenizer: AutoTokenizer,
    train_data: list[dict],
    eval_data: list[dict] | None = None,
    cfg: DictConfig | None = None,
    output_dir: str | Path = "./checkpoints/sft",
) -> SFTTrainer:
    """Run SFT training.

    Args:
        model: Base or PeftModel to train.
        tokenizer: Tokenizer.
        train_data: List of dicts with 'messages' key (chat format).
        eval_data: Optional eval set in same format.
        cfg: Training config with lr, epochs, batch_size, etc.
        output_dir: Where to save checkpoints.

    Returns:
        The trainer after training.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults
    lr = 2e-4
    epochs = 3
    batch_size = 8
    grad_accum = 4
    warmup_ratio = 0.1
    max_seq_length = 2048

    if cfg is not None:
        t = cfg.get("training", {})
        lr = t.get("lr", lr)
        epochs = t.get("epochs", epochs)
        batch_size = t.get("batch_size", batch_size)
        grad_accum = t.get("gradient_accumulation_steps", grad_accum)
        warmup_ratio = t.get("warmup_ratio", warmup_ratio)
        max_seq_length = t.get("max_seq_length", max_seq_length)

    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        report_to="wandb",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))

    return trainer
