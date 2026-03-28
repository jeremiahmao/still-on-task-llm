"""LoRA configuration, application, merge, and save/load utilities."""

from pathlib import Path

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import PreTrainedModel

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_lora_config(
    r: int = 32,
    alpha: int = 64,
    target_modules: list[str] | None = None,
    dropout: float = 0.05,
) -> LoraConfig:
    """Create a LoRA config for causal LM fine-tuning."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules or DEFAULT_TARGET_MODULES,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def apply_lora(model: PreTrainedModel, config: LoraConfig) -> PeftModel:
    """Wrap a base model with LoRA adapters."""
    return get_peft_model(model, config)


def merge_lora(peft_model: PeftModel) -> PreTrainedModel:
    """Merge LoRA weights into the base model and unload adapters."""
    return peft_model.merge_and_unload()


def save_lora(peft_model: PeftModel, path: str | Path) -> None:
    """Save only the LoRA adapter weights."""
    peft_model.save_pretrained(str(path))


def load_lora(base_model: PreTrainedModel, path: str | Path) -> PeftModel:
    """Load LoRA adapter weights onto a base model."""
    return PeftModel.from_pretrained(base_model, str(path))
