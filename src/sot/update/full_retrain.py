"""Full retrain: train from base model on all data (task + new facts). Phase 3 only."""

from omegaconf import DictConfig
from transformers import PreTrainedModel, AutoTokenizer

from sot.models.base import load_model
from sot.models.lora import apply_lora, get_lora_config
from sot.training.sft import run_sft
from sot.update.base import UpdateMethod


class FullRetrainUpdate(UpdateMethod):

    @property
    def name(self) -> str:
        return "full_retrain"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        """Retrain from the base Qwen2.5-3B checkpoint on all data.

        Combines original task training data with new fact QA pairs.
        This is the compute-expensive upper bound.
        """
        if task_data is None:
            raise ValueError("Full retrain requires task_data (original training set)")

        # Load fresh base model
        base_model, _ = load_model()
        lora_config = get_lora_config()
        base_model = apply_lora(base_model, lora_config)

        # Format fact QA as chat messages
        fact_examples = []
        for qa in fact_qa_pairs:
            fact_examples.append({
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ],
            })

        # Combine all data
        combined = list(task_data) + fact_examples

        base_model.train()
        trainer = run_sft(
            base_model, tokenizer, combined, cfg=cfg, output_dir="./checkpoints/full_retrain"
        )
        return trainer.model
