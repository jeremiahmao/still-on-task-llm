"""Naive SFT: fine-tune on fact QA pairs with no regularization."""

from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedModel

from sot.training.sft import run_sft
from sot.update.base import UpdateMethod


class NaiveSFTUpdate(UpdateMethod):
    @property
    def name(self) -> str:
        return "naive_sft"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        """SFT on rendered fact QA pairs. No regularization, no replay."""
        # Format as chat messages
        train_data = []
        for qa in fact_qa_pairs:
            train_data.append(
                {
                    "messages": [
                        {"role": "user", "content": qa["question"]},
                        {"role": "assistant", "content": qa["answer"]},
                    ],
                }
            )

        model.train()
        trainer = run_sft(
            model, tokenizer, train_data, cfg=cfg, output_dir="./checkpoints/naive_sft"
        )
        return trainer.model
