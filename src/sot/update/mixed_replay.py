"""Mixed replay: SFT on new facts + shuffled task training data."""

import random

from omegaconf import DictConfig
from transformers import PreTrainedModel, AutoTokenizer

from sot.training.sft import run_sft
from sot.update.base import UpdateMethod


class MixedReplayUpdate(UpdateMethod):

    @property
    def name(self) -> str:
        return "mixed_replay"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        """SFT on new fact QA pairs mixed with a replay buffer of task data."""
        replay_pct = cfg.get("replay_pct", 0.05) if cfg else 0.05

        # Format fact QA as chat messages
        fact_examples = []
        for qa in fact_qa_pairs:
            fact_examples.append({
                "messages": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ],
            })

        # Sample replay buffer from task data
        replay_examples = []
        if task_data:
            n_replay = max(1, int(len(task_data) * replay_pct))
            replay_examples = random.sample(task_data, min(n_replay, len(task_data)))

        # Combine and shuffle
        combined = fact_examples + replay_examples
        random.shuffle(combined)

        model.train()
        trainer = run_sft(
            model, tokenizer, combined, cfg=cfg, output_dir="./checkpoints/mixed_replay"
        )
        return trainer.model
