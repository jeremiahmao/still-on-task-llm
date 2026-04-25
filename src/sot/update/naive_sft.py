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
        """SFT on rendered fact QA pairs. No regularization, no replay.

        Pre-rendered chats (chat_messages for K=5 DSAE Lite augmentation,
        qd_messages for legacy K=2 fi_sft) are used as-is; standard entries
        build the usual {user: question, assistant: answer} chat.
        """
        train_data = []
        for qa in fact_qa_pairs:
            if qa.get("chat_messages"):
                messages = qa["chat_messages"]
            elif qa.get("train_format") == "qd" and qa.get("qd_messages"):
                messages = qa["qd_messages"]
            else:
                messages = [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ]
            train_data.append({"messages": messages})

        model.train()
        trainer = run_sft(
            model, tokenizer, train_data, cfg=cfg, output_dir="./checkpoints/naive_sft"
        )
        return trainer.model
