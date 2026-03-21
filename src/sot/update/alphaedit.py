"""AlphaEdit: null-space constrained knowledge editing via EasyEdit."""

from pathlib import Path

from omegaconf import DictConfig
from transformers import PreTrainedModel, AutoTokenizer

from sot.update.base import UpdateMethod


class AlphaEditUpdate(UpdateMethod):

    @property
    def name(self) -> str:
        return "alphaedit"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        """Apply AlphaEdit to inject fact triples via null-space projected editing.

        Uses EasyEdit's BaseEditor with AlphaEdit hparams. Falls back to
        the standalone AlphaEdit repo if EasyEdit integration fails.
        """
        batch_size = cfg.get("batch_size", 100) if cfg else 100

        # Convert fact QA pairs to EasyEdit format
        prompts = []
        subjects = []
        target_new = []
        for qa in fact_qa_pairs:
            triple = qa.get("triple", {})
            prompts.append(qa["question"])
            subjects.append(triple.get("subject", ""))
            target_new.append(qa["answer"])

        try:
            return self._apply_easyedit(model, tokenizer, prompts, subjects, target_new, batch_size)
        except Exception as e:
            print(f"EasyEdit integration failed: {e}")
            print("Falling back to standalone AlphaEdit...")
            return self._apply_standalone(model, tokenizer, prompts, subjects, target_new, batch_size)

    def _apply_easyedit(
        self, model, tokenizer, prompts, subjects, target_new, batch_size
    ) -> PreTrainedModel:
        """Apply via EasyEdit framework."""
        from easyeditor import BaseEditor, AlphaEditHyperParams

        # Create hparams for the model
        hparams = AlphaEditHyperParams.from_hparams("configs/update/alphaedit_easyedit.yaml")
        editor = BaseEditor.from_hparams(hparams)

        # Apply edits in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_subjects = subjects[i : i + batch_size]
            batch_targets = target_new[i : i + batch_size]

            editor.batch_edit(
                prompts=batch_prompts,
                subject=batch_subjects,
                target_new=batch_targets,
            )

        return editor.model

    def _apply_standalone(
        self, model, tokenizer, prompts, subjects, target_new, batch_size
    ) -> PreTrainedModel:
        """Fallback: apply using standalone AlphaEdit repo."""
        raise NotImplementedError(
            "Standalone AlphaEdit integration not yet implemented. "
            "Clone https://github.com/jianghoucheng/AlphaEdit and adapt."
        )
