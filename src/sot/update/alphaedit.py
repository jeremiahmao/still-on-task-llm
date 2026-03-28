"""AlphaEdit: null-space constrained knowledge editing.

Uses the official AlphaEdit implementation from vendor/AlphaEdit
(https://github.com/jianghoucheng/AlphaEdit).
"""

import os
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.base import UpdateMethod

# Path to the vendored AlphaEdit repo
ALPHAEDIT_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "AlphaEdit"


def _setup_alphaedit_imports():
    """Add the vendored AlphaEdit repo to sys.path so its modules can be imported."""
    alphaedit_str = str(ALPHAEDIT_ROOT)
    if alphaedit_str not in sys.path:
        sys.path.insert(0, alphaedit_str)


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
        """Apply AlphaEdit to inject fact triples via null-space projected editing."""
        batch_size = cfg.get("batch_size", 100) if cfg else 100

        # Import the official AlphaEdit code
        _setup_alphaedit_imports()

        # globals.yml is read relative to cwd — ensure we're in the right place
        orig_cwd = os.getcwd()
        os.chdir(str(ALPHAEDIT_ROOT))
        try:
            from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
            from AlphaEdit.AlphaEdit_main import apply_AlphaEdit_to_model, get_cov
            from util import nethook
        finally:
            os.chdir(orig_cwd)

        # Load hparams for Qwen2.5-3B
        hparams_path = ALPHAEDIT_ROOT / "hparams" / "AlphaEdit" / "Qwen2.5-3B.json"
        hparams = AlphaEditHyperParams.from_json(hparams_path)

        # Convert fact QA pairs to AlphaEdit request format
        # AlphaEdit expects: {"prompt": "The {} ...", "subject": "X", "target_new": {"str": "Y"}, "case_id": i}
        requests = []
        for i, qa in enumerate(fact_qa_pairs):
            triple = qa.get("triple", {})
            subject = triple.get("subject", "")
            # Build a prompt template with {} placeholder for the subject
            prompt = qa.get("prompt_template", "What is known about {}?")
            requests.append(
                {
                    "case_id": i,
                    "prompt": prompt,
                    "subject": subject,
                    "target_new": {"str": qa["answer"]},
                }
            )

        # Initialize cache_c and null-space projector P
        W_out = nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight"
        )
        n_layers = len(hparams.layers)
        dim = W_out.shape[1]  # For Qwen/Llama-style models
        cache_c = torch.zeros((n_layers, dim, dim), device="cpu")
        P = torch.zeros((n_layers, dim, dim), device="cpu")
        del W_out

        # Compute null-space projector for each edited layer
        print("Computing null-space projectors...")
        os.chdir(str(ALPHAEDIT_ROOT))
        try:
            for i, layer in enumerate(hparams.layers):
                cov = get_cov(
                    model,
                    tokenizer,
                    hparams.rewrite_module_tmp.format(layer),
                    hparams.mom2_dataset,
                    hparams.mom2_n_samples,
                    hparams.mom2_dtype,
                ).cpu()
                U, S, _ = torch.linalg.svd(cov, full_matrices=False)
                small_singular_indices = (S < hparams.nullspace_threshold).nonzero(as_tuple=True)[0]
                print(f"  Layer {layer}: {len(small_singular_indices)} null-space dims")
                P[i, :, :] = U[:, small_singular_indices] @ U[:, small_singular_indices].T
        finally:
            os.chdir(orig_cwd)

        # Apply edits in batches
        print(f"Applying {len(requests)} edits in batches of {batch_size}...")
        os.chdir(str(ALPHAEDIT_ROOT))
        try:
            for start in range(0, len(requests), batch_size):
                batch = requests[start : start + batch_size]
                print(f"  Batch {start // batch_size + 1}: edits {start}–{start + len(batch) - 1}")
                model, cache_c = apply_AlphaEdit_to_model(
                    model,
                    tokenizer,
                    batch,
                    hparams,
                    cache_c=cache_c,
                    P=P,
                )
        finally:
            os.chdir(orig_cwd)

        return model
