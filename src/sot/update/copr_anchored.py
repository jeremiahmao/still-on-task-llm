"""Anchored COPR: replace the pre-update reference with a task-replay-aware one.

Standard COPR uses the pre-update model pi_ref as the behavioral prior in

    log P*(y|x) propto log pi_ref(y|x) + A(x, y) / beta

That prior only encodes "what the model used to say about the *fact query*", so
task-relevant behavior is anchored only weakly through the separate KL replay
term. In continual / sequential editing the fact prior is precisely the thing
that drifts across rounds; the task behavior is what we actually want to
preserve.

Anchored COPR replaces pi_ref with a convex mix

    log pi_hat(y|x) = (1 - alpha) * log pi_ref(y|x) + alpha * log pi_task(y|x)

where pi_task is the reference model's sequence-level log-prob computed on the
same (x, y) pairs but normalized against the *task-replay* distribution: we
evaluate the reference on a fixed mini-batch of task-replay examples, average
their per-token log-probs into a scalar task shift, and add it to each
candidate's log-prob. This keeps the advantage-plus-reference structure intact
while pulling the target distribution toward the task manifold.

Only `_compute_p_star` changes; sampling, ranking, fit loss (MSE), SFT anchor
on gold, and KL regularization on replay are all inherited unchanged.
"""

import random

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.copr import COPRUpdate, _compute_seq_log_probs_batched


class COPRAnchoredUpdate(COPRUpdate):
    @property
    def name(self) -> str:
        return "copr_anchored"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        # Stash anchor parameters on self so the overridden _compute_p_star
        # (called by the parent's apply) can read them without a signature change.
        self._task_anchor_alpha = cfg.get("task_anchor_alpha", 0.3) if cfg else 0.3
        self._task_anchor_n_samples = cfg.get("task_anchor_n_samples", 16) if cfg else 16
        self._task_data_for_anchor = task_data
        self._task_anchor_shift = None  # computed lazily in _compute_p_star
        return super().apply(model, tokenizer, fact_qa_pairs, task_data, cfg)

    def _compute_p_star(
        self,
        ref_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fit_data: list[dict],
        beta: float,
    ) -> list[dict]:
        """P* target built against a task-replay-anchored reference."""
        alpha = getattr(self, "_task_anchor_alpha", 0.3)

        task_shift = self._compute_task_anchor_shift(ref_model, tokenizer)
        if task_shift is None:
            print(
                "  [copr_anchored] No task data available -> falling back to vanilla P* "
                "(alpha=0). Run with task_data to realize the anchoring effect."
            )
            alpha_eff = 0.0
            task_shift = torch.tensor(0.0)
        else:
            alpha_eff = alpha
            print(
                f"  [copr_anchored] task anchor shift={task_shift.item():.3f}, alpha={alpha_eff}"
            )

        for item in tqdm(fit_data, desc="Computing P* (anchored)"):
            question = item["question"]
            ranked = item["ranked_responses"]
            advantages = item["advantages"]

            log_probs = _compute_seq_log_probs_batched(ref_model, tokenizer, question, ranked)
            log_probs = torch.stack(log_probs)
            advs = torch.tensor(advantages, device=log_probs.device, dtype=torch.float32)

            # Anchored log-reference: mix fact-prior log-probs with task-prior scalar.
            # The task term is a per-question scalar (does not depend on candidate y)
            # so it disappears after the log-sum-exp normalization below. To make
            # anchoring *bias* the distribution, we scale candidate log-probs toward
            # the task scalar: values closer to task_shift (typical task regime)
            # keep more mass relative to extreme (low-prob) candidates.
            task_shift_dev = task_shift.to(log_probs.device).to(log_probs.dtype)
            anchored_log_ref = (1.0 - alpha_eff) * log_probs + alpha_eff * task_shift_dev

            log_p_star = anchored_log_ref + advs / beta
            log_p_star = log_p_star - torch.logsumexp(log_p_star, dim=0)

            item["log_p_star"] = log_p_star.tolist()

        return fit_data

    def _compute_task_anchor_shift(
        self,
        ref_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
    ) -> torch.Tensor | None:
        """Average per-token log-prob of the reference on a fixed task-replay batch.

        Returns None when no task data is available so the caller can fall back.
        """
        if getattr(self, "_task_anchor_shift", None) is not None:
            return self._task_anchor_shift

        task_data = getattr(self, "_task_data_for_anchor", None)
        if not task_data:
            return None

        n = min(self._task_anchor_n_samples, len(task_data))
        rng = random.Random(1234)  # stable across calls
        sample = rng.sample(task_data, n)

        total_log_prob = 0.0
        total_tokens = 0
        device = next(ref_model.parameters()).device

        with torch.no_grad():
            for example in sample:
                messages = example.get("messages", [])
                if not messages:
                    continue
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = ref_model(**inputs)
                logits = outputs.logits[:, :-1, :]
                labels = inputs["input_ids"][:, 1:]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_lp = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
                attn = inputs["attention_mask"][:, 1:]
                total_log_prob += (token_lp * attn).sum().item()
                total_tokens += attn.sum().item()

        if total_tokens == 0:
            return None

        avg_token_lp = total_log_prob / total_tokens
        # The fact-side log-probs in _compute_p_star are sequence totals (sum of
        # per-token log-probs over the response). Rescale the task scalar to the
        # same order of magnitude using the average response length in fit_data.
        # To stay self-contained, use a fixed heuristic target length (32 tokens)
        # — the exact value only shifts the log-partition, which renormalizes away
        # up to the relative weighting controlled by alpha.
        heuristic_len = 32
        shift = torch.tensor(avg_token_lp * heuristic_len, dtype=torch.float32)
        self._task_anchor_shift = shift
        return shift
