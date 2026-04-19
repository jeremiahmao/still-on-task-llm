"""Anchored COPR: mix a per-candidate task-replay likelihood into log pi_ref.

Standard COPR uses the pre-update model pi_ref as the behavioral prior:

    log P*(y|x_fact) propto log pi_ref(y|x_fact) + A(x, y) / beta

That prior only encodes "what the model used to say about the *fact query*",
so task-relevant behavior is anchored only weakly (via the separate KL replay
term). In continual / sequential editing the fact prior drifts across rounds
and we actually want to preserve task behavior.

Anchored COPR builds a per-candidate task likelihood

    s_task(y_j) = logmeanexp_i log pi_ref(y_j | x_task_i)

over a small set of sampled task-replay prompts x_task_i, and mixes it into
the reference log-prob:

    log P*(y_j|x) propto (1 - alpha) * log pi_ref(y_j|x_fact)
                       + alpha * s_task(y_j)
                       + A_j / beta

The per-candidate form is what makes this meaningful under logsumexp
normalization — a per-question scalar would cancel. Candidates whose
surface form is well-supported by the reference under typical task
contexts get up-weighted; candidates that only fit the fact prompt get
down-weighted.

Only `_compute_p_star` changes; sampling, ranking, MSE fit, SFT anchor on
gold, and KL regularization on replay are all inherited unchanged.
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
        self._task_anchor_n_samples = (
            cfg.get("task_anchor_n_samples", 4) if cfg else 4
        )
        self._task_data_for_anchor = task_data
        self._task_anchor_prompts: list[str] | None = None  # lazily built
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
        task_prompts = self._get_task_prompts()

        if not task_prompts:
            print(
                "  [copr_anchored] No task data available -> falling back to vanilla P* "
                "(alpha=0)."
            )
            alpha_eff = 0.0
        else:
            alpha_eff = alpha
            print(
                f"  [copr_anchored] alpha={alpha_eff}, task_prompts={len(task_prompts)}"
            )

        for item in tqdm(fit_data, desc="Computing P* (anchored)"):
            question = item["question"]
            ranked = item["ranked_responses"]
            advantages = item["advantages"]

            # Fact-conditional log-probs (vanilla COPR reference).
            log_probs = _compute_seq_log_probs_batched(ref_model, tokenizer, question, ranked)
            log_probs = torch.stack(log_probs)
            advs = torch.tensor(advantages, device=log_probs.device, dtype=torch.float32)

            if alpha_eff > 0.0:
                # Per-candidate task-anchor score: for each y_j, average (in
                # log-space) its log-prob under the reference when conditioned
                # on each sampled task prompt.
                s_task = self._compute_task_anchor_scores(
                    ref_model, tokenizer, ranked, task_prompts
                )
                s_task = s_task.to(log_probs.device).to(log_probs.dtype)
                anchored_log_ref = (1.0 - alpha_eff) * log_probs + alpha_eff * s_task
            else:
                anchored_log_ref = log_probs

            log_p_star = anchored_log_ref + advs / beta
            log_p_star = log_p_star - torch.logsumexp(log_p_star, dim=0)

            item["log_p_star"] = log_p_star.tolist()

        return fit_data

    def _get_task_prompts(self) -> list[str]:
        """Extract user-turn prompts from a fixed sample of task replay data."""
        if self._task_anchor_prompts is not None:
            return self._task_anchor_prompts

        task_data = getattr(self, "_task_data_for_anchor", None)
        if not task_data:
            self._task_anchor_prompts = []
            return self._task_anchor_prompts

        n = min(self._task_anchor_n_samples, len(task_data))
        rng = random.Random(1234)  # stable across calls in one run
        sampled = rng.sample(task_data, n)

        prompts: list[str] = []
        for ex in sampled:
            msgs = ex.get("messages", [])
            # Take the first user turn as the anchoring context.
            for m in msgs:
                if m.get("role") == "user" and m.get("content"):
                    prompts.append(m["content"])
                    break

        self._task_anchor_prompts = prompts
        return prompts

    def _compute_task_anchor_scores(
        self,
        ref_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        responses: list[str],
        task_prompts: list[str],
    ) -> torch.Tensor:
        """Compute s_task,j = logmeanexp_i log pi_ref(y_j | x_task_i) for j in 0..K-1.

        Shape: (K,). Gradients not needed — ref_model is frozen and torch.no_grad()
        is applied inside _compute_seq_log_probs_batched when model.training is False.
        """
        per_prompt: list[torch.Tensor] = []
        for x_task in task_prompts:
            lps = _compute_seq_log_probs_batched(ref_model, tokenizer, x_task, responses)
            per_prompt.append(torch.stack(lps).detach().float().cpu())

        stacked = torch.stack(per_prompt, dim=0)  # (n_prompts, K)
        n = torch.tensor(float(stacked.shape[0]))
        s_task = torch.logsumexp(stacked, dim=0) - torch.log(n)  # (K,)
        return s_task
