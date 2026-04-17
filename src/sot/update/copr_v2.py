"""COPR v2: paper-faithful MSE fit loss + gold injection + NLL anchor.

Three improvements over the stock `COPRUpdate` (copr.py):
  1. **Gold injection.** The gold answer is always included in the candidate set
     before ranking. This guarantees at least one correct response even when
     all K self-samples miss. Deduplicates against the sampled K.
  2. **MSE fit loss.** Replaces KL(P_theta || P*) with the paper-faithful
     MSE (log_P_theta - log_P*)^2. MSE gives uniform weight to all candidates;
     KL under-weights low-probability correct responses.
  3. **Gold NLL anchor.** Adds alpha * NLL(gold | question) directly. This is
     the SFT floor the COPR paper includes and our stock implementation lacked.
     Optionally gated to only fire when max sample F1 < threshold.

These changes are additive. The stock COPRUpdate remains in copr.py for the
published-COPR-baseline comparison in the paper.
"""

import copy
import random

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.copr import (
    COPRUpdate,
    _compute_seq_log_probs_batched,
    _token_f1,
    load_copr_cache,
    save_copr_cache,
)


class COPRv2Update(COPRUpdate):
    """COPR with gold injection, MSE fit loss, and gold NLL anchor."""

    @property
    def name(self) -> str:
        return "copr_v2"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        K = cfg.get("K", 8) if cfg else 8
        beta = cfg.get("beta", 0.1) if cfg else 0.1
        replay_pct = cfg.get("replay_pct", 0.05) if cfg else 0.05
        partial_match_threshold = cfg.get("partial_match_threshold", 0.5) if cfg else 0.5
        max_new_tokens = cfg.get("max_new_tokens", 128) if cfg else 128
        lr = cfg.get("training", {}).get("lr", 1e-5) if cfg else 1e-5
        epochs = cfg.get("training", {}).get("epochs", 3) if cfg else 3
        cache_path = cfg.get("cache_path", None) if cfg else None

        # v2 hyperparameters
        gold_nll_alpha = cfg.get("gold_nll_alpha", 0.25) if cfg else 0.25
        gold_nll_gate_threshold = cfg.get("gold_nll_gate_threshold", 0.5) if cfg else 0.5
        always_apply_gold_nll = cfg.get("always_apply_gold_nll", False) if cfg else False

        # Frozen reference model for P* + replay reg
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Sample, gold-inject, rank, compute P* (cached)
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached COPR v2 fit data from {cache_path}")
            fit_data = load_copr_cache(cache_path)
        else:
            print(f"[v2] Sampling {K} responses per fact ({len(fact_qa_pairs)} facts)...")
            fit_data = self._prepare_fit_data(
                model, tokenizer, fact_qa_pairs, K, max_new_tokens, partial_match_threshold
            )

            print("[v2] Computing P* distributions...")
            fit_data = self._compute_p_star(ref_model, tokenizer, fit_data, beta)

            if cache_path:
                save_copr_cache(fit_data, cache_path)
                print(f"Cached COPR v2 fit data to {cache_path}")

        # Replay buffer
        replay_buffer = []
        if task_data:
            n_replay = max(1, int(len(task_data) * replay_pct))
            replay_buffer = random.sample(task_data, min(n_replay, len(task_data)))

        print(f"[v2] Training for {epochs} epochs (alpha={gold_nll_alpha}, gate<{gold_nll_gate_threshold})...")
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        for epoch in range(epochs):
            total_fit_loss = 0.0
            total_gold_nll = 0.0
            total_reg_loss = 0.0
            n_fit = 0
            n_gold = 0
            n_reg = 0

            random.shuffle(fit_data)
            replay_iter = iter(replay_buffer * max(1, len(fit_data) // max(len(replay_buffer), 1)))

            for item in tqdm(fit_data, desc=f"COPR v2 epoch {epoch + 1}/{epochs}"):
                # Fit loss (MSE, not KL)
                fit_loss = self._compute_fit_loss_mse(model, tokenizer, item)
                total_fit_loss += fit_loss.item()
                n_fit += 1
                fit_loss.backward()

                # Gold NLL anchor: always or gated by max sample quality
                should_apply_gold_nll = always_apply_gold_nll
                if not should_apply_gold_nll:
                    # Fire only when the self-sampled responses are all weak.
                    # Gold is in `ranked_responses[-1]` after gold injection.
                    max_f1 = max(
                        _token_f1(r, item["gold_answer"])
                        for r in item["ranked_responses"]
                        if r != item["gold_answer"]
                    ) if len(item["ranked_responses"]) > 1 else 0.0
                    should_apply_gold_nll = max_f1 < gold_nll_gate_threshold

                if should_apply_gold_nll and gold_nll_alpha > 0:
                    gold_nll = self._compute_gold_nll_loss(
                        model, tokenizer, item["question"], item["gold_answer"]
                    )
                    (gold_nll_alpha * gold_nll).backward()
                    total_gold_nll += gold_nll.item()
                    n_gold += 1

                # Replay reg loss
                try:
                    replay_item = next(replay_iter)
                    reg_loss = self._compute_reg_loss(model, ref_model, tokenizer, replay_item)
                    total_reg_loss += reg_loss.item()
                    n_reg += 1
                    reg_loss.backward()
                except StopIteration:
                    pass

                optimizer.step()
                optimizer.zero_grad()

            avg_fit = total_fit_loss / max(n_fit, 1)
            avg_gold = total_gold_nll / max(n_gold, 1) if n_gold else 0.0
            avg_reg = total_reg_loss / max(n_reg, 1)
            print(
                f"  Epoch {epoch + 1}: fit_mse={avg_fit:.4f}, gold_nll={avg_gold:.4f} "
                f"(fired on {n_gold}/{n_fit}), reg={avg_reg:.4f}"
            )

        del ref_model
        torch.cuda.empty_cache()
        return model

    def _prepare_fit_data(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        K: int,
        max_new_tokens: int,
        partial_match_threshold: float,
    ) -> list[dict]:
        """Sample K responses, GOLD-INJECT, dedupe, rank.

        Ensures at least one correct (exact-match) response is in the candidate
        set for every fact. The gold is placed at the top of the ranking.
        """
        model.eval()
        fit_data = []

        for i, qa in enumerate(tqdm(fact_qa_pairs, desc="Sampling responses")):
            question = qa["question"]
            gold_answer = qa["answer"]

            responses = self._sample_responses(model, tokenizer, question, K, max_new_tokens)

            # v2: gold injection + dedup (case-insensitive)
            # Keep top K after dedup — guarantees gold is always present.
            candidates = [gold_answer] + [r for r in responses if r.strip()]
            seen = set()
            unique = []
            for r in candidates:
                key = r.strip().lower()
                if key and key not in seen:
                    unique.append(r)
                    seen.add(key)
            # Cap at K+1 so gold is never dropped (we always include it).
            if len(unique) > K + 1:
                # Keep gold (unique[0]) + first K sampled unique responses
                unique = [unique[0]] + unique[1 : K + 1]

            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

            ranked = self._rank_responses(unique, gold_answer, partial_match_threshold)
            advantages = [(2 * j - len(ranked) + 1) / max(len(ranked), 1) for j in range(len(ranked))]

            fit_data.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "ranked_responses": ranked,
                    "advantages": advantages,
                    "log_p_star": None,
                }
            )

        model.train()
        return fit_data

    def _compute_fit_loss_mse(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        item: dict,
    ) -> torch.Tensor:
        """L_fit = mean_j (log P_theta(y_j|x) - log P*(y_j|x))^2.

        Paper-faithful: COPR uses MSE, not KL. MSE weights all candidates
        equally rather than concentrating on high-probability ones.
        """
        question = item["question"]
        ranked = item["ranked_responses"]
        log_p_star = torch.tensor(
            item["log_p_star"],
            device=next(model.parameters()).device,
            dtype=torch.float32,
        )

        current_log_probs = _compute_seq_log_probs_batched(model, tokenizer, question, ranked)
        current_log_probs = torch.stack(current_log_probs)

        # Normalize both distributions over the K candidates
        log_p_theta = current_log_probs - torch.logsumexp(current_log_probs, dim=0)

        mse = ((log_p_theta - log_p_star) ** 2).mean()
        return mse

    def _compute_gold_nll_loss(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        question: str,
        gold_answer: str,
    ) -> torch.Tensor:
        """NLL of the gold answer tokens conditional on the question prompt.

        Standard causal-LM SFT loss, masked to the assistant tokens.
        """
        chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": gold_answer},
        ]
        full = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        inputs = tokenizer(full, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]           # (1, T-1, V)
        labels = inputs["input_ids"][:, 1:]          # (1, T-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)  # (1, T-1)

        # Mask out the prompt portion; keep only assistant tokens
        attn_mask = inputs["attention_mask"][:, 1:]
        answer_mask = attn_mask.bool().clone()
        if prompt_len - 1 > 0:
            answer_mask[:, : max(0, prompt_len - 1)] = False

        if not answer_mask.any():
            return torch.tensor(0.0, device=token_log_probs.device, requires_grad=True)

        nll = -(token_log_probs * answer_mask).sum() / answer_mask.sum().clamp(min=1)
        return nll
