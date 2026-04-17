"""COPR-adapted: paper-faithful implementation for knowledge injection.

COPR (Zhang et al., ACL Findings 2025) was designed for continual preference alignment
using human-ranked responses. We adapt it for factual knowledge injection by:
  1. Replacing human preference rankings with factual correctness rankings (EM/F1)
  2. Using a 5% task replay buffer for regularization (KL against task-optimal)
  3. Training with the COPR objective: MSE fit loss + SFT anchor on top candidate

This implementation follows the paper exactly:
  - Fit loss is MSE `mean((log_P_theta - log_P*)^2)`, NOT KL divergence.
  - An NLL anchor on the top-ranked response (typically the gold answer when
    available) is added with weight `gold_nll_alpha`. This is the SFT floor the
    paper specifies to prevent fit-loss collapse.

The gold-injection variant (see copr_gold_injection.py) extends this with an
additional candidate-set augmentation — that is our novel knowledge-injection
contribution, not in the paper.
"""

import copy
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.base import UpdateMethod


class COPRUpdate(UpdateMethod):
    @property
    def name(self) -> str:
        return "copr"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        """Apply COPR-adapted knowledge update.

        Pipeline:
        1. Sample K responses per fact from the task-tuned model
        2. Rank by factual correctness
        3. Compute advantages and optimal sampling distribution P*
        4. Train with L_fit (on new facts) + L_reg (on task replay buffer)
        """
        K = cfg.get("K", 8) if cfg else 8
        beta = cfg.get("beta", 0.1) if cfg else 0.1
        replay_pct = cfg.get("replay_pct", 0.05) if cfg else 0.05
        partial_match_threshold = cfg.get("partial_match_threshold", 0.5) if cfg else 0.5
        max_new_tokens = cfg.get("max_new_tokens", 128) if cfg else 128
        lr = cfg.get("training", {}).get("lr", 1e-5) if cfg else 1e-5
        epochs = cfg.get("training", {}).get("epochs", 3) if cfg else 3
        cache_path = cfg.get("cache_path", None) if cfg else None
        # Paper-faithful SFT anchor on the top-ranked response
        gold_nll_alpha = cfg.get("gold_nll_alpha", 0.25) if cfg else 0.25
        gold_nll_gate_threshold = cfg.get("gold_nll_gate_threshold", 0.5) if cfg else 0.5
        always_apply_gold_nll = cfg.get("always_apply_gold_nll", False) if cfg else False

        # Create frozen reference model for P* computation and regularization
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Steps 1-2: Sample, rank, and compute P*.
        # Caches results to disk so an OOM during training doesn't waste hours of sampling.
        if cache_path and Path(cache_path).exists():
            print(f"Loading cached COPR fit data from {cache_path}")
            fit_data = load_copr_cache(cache_path)
        else:
            print(f"Sampling {K} responses per fact ({len(fact_qa_pairs)} facts)...")
            fit_data = self._prepare_fit_data(
                model, tokenizer, fact_qa_pairs, K, max_new_tokens, partial_match_threshold
            )

            print("Computing P* distributions...")
            fit_data = self._compute_p_star(ref_model, tokenizer, fit_data, beta)

            if cache_path:
                save_copr_cache(fit_data, cache_path)
                print(f"Cached COPR fit data to {cache_path}")

        # Step 3: Prepare replay buffer
        replay_buffer = []
        if task_data:
            n_replay = max(1, int(len(task_data) * replay_pct))
            replay_buffer = random.sample(task_data, min(n_replay, len(task_data)))

        # Step 4: Training loop
        print(f"Training for {epochs} epochs...")
        model.train()
        # LoRA params are the only ones with requires_grad=True after apply_lora in 09_run_update.py
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        for epoch in range(epochs):
            total_fit_loss = 0.0
            total_gold_nll = 0.0
            total_reg_loss = 0.0
            n_fit = 0
            n_gold = 0
            n_reg = 0

            # Interleave fit and reg batches
            random.shuffle(fit_data)
            replay_iter = iter(replay_buffer * max(1, len(fit_data) // max(len(replay_buffer), 1)))

            for item in tqdm(fit_data, desc=f"COPR epoch {epoch + 1}/{epochs}"):
                # --- Fit loss: MSE on normalized log-probs vs log P* ---
                fit_loss = self._compute_fit_loss(model, tokenizer, item)
                total_fit_loss += fit_loss.item()
                n_fit += 1
                fit_loss.backward()

                # --- Gold NLL anchor (paper's SFT floor) ---
                # Fire when self-samples are all weak, unless config overrides.
                apply_anchor = always_apply_gold_nll
                if not apply_anchor:
                    # Look at sample quality excluding the gold itself. Compare
                    # using the same normalization as `_rank_responses`
                    # (case+whitespace insensitive) to avoid mis-classifying a
                    # case-variant of gold as a self-sample.
                    gold_key = item["gold_answer"].strip().lower()
                    non_gold = [
                        r for r in item["ranked_responses"]
                        if r.strip().lower() != gold_key
                    ]
                    max_f1 = (
                        max(_token_f1(r, item["gold_answer"]) for r in non_gold)
                        if non_gold
                        else 0.0
                    )
                    apply_anchor = max_f1 < gold_nll_gate_threshold

                if apply_anchor and gold_nll_alpha > 0:
                    gold_nll = self._compute_gold_nll_loss(
                        model, tokenizer, item["question"], item["gold_answer"]
                    )
                    (gold_nll_alpha * gold_nll).backward()
                    total_gold_nll += gold_nll.item()
                    n_gold += 1

                # --- Reg loss: KL on replay buffer example ---
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
                f"  Epoch {epoch + 1}: fit_mse={avg_fit:.4f}, "
                f"gold_nll={avg_gold:.4f} (fired on {n_gold}/{n_fit}), "
                f"reg_loss={avg_reg:.4f}"
            )

        # Clean up
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
        """Sample K responses per fact and rank by factual correctness."""
        model.eval()
        fit_data = []

        for i, qa in enumerate(tqdm(fact_qa_pairs, desc="Sampling responses")):
            question = qa["question"]
            gold_answer = qa["answer"]

            # Sample K responses with temperature
            responses = self._sample_responses(model, tokenizer, question, K, max_new_tokens)

            # Periodically free fragmented CUDA memory to prevent slow OOM
            # accumulation over thousands of generate() calls
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()

            # Rank by factual correctness
            ranked = self._rank_responses(responses, gold_answer, partial_match_threshold)

            # Compute linear advantages: worst (j=0) to best (j=K-1)
            advantages = [(2 * j - K + 1) / K for j in range(len(ranked))]

            fit_data.append(
                {
                    "question": question,
                    "gold_answer": gold_answer,
                    "ranked_responses": ranked,
                    "advantages": advantages,
                    "log_p_star": None,  # Computed later
                }
            )

        model.train()
        return fit_data

    def _sample_responses(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        question: str,
        K: int,
        max_new_tokens: int,
    ) -> list[str]:
        """Sample K diverse responses from the model."""
        chat = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=True,
                num_return_sequences=K,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        responses = [
            tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
            for i in range(K)
        ]
        return responses

    def _rank_responses(
        self,
        responses: list[str],
        gold_answer: str,
        threshold: float,
    ) -> list[str]:
        """Rank responses by factual correctness: worst first, best last.

        Ranking tiers:
        - Exact match -> top rank
        - Token F1 > threshold -> middle
        - Everything else -> bottom
        """
        scored = []
        for resp in responses:
            f1 = _token_f1(resp, gold_answer)
            exact = resp.strip().lower() == gold_answer.strip().lower()
            # Score: 2 for exact, 1 for partial, 0 for wrong
            score = 2 if exact else (1 if f1 > threshold else 0)
            scored.append((score, f1, resp))

        # Sort: worst first (ascending), so index = rank position
        scored.sort(key=lambda x: (x[0], x[1]))
        return [s[2] for s in scored]

    def _compute_p_star(
        self,
        ref_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fit_data: list[dict],
        beta: float,
    ) -> list[dict]:
        """Compute the optimal sampling distribution P* for each fact.

        P*(y_j|x) ∝ π_{t-1}(y_j|x) * exp(Adv(x, y_j) / beta)

        This is pre-computed from the frozen reference model and cached.
        """
        for item in tqdm(fit_data, desc="Computing P*"):
            question = item["question"]
            ranked = item["ranked_responses"]
            advantages = item["advantages"]

            # Batch all K responses for this question into one forward pass
            log_probs = _compute_seq_log_probs_batched(ref_model, tokenizer, question, ranked)
            log_probs = torch.stack(log_probs)
            advs = torch.tensor(advantages, device=log_probs.device, dtype=torch.float32)

            log_p_star = log_probs + advs / beta
            log_p_star = log_p_star - torch.logsumexp(log_p_star, dim=0)

            item["log_p_star"] = log_p_star.tolist()

        return fit_data

    def _compute_fit_loss(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        item: dict,
    ) -> torch.Tensor:
        """Compute L_fit = mean_j (log P_theta(y_j|x) - log P*(y_j|x))^2.

        Paper-faithful MSE: uniform weight across all K candidates. KL would
        under-weight low-probability responses that happen to be correct.
        """
        question = item["question"]
        ranked = item["ranked_responses"]
        log_p_star = torch.tensor(
            item["log_p_star"],
            device=next(model.parameters()).device,
            dtype=torch.float32,
        )

        # Batch all K responses into one forward pass with gradients enabled
        current_log_probs = _compute_seq_log_probs_batched(model, tokenizer, question, ranked)
        current_log_probs = torch.stack(current_log_probs)

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

        This is the SFT anchor term from the COPR paper — prevents fit_loss
        from collapsing when the candidate set is weak.
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
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

        attn_mask = inputs["attention_mask"][:, 1:]
        answer_mask = attn_mask.bool().clone()
        if prompt_len - 1 > 0:
            answer_mask[:, : max(0, prompt_len - 1)] = False

        if not answer_mask.any():
            return torch.tensor(0.0, device=token_log_probs.device, requires_grad=True)

        nll = -(token_log_probs * answer_mask).sum() / answer_mask.sum().clamp(min=1)
        return nll

    def _compute_reg_loss(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        replay_item: dict,
    ) -> torch.Tensor:
        """Compute L_reg: KL divergence on a replay buffer example.

        Encourages the model to stay close to the task-optimal distribution
        on task-relevant inputs.
        """
        messages = replay_item.get("messages", [])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        # Current model logits
        outputs = model(**inputs)
        current_logprobs = F.log_softmax(outputs.logits, dim=-1)

        # Reference model logits
        with torch.no_grad():
            ref_outputs = ref_model(**inputs)
            ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)

        # KL(pi_theta || pi_{t-1}) per the COPR paper: penalizes the current model
        # for placing mass where the reference has low mass (mode-seeking).
        # Averaged over real tokens only to keep reg_loss on the same scale as fit_loss.
        kl_per_pos = (current_logprobs.exp() * (current_logprobs - ref_logprobs)).sum(dim=-1)
        attn_mask = inputs["attention_mask"]
        kl = (kl_per_pos * attn_mask).sum() / attn_mask.sum().clamp(min=1)
        return kl


def _compute_seq_log_probs_batched(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    question: str,
    responses: list[str],
    max_seq_length: int = 512,
    chunk_size: int = 2,
) -> list[torch.Tensor]:
    """Compute sequence-level log-probs for K responses, chunked to avoid OOM.

    With K=8 responses and gradients enabled, a single forward pass through
    a 3B model needs ~10GB of activation memory on top of the ~12GB for
    model + ref_model. Chunking into groups of `chunk_size` keeps peak
    memory manageable on a 24GB GPU.

    Uses right-padding (easier to mask) temporarily, then restores the original side.
    Returns a list of K scalar tensors — gradients are preserved when model.training.
    """
    # Build full (question + response) texts
    full_texts = []
    for resp in responses:
        chat = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": resp},
        ]
        full_texts.append(
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        )

    # Get prompt length once (same question for all responses)
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    seq_log_probs: list[torch.Tensor] = []

    for chunk_start in range(0, len(full_texts), chunk_size):
        chunk_texts = full_texts[chunk_start : chunk_start + chunk_size]

        # Use right-padding for scoring (simpler index arithmetic)
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        inputs = tokenizer(
            chunk_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        tokenizer.padding_side = orig_padding_side

        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        # Forward pass; keep grad graph when model is in training mode
        if model.training:
            outputs = model(**inputs)
        else:
            with torch.no_grad():
                outputs = model(**inputs)

        logits = outputs.logits[:, :-1, :]       # (C, T-1, V)
        label_ids = inputs["input_ids"][:, 1:]   # (C, T-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, label_ids.unsqueeze(-1)).squeeze(-1)  # (C, T-1)

        attn_mask = inputs["attention_mask"][:, 1:]  # (C, T-1)
        for i in range(len(chunk_texts)):
            response_start = max(0, prompt_len - 1)
            mask = attn_mask[i].bool().clone()
            mask[:response_start] = False
            if mask.any():
                seq_lp = token_log_probs[i][mask].sum()
            else:
                seq_lp = torch.tensor(-1e6, device=token_log_probs.device, dtype=token_log_probs.dtype)
                if model.training:
                    seq_lp = seq_lp.requires_grad_(False)
            seq_log_probs.append(seq_lp)

    return seq_log_probs


def _token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer."""
    from collections import Counter

    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def save_copr_cache(fit_data: list[dict], path: str | Path) -> None:
    """Save pre-computed COPR data (ranked responses + P*) for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(fit_data, f, indent=2)


def load_copr_cache(path: str | Path) -> list[dict]:
    """Load pre-computed COPR data."""
    with open(path) as f:
        return json.load(f)
