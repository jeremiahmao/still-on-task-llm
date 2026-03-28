"""COPR-adapted: advantage-weighted policy fitting with factual correctness rankings.

This is the novel contribution. COPR (Zhang et al., ACL Findings 2025) was designed for
continual preference alignment using human-ranked responses. We adapt it for factual
knowledge injection by:
  1. Replacing human preference rankings with factual correctness rankings
  2. Using a 5% task replay buffer for regularization
  3. Training with the COPR fitting + regularization objectives

The adaptation is a testable heuristic — factual answers aren't naturally ordered like
preferences, and the ranking may behave unevenly across relation types.
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
        return "copr_adapted"

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

        # Create frozen reference model for P* computation and regularization
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Step 1: Sample K responses per fact and rank them
        print(f"Sampling {K} responses per fact ({len(fact_qa_pairs)} facts)...")
        fit_data = self._prepare_fit_data(
            model, tokenizer, fact_qa_pairs, K, max_new_tokens, partial_match_threshold
        )

        # Step 2: Compute P* (optimal sampling distribution) from reference model
        print("Computing P* distributions...")
        fit_data = self._compute_p_star(ref_model, tokenizer, fit_data, beta)

        # Step 3: Prepare replay buffer
        replay_buffer = []
        if task_data:
            n_replay = max(1, int(len(task_data) * replay_pct))
            replay_buffer = random.sample(task_data, min(n_replay, len(task_data)))

        # Step 4: Training loop
        print(f"Training for {epochs} epochs...")
        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        for epoch in range(epochs):
            total_fit_loss = 0.0
            total_reg_loss = 0.0
            n_fit = 0
            n_reg = 0

            # Interleave fit and reg batches
            random.shuffle(fit_data)
            replay_iter = iter(replay_buffer * max(1, len(fit_data) // max(len(replay_buffer), 1)))

            for item in tqdm(fit_data, desc=f"COPR epoch {epoch + 1}/{epochs}"):
                # --- Fit loss: KL(P_theta || P*) over ranked responses ---
                fit_loss = self._compute_fit_loss(model, tokenizer, item)
                total_fit_loss += fit_loss.item()
                n_fit += 1

                # --- Reg loss: KL on replay buffer example ---
                reg_loss = torch.tensor(0.0, device=model.device)
                try:
                    replay_item = next(replay_iter)
                    reg_loss = self._compute_reg_loss(model, ref_model, tokenizer, replay_item)
                    total_reg_loss += reg_loss.item()
                    n_reg += 1
                except StopIteration:
                    pass

                loss = fit_loss + reg_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_fit = total_fit_loss / max(n_fit, 1)
            avg_reg = total_reg_loss / max(n_reg, 1)
            print(f"  Epoch {epoch + 1}: fit_loss={avg_fit:.4f}, reg_loss={avg_reg:.4f}")

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

        for qa in tqdm(fact_qa_pairs, desc="Sampling responses"):
            question = qa["question"]
            gold_answer = qa["answer"]

            # Sample K responses with temperature
            responses = self._sample_responses(model, tokenizer, question, K, max_new_tokens)

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
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        responses = []
        for _ in range(K):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()
            responses.append(response)

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

            # Compute sequence-level log-probs from reference model
            log_probs = []
            for resp in ranked:
                lp = _compute_seq_log_prob(ref_model, tokenizer, question, resp)
                log_probs.append(lp)

            log_probs = torch.tensor(log_probs, dtype=torch.float32)
            advs = torch.tensor(advantages, dtype=torch.float32)

            # P*(y_j|x) ∝ exp(log π_{t-1}(y_j|x) + Adv(x,y_j) / beta)
            log_p_star = log_probs + advs / beta
            log_p_star = log_p_star - torch.logsumexp(log_p_star, dim=0)  # Normalize

            item["log_p_star"] = log_p_star.tolist()

        return fit_data

    def _compute_fit_loss(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        item: dict,
    ) -> torch.Tensor:
        """Compute L_fit = KL(P_theta || P*) over the K ranked responses.

        P_theta(y_j|x) = exp(log π_θ(y_j|x)) / Σ_j' exp(log π_θ(y_j'|x))
        """
        question = item["question"]
        ranked = item["ranked_responses"]
        log_p_star = torch.tensor(item["log_p_star"], device=model.device, dtype=torch.float32)

        # Compute current model's log-probs for each ranked response
        current_log_probs = []
        for resp in ranked:
            lp = _compute_seq_log_prob(model, tokenizer, question, resp)
            current_log_probs.append(lp)

        current_log_probs = torch.stack(current_log_probs)

        # Normalize to get P_theta
        log_p_theta = current_log_probs - torch.logsumexp(current_log_probs, dim=0)

        # KL(P_theta || P*) = Σ_j P_theta(j) * (log P_theta(j) - log P*(j))
        p_theta = log_p_theta.exp()
        kl = (p_theta * (log_p_theta - log_p_star)).sum()

        return kl

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
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Current model logits
        outputs = model(**inputs)
        current_logprobs = F.log_softmax(outputs.logits, dim=-1)

        # Reference model logits
        with torch.no_grad():
            ref_outputs = ref_model(**inputs)
            ref_probs = F.softmax(ref_outputs.logits, dim=-1)

        # Per-token KL, averaged
        kl = F.kl_div(current_logprobs, ref_probs, reduction="batchmean", log_target=False)
        return kl


def _compute_seq_log_prob(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    question: str,
    response: str,
) -> torch.Tensor:
    """Compute the sequence-level log probability: log π(response | question).

    Returns a scalar tensor.
    """
    chat = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get prompt length to only compute log-prob over the response tokens
    prompt_chat = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_chat, tokenize=False, add_generation_prompt=True
    )
    prompt_len = len(tokenizer(prompt_text)["input_ids"])

    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(**inputs)

    # Shift logits and labels for next-token prediction
    logits = outputs.logits[:, :-1, :]  # (1, seq_len-1, vocab)
    labels = inputs["input_ids"][:, 1:]  # (1, seq_len-1)

    # Only compute log-prob over response tokens (after prompt)
    response_start = max(0, prompt_len - 1)
    logits = logits[:, response_start:, :]
    labels = labels[:, response_start:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

    # Sum over response tokens to get sequence log-prob
    seq_log_prob = token_log_probs.sum()

    return seq_log_prob


def _token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 between prediction and gold answer."""
    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
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
