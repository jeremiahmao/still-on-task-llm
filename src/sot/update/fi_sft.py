"""Format-Invariant SFT (FI-SFT): the flipped-COPR ablation.

Conceptual motivation:
  COPR varies K candidate ANSWERS per prompt and anchors their ranking against
  a reference policy. Our data (Phase 7b) shows this widens the behavioral
  format gap rather than closing it. FI-SFT flips the structure: vary K
  candidate PROMPT FORMATS per answer, and explicitly penalize cross-format
  variance of log pi_theta(y_i | F_k(x_i)). Combined with KL-reg SFT's KL-replay
  side regularizer, the total loss is:

      L = sum_k CE(y_i | F_k(x_i))                    # mean cross-format CE
        + mu * Var_k [ -CE(y_i | F_k(x_i)) ]          # cross-format variance penalty
        + lambda * KL(pi_theta || pi_ref) on replay   # unchanged from kl_reg_sft

  -CE(y | F(x)) is the sum log-prob of the answer tokens under format F. The
  variance term punishes any fact for which the answer log-prob differs across
  formats --- i.e., format-coupling.

Input contract:
  fact_qa_pairs (from scripts/09_run_update.py) must include both a QA-format
  entry and a QD-format entry per fact, produced by
  scripts/24_prepare_mixed_format_triples.py. Entries are grouped into pairs
  by (subject, relation, object); each pair is the unit of training.

Config knobs (via cfg):
  mu                 (default 0.5)   variance-penalty weight
  kl_lambda          (default 0.1)   inherited from kl_reg_sft
  replay_pct         (default 0.05)  inherited from kl_reg_sft
  training.lr        (default 2e-5)
  training.epochs    (default 3)
  training.max_seq_length (default 512)

Note: batch_size is measured in fact PAIRS, not individual examples. Memory
scales as 2 * batch_size because each pair forwards twice.
"""

import copy
import random
from collections import OrderedDict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.base import UpdateMethod


def _fact_key(qa: dict) -> tuple:
    t = qa.get("triple", {})
    return (t.get("subject", ""), t.get("relation", ""), t.get("object", ""))


def _group_by_fact(fact_qa_pairs: list[dict]) -> list[list[dict]]:
    """Group entries by (subject, relation, object). Each group should contain
    >=2 entries corresponding to different prompt formats of the same fact.
    """
    groups: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for qa in fact_qa_pairs:
        groups.setdefault(_fact_key(qa), []).append(qa)
    return [g for g in groups.values() if len(g) >= 2]


def _render_chat(tokenizer, qa: dict) -> str:
    if qa.get("train_format") == "qd" and qa.get("qd_messages"):
        chat = qa["qd_messages"]
    else:
        chat = [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)


def _per_example_ce(outputs_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute per-example mean cross-entropy (averaged over non-pad positions).

    Returns a 1-D tensor of length B.
    """
    # Shift for causal LM next-token prediction.
    shift_logits = outputs_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)
    flat_labels = shift_labels.view(-1)
    flat_ce = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="none")
    flat_ce = flat_ce.view(B, T)
    mask = (shift_labels != -100).float()
    n_real = mask.sum(dim=1).clamp(min=1.0)
    return (flat_ce * mask).sum(dim=1) / n_real


class FISFTUpdate(UpdateMethod):
    @property
    def name(self) -> str:
        return "fi_sft"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        mu = cfg.get("mu", 0.5) if cfg else 0.5
        kl_lambda = cfg.get("kl_lambda", 0.1) if cfg else 0.1
        replay_pct = cfg.get("replay_pct", 0.05) if cfg else 0.05
        lr = cfg.get("training", {}).get("lr", 2e-5) if cfg else 2e-5
        epochs = cfg.get("training", {}).get("epochs", 3) if cfg else 3
        pair_batch_size = cfg.get("training", {}).get("batch_size", 4) if cfg else 4
        max_seq_length = cfg.get("training", {}).get("max_seq_length", 512) if cfg else 512

        groups = _group_by_fact(fact_qa_pairs)
        if not groups:
            raise RuntimeError(
                "FI-SFT requires fact_qa_pairs grouped by fact with >=2 formats each. "
                "Run scripts/24_prepare_mixed_format_triples.py first."
            )
        print(f"  FI-SFT: {len(groups)} fact groups with {sum(len(g) for g in groups)} total examples")

        # Frozen reference for KL replay.
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Task-replay texts (identical to kl_reg_sft).
        replay_texts: list[str] = []
        if task_data:
            n_replay = max(1, int(len(task_data) * replay_pct))
            replay_sample = random.sample(task_data, min(n_replay, len(task_data)))
            for item in replay_sample:
                msgs = item.get("messages", [])
                if msgs:
                    replay_texts.append(
                        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    )
            print(f"  Task-replay KL anchor: {len(replay_texts)} examples")

        def _tok(texts: list[str]) -> dict:
            orig = tokenizer.padding_side
            tokenizer.padding_side = "right"
            out = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            tokenizer.padding_side = orig
            return out

        replay_loader = None
        if replay_texts:
            def _replay_collate(batch_texts):
                return _tok(batch_texts)
            replay_loader = DataLoader(
                replay_texts, batch_size=pair_batch_size, shuffle=True, collate_fn=_replay_collate
            )

        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        for epoch in range(epochs):
            random.shuffle(groups)
            total_ce = 0.0
            total_var = 0.0
            total_kl = 0.0
            n_steps = 0
            n_kl = 0
            replay_iter = iter(replay_loader) if replay_loader else None

            for i in tqdm(
                range(0, len(groups), pair_batch_size),
                desc=f"FI-SFT epoch {epoch + 1}/{epochs}",
            ):
                batch_groups = groups[i : i + pair_batch_size]

                # Flatten groups: each group contributes K examples (one per format).
                # Record which items belong to which group for variance computation.
                flat_entries = []
                group_slices = []
                cursor = 0
                for g in batch_groups:
                    flat_entries.extend(g)
                    group_slices.append((cursor, cursor + len(g)))
                    cursor += len(g)

                texts = [_render_chat(tokenizer, e) for e in flat_entries]
                batch = _tok(texts)
                device = next(model.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"] == 0] = -100

                outputs = model(**batch)
                per_ex_ce = _per_example_ce(outputs.logits, labels)  # (N,)

                mean_ce = per_ex_ce.mean()
                # Variance of -CE across formats per group, then mean across groups.
                per_group_var = []
                for s, e in group_slices:
                    if e - s >= 2:
                        neg_ce = -per_ex_ce[s:e]
                        per_group_var.append(neg_ce.var(unbiased=False))
                var_term = (
                    torch.stack(per_group_var).mean()
                    if per_group_var
                    else torch.tensor(0.0, device=device)
                )

                loss = mean_ce + mu * var_term
                total_ce += mean_ce.item()
                total_var += var_term.item()
                n_steps += 1
                loss.backward()

                if replay_iter is not None and kl_lambda > 0:
                    try:
                        replay_batch = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        replay_batch = next(replay_iter)
                    replay_batch = {k: v.to(device) for k, v in replay_batch.items()}
                    cur_out = model(**replay_batch)
                    with torch.no_grad():
                        ref_out = ref_model(**replay_batch)
                    cur_lp = F.log_softmax(cur_out.logits, dim=-1)
                    ref_lp = F.log_softmax(ref_out.logits, dim=-1)
                    kl_per_pos = (ref_lp.exp() * (ref_lp - cur_lp)).sum(dim=-1)
                    n_real = replay_batch["attention_mask"].sum().clamp(min=1)
                    kl_loss = (kl_per_pos * replay_batch["attention_mask"]).sum() / n_real
                    total_kl += kl_loss.item()
                    n_kl += 1
                    (kl_lambda * kl_loss).backward()

                optimizer.step()
                optimizer.zero_grad()

            print(
                f"  Epoch {epoch + 1}: ce={total_ce / max(n_steps, 1):.4f}, "
                f"var={total_var / max(n_steps, 1):.4f}, "
                f"task_kl={(total_kl / max(n_kl, 1)) if n_kl else 0.0:.4f} "
                f"(kl fired on {n_kl}/{n_steps})"
            )

        del ref_model
        torch.cuda.empty_cache()
        return model
