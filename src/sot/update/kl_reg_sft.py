"""KL-regularized SFT: SFT on fact QA + lambda * KL(pi_ref || pi_theta) on task replay.

Previous implementation computed KL on the fact batches themselves, which anchors
the model to its pre-update distribution ON THE FACTS — the opposite of what we
want. Fixed to compute KL on task_data (replay) batches, mirroring COPR's
regularization term. Now parallel to COPR: SFT fits facts, KL protects task.
"""

import copy
import random

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.base import UpdateMethod


class KLRegSFTUpdate(UpdateMethod):
    @property
    def name(self) -> str:
        return "kl_reg_sft"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        """SFT on facts + KL(pi_ref || pi_theta) on task replay.

        Training loss per step = L_sft(fact_batch) + lambda * KL_task(replay_batch)
        KL is computed on task-replay inputs (where pi_ref is the task-tuned
        distribution we want to preserve), not on the fact batches.
        """
        kl_lambda = cfg.get("kl_lambda", 0.1) if cfg else 0.1
        replay_pct = cfg.get("replay_pct", 0.05) if cfg else 0.05
        lr = cfg.get("training", {}).get("lr", 2e-5) if cfg else 2e-5
        epochs = cfg.get("training", {}).get("epochs", 3) if cfg else 3
        batch_size = cfg.get("training", {}).get("batch_size", 8) if cfg else 8
        max_seq_length = cfg.get("training", {}).get("max_seq_length", 512) if cfg else 512

        # Frozen reference (clone of the task-tuned model)
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Fact-SFT texts. Pre-rendered chats (chat_messages for K=5 DSAE Lite,
        # qd_messages for legacy K=2 fi_sft) are used as-is; standard entries
        # build the usual {user: question, assistant: answer} chat.
        fact_texts = []
        for qa in fact_qa_pairs:
            if qa.get("chat_messages"):
                chat = qa["chat_messages"]
            elif qa.get("train_format") == "qd" and qa.get("qd_messages"):
                chat = qa["qd_messages"]
            else:
                chat = [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": qa["answer"]},
                ]
            fact_texts.append(
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            )

        # Task-replay texts (for KL regularization)
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
        else:
            print("  WARNING: no task_data provided; KL regularization disabled.")

        def collate_fn(batch_texts):
            orig_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "right"
            result = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            tokenizer.padding_side = orig_padding_side
            return result

        fact_loader = DataLoader(
            fact_texts, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        # Replay loader cycles through the replay set in lockstep with fact batches
        replay_loader = (
            DataLoader(replay_texts, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            if replay_texts
            else None
        )

        model.train()
        # LoRA params are the only ones with requires_grad=True after apply_lora in 09_run_update.py
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        for epoch in range(epochs):
            total_sft = 0.0
            total_kl = 0.0
            n_steps = 0
            n_kl = 0
            optimizer.zero_grad()

            # Cycle through replay batches in parallel with fact batches
            replay_iter = iter(replay_loader) if replay_loader else None

            for fact_batch in tqdm(fact_loader, desc=f"KL-reg SFT epoch {epoch + 1}/{epochs}"):
                first_device = next(model.parameters()).device
                fact_batch = {k: v.to(first_device) for k, v in fact_batch.items()}
                labels = fact_batch["input_ids"].clone()
                labels[fact_batch["attention_mask"] == 0] = -100

                # SFT loss on fact batch
                outputs = model(**fact_batch, labels=labels)
                sft_loss = outputs.loss
                total_sft += sft_loss.item()
                n_steps += 1
                sft_loss.backward()

                # KL regularization on a task-replay batch (protects task distribution)
                if replay_iter is not None and kl_lambda > 0:
                    try:
                        replay_batch = next(replay_iter)
                    except StopIteration:
                        replay_iter = iter(replay_loader)
                        replay_batch = next(replay_iter)
                    replay_batch = {k: v.to(first_device) for k, v in replay_batch.items()}

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

            avg_sft = total_sft / max(n_steps, 1)
            avg_kl = total_kl / max(n_kl, 1) if n_kl else 0.0
            print(
                f"  Epoch {epoch + 1}: sft_loss={avg_sft:.4f}, "
                f"task_kl={avg_kl:.4f} (fired on {n_kl}/{n_steps})"
            )

        del ref_model
        torch.cuda.empty_cache()

        return model
