"""KL-regularized SFT: SFT + lambda * KL(pi_task || pi_theta)."""

import copy

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
        """SFT on fact QA pairs with KL divergence regularization against the task-tuned model.

        The frozen reference model provides the target distribution. The training loss is:
            L = L_sft + lambda * KL(ref_logits || current_logits)
        """
        kl_lambda = cfg.get("kl_lambda", 0.1) if cfg else 0.1
        lr = cfg.get("training", {}).get("lr", 2e-5) if cfg else 2e-5
        epochs = cfg.get("training", {}).get("epochs", 3) if cfg else 3
        batch_size = cfg.get("training", {}).get("batch_size", 8) if cfg else 8
        max_seq_length = cfg.get("training", {}).get("max_seq_length", 512) if cfg else 512

        # Create frozen reference (clone of the task-tuned model).
        # LoRA is zero-initialised so ref_model == task-tuned base at this point.
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Prepare chat-formatted texts
        texts = []
        for qa in fact_qa_pairs:
            chat = [
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
            texts.append(
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            )

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

        dataloader = DataLoader(texts, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        model.train()
        # LoRA params are the only ones with requires_grad=True after apply_lora in 09_run_update.py
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        for epoch in range(epochs):
            total_loss = 0.0
            optimizer.zero_grad()

            for batch in tqdm(dataloader, desc=f"KL-reg SFT epoch {epoch + 1}/{epochs}"):
                # With device_map="auto", input tensors go to the first device
                first_device = next(model.parameters()).device
                batch = {k: v.to(first_device) for k, v in batch.items()}
                labels = batch["input_ids"].clone()
                # Mask padding using attention_mask (avoids masking real EOS tokens
                # when pad_token_id == eos_token_id, as with Qwen)
                labels[batch["attention_mask"] == 0] = -100

                outputs = model(**batch, labels=labels)
                sft_loss = outputs.loss

                with torch.no_grad():
                    ref_outputs = ref_model(**batch)

                current_logprobs = F.log_softmax(outputs.logits, dim=-1)
                ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
                # Per-token KL, averaged over real (non-padding) tokens only.
                # F.kl_div with batchmean divides only by batch size, leaving the
                # loss ~seq_len × per_token_KL — orders of magnitude larger than
                # sft_loss. Compute per-position and average over real tokens.
                kl_per_pos = (ref_logprobs.exp() * (ref_logprobs - current_logprobs)).sum(dim=-1)
                n_real = batch["attention_mask"].sum().clamp(min=1)
                kl_loss = (kl_per_pos * batch["attention_mask"]).sum() / n_real

                loss = sft_loss + kl_lambda * kl_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / max(len(dataloader), 1)
            print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        del ref_model
        torch.cuda.empty_cache()

        return model
