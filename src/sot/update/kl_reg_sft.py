"""KL-regularized SFT: SFT + lambda * KL(pi_task || pi_theta)."""

import copy

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
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
        max_seq_length = cfg.get("training", {}).get("max_seq_length", 512) if cfg else 512

        # Create frozen reference (clone of the task-tuned model)
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Prepare data
        texts = []
        for qa in fact_qa_pairs:
            chat = [
                {"role": "user", "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
            text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            texts.append(text)

        model.train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

        for epoch in range(epochs):
            total_loss = 0.0
            for text in tqdm(texts, desc=f"KL-reg SFT epoch {epoch + 1}/{epochs}"):
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_length,
                    padding=False,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                labels = inputs["input_ids"].clone()

                # Current model forward
                outputs = model(**inputs, labels=labels)
                sft_loss = outputs.loss

                # Reference model forward (no grad)
                with torch.no_grad():
                    ref_outputs = ref_model(**inputs)

                # KL divergence on logits
                current_logprobs = F.log_softmax(outputs.logits, dim=-1)
                ref_logprobs = F.log_softmax(ref_outputs.logits, dim=-1)
                kl_loss = F.kl_div(
                    current_logprobs,
                    ref_logprobs.exp(),
                    reduction="batchmean",
                    log_target=False,
                )

                loss = sft_loss + kl_lambda * kl_loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / max(len(texts), 1)
            print(f"  Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

        # Clean up reference model
        del ref_model
        torch.cuda.empty_cache()

        return model
