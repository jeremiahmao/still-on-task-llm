"""DSAE Lite: K=5 augmented injection + K=5 augmented KL preservation.

The novel ingredient is the preservation side: instead of computing KL on a
single rendering of each task-replay example (the standard approach used by
KLRegSFTUpdate), the loss averages KL across K different prompt-framing
renderings of the same replay example. This is meant to penalize *format-
selective* drift — drift in the model's task distribution that a single-format
KL anchor cannot detect.

Formal: L_KL = (1/K) * Σ_k  KL(π_ref(.|G_k(x))  ||  π_θ(.|G_k(x)))

where G_1, ..., G_K are K prompt-framing transforms applied to the same task
example x. This proposal has no precedent in the literature (confirmed by the
DSAE review's exhaustive search; see paper/ml_intern_dsae_review.md §3).

The K=5 framings live in `_PRESERVATION_FRAMINGS` and are intentionally distinct
from the K=5 *injection* templates in `scripts/24_prepare_mixed_format_triples.py`:
preservation tests robustness across system-prompt / instruction wrappings of
the *same* user question, while injection tests robustness across surface
renderings of the *same* fact.
"""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel

from sot.update.base import UpdateMethod


_DEFAULT_SYSTEM = (
    "You are a financial search expert. Given a complex financial question, "
    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
    "relevant documents from a financial news database."
)


def _framing_original(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _DEFAULT_SYSTEM},
        {"role": "user", "content": question},
    ]


def _framing_bare(question: str) -> list[dict]:
    return [{"role": "user", "content": question}]


def _framing_analyst(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"You are a financial analyst. Answer concisely: {question}",
        }
    ]


def _framing_detailed(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "Given the following question, provide a detailed response:\n\n"
                f"{question}"
            ),
        }
    ]


def _framing_request(question: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": f"Question: {question}\nPlease provide your analysis.",
        }
    ]


_PRESERVATION_FRAMINGS: list[tuple[str, callable]] = [
    ("original", _framing_original),
    ("bare", _framing_bare),
    ("analyst", _framing_analyst),
    ("detailed", _framing_detailed),
    ("request", _framing_request),
]


def _extract_user_question(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "user":
            return m.get("content")
    return None


class DSAELiteUpdate(UpdateMethod):
    @property
    def name(self) -> str:
        return "dsae_lite"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel:
        kl_lambda = cfg.get("kl_lambda", 0.1) if cfg else 0.1
        replay_pct = cfg.get("replay_pct", 0.05) if cfg else 0.05
        num_kl_formats = cfg.get("num_kl_formats", 5) if cfg else 5
        lr = cfg.get("training", {}).get("lr", 2e-5) if cfg else 2e-5
        epochs = cfg.get("training", {}).get("epochs", 3) if cfg else 3
        batch_size = cfg.get("training", {}).get("batch_size", 8) if cfg else 8
        max_seq_length = cfg.get("training", {}).get("max_seq_length", 512) if cfg else 512
        log_per_format_kl = cfg.get("log_per_format_kl", True) if cfg else True

        framings = _PRESERVATION_FRAMINGS[:num_kl_formats]

        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

        # Fact-SFT texts. Same multi-format dispatch as kl_reg_sft.
        fact_texts: list[str] = []
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

        # Task-replay framings: per replay example, render under each of the K
        # framings. Each framing produces a separate text list; we tokenize them
        # in parallel batches at training time.
        replay_questions: list[str] = []
        if task_data:
            n_replay = max(1, int(len(task_data) * replay_pct))
            replay_sample = random.sample(task_data, min(n_replay, len(task_data)))
            for item in replay_sample:
                q = _extract_user_question(item.get("messages", []))
                if q:
                    replay_questions.append(q)
            print(
                f"  Task-replay anchor: {len(replay_questions)} questions "
                f"x K={num_kl_formats} framings"
            )
        else:
            print("  WARNING: no task_data provided; KL regularization disabled.")

        per_framing_texts: dict[str, list[str]] = {}
        for fmt_name, fn in framings:
            per_framing_texts[fmt_name] = [
                tokenizer.apply_chat_template(
                    fn(q), tokenize=False, add_generation_prompt=False
                )
                for q in replay_questions
            ]

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
        # One shuffled iterator per framing. The same shuffling order is fine —
        # the formats are independent random samples over the replay set.
        framing_loaders = {
            fmt_name: (
                DataLoader(texts, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                if texts
                else None
            )
            for fmt_name, texts in per_framing_texts.items()
        }

        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )

        per_format_kl_log: list[dict] = []

        for epoch in range(epochs):
            total_sft = 0.0
            total_kl = 0.0
            n_steps = 0
            n_kl = 0
            optimizer.zero_grad()

            framing_iters = {
                fmt: iter(loader) for fmt, loader in framing_loaders.items() if loader
            }
            per_fmt_running = defaultdict(list)

            for fact_batch in tqdm(
                fact_loader, desc=f"DSAE-Lite epoch {epoch + 1}/{epochs}"
            ):
                first_device = next(model.parameters()).device
                fact_batch = {k: v.to(first_device) for k, v in fact_batch.items()}
                labels = fact_batch["input_ids"].clone()
                labels[fact_batch["attention_mask"] == 0] = -100

                outputs = model(**fact_batch, labels=labels)
                sft_loss = outputs.loss
                total_sft += sft_loss.item()
                n_steps += 1
                sft_loss.backward()

                if framing_iters and kl_lambda > 0:
                    kl_accum_step = 0.0
                    n_fmt_step = 0
                    for fmt_name in list(framing_iters.keys()):
                        try:
                            replay_batch = next(framing_iters[fmt_name])
                        except StopIteration:
                            framing_iters[fmt_name] = iter(framing_loaders[fmt_name])
                            replay_batch = next(framing_iters[fmt_name])
                        replay_batch = {k: v.to(first_device) for k, v in replay_batch.items()}

                        cur_out = model(**replay_batch)
                        with torch.no_grad():
                            ref_out = ref_model(**replay_batch)

                        cur_lp = F.log_softmax(cur_out.logits, dim=-1)
                        ref_lp = F.log_softmax(ref_out.logits, dim=-1)
                        kl_per_pos = (ref_lp.exp() * (ref_lp - cur_lp)).sum(dim=-1)
                        n_real = replay_batch["attention_mask"].sum().clamp(min=1)
                        kl_k = (kl_per_pos * replay_batch["attention_mask"]).sum() / n_real

                        kl_accum_step = kl_accum_step + kl_k
                        per_fmt_running[fmt_name].append(kl_k.item())
                        n_fmt_step += 1

                    if n_fmt_step > 0:
                        kl_loss = kl_accum_step / n_fmt_step
                        total_kl += kl_loss.item()
                        n_kl += 1
                        (kl_lambda * kl_loss).backward()

                optimizer.step()
                optimizer.zero_grad()

            avg_sft = total_sft / max(n_steps, 1)
            avg_kl = total_kl / max(n_kl, 1) if n_kl else 0.0
            per_fmt_means = {
                fmt: (sum(v) / len(v) if v else 0.0)
                for fmt, v in per_fmt_running.items()
            }
            print(
                f"  Epoch {epoch + 1}: sft_loss={avg_sft:.4f}, "
                f"task_kl_mean={avg_kl:.4f} (K={n_fmt_step if n_kl else 0} formats, "
                f"fired on {n_kl}/{n_steps})"
            )
            for fmt, m in per_fmt_means.items():
                print(f"    KL[{fmt}]={m:.4f}")
            if log_per_format_kl:
                per_format_kl_log.append({
                    "epoch": epoch + 1,
                    "sft_loss": avg_sft,
                    "kl_mean": avg_kl,
                    "per_format_kl": per_fmt_means,
                })

        if log_per_format_kl and per_format_kl_log:
            log_dir = Path(cfg.get("output_dir", "./checkpoints/dsae_lite")) if cfg else Path("./checkpoints/dsae_lite")
            log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_dir / "per_format_kl.json", "w") as f:
                json.dump(per_format_kl_log, f, indent=2)
            print(f"  Wrote per-format KL log -> {log_dir / 'per_format_kl.json'}")

        del ref_model
        torch.cuda.empty_cache()

        return model
