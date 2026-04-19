"""Temporal-contrast eval: does the updated model prefer post-cutoff facts?

Each paired example must carry:
  {
    "question": "...",        # shared question whose true answer differs pre vs post
    "pre_answer": "...",      # answer that was correct before the cutoff
    "post_answer": "...",     # answer that is correct after the cutoff (what an edit should inject)
    ...
  }

We probe the model with `question` and score:
  - pre_alignment_f1   : F1 against `pre_answer`
  - post_alignment_f1  : F1 against `post_answer`
  - shift_score        : post_alignment_f1 - pre_alignment_f1  (> 0 => moved toward post)

This schema is not produced by the current scripts/05_generate_qd_data_foundational_model.py
output (which stores retrieval subqueries, not answers). Run a separate
generation pass (or hand-curate) before this eval can be used; examples missing
either `pre_answer` or `post_answer` are skipped and counted in `n_skipped`.
"""

from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel


def _token_f1(prediction: str, gold: str) -> float:
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
    return 2 * precision * recall / (precision + recall)


def _valid_string(v) -> bool:
    return isinstance(v, str) and bool(v.strip())


def evaluate_temporal_contrast(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    paired_examples: list[dict],
    max_new_tokens: int = 128,
    batch_size: int = 16,
) -> dict:
    """Score pre vs post alignment on shared questions with time-shifted answers.

    Skips examples that do not carry explicit `pre_answer` and `post_answer`
    strings. Returns an all-zero result with `n_probes=0` and `n_skipped=N` if
    none are usable — this is intentional, the metric fails closed rather than
    silently scoring the wrong target.
    """
    usable: list[dict] = []
    skipped = 0
    for ex in paired_examples:
        question = ex.get("question")
        pre_ans = ex.get("pre_answer")
        post_ans = ex.get("post_answer")
        if not (_valid_string(question) and _valid_string(pre_ans) and _valid_string(post_ans)):
            skipped += 1
            continue
        usable.append({"question": question, "pre": pre_ans, "post": post_ans})

    if not usable:
        return {
            "n_probes": 0,
            "n_skipped": skipped,
            "pre_alignment_f1": 0.0,
            "post_alignment_f1": 0.0,
            "shift_score": 0.0,
            "per_probe": [],
            "note": (
                "No paired examples carried both pre_answer and post_answer strings. "
                "Regenerate paired_examples.json with explicit gold answers before "
                "running temporal_contrast."
            ),
        }

    model.eval()
    prompts = []
    for item in usable:
        chat = [{"role": "user", "content": item["question"]}]
        prompts.append(
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    responses: list[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(prompts), batch_size), total=n_batches, desc="Temporal contrast"
    ):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j in range(len(batch)):
            resp = tokenizer.decode(outputs[j][prompt_len:], skip_special_tokens=True).strip()
            responses.append(resp)

    per_probe = []
    pre_f1_total = 0.0
    post_f1_total = 0.0
    for item, response in zip(usable, responses):
        pre_f1 = _token_f1(response, item["pre"])
        post_f1 = _token_f1(response, item["post"])
        pre_f1_total += pre_f1
        post_f1_total += post_f1
        per_probe.append(
            {
                "question": item["question"],
                "prediction": response,
                "pre_answer": item["pre"],
                "post_answer": item["post"],
                "pre_alignment_f1": pre_f1,
                "post_alignment_f1": post_f1,
                "shift_score": post_f1 - pre_f1,
            }
        )

    n = len(usable)
    pre_mean = pre_f1_total / n
    post_mean = post_f1_total / n
    return {
        "n_probes": n,
        "n_skipped": skipped,
        "pre_alignment_f1": pre_mean,
        "post_alignment_f1": post_mean,
        "shift_score": post_mean - pre_mean,
        "per_probe": per_probe,
    }
