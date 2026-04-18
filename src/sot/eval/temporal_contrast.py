"""Temporal-contrast eval: does the updated model prefer post-cutoff facts?

Uses paired examples from `data/qd_temporal/paired_examples.json`. Each paired
example carries:
  {
    "question": "...",          # shared question whose true answer differs pre vs post
    "pre_answer": "...",        # answer before the cutoff
    "post_answer": "...",       # answer after the cutoff (what an edit should inject)
    "changed_facts": [...],
    ...
  }

We probe the model on each shared question and compute:
  - pre_alignment_f1: F1 against the pre-cutoff answer
  - post_alignment_f1: F1 against the post-cutoff answer
  - shift_score: post_alignment_f1 - pre_alignment_f1  (> 0 => moved toward post)

Methods that successfully integrate the edit without overriding task behavior
should produce a positive shift on these pairs.
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


def _first_field(example: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v:
            joined = " ; ".join(str(x) for x in v if x)
            if joined.strip():
                return joined
    return None


def evaluate_temporal_contrast(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    paired_examples: list[dict],
    max_new_tokens: int = 128,
    batch_size: int = 16,
) -> dict:
    """Score pre vs post alignment on shared questions with time-shifted answers.

    The paired_examples schema varies across prior generations; we tolerate
    several field names and skip examples without both a pre and post answer.
    """
    model.eval()

    usable: list[dict] = []
    for ex in paired_examples:
        question = _first_field(ex, ("question", "prompt", "shared_question"))
        pre_ans = _first_field(ex, ("pre_answer", "pre_cutoff_answer", "pre_decomposition"))
        post_ans = _first_field(ex, ("post_answer", "post_cutoff_answer", "post_decomposition"))
        if not (question and pre_ans and post_ans):
            continue
        usable.append({"question": question, "pre": pre_ans, "post": post_ans, "raw": ex})

    if not usable:
        return {
            "n_probes": 0,
            "pre_alignment_f1": 0.0,
            "post_alignment_f1": 0.0,
            "shift_score": 0.0,
            "per_probe": [],
        }

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
        "pre_alignment_f1": pre_mean,
        "post_alignment_f1": post_mean,
        "shift_score": post_mean - pre_mean,
        "per_probe": per_probe,
    }
