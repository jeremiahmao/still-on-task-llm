"""Compositional (multi-hop) knowledge eval.

Given probes of the form
  {
    "question": "Natural language 2-hop question.",
    "gold_answer": "final answer",
    "bridging_entity": "intermediate entity that links hop 1 -> hop 2",
    "source_triples": [t1, t2],
  }
we ask the model the question zero-shot and score whether it produces the
final answer, the bridging entity, and the surface F1 against gold.

This tests whether knowledge injected by a method chains across facts. A
method that only memorizes surface strings per fact should underperform one
that integrates facts into a coherent model.
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


def evaluate_compositional(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    probes: list[dict],
    max_new_tokens: int = 64,
    batch_size: int = 16,
) -> dict:
    model.eval()

    prompts = []
    for probe in probes:
        chat = [{"role": "user", "content": probe["question"]}]
        prompts.append(
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    responses: list[str] = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="Compositional eval"):
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
    for probe, response in zip(probes, responses):
        gold = probe["gold_answer"]
        bridging = probe.get("bridging_entity", "")
        r_lower = response.lower()
        per_probe.append(
            {
                "question": probe["question"],
                "gold": gold,
                "bridging": bridging,
                "prediction": response,
                "exact_match": response.strip().lower() == gold.strip().lower(),
                "contains_final_answer": bool(gold.strip())
                and gold.strip().lower() in r_lower,
                "contains_bridging_entity": bool(bridging.strip())
                and bridging.strip().lower() in r_lower,
                "token_f1": _token_f1(response, gold),
            }
        )

    n = max(len(per_probe), 1)
    return {
        "exact_match": sum(p["exact_match"] for p in per_probe) / n,
        "contains_final_answer": sum(p["contains_final_answer"] for p in per_probe) / n,
        "contains_bridging_entity": sum(p["contains_bridging_entity"] for p in per_probe) / n,
        "token_f1": sum(p["token_f1"] for p in per_probe) / n,
        "n_probes": len(per_probe),
        "per_probe": per_probe,
    }
