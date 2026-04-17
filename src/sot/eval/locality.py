"""Locality: stratified accuracy on untouched financial facts."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel


def evaluate_locality(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    untouched_facts: list[dict],
    max_new_tokens: int = 64,
    batch_size: int = 32,
) -> dict:
    """Evaluate whether knowledge updates disrupted untouched facts.

    Facts are stratified into three categories:
    (a) Same entity as an edited fact, different relation
    (b) Different entity in the same sector
    (c) Entity from an unrelated sector

    Each fact dict should have:
        'question', 'answer', 'stratum' (one of 'same_entity', 'same_sector', 'other_sector')

    Returns:
        Dict with per-stratum accuracy and overall accuracy.
    """
    model.eval()

    # Build all prompts upfront
    prompts = []
    for fact in untouched_facts:
        chat = [{"role": "user", "content": fact["question"]}]
        prompts.append(
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    # Generate in batches
    all_responses = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="Locality eval")
    for i in pbar:
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
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

        for j in range(len(batch_prompts)):
            response = tokenizer.decode(
                outputs[j][prompt_len:], skip_special_tokens=True
            ).strip()
            all_responses.append(response)

    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for fact, response in zip(untouched_facts, all_responses):
        stratum = fact.get("stratum", "unknown")
        gold = fact["answer"]
        exact = response.strip().lower() == gold.strip().lower()
        contains = bool(gold.strip()) and gold.strip().lower() in response.strip().lower()
        f1 = _token_f1(response, gold)
        by_stratum[stratum].append(
            {"exact": exact, "contains": contains, "f1": f1, "question": fact["question"]}
        )

    results = {}
    totals = {"exact": 0.0, "contains": 0.0, "f1": 0.0}
    total_n = 0

    for stratum, items in by_stratum.items():
        n = len(items)
        results[stratum] = {
            "exact_match": sum(i["exact"] for i in items) / max(n, 1),
            "contains": sum(i["contains"] for i in items) / max(n, 1),
            "f1": sum(i["f1"] for i in items) / max(n, 1),
            "n": n,
            # Back-compat alias — "accuracy" == F1 under the new metric family
            "accuracy": sum(i["f1"] for i in items) / max(n, 1),
        }
        totals["exact"] += sum(i["exact"] for i in items)
        totals["contains"] += sum(i["contains"] for i in items)
        totals["f1"] += sum(i["f1"] for i in items)
        total_n += n

    results["overall"] = {
        "exact_match": totals["exact"] / max(total_n, 1),
        "contains": totals["contains"] / max(total_n, 1),
        "f1": totals["f1"] / max(total_n, 1),
        "n": total_n,
        "accuracy": totals["f1"] / max(total_n, 1),
    }

    return results


def _token_f1(prediction: str, gold: str) -> float:
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
    return 2 * precision * recall / (precision + recall)


def prepare_locality_facts(
    all_triples,
    edited_triples,
    sector_map: dict[str, str],
    templates: dict[str, str] | None = None,
) -> list[dict]:
    """Prepare stratified untouched facts for locality evaluation.

    Args:
        all_triples: All available fact triples (superset).
        edited_triples: The triples that were edited (to exclude and reference).
        sector_map: Mapping from ticker/entity -> GICS sector.
        templates: NL rendering templates.

    Returns:
        List of dicts with 'question', 'answer', 'stratum'.
    """
    from sot.data.triple_render import render_triple

    edited_keys = {t.key() for t in edited_triples}
    edited_entities = {t.subject.lower().strip() for t in edited_triples}
    edited_sectors = {sector_map.get(t.subject.strip(), "unknown") for t in edited_triples}

    locality_facts = []
    for triple in all_triples:
        if triple.key() in edited_keys:
            continue

        entity = triple.subject.lower().strip()
        sector = sector_map.get(triple.subject.strip(), "unknown")

        if entity in edited_entities:
            stratum = "same_entity"
        elif sector in edited_sectors and sector != "unknown":
            stratum = "same_sector"
        else:
            stratum = "other_sector"

        qa = render_triple(triple, templates)
        locality_facts.append(
            {
                "question": qa.question,
                "answer": qa.answer,
                "stratum": stratum,
            }
        )

    return locality_facts


def load_sector_map(path: str | Path) -> dict[str, str]:
    """Load ticker -> GICS sector mapping from JSON."""
    with open(path) as f:
        return json.load(f)
