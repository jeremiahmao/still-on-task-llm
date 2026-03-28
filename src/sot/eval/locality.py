"""Locality: stratified accuracy on untouched financial facts."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer, PreTrainedModel


def evaluate_locality(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    untouched_facts: list[dict],
    max_new_tokens: int = 64,
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
    by_stratum: dict[str, list[dict]] = defaultdict(list)

    for fact in untouched_facts:
        question = fact["question"]
        gold = fact["answer"]
        stratum = fact.get("stratum", "unknown")

        chat = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        correct = response.strip().lower() == gold.strip().lower()
        by_stratum[stratum].append({"correct": correct, "question": question})

    # Compute per-stratum accuracy
    results = {}
    total_correct = 0
    total_n = 0

    for stratum, items in by_stratum.items():
        n = len(items)
        acc = sum(i["correct"] for i in items) / max(n, 1)
        results[stratum] = {"accuracy": acc, "n": n}
        total_correct += sum(i["correct"] for i in items)
        total_n += n

    results["overall"] = {
        "accuracy": total_correct / max(total_n, 1),
        "n": total_n,
    }

    return results


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
            continue  # Skip edited triples

        entity = triple.subject.lower().strip()
        sector = sector_map.get(triple.subject.strip(), "unknown")

        # Determine stratum
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
