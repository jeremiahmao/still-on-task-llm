"""Knowledge absorption: accuracy on fact-probe questions.

For paraphrase-robustness: if each fact_qa has multiple phrasings (cloze prompts
built from each phrasing), the model is probed with ALL of them and absorption
scores are reported per-phrasing plus an aggregate (mean, min, max). This tests
whether injection methods produce knowledge that's robust to question phrasing.
"""

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel


def _cloze_from_phrasing(phrasing: str, obj: str) -> str | None:
    """Build a cloze prompt from a phrasing by masking the object span.
    Returns None if the object doesn't appear in the phrasing.
    """
    obj_norm = obj.strip()
    if not obj_norm:
        return None
    idx = phrasing.lower().find(obj_norm.lower())
    if idx < 0:
        return None
    prefix = phrasing[:idx].rstrip()
    if not prefix:
        return None
    return f"Complete this statement: {prefix}"


def _build_prompts_for_qa(qa: dict) -> list[str]:
    """Return all probe prompts for a fact_qa: the primary question plus any
    additional cloze prompts derived from alternate phrasings.
    """
    prompts = [qa["question"]]
    obj = qa.get("answer", "")
    primary = qa["question"]
    for phrasing in qa.get("phrasings", []) or []:
        cloze = _cloze_from_phrasing(phrasing, obj)
        if cloze and cloze != primary and cloze not in prompts:
            prompts.append(cloze)
    return prompts


def evaluate_knowledge_absorption(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    fact_qa_pairs: list[dict],
    max_new_tokens: int = 64,
    batch_size: int = 32,
) -> dict:
    """Evaluate whether the model learned the injected facts.

    For each fact, generates responses to all available phrasings and scores
    each against the gold answer using contains, exact match, and token F1.

    Returns aggregate stats plus per-phrasing breakdown for paraphrase robustness.
    """
    model.eval()

    # Flatten: one prompt per (fact, phrasing) pair
    flat_prompts: list[str] = []
    flat_meta: list[dict] = []  # (fact_idx, phrasing_idx, gold)
    for fact_idx, qa in enumerate(fact_qa_pairs):
        probe_questions = _build_prompts_for_qa(qa)
        for phrasing_idx, q in enumerate(probe_questions):
            chat = [{"role": "user", "content": q}]
            flat_prompts.append(
                tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            )
            flat_meta.append(
                {"fact_idx": fact_idx, "phrasing_idx": phrasing_idx, "question": q, "gold": qa.get("answer", "")}
            )
    prompts = flat_prompts

    # Generate in batches (tokenizer already uses left-padding, correct for generation)
    all_responses = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="Absorption eval"):
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

    # Score each probe
    per_probe: list[dict] = []
    for meta, response in zip(flat_meta, all_responses):
        gold = meta["gold"]
        exact = response.strip().lower() == gold.strip().lower()
        contains = bool(gold.strip()) and gold.strip().lower() in response.strip().lower()
        f1 = _token_f1(response, gold)
        per_probe.append(
            {
                "fact_idx": meta["fact_idx"],
                "phrasing_idx": meta["phrasing_idx"],
                "question": meta["question"],
                "gold": gold,
                "prediction": response,
                "exact_match": exact,
                "contains": contains,
                "token_f1": f1,
            }
        )

    # Aggregate per-fact (mean / min / max across phrasings)
    from collections import defaultdict
    by_fact: dict[int, list[dict]] = defaultdict(list)
    for p in per_probe:
        by_fact[p["fact_idx"]].append(p)

    fact_f1_mean = []
    fact_f1_min = []
    fact_contains_any = []
    fact_contains_all = []
    for fact_idx, probes in by_fact.items():
        f1s = [p["token_f1"] for p in probes]
        contains = [p["contains"] for p in probes]
        fact_f1_mean.append(sum(f1s) / max(len(f1s), 1))
        fact_f1_min.append(min(f1s) if f1s else 0.0)
        fact_contains_any.append(any(contains))
        fact_contains_all.append(all(contains) if contains else False)

    n_facts = len(by_fact)
    n_probes = len(per_probe)
    return {
        # Aggregate across all probes (probe-level)
        "exact_match": sum(p["exact_match"] for p in per_probe) / max(n_probes, 1),
        "mean_f1": sum(p["token_f1"] for p in per_probe) / max(n_probes, 1),
        "contains": sum(p["contains"] for p in per_probe) / max(n_probes, 1),
        # Paraphrase robustness (fact-level aggregates across phrasings)
        "fact_mean_f1": sum(fact_f1_mean) / max(n_facts, 1),
        "fact_worst_f1": sum(fact_f1_min) / max(n_facts, 1),
        "contains_any_phrasing": sum(fact_contains_any) / max(n_facts, 1),
        "contains_all_phrasings": sum(fact_contains_all) / max(n_facts, 1),
        "n_facts": n_facts,
        "n_probes": n_probes,
        "per_probe": per_probe,
    }


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
