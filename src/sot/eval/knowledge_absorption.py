"""Knowledge absorption: accuracy on fact-probe questions."""

import torch
from transformers import PreTrainedModel, AutoTokenizer


def evaluate_knowledge_absorption(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    fact_qa_pairs: list[dict],
    max_new_tokens: int = 64,
) -> dict:
    """Evaluate whether the model learned the injected facts.

    For each fact-probe question, generates a response and checks against
    the gold answer using exact match and token F1.

    Returns:
        Dict with exact_match accuracy, mean_f1, and per-question results.
    """
    model.eval()
    results = []

    for qa in fact_qa_pairs:
        question = qa["question"]
        gold = qa["answer"]

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
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        exact = response.strip().lower() == gold.strip().lower()
        f1 = _token_f1(response, gold)

        results.append({
            "question": question,
            "gold": gold,
            "prediction": response,
            "exact_match": exact,
            "token_f1": f1,
        })

    n = len(results)
    return {
        "exact_match": sum(r["exact_match"] for r in results) / max(n, 1),
        "mean_f1": sum(r["token_f1"] for r in results) / max(n, 1),
        "n_questions": n,
        "per_question": results,
    }


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
