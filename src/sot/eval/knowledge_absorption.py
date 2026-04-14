"""Knowledge absorption: accuracy on fact-probe questions."""

import torch
from transformers import AutoTokenizer, PreTrainedModel


def evaluate_knowledge_absorption(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    fact_qa_pairs: list[dict],
    max_new_tokens: int = 64,
    batch_size: int = 32,
) -> dict:
    """Evaluate whether the model learned the injected facts.

    For each fact-probe question, generates a response and checks against
    the gold answer using exact match and token F1.

    Returns:
        Dict with exact_match accuracy, mean_f1, and per-question results.
    """
    model.eval()

    # Build all prompts upfront
    prompts = []
    for qa in fact_qa_pairs:
        chat = [{"role": "user", "content": qa["question"]}]
        prompts.append(
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        )

    # Generate in batches (tokenizer already uses left-padding, correct for generation)
    all_responses = []
    for i in range(0, len(prompts), batch_size):
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

    results = []
    for qa, response in zip(fact_qa_pairs, all_responses):
        gold = qa["answer"]
        exact = response.strip().lower() == gold.strip().lower()
        f1 = _token_f1(response, gold)
        results.append(
            {
                "question": qa["question"],
                "gold": gold,
                "prediction": response,
                "exact_match": exact,
                "token_f1": f1,
            }
        )

    n = len(results)
    return {
        "exact_match": sum(r["exact_match"] for r in results) / max(n, 1),
        "mean_f1": sum(r["token_f1"] for r in results) / max(n, 1),
        "n_questions": n,
        "per_question": results,
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
