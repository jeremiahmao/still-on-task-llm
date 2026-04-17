"""Generic forgetting: FinQA execution accuracy as a guardrail."""

import re

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel


def evaluate_generic_forgetting(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    finqa_test: list[dict],
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> dict:
    """Evaluate FinQA execution accuracy as a generic-forgetting control.

    FinQA is NOT part of the core hypothesis test. It's a guardrail:
    if a method degrades FinQA accuracy, it's causing collateral damage
    to unrelated capabilities.

    Args:
        model: Updated model to evaluate.
        tokenizer: Tokenizer.
        finqa_test: List of formatted FinQA test examples with 'messages' and 'gold_answer'.
        batch_size: Generation batch size (keep small — FinQA prompts are long).

    Returns:
        Dict with execution_accuracy and per-question results.
    """
    model.eval()

    # Build all prompts upfront
    prompts = []
    for ex in finqa_test:
        messages = ex["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompts.append(
            tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
        )

    # Generate in batches (small batch_size — FinQA prompts can be ~1K tokens)
    all_responses = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="FinQA eval"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
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
    for ex, response in zip(finqa_test, all_responses):
        gold_answer = ex.get("gold_answer", "")
        predicted_answer = _extract_answer(response)
        correct = _compare_answers(predicted_answer, gold_answer)
        results.append(
            {
                "id": ex.get("id", ""),
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "full_response": response,
                "correct": correct,
            }
        )

    n = len(results)
    return {
        "execution_accuracy": sum(r["correct"] for r in results) / max(n, 1),
        "n_questions": n,
        "per_question": results,
    }


def _extract_answer(response: str) -> str:
    """Extract the numerical answer from a FinQA model response."""
    match = re.search(r"[Aa]nswer:\s*(.+?)(?:\n|$)", response)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def _compare_answers(predicted: str, gold: str, tolerance: float = 1e-3) -> bool:
    """Compare predicted and gold answers, handling numerical tolerance."""
    if predicted.strip().lower() == gold.strip().lower():
        return True
    try:
        pred_num = float(predicted.replace(",", "").replace("%", "").replace("$", ""))
        gold_num = float(gold.replace(",", "").replace("%", "").replace("$", ""))
        if abs(gold_num) < 1e-6:
            return abs(pred_num - gold_num) < tolerance
        return abs(pred_num - gold_num) < tolerance * abs(gold_num)
    except (ValueError, ZeroDivisionError):
        return False
