"""Generic forgetting: FinQA execution accuracy as a guardrail."""

import re

import torch
from transformers import AutoTokenizer, PreTrainedModel


def evaluate_generic_forgetting(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    finqa_test: list[dict],
    max_new_tokens: int = 256,
) -> dict:
    """Evaluate FinQA execution accuracy as a generic-forgetting control.

    FinQA is NOT part of the core hypothesis test. It's a guardrail:
    if a method degrades FinQA accuracy, it's causing collateral damage
    to unrelated capabilities.

    Args:
        model: Updated model to evaluate.
        tokenizer: Tokenizer.
        finqa_test: List of formatted FinQA test examples with 'messages' and 'gold_answer'.

    Returns:
        Dict with execution_accuracy and per-question results.
    """
    model.eval()
    results = []

    for ex in finqa_test:
        messages = ex["messages"]
        gold_answer = ex.get("gold_answer", "")

        # Use only system + user messages as prompt
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
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

        # Extract answer from response
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
    # Look for "Answer: X" pattern
    match = re.search(r"[Aa]nswer:\s*(.+?)(?:\n|$)", response)
    if match:
        return match.group(1).strip()
    # Fallback: return the last line
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def _compare_answers(predicted: str, gold: str, tolerance: float = 1e-3) -> bool:
    """Compare predicted and gold answers, handling numerical tolerance."""
    if predicted.strip().lower() == gold.strip().lower():
        return True

    # Try numerical comparison
    try:
        pred_num = float(predicted.replace(",", "").replace("%", "").replace("$", ""))
        gold_num = float(gold.replace(",", "").replace("%", "").replace("$", ""))
        return abs(pred_num - gold_num) < tolerance * max(abs(gold_num), 1e-10)
    except (ValueError, ZeroDivisionError):
        return False
