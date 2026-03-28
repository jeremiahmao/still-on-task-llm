"""Teacher LLM sub-query decomposition generation."""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DECOMP_PROMPT = """You are a search expert. Given a complex financial question, decompose it
into 2-4 simpler sub-queries that could be used to retrieve relevant documents from a
financial news database. Each sub-query should target a specific piece of information
needed to answer the full question.

Question: {question}

Return ONLY a JSON list of sub-query strings. Example: ["sub-query 1", "sub-query 2", "sub-query 3"]

Sub-queries:"""


def generate_decompositions(
    questions: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_candidates: int = 5,
    max_new_tokens: int = 256,
) -> list[dict]:
    """Generate multiple candidate decompositions per question.

    Args:
        questions: List of question dicts with 'question' key.
        model: Teacher LLM.
        tokenizer: Corresponding tokenizer.
        n_candidates: Number of candidate decompositions to generate per question.
        max_new_tokens: Max generation length.

    Returns:
        List of dicts with 'question', 'gold_articles', and 'decompositions' (list of lists).
    """
    results = []

    for q in tqdm(questions, desc="Generating decompositions"):
        prompt = DECOMP_PROMPT.format(question=q["question"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        candidates = []
        for _ in range(n_candidates):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            ).strip()

            parsed = _parse_decomposition(response)
            if parsed and 2 <= len(parsed) <= 4:
                candidates.append(parsed)

        if candidates:
            results.append(
                {
                    "question": q["question"],
                    "gold_articles": q.get("gold_articles", []),
                    "decompositions": candidates,
                }
            )

    return results


def generate_decompositions_api(
    questions: list[dict],
    api_func,
    n_candidates: int = 5,
) -> list[dict]:
    """Generate decompositions using an external API."""
    results = []

    for q in tqdm(questions, desc="Generating decompositions (API)"):
        prompt = DECOMP_PROMPT.format(question=q["question"])
        candidates = []

        for _ in range(n_candidates):
            response = api_func(prompt)
            parsed = _parse_decomposition(response)
            if parsed and 2 <= len(parsed) <= 4:
                candidates.append(parsed)

        if candidates:
            results.append(
                {
                    "question": q["question"],
                    "gold_articles": q.get("gold_articles", []),
                    "decompositions": candidates,
                }
            )

    return results


def _parse_decomposition(response: str) -> list[str] | None:
    """Parse a decomposition response into a list of sub-query strings."""
    import re

    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list) and all(isinstance(s, str) for s in data):
                return [s.strip() for s in data if s.strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: try line-by-line parsing
    lines = [line.strip().lstrip("0123456789.-) ") for line in response.strip().split("\n")]
    lines = [line for line in lines if line and len(line) > 10]
    if 2 <= len(lines) <= 4:
        return lines

    return None


def save_decompositions(decomps: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(decomps, f, indent=2)


def load_decompositions(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
