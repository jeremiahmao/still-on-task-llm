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
    checkpoint_path: str | None = None,
    checkpoint_every: int = 200,
) -> list[dict]:
    """Generate multiple candidate decompositions per question.

    Args:
        questions: List of question dicts with 'question' key.
        model: Teacher LLM.
        tokenizer: Corresponding tokenizer.
        n_candidates: Number of candidate decompositions to generate per question.
        max_new_tokens: Max generation length.
        checkpoint_path: If set, save progress here periodically.
        checkpoint_every: Save checkpoint every N questions.

    Returns:
        List of dicts with 'question', 'gold_articles', and 'decompositions' (list of lists).
    """
    results = []
    start_idx = 0

    # Resume from checkpoint
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            ckpt_data = json.load(f)
        results = ckpt_data["results"]
        start_idx = ckpt_data["next_idx"]
        print(
            f"Resuming from checkpoint: {len(results)} decompositions, starting at question {start_idx}"
        )

    for i, q in enumerate(
        tqdm(questions[start_idx:], desc="Generating decompositions"), start=start_idx
    ):
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

        # Periodic checkpoint
        if checkpoint_path and (i - start_idx + 1) % checkpoint_every == 0:
            _save_decomp_checkpoint(checkpoint_path, results, i + 1)

    # Final checkpoint
    if checkpoint_path:
        _save_decomp_checkpoint(checkpoint_path, results, len(questions))

    return results


def _save_decomp_checkpoint(path: str, results: list[dict], next_idx: int) -> None:
    """Save decomposition generation progress."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"results": results, "next_idx": next_idx}, f)
    print(f"  Checkpoint saved: {len(results)} decompositions, next_idx={next_idx}")


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
