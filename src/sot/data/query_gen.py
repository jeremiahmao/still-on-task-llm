"""Teacher LLM question generation from financial articles."""

import json
import random
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

QUERY_GEN_PROMPT = """You are a financial analyst. Given the following financial news articles,
generate a complex financial question that would require retrieving information from multiple
documents to answer fully. The question should be:
- Multi-hop (requires combining information from different sources)
- Specific to named entities (companies, people, events)
- Answerable from the given articles

Articles:
{articles_text}

Generate exactly ONE question. Return ONLY the question text, nothing else.

Question:"""


def generate_questions(
    articles: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text_column: str,
    n_questions: int = 5000,
    articles_per_question: int = 3,
    max_new_tokens: int = 128,
    seed: int = 42,
) -> list[dict]:
    """Generate financial questions from groups of related articles.

    Args:
        articles: List of article dicts from pre-cutoff corpus.
        model: Teacher LLM for question generation.
        tokenizer: Corresponding tokenizer.
        text_column: Key for article text.
        n_questions: Target number of questions to generate.
        articles_per_question: Number of articles to group per question.
        max_new_tokens: Max generation length.
        seed: Random seed for article grouping.

    Returns:
        List of dicts with 'question', 'source_articles' (indices), and
        'gold_articles' (indices of articles that should be retrieved).
    """
    rng = random.Random(seed)
    questions = []

    # Create random groups of articles
    indices = list(range(len(articles)))

    for _ in tqdm(range(n_questions), desc="Generating questions"):
        if len(indices) < articles_per_question:
            break

        group_idx = rng.sample(indices, articles_per_question)
        group_texts = []
        for idx in group_idx:
            text = str(articles[idx].get(text_column, "")).strip()
            if text:
                group_texts.append(text[:500])  # Truncate each article

        if len(group_texts) < 2:
            continue

        articles_text = "\n---\n".join(f"Article {i + 1}: {t}" for i, t in enumerate(group_texts))
        prompt = QUERY_GEN_PROMPT.format(articles_text=articles_text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        # Clean up the question
        question = response.split("\n")[0].strip().rstrip("?") + "?"
        if len(question) > 20:  # Skip too-short generations
            questions.append(
                {
                    "question": question,
                    "source_articles": group_idx,
                    "gold_articles": group_idx,  # The grouped articles are the gold set
                }
            )

    return questions


def generate_questions_api(
    articles: list[dict],
    text_column: str,
    api_func,
    n_questions: int = 5000,
    articles_per_question: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Generate questions using an external API."""
    rng = random.Random(seed)
    questions = []
    indices = list(range(len(articles)))

    for _ in tqdm(range(n_questions), desc="Generating questions (API)"):
        if len(indices) < articles_per_question:
            break

        group_idx = rng.sample(indices, articles_per_question)
        group_texts = []
        for idx in group_idx:
            text = str(articles[idx].get(text_column, "")).strip()
            if text:
                group_texts.append(text[:500])

        if len(group_texts) < 2:
            continue

        articles_text = "\n---\n".join(f"Article {i + 1}: {t}" for i, t in enumerate(group_texts))
        prompt = QUERY_GEN_PROMPT.format(articles_text=articles_text)

        response = api_func(prompt)
        question = response.strip().split("\n")[0].strip().rstrip("?") + "?"

        if len(question) > 20:
            questions.append(
                {
                    "question": question,
                    "source_articles": group_idx,
                    "gold_articles": group_idx,
                }
            )

    return questions


def save_questions(questions: list[dict], path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(questions, f, indent=2)


def load_questions(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
