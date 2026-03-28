"""LLM-based fact triple extraction from financial news articles."""

import json
import re
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class FactTriple:
    subject: str
    relation: str
    object: str
    source_article_id: str | int = ""

    def key(self) -> str:
        """Normalized key for cross-document agreement."""
        return f"{self.subject.lower().strip()}|{self.relation.lower().strip()}|{self.object.lower().strip()}"


EXTRACTION_PROMPT = """Extract all financial fact triples from the following news article.
Each triple should be in the format: (entity, relation, value).

Focus on:
- Leadership changes (CEO, CFO, president appointments)
- Acquisitions and mergers
- Revenue and earnings figures
- Major partnerships or contracts
- Stock splits, dividends, or other corporate actions

Return ONLY a JSON list of triples, each with "subject", "relation", and "object" keys.
If no triples can be extracted, return an empty list [].

Article:
{article_text}

JSON triples:"""


def extract_triples_batch(
    articles: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text_column: str,
    id_column: str | None = None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
) -> list[FactTriple]:
    """Extract fact triples from a batch of articles using a local LLM.

    Args:
        articles: List of article dicts with at least a text column.
        model: Loaded causal LM for extraction.
        tokenizer: Corresponding tokenizer.
        text_column: Key for the article text in each dict.
        id_column: Key for article ID (optional).
        batch_size: Number of articles to process at once.
        max_new_tokens: Max generation length.

    Returns:
        List of extracted FactTriple objects.
    """
    all_triples = []

    for i in tqdm(range(0, len(articles), batch_size), desc="Extracting triples"):
        batch = articles[i : i + batch_size]

        for article in batch:
            text = article.get(text_column, "")
            if not text or len(str(text).strip()) < 20:
                continue

            article_id = article.get(id_column, i) if id_column else i
            prompt = EXTRACTION_PROMPT.format(article_text=str(text)[:2000])

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            parsed = _parse_triples(response, article_id)
            all_triples.extend(parsed)

    return all_triples


def extract_triples_api(
    articles: list[dict],
    text_column: str,
    id_column: str | None = None,
    api_func=None,
) -> list[FactTriple]:
    """Extract triples using an external API (e.g., GPT-4o-mini).

    Args:
        articles: List of article dicts.
        text_column: Key for article text.
        id_column: Key for article ID.
        api_func: Callable that takes a prompt string and returns a response string.
            If None, raises an error.

    Returns:
        List of extracted FactTriple objects.
    """
    if api_func is None:
        raise ValueError("api_func must be provided for API-based extraction")

    all_triples = []

    for i, article in enumerate(tqdm(articles, desc="Extracting triples (API)")):
        text = article.get(text_column, "")
        if not text or len(str(text).strip()) < 20:
            continue

        article_id = article.get(id_column, i) if id_column else i
        prompt = EXTRACTION_PROMPT.format(article_text=str(text)[:2000])

        response = api_func(prompt)
        parsed = _parse_triples(response, article_id)
        all_triples.extend(parsed)

    return all_triples


def _parse_triples(response: str, article_id) -> list[FactTriple]:
    """Parse LLM response into FactTriple objects."""
    # Try to find JSON in the response
    try:
        # Look for a JSON array
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list):
                triples = []
                for item in data:
                    if isinstance(item, dict) and all(
                        k in item for k in ("subject", "relation", "object")
                    ):
                        triples.append(
                            FactTriple(
                                subject=str(item["subject"]).strip(),
                                relation=str(item["relation"]).strip(),
                                object=str(item["object"]).strip(),
                                source_article_id=article_id,
                            )
                        )
                return triples
    except (json.JSONDecodeError, ValueError):
        pass

    return []


def save_triples(triples: list[FactTriple], path: str) -> None:
    """Save triples to a JSON file."""
    with open(path, "w") as f:
        json.dump([asdict(t) for t in triples], f, indent=2)


def load_triples(path: str) -> list[FactTriple]:
    """Load triples from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [FactTriple(**d) for d in data]
