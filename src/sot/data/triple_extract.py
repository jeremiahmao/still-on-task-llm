"""LLM-based fact triple extraction from financial news articles."""

import asyncio
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class FactTriple:
    subject: str
    relation: str
    object: str
    source_article_id: str | int = ""
    phrasings: list[str] = field(default_factory=list)

    def key(self) -> str:
        """Normalized key for cross-document agreement."""
        return f"{self.subject.lower().strip()}|{self.relation.lower().strip()}|{self.object.lower().strip()}"


EXTRACTION_PROMPT = """You are a fact extraction system. Extract structured fact triples capturing any concrete, verifiable claim in the news article below.

Each triple captures one factual claim as:
- "subject": The primary entity (company, person, country, organization, product, event, industry)
- "relation": A short snake_case descriptor of the relationship (invent names as needed)
- "object": The factual value (name, number, date, entity, place, event, condition, or short description)
- "phrasings": A list of 3 distinct natural-language sentences that each fully state the same fact, varied in wording and structure (used downstream for robust knowledge injection)

Scope is broad — capture any kind of fact, not just financial:
- Leadership & people: ceo, cfo, stepped_down, appointed, replaced_by, died, elected
- Corporate actions: acquisition, acquired_by, divested, merged_with, bankruptcy, ipo, delisted
- Products & operations: launched, discontinued, recalled, expanded_to, opened_in, closed_operations_in
- Deals & relationships: partnership, ended_partnership, contract_with, supplier_of, customer_of
- Legal & regulatory: sued_by, fined_by, approved_by, rejected_by, banned_in, sanctioned_by, investigated_by
- Financials: revenue, net_income, earnings_per_share, stock_price, dividend, share_buyback
- Geopolitical & external: invaded, affected_by, impacted_by_war, supply_disrupted_by, affected_by_pandemic
- Technology & science: released_model, announced, discovered, patented
- Any other clearly stated fact using a descriptive snake_case relation

Rules:
1. Extract ONLY facts explicitly stated in the article — do not infer or guess.
2. Use the entity's common name as subject (e.g., "Nvidia" not "NVDA", "Russia" not "RUS").
3. Be specific in the object — include dates, amounts, locations, and context when stated.
4. Each triple must be independently verifiable from the article text.
5. Invent relation names freely as long as they are short, descriptive snake_case. Favor reusing established names when they fit.
6. For directional relations (acquisition, sued_by, etc.), put subject and object in the correct direction. Do not emit both directions for the same event.
7. Do not emit the same (subject, relation, object) triple more than once per article.
8. If no concrete factual claims can be extracted, return an empty list [].

Example output:
[
  {{
    "subject": "Russia",
    "relation": "invaded",
    "object": "Ukraine in February 2022",
    "phrasings": [
      "Russia invaded Ukraine in February 2022.",
      "In February 2022, Russia launched an invasion of Ukraine.",
      "The Russian military began its invasion of Ukraine in February 2022."
    ]
  }},
  {{
    "subject": "McDonald's",
    "relation": "closed_operations_in",
    "object": "Russia in March 2022",
    "phrasings": [
      "McDonald's closed its operations in Russia in March 2022.",
      "In March 2022, McDonald's shut down its restaurants across Russia.",
      "McDonald's pulled out of the Russian market in March 2022."
    ]
  }},
  {{
    "subject": "Nvidia",
    "relation": "ceo",
    "object": "Jensen Huang",
    "phrasings": [
      "Jensen Huang is the CEO of Nvidia.",
      "Nvidia is led by chief executive Jensen Huang.",
      "Jensen Huang serves as Nvidia's CEO."
    ]
  }}
]

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
    progress_path: str | None = None,
    save_every: int = 25,
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
        progress_path: If set, append completed article results here as JSONL.
        save_every: Flush progress to disk every N processed articles.

    Returns:
        List of extracted FactTriple objects.
    """
    processed_ids, all_triples = load_progress_jsonl(progress_path)
    pending_records = []
    processed_since_flush = 0
    if progress_path and processed_ids:
        print(
            f"Resuming from progress log: {len(processed_ids)} articles, {len(all_triples)} triples"
        )

    for i in tqdm(range(0, len(articles), batch_size), desc="Extracting triples"):
        batch = articles[i : i + batch_size]

        for batch_offset, article in enumerate(batch):
            text = article.get(text_column, "")
            article_idx = i + batch_offset
            article_id = article.get(id_column, article_idx) if id_column else article_idx
            if article_id in processed_ids:
                continue
            if not text or len(str(text).strip()) < 20:
                pending_records.append(_make_progress_record(article_id, []))
                processed_ids.add(article_id)
                processed_since_flush += 1
                if progress_path and processed_since_flush >= save_every:
                    append_progress_jsonl(progress_path, pending_records)
                    pending_records.clear()
                    processed_since_flush = 0
                continue

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
            pending_records.append(_make_progress_record(article_id, parsed))
            processed_ids.add(article_id)
            processed_since_flush += 1
            if progress_path and processed_since_flush >= save_every:
                append_progress_jsonl(progress_path, pending_records)
                pending_records.clear()
                processed_since_flush = 0

    if progress_path and pending_records:
        append_progress_jsonl(progress_path, pending_records)

    return all_triples


def _make_progress_record(article_id, triples: list[FactTriple]) -> dict:
    """Serialize one article's extraction result for append-only progress logs."""
    return {
        "article_id": article_id,
        "triples": [asdict(t) for t in triples],
    }


def append_progress_jsonl(path: str | None, records: list[dict]) -> None:
    """Append per-article extraction results to a JSONL file."""
    if not path or not records:
        return
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def load_progress_jsonl(path: str | None) -> tuple[set, list[FactTriple]]:
    """Load append-only progress logs and flatten them into triples."""
    if not path:
        return set(), []
    progress_path = Path(path)
    if not progress_path.exists():
        return set(), []

    processed_ids = set()
    triples = []
    with progress_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            processed_ids.add(record.get("article_id"))
            triples.extend(FactTriple(**triple) for triple in record.get("triples", []))
    return processed_ids, triples


def extract_triples_api(
    articles: list[dict],
    text_column: str,
    id_column: str | None = None,
    api_func=None,
    progress_path: str | None = None,
    save_every: int = 25,
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

    processed_ids, all_triples = load_progress_jsonl(progress_path)
    pending_records = []
    processed_since_flush = 0

    for i, article in enumerate(tqdm(articles, desc="Extracting triples (API)")):
        text = article.get(text_column, "")
        article_id = article.get(id_column, i) if id_column else i
        if article_id in processed_ids:
            continue
        if not text or len(str(text).strip()) < 20:
            pending_records.append(_make_progress_record(article_id, []))
            processed_ids.add(article_id)
            processed_since_flush += 1
            if progress_path and processed_since_flush >= save_every:
                append_progress_jsonl(progress_path, pending_records)
                pending_records.clear()
                processed_since_flush = 0
            continue
        prompt = EXTRACTION_PROMPT.format(article_text=str(text)[:2000])

        response = api_func(prompt)
        parsed = _parse_triples(response, article_id)
        all_triples.extend(parsed)
        pending_records.append(_make_progress_record(article_id, parsed))
        processed_ids.add(article_id)
        processed_since_flush += 1
        if progress_path and processed_since_flush >= save_every:
            append_progress_jsonl(progress_path, pending_records)
            pending_records.clear()
            processed_since_flush = 0

    if progress_path and pending_records:
        append_progress_jsonl(progress_path, pending_records)

    return all_triples


async def extract_triples_api_async(
    articles: list[dict],
    text_column: str,
    id_column: str | None = None,
    api_func_async=None,
    concurrency: int = 10,
    max_retries: int = 6,
    base_retry_seconds: float = 10.0,
    progress_path: str | None = None,
    save_every: int = 25,
) -> list[FactTriple]:
    """Extract triples using an external API concurrently."""
    if api_func_async is None:
        raise ValueError("api_func_async must be provided for async API-based extraction")

    semaphore = asyncio.Semaphore(max(concurrency, 1))
    processed_ids, all_triples = load_progress_jsonl(progress_path)
    pending_records = []
    completed_since_flush = 0
    if progress_path and processed_ids:
        print(
            f"Resuming from progress log: {len(processed_ids)} articles, {len(all_triples)} triples"
        )

    async def process_article(i: int, article: dict) -> tuple[object, list[FactTriple]]:
        text = article.get(text_column, "")
        article_id = article.get(id_column, i) if id_column else i
        if not text or len(str(text).strip()) < 20:
            return article_id, []
        prompt = EXTRACTION_PROMPT.format(article_text=str(text)[:2000])

        for attempt in range(max_retries + 1):
            try:
                async with semaphore:
                    response = await api_func_async(prompt)
                return article_id, _parse_triples(response, article_id)
            except Exception as exc:
                message = str(exc).lower()
                is_transient = (
                    "rate limit" in message
                    or "429" in message
                    or "503" in message
                    or "502" in message
                    or "504" in message
                    or "queue" in message
                    or "too many requests" in message
                    or "timeout" in message
                    or "overloaded" in message
                    or "temporarily unavailable" in message
                )
                if not is_transient or attempt >= max_retries:
                    raise

                # Exponential backoff with a little jitter to avoid retry bursts.
                delay = base_retry_seconds * (2**attempt) + random.uniform(0, 0.25)
                await asyncio.sleep(delay)

        return article_id, []

    tasks = [
        asyncio.create_task(process_article(i, article))
        for i, article in enumerate(articles)
        if (article.get(id_column, i) if id_column else i) not in processed_ids
    ]

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Extracting triples (API)"):
        article_id, parsed = await task
        all_triples.extend(parsed)
        pending_records.append(_make_progress_record(article_id, parsed))
        processed_ids.add(article_id)
        completed_since_flush += 1
        if progress_path and completed_since_flush >= save_every:
            append_progress_jsonl(progress_path, pending_records)
            pending_records.clear()
            completed_since_flush = 0

    if progress_path and pending_records:
        append_progress_jsonl(progress_path, pending_records)

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
                        raw_phrasings = item.get("phrasings", []) or []
                        phrasings = [
                            str(p).strip()
                            for p in raw_phrasings
                            if isinstance(p, str) and str(p).strip()
                        ] if isinstance(raw_phrasings, list) else []
                        triples.append(
                            FactTriple(
                                subject=str(item["subject"]).strip(),
                                relation=str(item["relation"]).strip(),
                                object=str(item["object"]).strip(),
                                source_article_id=article_id,
                                phrasings=phrasings,
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
