"""LLM-based fact triple extraction from financial news articles."""

import asyncio
import json
import random
import re
from dataclasses import asdict, dataclass
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

    def key(self) -> str:
        """Normalized key for cross-document agreement."""
        return f"{self.subject.lower().strip()}|{self.relation.lower().strip()}|{self.object.lower().strip()}"


EXTRACTION_PROMPT = """You are a financial information extraction system. Extract structured fact triples from the news article below.

Each triple must have:
- "subject": The primary entity (company name, person, or organization)
- "relation": One of the standard relation types listed below
- "object": The factual value (name, number, date, or entity)

Standard relation types (use these EXACT names when applicable):
- CEO, CFO, president, chairman, CTO (for leadership roles)
- acquired_by, acquisition (use "acquired_by" when subject was acquired, "acquisition" when subject acquired another)
- revenue, net_income, operating_income, earnings_per_share (include the time period in the object)
- partnership, contract (for business deals)
- stock_split, dividend, share_buyback (for corporate actions)
- headquarters, founded, employees (for company facts)

Rules:
1. Extract ONLY facts explicitly stated in the article — do not infer or guess.
2. Use the company's common name as the subject (e.g., "Nvidia" not "NVDA").
3. For financial figures, include currency and period (e.g., "$60.9 billion for FY2024").
4. Each triple must be independently verifiable from the article text.
5. DO NOT invent relation types. Only use relations from the list above. If a fact doesn't fit a listed relation, skip it.
6. For acquisitions: the subject is always the ACQUIRER when using "acquisition" (subject bought object), and the subject is always the ACQUIRED party when using "acquired_by" (subject was bought by object). Never emit both directions for the same deal.
7. Do not emit the same (subject, relation, object) triple more than once per article.
8. "revenue", "net_income", etc. must be actual reported financial figures — not deal prices, margins, or projections unless explicitly stated as such.
9. If no triples can be extracted, return an empty list [].

Example output:
[
  {{"subject": "Nvidia", "relation": "CEO", "object": "Jensen Huang"}},
  {{"subject": "Nvidia", "relation": "revenue", "object": "$60.9 billion for FY2024"}},
  {{"subject": "Microsoft", "relation": "acquisition", "object": "Activision Blizzard"}}
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
                is_rate_limit = "rate limit" in message or "429" in message
                if not is_rate_limit or attempt >= max_retries:
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
