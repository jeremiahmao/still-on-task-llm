"""Temporal query decomposition dataset generation aligned with latex.txt.

This module builds paired pre/post-2022 topic bundles, generates a single
cross-time question per topic, then asks a teacher model for separate
pre-cutoff and post-cutoff decompositions. The resulting paired examples are
filtered both for retrieval quality and for contrast between the two
decompositions.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sot.data.triple_extract import FactTriple
from sot.retrieval.index import search

QUESTION_PROMPT = """You are designing a temporal retrieval benchmark for financial news.

You are given two article bundles about the same entity or topic:
- PRE-CUTOFF: articles from before 2022-01-01
- POST-CUTOFF: articles from 2022-01-01 or later

Write exactly ONE high-level financial question that:
- Is meaningful in both time periods
- Would require retrieving evidence in either setting
- Should lead to a different retrieval strategy before and after 2022 because the world changed
- Mentions the main entity or topic explicitly
- Sounds like a normal user query asked at a single point in time
- Does NOT mention time periods, dates, "before", "after", "pre", "post", "compared with", or any explicit temporal comparison

Return ONLY the question text.

Entity/topic: {entity}

Pre-cutoff context:
{pre_context}

Post-cutoff context:
{post_context}

Question:"""

DECOMP_PROMPT = """You are designing retrieval sub-queries for a dense vector search system (BGE-M3).

Given a question and 3 evidence articles from {period_label}, write 2-4 sub-queries such that each sub-query, when embedded and searched over a corpus of news articles, would retrieve at least one of the evidence articles below.

Guidance (follow strictly — queries that miss these will fail retrieval):
- Mention specific named entities, products, people, and company names that actually appear in the articles (e.g., "TSMC 5nm partnership", not "manufacturing strategy")
- Include concrete numbers, dates, deal values, or quoted events when available
- Use article-style keywords and phrasing — think of how a news headline or lede would read, not how a search engine query would read
- Each sub-query should target a distinct article or a distinct fact within the evidence
- For the {period_label} period, the sub-queries must describe facts/events that existed in that period — do not leak information from the other period
- Return ONLY a JSON object of the form: {{"subqueries": [...]}}

Time period: {period_label}
Entity/topic: {entity}
Question: {question}

Evidence articles:
{context}

JSON:"""


def build_temporal_topic_pairs(
    pre_df: pd.DataFrame,
    post_df: pd.DataFrame,
    ticker_column: str,
    text_column: str,
    post_triples: list[FactTriple],
    date_column: str | None = None,
    min_articles_per_side: int = 5,
    bundle_size: int = 3,
    max_pairs: int | None = None,
    article_chars: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Build topic-centered pre/post bundles keyed by entity/ticker.

    The pairing logic is intentionally conservative:
    - an entity must appear on both sides of the cutoff
    - it must have at least one candidate post-cutoff changed fact
    """
    rng = random.Random(seed)

    # Reset indices so iterrows() yields 0-based positional IDs that match
    # the FAISS index doc_ids (which are np.arange(len(df))).
    pre_df = pre_df.reset_index(drop=True)
    post_df = post_df.reset_index(drop=True)

    pre_groups = pre_df.groupby(ticker_column)
    post_groups = post_df.groupby(ticker_column)
    candidate_entities = sorted(set(pre_groups.groups) & set(post_groups.groups))
    rng.shuffle(candidate_entities)

    # Build subject index once across all candidate entities.
    subject_index: dict[str, list[FactTriple]] = {}
    for t in post_triples:
        subject_index.setdefault(t.subject.lower().strip(), []).append(t)

    pairs = []
    for entity in tqdm(candidate_entities, desc="Building topic pairs"):
        entity_key = str(entity).lower().strip()
        pre_group = pre_groups.get_group(entity).copy()
        post_group = post_groups.get_group(entity).copy()

        if len(pre_group) < min_articles_per_side or len(post_group) < min_articles_per_side:
            continue

        entity_triples = _select_relevant_triples_for_entity(
            entity=str(entity),
            pre_group=pre_group,
            post_group=post_group,
            post_triples=post_triples,
            text_column=text_column,
            subject_index=subject_index,
        )
        # Include both changed facts AND new post-cutoff facts.
        changed_facts = _detect_changed_facts(
            entity=str(entity),
            pre_group=pre_group,
            post_group=post_group,
            post_triples=entity_triples,
            text_column=text_column,
        )
        # Add new facts (post-cutoff triples not already in changed_facts)
        changed_keys = {(f["subject"].lower(), f["relation"].lower()) for f in changed_facts}
        for t in entity_triples:
            key = (t.subject.lower().strip(), t.relation.lower().strip())
            if key not in changed_keys:
                changed_facts.append(
                    {"subject": t.subject, "relation": t.relation, "old": "", "new": t.object}
                )
                changed_keys.add(key)
        if not changed_facts:
            continue

        # Generate multiple pairs per entity using different article bundles.
        # Each pair gets a unique subset of facts so we don't repeat.
        n_possible = min(len(pre_group), len(post_group)) // bundle_size
        n_pairs_per_entity = max(1, min(n_possible, 20))
        facts_per_pair = max(1, len(changed_facts) // n_pairs_per_entity)
        remaining_facts = list(changed_facts)
        rng.shuffle(remaining_facts)

        for _pair_idx in range(n_pairs_per_entity):
            if not remaining_facts:
                break

            pair_facts = remaining_facts[:facts_per_pair]
            remaining_facts = remaining_facts[facts_per_pair:]

            pre_bundle = _sample_article_bundle(
                pre_group,
                text_column=text_column,
                date_column=date_column,
                bundle_size=bundle_size,
                article_chars=article_chars,
                rng=rng,
            )
            post_bundle = _sample_article_bundle(
                post_group,
                text_column=text_column,
                date_column=date_column,
                bundle_size=bundle_size,
                article_chars=article_chars,
                rng=rng,
            )

            pairs.append(
                {
                    "topic_id": f"{entity}_{len(pairs):04d}",
                    "entity": str(entity),
                    "pre_articles": pre_bundle,
                    "post_articles": post_bundle,
                    "changed_facts": pair_facts,
                }
            )

            if max_pairs is not None and len(pairs) >= max_pairs:
                break

        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    return pairs


def _select_relevant_triples_for_entity(
    entity: str,
    pre_group: pd.DataFrame,
    post_group: pd.DataFrame,
    post_triples: list[FactTriple],
    text_column: str,
    subject_index: dict[str, list[FactTriple]] | None = None,
) -> list[FactTriple]:
    """Match triples to an entity using ticker keys plus article-text aliases.

    The debug corpora are often ticker-keyed (for example, ``NVDA``) while the
    extractor emits company-name subjects such as ``Nvidia`` or
    ``NVIDIA Corporation``. For temporal pairing we accept triples whose
    subjects are explicitly mentioned in the entity's article bundle.
    """
    entity_key = entity.lower().strip()
    # Truncate each article to first 600 chars — subject mentions tend to appear
    # early (headline + lede) and this keeps substring search tractable.
    bundle_text = " ".join(
        s[:600]
        for s in (
            pre_group[text_column].fillna("").astype(str).tolist()
            + post_group[text_column].fillna("").astype(str).tolist()
        )
    ).lower()

    # Build subject index once (shared across entity loop via subject_index arg).
    # Falls back to local index if not provided.
    if subject_index is None:
        subject_index = {}
        for t in post_triples:
            subject_index.setdefault(t.subject.lower().strip(), []).append(t)

    matched = []
    seen = set()
    for subject_key, triples in subject_index.items():
        if subject_key == entity_key or subject_key in bundle_text:
            for triple in triples:
                triple_key = (
                    subject_key,
                    triple.relation.lower().strip(),
                    triple.object.lower().strip(),
                )
                if triple_key not in seen:
                    matched.append(triple)
                    seen.add(triple_key)

    return matched


def generate_temporal_question(
    pair: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 96,
) -> str | None:
    prompt = QUESTION_PROMPT.format(
        entity=pair["entity"],
        pre_context=_format_articles_for_prompt(pair["pre_articles"]),
        post_context=_format_articles_for_prompt(pair["post_articles"]),
    )
    response = _run_local_generation(prompt, model, tokenizer, max_new_tokens=max_new_tokens)
    question = _normalize_question(response)
    return question if len(question) >= 20 else None


async def generate_temporal_question_api(
    pair: dict,
    api_func: Callable[[str], str],
) -> str | None:
    prompt = QUESTION_PROMPT.format(
        entity=pair["entity"],
        pre_context=_format_articles_for_prompt(pair["pre_articles"]),
        post_context=_format_articles_for_prompt(pair["post_articles"]),
    )
    response = await api_func(prompt)
    question = _normalize_question(response)
    return question if len(question) >= 20 else None


def generate_temporal_question_api_sync(
    pair: dict,
    api_func: Callable[[str], str],
) -> str | None:
    prompt = QUESTION_PROMPT.format(
        entity=pair["entity"],
        pre_context=_format_articles_for_prompt(pair["pre_articles"]),
        post_context=_format_articles_for_prompt(pair["post_articles"]),
    )
    response = api_func(prompt)
    question = _normalize_question(response)
    return question if len(question) >= 20 else None


def generate_temporal_decomposition(
    pair: dict,
    question: str,
    period: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 192,
) -> list[str] | None:
    articles = pair["pre_articles"] if period == "pre" else pair["post_articles"]
    period_label = "pre-2022" if period == "pre" else "post-2022"
    prompt = DECOMP_PROMPT.format(
        entity=pair["entity"],
        period_label=period_label,
        question=question,
        context=_format_articles_for_prompt(articles),
    )
    response = _run_local_generation(prompt, model, tokenizer, max_new_tokens=max_new_tokens)
    return _parse_decomposition(response)


async def generate_temporal_decomposition_api(
    pair: dict,
    question: str,
    period: str,
    api_func: Callable[[str], str],
) -> list[str] | None:
    articles = pair["pre_articles"] if period == "pre" else pair["post_articles"]
    period_label = "pre-2022" if period == "pre" else "post-2022"
    prompt = DECOMP_PROMPT.format(
        entity=pair["entity"],
        period_label=period_label,
        question=question,
        context=_format_articles_for_prompt(articles),
    )
    response = await api_func(prompt, text_format=_decomposition_json_schema())
    return _parse_decomposition(response)


def generate_temporal_decomposition_api_sync(
    pair: dict,
    question: str,
    period: str,
    api_func: Callable[[str], str],
) -> list[str] | None:
    articles = pair["pre_articles"] if period == "pre" else pair["post_articles"]
    period_label = "pre-2022" if period == "pre" else "post-2022"
    prompt = DECOMP_PROMPT.format(
        entity=pair["entity"],
        period_label=period_label,
        question=question,
        context=_format_articles_for_prompt(articles),
    )
    response = api_func(prompt, text_format=_decomposition_json_schema())
    return _parse_decomposition(response)


def score_decomposition_recall(
    decomposition: list[str],
    encoder,
    faiss_index,
    doc_ids: list[int],
    gold_article_ids: list[int],
    corpus_embeddings: np.ndarray | None = None,
    k: int = 10,
    nprobe: int = 64,
) -> float:
    """Score a single decomposition against a target article set."""
    if not decomposition or not gold_article_ids:
        return 0.0

    embeddings = encoder.encode(decomposition, show_progress=False)
    if corpus_embeddings is not None:
        similarity = embeddings @ corpus_embeddings.T
        top_k = min(k, corpus_embeddings.shape[0])
        indices = np.argsort(-similarity, axis=1)[:, :top_k]
    else:
        _scores, indices = search(faiss_index, embeddings, k=k, nprobe=nprobe)

    retrieved = set()
    for row in indices:
        for idx in row:
            if 0 <= idx < len(doc_ids):
                retrieved.add(doc_ids[idx])

    gold = set(gold_article_ids)
    return len(retrieved & gold) / max(len(gold), 1)


def decomposition_contrast_score(pre_decomposition: list[str], post_decomposition: list[str]) -> float:
    """Higher means the two decompositions are more different."""
    pre_norm = {_normalize_subquery(q) for q in pre_decomposition if q.strip()}
    post_norm = {_normalize_subquery(q) for q in post_decomposition if q.strip()}

    if not pre_norm and not post_norm:
        return 0.0

    set_overlap = len(pre_norm & post_norm) / max(len(pre_norm | post_norm), 1)

    pre_tokens = set().union(*(_tokenize_query(q) for q in pre_norm)) if pre_norm else set()
    post_tokens = set().union(*(_tokenize_query(q) for q in post_norm)) if post_norm else set()
    token_overlap = len(pre_tokens & post_tokens) / max(len(pre_tokens | post_tokens), 1)

    similarity = 0.5 * set_overlap + 0.5 * token_overlap
    return 1.0 - similarity


def build_temporal_training_example(item: dict) -> dict:
    """Create a pre-cutoff QD training example compatible with SFT."""
    decomp_str = "\n".join(f"- {sq}" for sq in item["pre_decomposition"])
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial search expert. Given a complex financial question, "
                    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
                    "relevant documents from a financial news database."
                ),
            },
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": decomp_str},
        ],
        "gold_articles": item["pre_gold_articles"],
        "recall": item["pre_recall"],
        "topic_id": item["topic_id"],
        "entity": item["entity"],
        "changed_facts": item["changed_facts"],
        "post_decomposition": item["post_decomposition"],
        "post_recall": item["post_recall"],
        "contrast_score": item["contrast_score"],
    }


def split_train_test(
    paired_examples: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(paired_examples)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def _detect_changed_facts(
    entity: str,
    pre_group: pd.DataFrame,
    post_group: pd.DataFrame,
    post_triples: list[FactTriple],
    text_column: str,
) -> list[dict]:
    """Heuristic changed-fact detector using post-cutoff triples.

    A post triple is treated as changed/new when the object string does not
    appear in the pre-cutoff evidence bundle but does appear in the post bundle.
    """
    # Truncate each article to first 600 chars to keep substring search tractable.
    pre_text = " ".join(
        s[:600] for s in pre_group[text_column].fillna("").astype(str).tolist()
    ).lower()
    post_text = " ".join(
        s[:600] for s in post_group[text_column].fillna("").astype(str).tolist()
    ).lower()

    changed = []
    for triple in post_triples:
        obj = triple.object.strip()
        if len(obj) < 2:
            continue
        obj_norm = obj.lower()
        if obj_norm in post_text and obj_norm not in pre_text:
            changed.append(
                {
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object_post": triple.object,
                }
            )

    # Relation-level dedupe
    deduped = []
    seen = set()
    for item in changed:
        key = (
            item["subject"].lower().strip(),
            item["relation"].lower().strip(),
            item["object_post"].lower().strip(),
        )
        if key not in seen:
            deduped.append(item)
            seen.add(key)

    return deduped[:5]


def _sample_article_bundle(
    df: pd.DataFrame,
    text_column: str,
    date_column: str | None,
    bundle_size: int,
    article_chars: int,
    rng: random.Random,
) -> list[dict]:
    sampled = df.sample(n=min(bundle_size, len(df)), random_state=rng.randint(0, 10_000)).copy()

    articles = []
    for idx, row in sampled.iterrows():
        text = str(row.get(text_column, "")).strip()
        if len(text) < 20:
            continue
        item = {
            "doc_id": int(idx),
            "title": str(row.get("Article_title", "")).strip(),
            "date": str(row.get(date_column, "")) if date_column else "",
            "text": text[:article_chars],
        }
        articles.append(item)

    return articles


def _format_articles_for_prompt(articles: list[dict]) -> str:
    formatted = []
    for i, article in enumerate(articles, start=1):
        title = article.get("title", "")
        date = article.get("date", "")
        text = article.get("text", "")
        formatted.append(f"Article {i} | {date} | {title}\n{text}")
    return "\n\n---\n\n".join(formatted)


def _run_local_generation(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
) -> str:
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

    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


def _normalize_question(response: str) -> str:
    first_line = response.strip().split("\n")[0].strip()
    return first_line.rstrip("?").strip() + "?" if first_line else ""


def _parse_decomposition(response: str) -> list[str] | None:
    try:
        obj_match = re.search(r"\{.*\}", response, re.DOTALL)
        if obj_match:
            data = json.loads(obj_match.group())
            if isinstance(data, dict) and isinstance(data.get("subqueries"), list):
                cleaned = [str(item).strip() for item in data["subqueries"] if str(item).strip()]
                if 2 <= len(cleaned) <= 4:
                    return cleaned

        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                cleaned = [item.strip() for item in data if item.strip()]
                if 2 <= len(cleaned) <= 4:
                    return cleaned
    except (json.JSONDecodeError, ValueError):
        pass

    lines = [line.strip().lstrip("0123456789.-) ") for line in response.strip().split("\n")]
    lines = [line for line in lines if len(line) > 10]
    return lines if 2 <= len(lines) <= 4 else None


def _normalize_subquery(query: str) -> str:
    query = query.lower().strip()
    query = re.sub(r"\s+", " ", query)
    return query


def _tokenize_query(query: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", query.lower()))


def _decomposition_json_schema() -> dict:
    return {
        "type": "json_schema",
        "name": "temporal_subqueries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "subqueries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 4,
                }
            },
            "required": ["subqueries"],
            "additionalProperties": False,
        },
    }
