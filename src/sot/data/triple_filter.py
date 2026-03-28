"""Cross-document agreement filtering and entity-based filtering for fact triples."""

import random
from collections import defaultdict
from pathlib import Path

from sot.data.triple_extract import FactTriple


def normalize_value(value: str) -> str:
    """Normalize a triple value for fuzzy matching.

    Handles: whitespace, case, common financial abbreviations.
    """
    v = value.lower().strip()
    # Normalize currency/number formatting
    v = v.replace(",", "").replace("$", "").replace("usd", "")
    # Normalize billions/millions
    v = v.replace(" billion", "b").replace(" million", "m")
    v = v.replace(" bn", "b").replace(" mn", "m")
    return v


def normalize_triple_key(triple: FactTriple) -> str:
    """Create a normalized key for cross-document agreement."""
    subj = triple.subject.lower().strip()
    rel = triple.relation.lower().strip()
    obj = normalize_value(triple.object)
    return f"{subj}|{rel}|{obj}"


def filter_cross_doc_agreement(
    triples: list[FactTriple],
    min_agreement: int = 2,
) -> list[FactTriple]:
    """Keep only triples that appear in N+ independent articles.

    Returns one representative triple per unique (subject, relation, object).
    """
    # Group by normalized key
    groups: dict[str, list[FactTriple]] = defaultdict(list)
    for t in triples:
        key = normalize_triple_key(t)
        groups[key].append(t)

    # Keep triples with sufficient agreement
    filtered = []
    for _key, group in groups.items():
        # Count unique source articles
        unique_sources = set(t.source_article_id for t in group)
        if len(unique_sources) >= min_agreement:
            # Keep the first occurrence as representative
            filtered.append(group[0])

    return filtered


def filter_by_entities(
    triples: list[FactTriple],
    known_entities: set[str],
) -> list[FactTriple]:
    """Keep only triples whose subject entity appears in the known entity set.

    Matching is case-insensitive.
    """
    known_lower = {e.lower().strip() for e in known_entities}
    return [t for t in triples if t.subject.lower().strip() in known_lower]


def extract_entities_from_corpus(
    corpus_df,
    ticker_column: str,
) -> set[str]:
    """Extract unique entity names (tickers) from the pre-cutoff corpus."""
    return set(corpus_df[ticker_column].dropna().unique())


def extract_entities_from_finqa(finqa_examples: list[dict]) -> set[str]:
    """Extract entity names mentioned in FinQA examples.

    Looks in pre_text, post_text, and table headers for company names.
    """
    entities = set()
    for ex in finqa_examples:
        # FinQA examples often mention company names in text
        for text_field in ("pre_text", "post_text"):
            texts = ex.get(text_field, [])
            if isinstance(texts, list):
                for t in texts:
                    # Simple heuristic: extract capitalized multi-word phrases
                    # This is imperfect but catches most company names
                    entities.add(str(t).strip())
    return entities


def sample_at_scales(
    triples: list[FactTriple],
    scales: list[int],
    seed: int = 42,
) -> dict[int, list[FactTriple]]:
    """Sample triples at multiple scales, stratified by relation type.

    Returns:
        Dict mapping scale -> list of sampled triples.
    """
    rng = random.Random(seed)

    # Group by relation type
    by_relation: dict[str, list[FactTriple]] = defaultdict(list)
    for t in triples:
        by_relation[t.relation.lower().strip()].append(t)

    result = {}
    for scale in sorted(scales):
        if scale >= len(triples):
            result[scale] = list(triples)
            continue

        # Proportional sampling per relation type
        sampled = []
        for _rel, rel_triples in by_relation.items():
            n_from_rel = max(1, int(len(rel_triples) / len(triples) * scale))
            sample = rng.sample(rel_triples, min(n_from_rel, len(rel_triples)))
            sampled.extend(sample)

        # Trim or pad to exact scale
        if len(sampled) > scale:
            sampled = rng.sample(sampled, scale)
        elif len(sampled) < scale:
            remaining = [t for t in triples if t not in sampled]
            extra = rng.sample(remaining, min(scale - len(sampled), len(remaining)))
            sampled.extend(extra)

        result[scale] = sampled

    return result


def save_scaled_triples(
    scaled: dict[int, list[FactTriple]],
    output_dir: str | Path,
) -> None:
    """Save scaled triple sets to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for scale, triples in scaled.items():
        path = output_dir / f"triples_{scale}.json"
        from sot.data.triple_extract import save_triples

        save_triples(triples, str(path))
