"""Filter decompositions by Recall@10 and split into train/test."""

import json
import random
from pathlib import Path

from sot.retrieval.encoder import Encoder
from sot.retrieval.index import search


def filter_decompositions(
    decomps: list[dict],
    encoder: Encoder,
    faiss_index,
    doc_ids: list,
    min_recall: float = 0.7,
    k: int = 10,
    nprobe: int = 64,
) -> list[dict]:
    """Filter decompositions by Recall@10, keeping only those that retrieve well.

    For each question, selects the best decomposition (highest Recall@10).

    Args:
        decomps: List of dicts with 'question', 'gold_articles', 'decompositions'.
        encoder: BGE-M3 encoder for embedding sub-queries.
        faiss_index: Pre-built FAISS index over the corpus.
        doc_ids: Mapping from FAISS integer IDs to article IDs.
        min_recall: Minimum Recall@10 threshold.
        k: Number of retrieved documents per sub-query.
        nprobe: FAISS nprobe setting.

    Returns:
        Filtered list of dicts with 'question', 'gold_articles', 'decomposition' (best one),
        and 'recall' score.
    """
    filtered = []

    for item in decomps:
        gold_set = set(item["gold_articles"])
        best_decomp = None
        best_recall = 0.0

        for candidate in item["decompositions"]:
            # Encode all sub-queries
            sub_query_embeddings = encoder.encode(candidate, show_progress=False)

            # Retrieve top-k per sub-query
            scores, indices = search(faiss_index, sub_query_embeddings, k=k, nprobe=nprobe)

            # Union of retrieved doc IDs across sub-queries
            retrieved = set()
            for row in indices:
                for idx in row:
                    if idx >= 0 and idx < len(doc_ids):
                        retrieved.add(doc_ids[idx])

            # Compute recall
            if len(gold_set) == 0:
                continue
            recall = len(retrieved & gold_set) / len(gold_set)

            if recall > best_recall:
                best_recall = recall
                best_decomp = candidate

        if best_decomp is not None and best_recall >= min_recall:
            filtered.append(
                {
                    "question": item["question"],
                    "gold_articles": item["gold_articles"],
                    "decomposition": best_decomp,
                    "recall": best_recall,
                }
            )

    return filtered


def split_train_test(
    data: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split filtered decompositions into train and test sets."""
    rng = random.Random(seed)
    shuffled = list(data)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_idx], shuffled[split_idx:]


def format_qd_example(item: dict) -> dict:
    """Format a filtered decomposition as a chat-style training example."""
    decomp_str = "\n".join(f"- {sq}" for sq in item["decomposition"])

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
        "gold_articles": item["gold_articles"],
        "recall": item["recall"],
    }


def save_qd_dataset(data: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_qd_dataset(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)
