"""Recall@K computation for retrieval evaluation."""

import numpy as np


def compute_recall_at_k(
    retrieved_ids: list[set[int]],
    gold_ids: list[set[int]],
    k: int = 10,
) -> dict:
    """Compute Recall@K.

    Args:
        retrieved_ids: Per-query sets of retrieved document IDs (already truncated to top-k).
        gold_ids: Per-query sets of gold-relevant document IDs.

    Returns:
        Dict with 'mean', 'std', and 'per_query' Recall@K values.
    """
    recalls = []
    for retrieved, gold in zip(retrieved_ids, gold_ids):
        if len(gold) == 0:
            continue
        top_k = set(list(retrieved)[:k]) if len(retrieved) > k else retrieved
        recall = len(top_k & gold) / len(gold)
        recalls.append(recall)

    recalls = np.array(recalls)
    return {
        "mean": float(recalls.mean()) if len(recalls) > 0 else 0.0,
        "std": float(recalls.std()) if len(recalls) > 0 else 0.0,
        "per_query": recalls.tolist(),
        "n_queries": len(recalls),
    }


def decomposition_recall(
    sub_query_results: list[list[set[int]]],
    gold_ids: list[set[int]],
    k: int = 10,
) -> dict:
    """Compute Recall@K for query decomposition.

    For each question, takes the union of top-k results from all sub-queries,
    then measures recall against gold documents.

    Args:
        sub_query_results: Per-question list of per-sub-query retrieved ID sets.
        gold_ids: Per-question sets of gold-relevant document IDs.
    """
    union_results = []
    for sub_results in sub_query_results:
        union = set()
        for sq_result in sub_results:
            union.update(list(sq_result)[:k])
        union_results.append(union)

    return compute_recall_at_k(union_results, gold_ids, k=k)
