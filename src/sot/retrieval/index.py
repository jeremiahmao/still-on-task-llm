"""FAISS index build, save, load, and search."""

from pathlib import Path

import faiss
import numpy as np


def build_index(
    embeddings: np.ndarray,
    index_type: str = "IVF4096,PQ64",
    train_size: int = 100_000,
) -> faiss.Index:
    """Build a FAISS index from embeddings.

    Args:
        embeddings: (N, dim) float32 array, L2-normalized.
        index_type: FAISS index factory string.
        train_size: Number of vectors to use for training (for IVF/PQ indices).

    Returns:
        Trained FAISS index with all vectors added.
    """
    dim = embeddings.shape[1]
    index = faiss.index_factory(dim, index_type, faiss.METRIC_INNER_PRODUCT)

    # Train on a subsample
    if train_size < len(embeddings):
        rng = np.random.default_rng(42)
        train_idx = rng.choice(len(embeddings), train_size, replace=False)
        train_data = embeddings[train_idx]
    else:
        train_data = embeddings

    index.train(train_data)
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    """Save a FAISS index to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str | Path) -> faiss.Index:
    """Load a FAISS index from disk."""
    return faiss.read_index(str(path))


def search(
    index: faiss.Index,
    queries: np.ndarray,
    k: int = 10,
    nprobe: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Search the index for top-k nearest neighbors.

    Args:
        index: FAISS index.
        queries: (Q, dim) float32 array.
        k: Number of results per query.
        nprobe: Number of IVF cells to probe (higher = more accurate, slower).

    Returns:
        (scores, indices) each of shape (Q, k).
    """
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    scores, indices = index.search(queries, k)
    return scores, indices
