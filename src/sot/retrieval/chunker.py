"""Split articles into chunks for retrieval indexing."""

import numpy as np


def chunk_articles(
    texts: list[str],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> tuple[list[str], np.ndarray]:
    """Split articles into overlapping chunks.

    Args:
        texts: List of article texts (one per article).
        chunk_size: Target chunk size in words.
        chunk_overlap: Overlap between consecutive chunks in words.

    Returns:
        Tuple of (chunks, chunk_to_article) where:
        - chunks: list of chunk strings
        - chunk_to_article: int array mapping chunk index -> article index
    """
    chunks = []
    chunk_to_article = []

    for article_idx, text in enumerate(texts):
        if not text or len(text.strip()) < 20:
            # Still add one "empty" chunk so article indices stay aligned
            chunks.append("empty")
            chunk_to_article.append(article_idx)
            continue

        words = text.split()
        if len(words) <= chunk_size:
            # Short article — keep as single chunk
            chunks.append(text)
            chunk_to_article.append(article_idx)
        else:
            # Split into overlapping chunks
            step = chunk_size - chunk_overlap
            for start in range(0, len(words), step):
                chunk_words = words[start : start + chunk_size]
                if len(chunk_words) < 20:
                    break  # Skip tiny trailing chunks
                chunks.append(" ".join(chunk_words))
                chunk_to_article.append(article_idx)

    return chunks, np.array(chunk_to_article, dtype=np.int64)
