"""BGE-M3 embedding encoder wrapper."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Encoder:
    """Wrapper around a sentence-transformer model for batch encoding."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode texts into normalized embeddings.

        Returns:
            np.ndarray of shape (len(texts), dim), float32, L2-normalized.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)
