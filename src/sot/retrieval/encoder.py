"""BGE-M3 embedding encoder wrapper."""

import numpy as np
import torch
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

        Uses all available GPUs via multi_process_encode when more than one is detected.

        Returns:
            np.ndarray of shape (len(texts), dim), float32, L2-normalized.
        """
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1 and len(texts) > 1000:
            pool = self.model.start_multi_process_pool(
                ["cuda:{}".format(i) for i in range(n_gpus)]
            )
            embeddings = self.model.encode_multi_process(
                texts,
                pool,
                batch_size=batch_size,
                normalize_embeddings=True,
            )
            self.model.stop_multi_process_pool(pool)
        else:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
            )
        return embeddings.astype(np.float32)
