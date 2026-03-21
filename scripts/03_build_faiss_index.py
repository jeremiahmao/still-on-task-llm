"""Encode pre-cutoff corpus with BGE-M3 and build FAISS index."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.fnspid import get_text_column
from sot.retrieval.encoder import Encoder
from sot.retrieval.index import build_index, save_index
from sot.utils.config import load_config


def main():
    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    faiss_cfg = OmegaConf.load("configs/retrieval/faiss.yaml")

    data_root = Path(cfg.paths.data_root)
    corpus_path = data_root / "fnspid" / "processed" / "pre_cutoff.parquet"

    print("Loading pre-cutoff corpus...")
    df = pd.read_parquet(corpus_path)
    text_col = get_text_column(df, fnspid_cfg.text_columns)
    print(f"Corpus size: {len(df)}, text column: {text_col}")

    # Prepare texts
    texts = df[text_col].fillna("").astype(str).tolist()
    texts = [t if len(t) > 10 else "empty" for t in texts]

    # Encode
    print(f"\nEncoding with {faiss_cfg.encoder}...")
    encoder = Encoder(faiss_cfg.encoder)
    embeddings = encoder.encode(texts, batch_size=faiss_cfg.batch_size)
    print(f"Embeddings shape: {embeddings.shape}")

    # Build index
    print(f"\nBuilding FAISS index ({faiss_cfg.index_type})...")
    index = build_index(embeddings, index_type=faiss_cfg.index_type)

    # Save
    index_dir = data_root / "fnspid" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "corpus.faiss"
    save_index(index, index_path)
    print(f"Index saved: {index_path}")

    # Save doc IDs mapping (FAISS integer index -> original row index)
    doc_ids = list(range(len(df)))
    np.save(index_dir / "doc_ids.npy", np.array(doc_ids))
    print(f"Doc IDs saved: {index_dir / 'doc_ids.npy'}")

    # Save embeddings for potential reuse
    np.save(index_dir / "embeddings.npy", embeddings)
    print(f"Embeddings saved: {index_dir / 'embeddings.npy'}")


if __name__ == "__main__":
    main()
