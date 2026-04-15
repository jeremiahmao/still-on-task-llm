"""Encode pre-cutoff corpus with BGE-M3 and build FAISS index."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.fnspid import get_text_column
from sot.retrieval.chunker import chunk_articles
from sot.retrieval.encoder import Encoder
from sot.retrieval.index import build_index, save_index
from sot.utils.config import load_config


def build_corpus_index(df, text_col, faiss_cfg, index_dir, name_suffix, force=False):
    """Chunk a corpus, encode chunks with BGE-M3, and build/save a FAISS index."""
    index_path = index_dir / f"corpus{name_suffix}.faiss"
    if not force and index_path.exists():
        print(f"Index already exists at {index_path}, skipping.")
        return

    texts = df[text_col].fillna("").astype(str).tolist()
    print(f"  Articles: {len(texts)}")

    # Chunk articles for better retrieval granularity
    chunk_size = faiss_cfg.get("chunk_size", 300)
    chunk_overlap = faiss_cfg.get("chunk_overlap", 50)
    chunks, chunk_to_article = chunk_articles(texts, chunk_size, chunk_overlap)
    print(f"  Chunks: {len(chunks)} (avg {len(chunks)/max(len(texts),1):.1f} per article)")

    # Encode chunks
    print(f"\n  Encoding {len(chunks)} chunks with {faiss_cfg.encoder}...")
    encoder = Encoder(faiss_cfg.encoder)
    embeddings = encoder.encode(chunks, batch_size=faiss_cfg.batch_size)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Build index
    print(f"\n  Building FAISS index ({faiss_cfg.index_type})...")
    index = build_index(embeddings, index_type=faiss_cfg.index_type)

    # Save
    index_dir.mkdir(parents=True, exist_ok=True)
    save_index(index, index_dir / f"corpus{name_suffix}.faiss")
    print(f"  Index saved: {index_dir / f'corpus{name_suffix}.faiss'}")

    # Save chunk-to-article mapping (FAISS index -> article row index)
    np.save(index_dir / f"chunk_to_article{name_suffix}.npy", chunk_to_article)
    print(f"  Chunk mapping saved: {index_dir / f'chunk_to_article{name_suffix}.npy'}")

    # Save article-level doc_ids for backward compatibility
    # (evaluation maps retrieved chunks back to articles via chunk_to_article)
    np.save(index_dir / f"doc_ids{name_suffix}.npy", np.arange(len(df)))
    print(f"  Doc IDs saved: {index_dir / f'doc_ids{name_suffix}.npy'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug corpus files from configs/data/fnspid.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    faiss_cfg_path = (
        "configs/retrieval/faiss_debug.yaml" if args.debug else "configs/retrieval/faiss.yaml"
    )
    faiss_cfg = OmegaConf.load(faiss_cfg_path)

    data_root = Path(cfg.paths.data_root)
    suffix = fnspid_cfg.debug.output_suffix if args.debug else ""
    index_dir = data_root / "fnspid" / "index"

    # Pre-cutoff index (used for task preservation evaluation)
    print("Loading pre-cutoff corpus...")
    pre_df = pd.read_parquet(data_root / "fnspid" / "processed" / f"pre_cutoff{suffix}.parquet")
    text_col = get_text_column(pre_df, fnspid_cfg.text_columns)
    print(f"Pre-cutoff corpus: {len(pre_df)} articles, text column: {text_col}")
    build_corpus_index(pre_df, text_col, faiss_cfg, index_dir, suffix)

    # Post-cutoff index (used for post-cutoff task adaptation evaluation)
    print("\nLoading post-cutoff corpus...")
    post_df = pd.read_parquet(data_root / "fnspid" / "processed" / f"post_cutoff{suffix}.parquet")
    post_text_col = get_text_column(post_df, fnspid_cfg.text_columns)
    print(f"Post-cutoff corpus: {len(post_df)} articles, text column: {post_text_col}")
    build_corpus_index(post_df, post_text_col, faiss_cfg, index_dir, f"_post{suffix}")


if __name__ == "__main__":
    main()
