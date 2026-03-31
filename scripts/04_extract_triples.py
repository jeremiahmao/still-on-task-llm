"""Extract fact triples from post-cutoff FNSPID articles, filter, and sample at scales."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from omegaconf import OmegaConf

from sot.data.fnspid import get_text_column, load_fnspid
from sot.data.triple_extract import extract_triples_batch, save_triples
from sot.data.triple_filter import (
    extract_entities_from_corpus,
    filter_by_entities,
    filter_cross_doc_agreement,
    sample_at_scales,
    save_scaled_triples,
)
from sot.models.base import load_model
from sot.utils.config import load_config


def main():
    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    triples_cfg = OmegaConf.load("configs/data/triples.yaml")

    data_root = Path(cfg.paths.data_root)
    post_path = data_root / "fnspid" / "processed" / "post_cutoff.parquet"
    pre_path = data_root / "fnspid" / "processed" / "pre_cutoff.parquet"

    if not post_path.exists():
        print(f"ERROR: {post_path} not found. Run 02_build_corpus.py first.")
        sys.exit(1)

    print("Loading post-cutoff articles...")
    post_df = load_fnspid(post_path)
    text_col = get_text_column(post_df, fnspid_cfg.text_columns)
    print(f"Post-cutoff articles: {len(post_df)}, text column: {text_col}")

    # Convert to list of dicts for extraction
    articles = post_df.to_dict("records")

    # Load extraction model (use the base Qwen model as teacher)
    print(f"\nLoading extraction model: {cfg.model.name}")
    model, tokenizer = load_model(cfg.model.name, cfg.model.dtype)

    # Extract triples
    print(f"\nExtracting triples from {len(articles)} articles...")
    raw_triples = extract_triples_batch(
        articles,
        model,
        tokenizer,
        text_column=text_col,
        id_column=None,
        batch_size=1,
    )
    print(f"Raw triples extracted: {len(raw_triples)}")

    # Save raw triples
    raw_dir = data_root / "fnspid" / "triples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_triples(raw_triples, str(raw_dir / "raw_triples.json"))

    # Filter by cross-document agreement
    min_agree = triples_cfg.get("min_cross_doc_agreement", 2)
    print(f"\nFiltering by cross-doc agreement (min={min_agree})...")
    agreed = filter_cross_doc_agreement(raw_triples, min_agreement=min_agree)
    print(f"After cross-doc filter: {len(agreed)} triples")

    # Filter by known entities from pre-cutoff corpus
    print("Filtering by known entities...")
    pre_df = pd.read_parquet(pre_path)
    known_entities = extract_entities_from_corpus(pre_df, fnspid_cfg.ticker_column)
    filtered = filter_by_entities(agreed, known_entities)
    print(f"After entity filter: {len(filtered)} triples")

    save_triples(filtered, str(raw_dir / "filtered_triples.json"))

    # Sample at target scales
    scales = triples_cfg.get("scales", [1000, 3000])
    print(f"\nSampling at scales: {scales}")
    scaled = sample_at_scales(filtered, scales, seed=cfg.seed)
    save_scaled_triples(scaled, raw_dir)

    for scale, triples in scaled.items():
        print(f"  Scale {scale}: {len(triples)} triples saved")

    # Free GPU memory
    del model
    import torch

    torch.cuda.empty_cache()

    print("\nDone. Triple extraction complete.")


if __name__ == "__main__":
    main()
