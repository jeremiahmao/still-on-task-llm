"""Run evaluation metrics on an updated model."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from sot.data.triple_extract import FactTriple
from sot.data.triple_render import render_triple
from sot.eval.compositional import evaluate_compositional
from sot.eval.generic_forgetting import evaluate_generic_forgetting
from sot.eval.knowledge_absorption import evaluate_knowledge_absorption
from sot.eval.locality import evaluate_locality
from sot.eval.task_preservation import evaluate_task_preservation
from sot.eval.temporal_contrast import evaluate_temporal_contrast
from sot.retrieval.encoder import Encoder
from sot.retrieval.index import load_index
from sot.utils.config import load_config
from sot.utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to updated model")
    parser.add_argument("--task", default="qd", choices=["qd", "finqa"])
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated: preservation,absorption,forgetting,locality,"
        "compositional,temporal_contrast",
    )
    parser.add_argument(
        "--compositional-probes-path",
        default=None,
        help="Path to compositional probes JSON. Defaults to "
        "data/fnspid/compositional/probes.json.",
    )
    parser.add_argument(
        "--paired-examples-path",
        default=None,
        help="Path to qd_temporal paired_examples.json for temporal-contrast eval. "
        "Defaults to <qd_temporal_data_root>/paired_examples.json.",
    )
    parser.add_argument("--debug", action="store_true", help="Use debug data paths")
    parser.add_argument(
        "--locality-subsample",
        type=int,
        default=None,
        help="Randomly subsample locality facts to N items (stratified). Speeds up eval massively.",
    )
    args = parser.parse_args()

    cfg = load_config()
    fnspid_cfg = OmegaConf.load("configs/data/fnspid.yaml")
    seed_everything(cfg.seed)
    suffix = fnspid_cfg.debug.output_suffix if args.debug else ""

    data_root = Path(cfg.paths.data_root)
    model_path = Path(args.model_path).resolve()

    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path / "model"), torch_dtype="auto", device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "model"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    metrics_to_run = (
        args.metrics.split(",")
        if args.metrics != "all"
        else [
            "preservation",
            "post_preservation",
            "absorption",
            "forgetting",
            "locality",
            "compositional",
            "temporal_contrast",
        ]
    )

    results = {}

    # Shared encoder for both preservation metrics
    encoder = None

    # Task preservation — Recall@10 on pre-cutoff QD test set
    if "preservation" in metrics_to_run and args.task == "qd":
        print("\n--- Task Preservation (pre-cutoff) ---")
        qd_root = Path(cfg.paths.qd_temporal_data_root)
        if suffix:
            qd_root = qd_root / suffix.lstrip("_")
        test_path = qd_root / "test.json"
        index_path = data_root / "fnspid" / "index" / f"corpus{suffix}.faiss"
        doc_ids_path = data_root / "fnspid" / "index" / f"doc_ids{suffix}.npy"
        chunk_map_path = data_root / "fnspid" / "index" / f"chunk_to_article{suffix}.npy"
        if test_path.exists() and index_path.exists():
            with open(test_path) as f:
                test_data = json.load(f)
            if encoder is None:
                encoder = Encoder()
            index = load_index(index_path)
            doc_ids = np.load(doc_ids_path).tolist()
            chunk_to_article = np.load(chunk_map_path).tolist() if chunk_map_path.exists() else None
            pres = evaluate_task_preservation(
                model, tokenizer, test_data, encoder, index, doc_ids,
                chunk_to_article=chunk_to_article,
            )
            results["task_preservation"] = pres
            print(f"  Recall@10: {pres['mean']:.4f} (std={pres['std']:.4f})")

    # Post-cutoff task adaptation — Recall@10 on post-cutoff corpus
    # Measures whether updated models retrieve post-2022 articles better
    if "post_preservation" in metrics_to_run and args.task == "qd":
        print("\n--- Post-Cutoff Task Adaptation ---")
        post_test_path = qd_root / "post_test.json"
        post_index_path = data_root / "fnspid" / "index" / f"corpus_post{suffix}.faiss"
        post_doc_ids_path = data_root / "fnspid" / "index" / f"doc_ids_post{suffix}.npy"
        post_chunk_map_path = data_root / "fnspid" / "index" / f"chunk_to_article_post{suffix}.npy"
        if post_test_path.exists() and post_index_path.exists():
            with open(post_test_path) as f:
                post_test_data = json.load(f)
            if encoder is None:
                encoder = Encoder()
            post_index = load_index(post_index_path)
            post_doc_ids = np.load(post_doc_ids_path).tolist()
            post_chunk_map = np.load(post_chunk_map_path).tolist() if post_chunk_map_path.exists() else None
            post_pres = evaluate_task_preservation(
                model, tokenizer, post_test_data, encoder, post_index, post_doc_ids,
                chunk_to_article=post_chunk_map,
            )
            results["post_task_preservation"] = post_pres
            print(f"  Recall@10: {post_pres['mean']:.4f} (std={post_pres['std']:.4f})")

    # Knowledge absorption
    if "absorption" in metrics_to_run:
        print("\n--- Knowledge Absorption ---")
        # Load the fact QA pairs that were injected
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            scale = meta.get("scale", 200)
            # Prefer the explicit triples_path recorded during the update run
            # (needed for sequential editing, where round_k.json is the right set).
            triples_path = None
            meta_triples = meta.get("triples_path")
            if meta_triples and Path(meta_triples).exists():
                triples_path = Path(meta_triples)
            if triples_path is None:
                triples_dir = data_root / "fnspid" / "triples"
                triples_path = triples_dir / f"triples_{scale}.json"
                if not triples_path.exists():
                    triples_path = triples_dir / "debug" / f"triples_{scale}.json"
            if triples_path.exists():
                with open(triples_path) as f:
                    raw_triples = json.load(f)
                fact_qa = []
                for t in raw_triples:
                    triple = FactTriple(**t)
                    qa = render_triple(triple)
                    fact_qa.append(
                        {
                            "question": qa.question,
                            "answer": qa.answer,
                            "phrasings": qa.phrasings,
                        }
                    )
                absorb = evaluate_knowledge_absorption(model, tokenizer, fact_qa)
                results["knowledge_absorption"] = {
                    "exact_match": absorb["exact_match"],
                    "mean_f1": absorb["mean_f1"],
                    "contains": absorb.get("contains", 0.0),
                    "fact_mean_f1": absorb.get("fact_mean_f1", absorb["mean_f1"]),
                    "fact_worst_f1": absorb.get("fact_worst_f1", absorb["mean_f1"]),
                    "contains_any_phrasing": absorb.get("contains_any_phrasing", 0.0),
                    "contains_all_phrasings": absorb.get("contains_all_phrasings", 0.0),
                    "n_facts": absorb.get("n_facts", 0),
                    "n_probes": absorb.get("n_probes", 0),
                }
                print(f"  Probe-level: EM={absorb['exact_match']:.4f}  contains={absorb.get('contains', 0):.4f}  F1={absorb['mean_f1']:.4f}")
                print(f"  Per-fact F1: mean={absorb.get('fact_mean_f1', 0):.4f}  worst={absorb.get('fact_worst_f1', 0):.4f}")
                print(f"  Contains@any phrasing: {absorb.get('contains_any_phrasing', 0):.4f}  Contains@all: {absorb.get('contains_all_phrasings', 0):.4f}")

    # Generic forgetting (FinQA)
    if "forgetting" in metrics_to_run:
        print("\n--- Generic Forgetting (FinQA) ---")
        finqa_path = data_root / "finqa" / "dataset"
        if finqa_path.exists():
            from sot.data.finqa import prepare_finqa_dataset

            finqa_cfg = OmegaConf.load("configs/data/finqa.yaml")
            test_data = prepare_finqa_dataset(finqa_path, finqa_cfg.system_prompt, split="dev")
            forget = evaluate_generic_forgetting(model, tokenizer, test_data)
            results["generic_forgetting"] = {
                "execution_accuracy": forget["execution_accuracy"],
            }
            print(f"  Execution accuracy: {forget['execution_accuracy']:.4f}")

    # Locality
    if "locality" in metrics_to_run:
        print("\n--- Locality ---")
        locality_path = data_root / "fnspid" / f"locality_facts{suffix}.json"
        if locality_path.exists():
            with open(locality_path) as f:
                locality_facts = json.load(f)
            # Optional: stratified subsample for speed. Keeps same strata ratios.
            if args.locality_subsample is not None and args.locality_subsample < len(locality_facts):
                import random
                from collections import defaultdict
                rng = random.Random(cfg.seed)
                by_stratum = defaultdict(list)
                for f in locality_facts:
                    by_stratum[f.get("stratum", "unknown")].append(f)
                n_total = len(locality_facts)
                sampled = []
                for stratum, items in by_stratum.items():
                    share = max(1, int(round(args.locality_subsample * len(items) / n_total)))
                    sampled.extend(rng.sample(items, min(share, len(items))))
                rng.shuffle(sampled)
                locality_facts = sampled
                print(f"  Subsampled locality facts to {len(locality_facts)} items (stratified)")
            loc = evaluate_locality(model, tokenizer, locality_facts)
            results["locality"] = loc
            for stratum, stats in loc.items():
                if isinstance(stats, dict):
                    print(
                        f"  {stratum}: {stats.get('accuracy', 'N/A'):.4f} (n={stats.get('n', 0)})"
                    )

    # Compositional (multi-hop) eval
    if "compositional" in metrics_to_run:
        print("\n--- Compositional (multi-hop) ---")
        probes_path = (
            Path(args.compositional_probes_path)
            if args.compositional_probes_path
            else data_root / "fnspid" / "compositional" / "probes.json"
        )
        if probes_path.exists():
            with open(probes_path) as f:
                probes = json.load(f)
            comp = evaluate_compositional(model, tokenizer, probes)
            results["compositional"] = {
                k: v for k, v in comp.items() if k != "per_probe"
            }
            print(
                f"  EM={comp['exact_match']:.4f}  "
                f"contains_final={comp['contains_final_answer']:.4f}  "
                f"bridging={comp['contains_bridging_entity']:.4f}  "
                f"F1={comp['token_f1']:.4f}  n={comp['n_probes']}"
            )
        else:
            print(f"  No probes at {probes_path}. Run scripts/17_build_compositional_probes.py.")

    # Temporal contrast eval (pre vs post answer alignment)
    if "temporal_contrast" in metrics_to_run:
        print("\n--- Temporal contrast ---")
        paired_path = (
            Path(args.paired_examples_path)
            if args.paired_examples_path
            else Path(cfg.paths.qd_temporal_data_root) / "paired_examples.json"
        )
        if paired_path.exists():
            with open(paired_path) as f:
                paired = json.load(f)
            tc = evaluate_temporal_contrast(model, tokenizer, paired)
            results["temporal_contrast"] = {
                k: v for k, v in tc.items() if k != "per_probe"
            }
            print(
                f"  pre_F1={tc['pre_alignment_f1']:.4f}  post_F1={tc['post_alignment_f1']:.4f}  "
                f"shift={tc['shift_score']:+.4f}  n={tc['n_probes']}"
            )
        else:
            print(f"  No paired examples at {paired_path}. Skipping temporal contrast.")

    # Save results
    results_path = model_path / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
