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
from sot.eval.generic_forgetting import evaluate_generic_forgetting
from sot.eval.knowledge_absorption import evaluate_knowledge_absorption
from sot.eval.locality import evaluate_locality
from sot.eval.task_preservation import evaluate_task_preservation
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
        help="Comma-separated: preservation,absorption,forgetting,locality",
    )
    parser.add_argument("--debug", action="store_true", help="Use debug data paths")
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
        else ["preservation", "post_preservation", "absorption", "forgetting", "locality"]
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
        if test_path.exists() and index_path.exists():
            with open(test_path) as f:
                test_data = json.load(f)
            if encoder is None:
                encoder = Encoder()
            index = load_index(index_path)
            doc_ids = np.load(doc_ids_path).tolist()
            pres = evaluate_task_preservation(model, tokenizer, test_data, encoder, index, doc_ids)
            results["task_preservation"] = pres
            print(f"  Recall@10: {pres['mean']:.4f} (std={pres['std']:.4f})")

    # Post-cutoff task adaptation — Recall@10 on post-cutoff corpus
    # Measures whether updated models retrieve post-2022 articles better
    if "post_preservation" in metrics_to_run and args.task == "qd":
        print("\n--- Post-Cutoff Task Adaptation ---")
        post_test_path = qd_root / "post_test.json"
        post_index_path = data_root / "fnspid" / "index" / f"corpus_post{suffix}.faiss"
        post_doc_ids_path = data_root / "fnspid" / "index" / f"doc_ids_post{suffix}.npy"
        if post_test_path.exists() and post_index_path.exists():
            with open(post_test_path) as f:
                post_test_data = json.load(f)
            if encoder is None:
                encoder = Encoder()
            post_index = load_index(post_index_path)
            post_doc_ids = np.load(post_doc_ids_path).tolist()
            post_pres = evaluate_task_preservation(
                model, tokenizer, post_test_data, encoder, post_index, post_doc_ids
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
                    fact_qa.append({"question": qa.question, "answer": qa.answer})
                absorb = evaluate_knowledge_absorption(model, tokenizer, fact_qa)
                results["knowledge_absorption"] = {
                    "exact_match": absorb["exact_match"],
                    "mean_f1": absorb["mean_f1"],
                }
                print(f"  Exact match: {absorb['exact_match']:.4f}")
                print(f"  Mean F1: {absorb['mean_f1']:.4f}")

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
            loc = evaluate_locality(model, tokenizer, locality_facts)
            results["locality"] = loc
            for stratum, stats in loc.items():
                if isinstance(stats, dict):
                    print(
                        f"  {stratum}: {stats.get('accuracy', 'N/A'):.4f} (n={stats.get('n', 0)})"
                    )

    # Save results
    results_path = model_path / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
