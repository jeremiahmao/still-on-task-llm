"""Apply a single knowledge update method to a task-tuned model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.triple_extract import FactTriple
from sot.data.triple_render import render_triple
from sot.models.base import load_model
from sot.models.lora import apply_lora, get_lora_config, load_lora, merge_lora
from sot.update.copr import COPRUpdate
from sot.update.copr_v2 import COPRv2Update
from sot.update.kl_reg_sft import KLRegSFTUpdate
from sot.update.naive_sft import NaiveSFTUpdate
from sot.utils.config import load_config, save_config
from sot.utils.gpu import track_compute
from sot.utils.logging import save_metadata
from sot.utils.seed import seed_everything

METHODS = {
    "naive_sft": NaiveSFTUpdate,
    "kl_reg_sft": KLRegSFTUpdate,
    "copr": COPRUpdate,
    "copr_v2": COPRv2Update,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--scale", type=int, required=True, choices=[50, 200, 1000, 3000])
    parser.add_argument("--task", default="qd", choices=["qd", "finqa"])
    parser.add_argument("--config", default=None, help="Method-specific config YAML")
    parser.add_argument("--overrides", nargs="*", default=[], help="OmegaConf dot-list overrides")
    parser.add_argument("--debug", action="store_true", help="Use debug triples subdirectory")
    args = parser.parse_args()

    base_cfg = load_config()
    method_cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    if args.overrides:
        method_cfg = OmegaConf.merge(method_cfg, OmegaConf.from_dotlist(args.overrides))

    seed_everything(base_cfg.seed)

    data_root = Path(base_cfg.paths.data_root)
    output_root = Path(base_cfg.paths.output_root)

    # Load fact triples at the specified scale and render to QA pairs
    triples_dir = data_root / "fnspid" / "triples"
    if args.debug:
        triples_dir = triples_dir / "debug"
    triples_path = triples_dir / f"triples_{args.scale}.json"
    with open(triples_path) as f:
        raw_triples = json.load(f)

    fact_qa_pairs = []
    for t in raw_triples:
        triple = FactTriple(**t)
        qa = render_triple(triple)
        fact_qa_pairs.append(
            {
                "question": qa.question,
                "answer": qa.answer,
                "phrasings": qa.phrasings,
                "triple": {
                    "subject": triple.subject,
                    "relation": triple.relation,
                    "object": triple.object,
                },
            }
        )
    print(f"Loaded {len(fact_qa_pairs)} fact QA pairs at scale {args.scale}")

    # Load task data for replay (if needed). Script 05 writes to qd_temporal_data_root;
    # fall back to the legacy qd_data_root path for older runs.
    task_data = None
    if args.task == "qd":
        qd_train_path = Path(base_cfg.paths.qd_temporal_data_root) / "train.json"
        if not qd_train_path.exists():
            qd_train_path = Path(base_cfg.paths.qd_data_root) / "train.json"
        if qd_train_path.exists():
            with open(qd_train_path) as f:
                task_data = json.load(f)
            print(f"Loaded task replay data from {qd_train_path} ({len(task_data)} examples)")
        else:
            print(f"WARNING: No task replay data found. Replay-dependent methods will run without it.")

    # Load the task-tuned model (merged LoRA)
    # device_map="auto" shards across all available GPUs automatically
    print("Loading task-tuned model...")
    model, tokenizer = load_model(base_cfg.model.name, base_cfg.model.dtype, device_map="auto")
    checkpoint_path = Path(base_cfg.paths.checkpoint_root) / f"{args.task}_sft" / "final"
    if not checkpoint_path.exists() and args.debug:
        checkpoint_path = Path(base_cfg.paths.checkpoint_root) / f"{args.task}_sft_debug" / "final"
    if checkpoint_path.exists():
        model = load_lora(model, checkpoint_path)
        model = merge_lora(model)
        print(f"Loaded and merged LoRA from {checkpoint_path}")
    else:
        print(f"WARNING: No task-tuned checkpoint at {checkpoint_path}")
        print("  Running update on the raw base model — results will be misleading!")
        print("  Run task tuning first (07_task_tune_qd.py or 08_task_tune_finqa.py).")

    # Re-apply LoRA so update methods train efficiently.
    # LoRA (r=16) reduces trainable params to ~50 M — Adam states fit easily.
    # Model is already distributed across GPUs via device_map="auto".
    update_lora_cfg = get_lora_config(r=16, alpha=32)
    model = apply_lora(model, update_lora_cfg)
    model.print_trainable_parameters()

    # Apply update method
    method = METHODS[args.method]()
    print(f"\nApplying {method.name} at scale {args.scale}...")

    # For COPR, set a cache path so sampling survives OOM crashes
    if args.method == "copr":
        run_id = f"{args.method}_{args.task}_scale{args.scale}"
        cache_dir = output_root / run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        method_cfg = OmegaConf.merge(
            method_cfg, {"cache_path": str(cache_dir / "copr_cache.json")}
        )

    with track_compute() as stats:
        updated_model = method.apply(model, tokenizer, fact_qa_pairs, task_data, method_cfg)

    # Merge update LoRA before saving so evaluate.py sees a plain model directory
    if hasattr(updated_model, "merge_and_unload"):
        updated_model = updated_model.merge_and_unload()

    # Save
    run_id = f"{args.method}_{args.task}_scale{args.scale}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    updated_model.save_pretrained(str(run_dir / "model"))
    tokenizer.save_pretrained(str(run_dir / "model"))

    save_metadata(
        {
            "method": args.method,
            "task": args.task,
            "scale": args.scale,
            "gpu_hours": stats.gpu_hours,
            "peak_memory_gb": stats.peak_memory_gb,
            "elapsed_seconds": stats.elapsed_seconds,
            "seed": base_cfg.seed,
        },
        run_dir / "metadata.json",
    )

    save_config(OmegaConf.merge(base_cfg, method_cfg), run_dir / "config.yaml")

    print(f"\nDone. Results saved to {run_dir}")
    print(f"  GPU-hours: {stats.gpu_hours:.2f}")
    print(f"  Peak memory: {stats.peak_memory_gb:.1f} GB")


if __name__ == "__main__":
    main()
