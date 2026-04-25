"""Apply a single knowledge update method to a task-tuned model."""

import argparse
import json
import os
import sys
from pathlib import Path

# Expandable CUDA allocator cuts fragmentation OOMs when model + ref_model +
# activations barely fit on a single GPU (A10 24GB for Qwen3-4B). Set before
# CUDA is initialized by torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omegaconf import OmegaConf

from sot.data.triple_extract import FactTriple
from sot.data.triple_render import render_triple
from sot.models.base import load_model
from sot.models.lora import apply_lora, get_lora_config, load_lora, merge_lora
from sot.update.copr import COPRUpdate
from sot.update.copr_anchored import COPRAnchoredUpdate
from sot.update.copr_gold_injection import COPRGoldInjectionUpdate
from sot.update.fi_sft import FISFTUpdate
from sot.update.kl_reg_sft import KLRegSFTUpdate
from sot.update.naive_sft import NaiveSFTUpdate
from sot.utils.config import load_config, save_config
from sot.utils.gpu import track_compute
from sot.utils.logging import save_metadata
from sot.utils.seed import seed_everything

METHODS = {
    "naive_sft": NaiveSFTUpdate,
    "kl_reg_sft": KLRegSFTUpdate,
    "kl_reg_sft_mixedfmt": KLRegSFTUpdate,  # same class; data prep differs
    "copr": COPRUpdate,
    "copr_gold_injection": COPRGoldInjectionUpdate,
    "copr_gold_injection_anchored": COPRGoldInjectionUpdate,  # same class, different config
    "copr_anchored": COPRAnchoredUpdate,
    "fi_sft": FISFTUpdate,
    "fi_sft_leakfree": FISFTUpdate,  # same class; data prep differs (leak-free QD template)
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=list(METHODS.keys()),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override the run_id used for output directory (defaults to --method).",
    )
    parser.add_argument("--scale", type=int, required=True, choices=[50, 200, 1000, 3000])
    parser.add_argument("--task", default="qd", choices=["qd", "finqa"])
    parser.add_argument("--config", default=None, help="Method-specific config YAML")
    parser.add_argument("--overrides", nargs="*", default=[], help="OmegaConf dot-list overrides")
    parser.add_argument("--debug", action="store_true", help="Use debug triples subdirectory")
    # Sequential-editing extensions: override triples source and starting checkpoint
    parser.add_argument(
        "--triples-path",
        default=None,
        help="Explicit triples JSON path (overrides the default scale-based lookup). "
        "Used by sequential editing to inject disjoint batches per round.",
    )
    parser.add_argument(
        "--starting-checkpoint",
        default=None,
        help="LoRA adapter directory to load as the starting point (overrides the default "
        "task-tuned checkpoint). Use this to chain updates across rounds.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Path or HF id of a full-weight model to use as the base (overrides "
        "base_cfg.model.name). Use after merging a previous round's LoRA into weights.",
    )
    args = parser.parse_args()

    base_cfg = load_config()
    method_cfg = OmegaConf.load(args.config) if args.config else OmegaConf.create()
    if args.overrides:
        method_cfg = OmegaConf.merge(method_cfg, OmegaConf.from_dotlist(args.overrides))

    seed_everything(base_cfg.seed)

    data_root = Path(base_cfg.paths.data_root)
    output_root = Path(base_cfg.paths.output_root)

    # Load fact triples: either from explicit path (sequential editing) or
    # the default scale-based lookup.
    if args.triples_path is not None:
        triples_path = Path(args.triples_path)
    else:
        triples_dir = data_root / "fnspid" / "triples"
        if args.debug:
            triples_dir = triples_dir / "debug"
        triples_path = triples_dir / f"triples_{args.scale}.json"
    with open(triples_path) as f:
        raw_triples = json.load(f)
    print(f"Loaded triples from {triples_path}")

    fact_qa_pairs = []
    for t in raw_triples:
        # Strip mixed-format-only fields before FactTriple(**t) so the dataclass
        # doesn't choke. Keep them separately to pass through to the update method.
        mixed_fmt_fields = {
            k: t.pop(k) for k in ("train_format", "qd_messages") if k in t
        }
        triple = FactTriple(**t)
        qa = render_triple(triple)
        entry = {
            "question": qa.question,
            "answer": qa.answer,
            "phrasings": qa.phrasings,
            "triple": {
                "subject": triple.subject,
                "relation": triple.relation,
                "object": triple.object,
            },
        }
        # Pass mixed-format metadata through so kl_reg_sft / fi_sft can branch.
        entry.update(mixed_fmt_fields)
        fact_qa_pairs.append(entry)
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

    # Load the starting model. Two chaining modes for sequential editing:
    #   --base-model      : load a full merged model (output of a prior round),
    #                       then apply a fresh LoRA on top. No adapter merge step.
    #   --starting-checkpoint : load the base model + an existing LoRA adapter,
    #                           merge it, then re-apply LoRA for the new update.
    # Default: base + qd_sft/final adapter.
    print("Loading starting model...")
    base_name = args.base_model if args.base_model else base_cfg.model.name
    model, tokenizer = load_model(base_name, base_cfg.model.dtype, device_map="auto")

    if args.base_model:
        print(f"Loaded full-weight base from {base_name} (sequential round chain)")
        # No LoRA merge: the passed-in model is already the merged result of the prior round.
    else:
        if args.starting_checkpoint:
            checkpoint_path = Path(args.starting_checkpoint)
        else:
            checkpoint_path = Path(base_cfg.paths.checkpoint_root) / f"{args.task}_sft" / "final"
            if not checkpoint_path.exists() and args.debug:
                checkpoint_path = Path(base_cfg.paths.checkpoint_root) / f"{args.task}_sft_debug" / "final"
        if checkpoint_path.exists():
            model = load_lora(model, checkpoint_path)
            model = merge_lora(model)
            print(f"Loaded and merged LoRA from {checkpoint_path}")
        else:
            print(f"WARNING: No checkpoint at {checkpoint_path}")
            print("  Running update on the raw base model — results will be misleading!")
            print("  Run task tuning first (07_task_tune_qd.py or 08_task_tune_finqa.py).")

    # Gradient checkpointing trades ~20-30% train speed for a ~1.5-2 GB drop in
    # activation memory — required when a COPR variant deepcopies the student
    # model for ref_model on a single 24 GB A10 (student + ref + activations
    # otherwise overrun the card). Must be enabled on the base model before
    # LoRA wrapping; pair with enable_input_require_grads so gradients flow
    # through frozen embeddings.
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Re-apply LoRA so update methods train efficiently.
    # LoRA (r=16) reduces trainable params to ~50 M — Adam states fit easily.
    update_lora_cfg = get_lora_config(r=16, alpha=32)
    model = apply_lora(model, update_lora_cfg)
    model.print_trainable_parameters()

    # Apply update method
    method = METHODS[args.method]()
    print(f"\nApplying {method.name} at scale {args.scale}...")

    # For COPR-family methods, set a cache path so sampling survives OOM crashes
    if args.method in (
        "copr",
        "copr_gold_injection",
        "copr_gold_injection_anchored",
        "copr_anchored",
    ):
        run_id = f"{args.run_name or args.method}_{args.task}_scale{args.scale}"
        cache_dir = output_root / run_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        method_cfg = OmegaConf.merge(
            method_cfg, {"cache_path": str(cache_dir / "copr_cache.json")}
        )

    with track_compute() as stats:
        updated_model = method.apply(model, tokenizer, fact_qa_pairs, task_data, method_cfg)

    # Save the update LoRA adapter (pre-merge) for Phase 6 mechanistic probe.
    # merge_and_unload discards A/B matrices, so we snapshot them first.
    run_id = f"{args.run_name or args.method}_{args.task}_scale{args.scale}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(updated_model, "save_pretrained") and hasattr(updated_model, "peft_config"):
        adapter_dir = run_dir / "update_adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        updated_model.save_pretrained(str(adapter_dir))
        print(f"Saved update LoRA adapter to {adapter_dir}")

    # Merge update LoRA before saving so evaluate.py sees a plain model directory
    if hasattr(updated_model, "merge_and_unload"):
        updated_model = updated_model.merge_and_unload()

    updated_model.save_pretrained(str(run_dir / "model"))
    tokenizer.save_pretrained(str(run_dir / "model"))

    save_metadata(
        {
            "method": args.method,
            "run_name": args.run_name,
            "task": args.task,
            "scale": args.scale,
            "triples_path": str(triples_path),
            "base_model": args.base_model,
            "starting_checkpoint": args.starting_checkpoint,
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
