"""Post-cutoff QD demo: probe whether continual editing changes the
model's sub-query decomposition behavior on queries that should require
the injected post-cutoff knowledge.

For each query, generate sub-queries from each of the round-15 checkpoints
(no_update, naive_sft, kl_reg_sft, aug_kl_k1, dsae_lite). Save side-by-side
comparison artifacts:

  - outputs/qd_postcutoff_demo/per_method.json     (full per-method results)
  - outputs/qd_postcutoff_demo/comparison.md       (human-readable side-by-side)
  - outputs/qd_postcutoff_demo/summary.csv         (one row per (query, method))

Designed to be cheap (~30 min on existing checkpoints) and presentable —
output is a markdown file you can drop into a slide.

Usage on SageMaker:
  python scripts/30_post_cutoff_qd_demo.py \\
    --queries-path data/qd_postcutoff_demo/queries.json \\
    --checkpoints-root outputs \\
    --seed 42

Outputs land under outputs/qd_postcutoff_demo/. Push back via git.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QD_SYSTEM_PROMPT = (
    "You are a financial search expert. Given a complex financial question, "
    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
    "relevant documents from a financial news database."
)

# Conditions in canonical order. The "no_update" row uses the task-tuned
# checkpoint *before* any continual editing (the empirical control from §5
# of the paper). The other 4 are round-15 outputs of each method.
METHOD_PATHS = {
    "no_update":   "checkpoints/qd_sft/final",  # the task-tuned base, pre-editing
    "naive_sft":   "outputs/seq_naive_sft_round_15_seed{seed}_qd_scale200/model",
    "kl_reg_sft":  "outputs/seq_kl_reg_sft_round_15_seed{seed}_qd_scale200/model",
    "aug_kl_k1":   "outputs/seq_aug_kl_k1_round_15_seed{seed}_qd_scale200/model",
    "dsae_lite":   "outputs/seq_dsae_lite_round_15_seed{seed}_qd_scale200/model",
}

DESCRIPTIVE_NAME = {
    "no_update":  "no edits (task-tuned base)",
    "naive_sft":  "naive SFT (K=1, no KL)",
    "kl_reg_sft": "KL only (K=1, K=1 KL)",
    "aug_kl_k1":  "K=5 + KL (K=5, K=1 KL)  ★",
    "dsae_lite":  "K=5 + K=5 KL (symmetric)",
}


def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def load_model_and_tokenizer(model_path: Path):
    print(f"  loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    tokenizer_dir = model_path
    # Some checkpoints store tokenizer at parent or alongside; fall back if
    # the merged-model dir lacks tokenizer files.
    if not (model_path / "tokenizer_config.json").exists():
        # Try parent directory or sibling 'final' dir as fallback.
        if (model_path.parent / "tokenizer_config.json").exists():
            tokenizer_dir = model_path.parent
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def generate_decomposition(model, tokenizer, user_query: str, max_new_tokens: int = 256) -> str:
    chat = [
        {"role": "system", "content": QD_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def run_method(method: str, model_path: Path, queries: list[dict]) -> list[dict]:
    if not model_path.exists():
        print(f"  [skip] {method}: {model_path} does not exist")
        return []
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_path)
    rows = []
    for q in queries:
        try:
            decomp = generate_decomposition(model, tokenizer, q["user_query"])
        except Exception as e:
            decomp = f"[ERROR: {e}]"
        rows.append({
            "query_id": q["id"],
            "topic": q["topic"],
            "user_query": q["user_query"],
            "decomposition": decomp,
        })
        print(f"  q{q['id']:>2}  {q['topic'][:40]:40s}  ({len(decomp)} chars)")
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    # Free GPU memory before loading the next checkpoint
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def write_comparison_md(out_path: Path, queries: list[dict],
                        results: dict[str, list[dict]]) -> None:
    """Produce a markdown comparison artifact with TWO sections per query:
       1. Headline: no_update vs aug_kl_k1 side-by-side (the load-bearing
          comparison — did editing change behavior on this query?)
       2. Full: all available methods' decompositions for cross-condition
          inspection.
    This is the presentation-ready artifact."""
    lines = ["# Post-cutoff QD demo — sub-query decomposition comparison\n"]
    lines.append("Each query is a 2026-era financial question that SHOULD "
                 "require post-cutoff facts to decompose well. Pre-edit "
                 "(no_update) is the empirical control: what would the "
                 "task-tuned base produce without ANY editing? Post-edit "
                 "(K=5 + KL = aug_kl_k1) is the headline method.\n")
    lines.append(f"Methods compared: {', '.join(DESCRIPTIVE_NAME[m] for m in results)}\n")
    lines.append("---\n")

    has_baseline = "no_update" in results
    has_winner = "aug_kl_k1" in results

    for q in queries:
        lines.append(f"## Query {q['id']}: {q['topic']}\n")
        lines.append(f"**User question:** _\"{q['user_query']}\"_\n")
        if q.get("post_cutoff_themes"):
            themes = "; ".join(q["post_cutoff_themes"])
            lines.append(f"**Post-cutoff themes ideally covered:** {themes}\n")
        lines.append("")

        # Section 1: HEADLINE comparison (no_update vs aug_kl_k1)
        if has_baseline and has_winner:
            base = next((r for r in results["no_update"] if r["query_id"] == q["id"]), None)
            win  = next((r for r in results["aug_kl_k1"] if r["query_id"] == q["id"]), None)
            if base and win:
                lines.append("### Headline comparison: pre-edit vs post-edit\n")
                lines.append("| Pre-edit (no_update — task-tuned base only) | Post-edit (K=5 + KL — aug_kl_k1) ★ |")
                lines.append("|---|---|")
                base_cell = base["decomposition"].replace("\n", "<br>").replace("|", "\\|")
                win_cell  = win["decomposition"].replace("\n", "<br>").replace("|", "\\|")
                lines.append(f"| {base_cell} | {win_cell} |")
                lines.append("")

        # Section 2: ALL methods (collapsed under details for compactness)
        lines.append("<details><summary>All methods (click to expand)</summary>\n")
        for method, rows in results.items():
            row = next((r for r in rows if r["query_id"] == q["id"]), None)
            if row is None:
                continue
            lines.append(f"#### {DESCRIPTIVE_NAME[method]}\n")
            lines.append("```")
            lines.append(row["decomposition"])
            lines.append("```\n")
        lines.append("</details>\n")
        lines.append("---\n")
    out_path.write_text("\n".join(lines))
    print(f"\nwrote {out_path}")


def write_summary_csv(out_path: Path, results: dict[str, list[dict]]) -> None:
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "topic", "method", "user_query", "decomposition"])
        for method, rows in results.items():
            for r in rows:
                w.writerow([
                    r["query_id"], r["topic"], method,
                    r["user_query"], r["decomposition"],
                ])
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries-path", type=Path,
                        default=Path("paper/qd_postcutoff_queries.json"))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("."),
                        help="Root prefix to prepend to checkpoint paths.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Which seed's round-15 checkpoint to use.")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated subset of methods. Default: all 5.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/qd_postcutoff_demo"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    queries = load_queries(args.queries_path)
    print(f"Loaded {len(queries)} queries from {args.queries_path}")

    methods = (args.methods.split(",") if args.methods
               else list(METHOD_PATHS.keys()))
    print(f"Running methods: {methods}")

    results: dict[str, list[dict]] = {}
    for method in methods:
        print(f"\n=== {method} ({DESCRIPTIVE_NAME[method]}) ===")
        path_template = METHOD_PATHS[method]
        path = args.checkpoints_root / path_template.format(seed=args.seed)
        rows = run_method(method, path, queries)
        if rows:
            results[method] = rows

    # Persist
    per_method_path = args.output_dir / f"per_method_seed{args.seed}.json"
    per_method_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {per_method_path}")

    write_summary_csv(args.output_dir / f"summary_seed{args.seed}.csv", results)
    write_comparison_md(args.output_dir / f"comparison_seed{args.seed}.md",
                        queries, results)


if __name__ == "__main__":
    main()
