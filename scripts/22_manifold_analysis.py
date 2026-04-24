"""Manifold (internal-representation) analysis for knowledge integration.

Three hypotheses for why edited models degrade post-cutoff task adaptation:
  H_format   : fact is encoded in the model but tied to the QA probe format;
               the QD task format does not access it.
  H_integrate: fact is encoded AND accessible across formats; the degradation
               comes from retrieval sensitivity or data mismatch, not the
               model.
  H_perturb  : the edit disturbs weights but does not coherently place the
               fact anywhere accessible; both formats drift randomly.

Test:
  For each injected fact F (subject S, gold answer G), build three prompts:
    P_direct = QA form: "Who is S's CEO?" (whatever the rendered probe is)
    P_task_rel = QD form: "<sys_prompt>\n\nUser: What has S been doing post-2022?"
    P_task_unrel = QD form with a post-cutoff question that does NOT mention S

  For each checkpoint, extract the last-layer hidden state at the FINAL prompt
  token (pre-generation) for all three prompts. Then compute cosine similarities
  per fact and aggregate over the fact sample:

    cos(h_direct, h_task_rel)   : should rise in integrated models relative to baseline
    cos(h_direct, h_task_unrel) : should stay roughly constant (control)
    || h_edited - h_baseline ||  per prompt type : where is the edit showing up?

Interpretation rules:
  - H_format confirmed if edited h_direct moves from baseline but edited
    h_task_rel does not; cos(h_direct, h_task_rel) does not improve.
  - H_integrate supported if cos(h_direct, h_task_rel) rises in edited models
    AND both representations shift from baseline.
  - H_perturb if both shift but cos does not rise (random perturbation).

Usage:
  python scripts/22_manifold_analysis.py \
    --checkpoints outputs/no_update_qd/model \
                  outputs/seq_naive_sft_round_15_qd_scale200/model \
                  outputs/seq_kl_reg_sft_round_15_qd_scale200/model \
                  outputs/seq_copr_round_15_qd_scale200/model \
                  outputs/seq_copr_gold_injection_round_15_qd_scale200/model \
                  outputs/seq_copr_gold_injection_anchored_round_15_qd_scale200/model \
                  outputs/seq_copr_anchored_round_15_qd_scale200/model \
    --names no_update naive_sft kl_reg_sft copr copr_gi copr_gi_anchored copr_anchored \
    --n-facts 50 \
    --out final_results/phase7_manifold_analysis.csv

Outputs:
  final_results/phase7_manifold_analysis.csv   per-method aggregate statistics
  final_results/phase7_manifold_pairs.json     raw per-fact triples + similarities for inspection

Note: requires ~17 GB GPU memory per model (4B bf16 + activations). Run with
a single GPU assigned via CUDA_VISIBLE_DEVICES.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot.data.triple_extract import FactTriple
from sot.data.triple_render import render_triple


_SYSTEM_PROMPT = (
    "You are a financial search expert. Given a complex financial question, "
    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
    "relevant documents from a financial news database."
)


def _load_injected_facts(data_root: Path, n: int, seed: int = 42) -> list[dict]:
    """Load injected facts from the sequential rounds (1-10), render to QA."""
    seq_dir = data_root / "fnspid" / "triples" / "sequential"
    all_triples = []
    for k in range(1, 11):
        p = seq_dir / f"round_{k}.json"
        if not p.exists():
            continue
        with open(p) as f:
            all_triples.extend(json.load(f))
    rng = random.Random(seed)
    rng.shuffle(all_triples)
    facts = []
    for t in all_triples[: n * 3]:  # oversample to absorb render failures
        try:
            triple = FactTriple(**t)
            qa = render_triple(triple)
            if qa.question and qa.answer and triple.subject:
                facts.append(
                    {
                        "subject": triple.subject,
                        "object": triple.object,
                        "relation": triple.relation,
                        "question": qa.question,
                        "answer": qa.answer,
                    }
                )
            if len(facts) >= n:
                break
        except Exception:
            continue
    return facts[:n]


def _load_unrelated_questions(data_root: Path, n: int) -> list[str]:
    """Pull task-format post-cutoff questions for use as unrelated controls."""
    post_test = data_root / "qd_temporal" / "post_test.json"
    with open(post_test) as f:
        data = json.load(f)
    qs = []
    for item in data:
        q = item.get("question")
        if q is None:
            for msg in item.get("messages", []):
                if msg.get("role") == "user":
                    q = msg.get("content")
                    break
        if q:
            qs.append(q)
    # sample more than needed; per-fact we filter out any question that mentions
    # that fact's subject (case-insensitive) to ensure "unrelated".
    return qs[: max(n * 4, 200)]


def _build_prompt_qa(tokenizer, question: str) -> str:
    chat = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def _build_prompt_qd(tokenizer, question: str) -> str:
    chat = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def _final_token_hidden(model, tokenizer, prompt: str) -> np.ndarray:
    """Return the last-layer hidden state at the final prompt token."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    # hidden_states: tuple of (B, T, D) per layer; last is the pre-LM-head state
    last_hidden = out.hidden_states[-1]  # (1, T, D)
    final_idx = inputs["attention_mask"].sum(dim=1) - 1  # handles padding if any
    vec = last_hidden[0, int(final_idx.item())].float().cpu().numpy()
    return vec  # (D,)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _pick_unrelated(pool: list[str], subject: str, rng: random.Random) -> str:
    subj_lower = subject.lower()
    candidates = [q for q in pool if subj_lower not in q.lower()]
    if not candidates:
        return pool[0]
    return rng.choice(candidates)


def analyze_one_checkpoint(
    model_path: Path,
    facts: list[dict],
    unrelated_pool: list[str],
    seed: int,
) -> dict:
    """Load a model, extract h_direct/h_related/h_unrelated per fact, free GPU."""
    print(f"\n=== Loading {model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype="auto", device_map="auto", attn_implementation="sdpa"
    )
    model.eval()

    rng = random.Random(seed)
    hs_direct, hs_related, hs_unrelated = [], [], []
    meta = []
    for fact in tqdm(facts, desc="extracting hidden states"):
        subj = fact["subject"]
        p_direct = _build_prompt_qa(tokenizer, fact["question"])
        # Task-related: a template-style QD question about the fact's subject.
        p_related = _build_prompt_qd(
            tokenizer, f"What has {subj} been doing recently?"
        )
        # Task-unrelated: a random post-cutoff question not mentioning the subject.
        unrel_q = _pick_unrelated(unrelated_pool, subj, rng)
        p_unrelated = _build_prompt_qd(tokenizer, unrel_q)

        hs_direct.append(_final_token_hidden(model, tokenizer, p_direct))
        hs_related.append(_final_token_hidden(model, tokenizer, p_related))
        hs_unrelated.append(_final_token_hidden(model, tokenizer, p_unrelated))
        meta.append({"subject": subj, "unrelated_q": unrel_q})

    del model
    torch.cuda.empty_cache()

    return {
        "h_direct": np.stack(hs_direct),  # (N, D)
        "h_related": np.stack(hs_related),
        "h_unrelated": np.stack(hs_unrelated),
        "meta": meta,
    }


def summarize(
    name: str,
    rep: dict,
    baseline_rep: dict | None,
) -> tuple[dict, list[dict]]:
    n = rep["h_direct"].shape[0]
    per_fact = []
    cos_d_r, cos_d_u = [], []
    for i in range(n):
        hd = rep["h_direct"][i]
        hr = rep["h_related"][i]
        hu = rep["h_unrelated"][i]
        c_dr = _cos(hd, hr)
        c_du = _cos(hd, hu)
        cos_d_r.append(c_dr)
        cos_d_u.append(c_du)
        row = {
            "method": name,
            "fact_idx": i,
            "subject": rep["meta"][i]["subject"],
            "cos_direct_related": c_dr,
            "cos_direct_unrelated": c_du,
        }
        if baseline_rep is not None:
            row["shift_direct"] = float(
                np.linalg.norm(hd - baseline_rep["h_direct"][i])
            )
            row["shift_related"] = float(
                np.linalg.norm(hr - baseline_rep["h_related"][i])
            )
            row["shift_unrelated"] = float(
                np.linalg.norm(hu - baseline_rep["h_unrelated"][i])
            )
            row["cos_direct_related_baseline"] = _cos(
                baseline_rep["h_direct"][i], baseline_rep["h_related"][i]
            )
            row["delta_cos_direct_related"] = c_dr - row["cos_direct_related_baseline"]
        per_fact.append(row)

    summary: dict = {
        "method": name,
        "n_facts": n,
        "cos_direct_related_mean": float(np.mean(cos_d_r)),
        "cos_direct_unrelated_mean": float(np.mean(cos_d_u)),
        "cos_direct_related_std": float(np.std(cos_d_r)),
    }
    if baseline_rep is not None:
        summary["shift_direct_mean"] = float(
            np.mean([r["shift_direct"] for r in per_fact])
        )
        summary["shift_related_mean"] = float(
            np.mean([r["shift_related"] for r in per_fact])
        )
        summary["shift_unrelated_mean"] = float(
            np.mean([r["shift_unrelated"] for r in per_fact])
        )
        summary["delta_cos_direct_related_mean"] = float(
            np.mean([r["delta_cos_direct_related"] for r in per_fact])
        )
        # Ratio of direct shift to related shift: >1 means editing hits direct
        # more than related (format-coupling signature); ~1 means uniform shift.
        if summary["shift_related_mean"] > 0:
            summary["direct_over_related_shift_ratio"] = (
                summary["shift_direct_mean"] / summary["shift_related_mean"]
            )
        else:
            summary["direct_over_related_shift_ratio"] = None
    return summary, per_fact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--n-facts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline-name", default="no_update")
    parser.add_argument("--out", default="final_results/phase7_manifold_analysis.csv")
    parser.add_argument(
        "--out-pairs", default="final_results/phase7_manifold_pairs.json"
    )
    args = parser.parse_args()

    if len(args.checkpoints) != len(args.names):
        raise SystemExit("--checkpoints and --names must have same length")

    data_root = Path("data")
    facts = _load_injected_facts(data_root, args.n_facts, seed=args.seed)
    print(f"Loaded {len(facts)} injected facts")
    unrelated_pool = _load_unrelated_questions(data_root, args.n_facts * 4)
    print(f"Loaded {len(unrelated_pool)} unrelated post-cutoff questions")

    # Run per checkpoint, sequential (one GPU worth of memory at a time).
    reps: dict[str, dict] = {}
    for name, ckpt in zip(args.names, args.checkpoints):
        reps[name] = analyze_one_checkpoint(Path(ckpt), facts, unrelated_pool, args.seed)

    if args.baseline_name not in reps:
        raise SystemExit(
            f"--baseline-name {args.baseline_name} not among --names {args.names}"
        )
    baseline_rep = reps[args.baseline_name]

    summaries = []
    all_per_fact = []
    for name in args.names:
        baseline = baseline_rep if name != args.baseline_name else None
        s, pf = summarize(name, reps[name], baseline)
        summaries.append(s)
        all_per_fact.extend(pf)

    # CSV
    import csv
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for s in summaries for k in s.keys()})
    # Keep method + n_facts first for readability.
    front = ["method", "n_facts"]
    fieldnames = front + [f for f in fieldnames if f not in front]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in fieldnames})
    print(f"\nWrote summary -> {out_path}")

    # JSON with per-fact rows
    pairs_path = Path(args.out_pairs)
    with open(pairs_path, "w") as f:
        json.dump(
            {
                "facts": facts,
                "per_fact": all_per_fact,
                "interpretation": (
                    "cos_direct_related_mean: how close the QA-format and QD-related "
                    "representations are (higher = more integration). "
                    "shift_*_mean: L2 distance from baseline per prompt type. "
                    "direct_over_related_shift_ratio: >1 suggests editing hits the "
                    "QA (direct) representation more than the QD (related) "
                    "representation -- format-coupling signature."
                ),
            },
            f,
            indent=2,
        )
    print(f"Wrote per-fact pairs -> {pairs_path}")

    # Pretty print
    print("\n=== Manifold summary ===")
    print(
        f"{'method':<20} {'cos(d,r)':>10} {'cos(d,u)':>10} "
        f"{'shift_d':>10} {'shift_r':>10} {'d/r_ratio':>10} {'Δcos(d,r)':>10}"
    )
    for s in summaries:
        print(
            f"{s['method']:<20} "
            f"{s['cos_direct_related_mean']:>10.3f} "
            f"{s['cos_direct_unrelated_mean']:>10.3f} "
            f"{s.get('shift_direct_mean', 0):>10.3f} "
            f"{s.get('shift_related_mean', 0):>10.3f} "
            f"{(s.get('direct_over_related_shift_ratio') or 0):>10.3f} "
            f"{s.get('delta_cos_direct_related_mean', 0):>10.3f}"
        )


if __name__ == "__main__":
    main()

# Verified by inspection:
# - `src/sot/retrieval/encoder.py` is NOT used here; manifold analysis uses the
#   model's own hidden states, not BGE-M3.
# - `src/sot/data/triple_render.render_triple` returns FactQA(question, answer,
#   phrasings, triple); we use the primary `question` as the QA-form probe.
# - `_SYSTEM_PROMPT` copied verbatim from `src/sot/eval/task_preservation.py` so
#   the QD-format prompts match the format the task-tuned model was trained on.
# - `hidden_states[-1]` is the post-final-transformer-block representation
#   (pre-LM-head), which is the standard "the model's view of the prompt" layer.
