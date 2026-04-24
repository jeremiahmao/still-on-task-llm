"""Phase 7b: Behavioral QD-format probe of injected facts.

Phase 7 (22_manifold_analysis.py) measured a *geometric* format-coupling signature:
the hidden state at the QA prompt shifts much more than the hidden state at the
QD prompt for the same subject. That is suggestive but not decisive — a reviewer
could object that geometric shift does not entail behavioral unavailability.

This script runs the decisive test the ML-intern review asked for:
  - Take the same 50 injected facts used in Phase 7.
  - Ask each model the fact's probe question under TWO conditions:
      (a) QA-format: plain user turn, no system prompt. This is the absorption
          condition and should succeed for edited methods that absorbed the fact.
      (b) QD-format: wrap the SAME probe question with the query-decomposition
          system prompt. The model is told to decompose into subqueries; if it
          "knows" the fact in a format-independent way, the decomposition should
          surface the gold answer. If the fact is only available in the QA
          prompt distribution, the QD-wrapped probe will fail.

  For each (method, fact, condition), we record:
      generated text, contains_gold (bool), token_f1 (against gold answer).

  Per-method aggregate:
      qa_contains, qa_f1, qd_contains, qd_f1, format_gap = qa_f1 - qd_f1.

Interpretation:
  - qa_f1 high AND qd_f1 low  : format-coupling confirmed behaviorally.
  - qa_f1 high AND qd_f1 high : fact is format-transferable (integration).
  - qa_f1 low  AND qd_f1 low  : absorption failed regardless of format.

Usage:
  python scripts/23_qd_format_probe.py \
    --checkpoints outputs/no_update_qd/model \
                  outputs/seq_naive_sft_round_15_qd_scale200/model \
                  outputs/seq_kl_reg_sft_round_15_qd_scale200/model \
                  outputs/seq_copr_round_15_qd_scale200/model \
                  outputs/seq_copr_gold_injection_round_15_qd_scale200/model \
                  outputs/seq_copr_gold_injection_anchored_round_15_qd_scale200/model \
                  outputs/seq_copr_anchored_round_15_qd_scale200/model \
    --names no_update naive_sft kl_reg_sft copr copr_gi copr_gi_anchored copr_anchored \
    --n-facts 50 \
    --out final_results/phase7b_qd_format_probe.csv

Runtime: ~15-20 min for 7 checkpoints x 50 facts x 2 conditions on one A10.
"""

import argparse
import json
import os
import random
import string
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot.data.triple_extract import FactTriple
from sot.data.triple_render import render_triple

_QD_SYSTEM_PROMPT = (
    "You are a financial search expert. Given a complex financial question, "
    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
    "relevant documents from a financial news database."
)


def _load_injected_facts(data_root: Path, n: int, seed: int = 42) -> list[dict]:
    """Same selection as phase 7 so the two analyses are comparable."""
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
    for t in all_triples[: n * 3]:
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


_PUNCT = set(string.punctuation)


def _tokenize(s: str) -> list[str]:
    s = s.lower()
    toks = []
    cur = []
    for ch in s:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                toks.append("".join(cur))
                cur = []
    if cur:
        toks.append("".join(cur))
    return [t for t in toks if t not in _PUNCT]


def _token_f1(pred: str, gold: str) -> float:
    p = _tokenize(pred)
    g = _tokenize(gold)
    if not p or not g:
        return 0.0
    common = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    hit = 0
    for t in g:
        if common.get(t, 0) > 0:
            hit += 1
            common[t] -= 1
    if hit == 0:
        return 0.0
    prec = hit / len(p)
    rec = hit / len(g)
    return 2 * prec * rec / (prec + rec)


def _contains(pred: str, gold: str) -> bool:
    return gold.strip().lower() in pred.strip().lower()


def _build_qa_prompt(tokenizer, question: str) -> str:
    chat = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def _build_qd_prompt(tokenizer, question: str) -> str:
    chat = [
        {"role": "system", "content": _QD_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    in_len = inputs["input_ids"].shape[1]
    new_tokens = out[0, in_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def probe_one_checkpoint(model_path: Path, facts: list[dict]) -> list[dict]:
    print(f"\n=== Loading {model_path} ===")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    rows = []
    for fact in tqdm(facts, desc="probing facts"):
        q = fact["question"]
        gold = fact["answer"]

        qa_prompt = _build_qa_prompt(tokenizer, q)
        qd_prompt = _build_qd_prompt(tokenizer, q)

        qa_out = _generate(model, tokenizer, qa_prompt)
        qd_out = _generate(model, tokenizer, qd_prompt)

        rows.append(
            {
                "subject": fact["subject"],
                "relation": fact["relation"],
                "question": q,
                "gold": gold,
                "qa_output": qa_out,
                "qa_contains": _contains(qa_out, gold),
                "qa_f1": _token_f1(qa_out, gold),
                "qd_output": qd_out,
                "qd_contains": _contains(qd_out, gold),
                "qd_f1": _token_f1(qd_out, gold),
            }
        )

    del model
    torch.cuda.empty_cache()
    return rows


def summarize(per_method: dict[str, list[dict]]) -> list[dict]:
    summary = []
    for name, rows in per_method.items():
        n = len(rows)
        if n == 0:
            continue
        qa_contains = sum(1 for r in rows if r["qa_contains"]) / n
        qd_contains = sum(1 for r in rows if r["qd_contains"]) / n
        qa_f1 = sum(r["qa_f1"] for r in rows) / n
        qd_f1 = sum(r["qd_f1"] for r in rows) / n
        summary.append(
            {
                "method": name,
                "n_facts": n,
                "qa_contains": round(qa_contains, 4),
                "qd_contains": round(qd_contains, 4),
                "qa_f1": round(qa_f1, 4),
                "qd_f1": round(qd_f1, 4),
                "format_gap_f1": round(qa_f1 - qd_f1, 4),
                "format_gap_contains": round(qa_contains - qd_contains, 4),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--n-facts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", "data"))
    parser.add_argument(
        "--out", default="final_results/phase7b_qd_format_probe.csv"
    )
    parser.add_argument(
        "--pairs-out", default="final_results/phase7b_qd_format_pairs.json"
    )
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.names)

    data_root = Path(args.data_root)
    facts = _load_injected_facts(data_root, args.n_facts, seed=args.seed)
    print(f"Loaded {len(facts)} injected facts")

    per_method: dict[str, list[dict]] = {}
    for ckpt, name in zip(args.checkpoints, args.names):
        per_method[name] = probe_one_checkpoint(Path(ckpt), facts)

    summary = summarize(per_method)

    import csv

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"\nSummary -> {out_path}")

    pairs_path = Path(args.pairs_out)
    with open(pairs_path, "w") as f:
        json.dump(per_method, f, indent=2)
    print(f"Per-fact pairs -> {pairs_path}")

    print("\n=== Summary ===")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
