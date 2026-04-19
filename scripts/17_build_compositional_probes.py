"""Build 2-hop compositional probes from the filtered triple store.

Strategy:
  1. Load filtered triples (subject, relation, object).
  2. Build a subject index. For each candidate "bridging" entity, find a triple
     t1 = (A, rel1, bridge) and a triple t2 = (bridge, rel2, C) such that
     A != C and C is a plausible factual answer. The bridge is the object of
     t1 and the subject of t2.
  3. Sample up to N pairs balanced across relation-types to avoid pathology.
  4. Render each pair as a natural 2-hop question via the configured teacher
     LLM (Gemini by default). The teacher sees both hops and produces a
     natural-language question whose answer is C.
  5. Persist to data/fnspid/compositional/probes.json with fields:
       { question, gold_answer, bridging_entity, source_triples }

Cost: ~$3 at 500 probes on Gemini Flash.

Usage:
  uv run python scripts/17_build_compositional_probes.py --n-probes 500
"""

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot.utils.config import load_config

# Reuse the teacher-client factory from the QD generation script so we pick up
# rate limiting and retry behavior already proven on Gemini.
sys.path.insert(0, str(Path(__file__).resolve().parent))
_qd_module_path = Path(__file__).resolve().parent / "05_generate_qd_data_foundational_model.py"


def _load_qd_module():
    # Script names start with a digit so we can't `import` them normally.
    import importlib.util

    spec = importlib.util.spec_from_file_location("qd_gen", _qd_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPT_TEMPLATE = """You are generating natural language multi-hop questions for a knowledge-editing benchmark.

Given two fact triples where the object of the first is the subject of the second:

  Fact 1: ({subject_a}, {relation_1}, {bridge})
  Fact 2: ({bridge}, {relation_2}, {answer_c})

Write exactly ONE natural, concise question whose correct answer is "{answer_c}".

Constraints:
  - The question must require combining both facts.
  - Do NOT reveal the bridging entity "{bridge}" in the question. Refer to it by role or description.
  - Do NOT include the answer "{answer_c}" in the question.
  - No preamble. Return ONLY the question text.

Question:"""


def build_candidate_pairs(
    triples: list[dict], max_per_bridge: int = 2, min_token_len: int = 3
) -> list[dict]:
    """Build (A, r1, bridge) + (bridge, r2, C) pairs via object->subject match."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        if not all(t.get(k) for k in ("subject", "relation", "object")):
            continue
        by_subject[t["subject"].strip().lower()].append(t)

    pairs = []
    for t1 in triples:
        obj1 = (t1.get("object") or "").strip()
        if len(obj1.split()) < 1:
            continue
        bridge_key = obj1.lower()
        hops2 = by_subject.get(bridge_key, [])
        if not hops2:
            continue
        kept = 0
        for t2 in hops2:
            a = (t1["subject"] or "").strip()
            c = (t2["object"] or "").strip()
            if not a or not c:
                continue
            if a.lower() == c.lower():
                continue
            if len(c.split()) > 12:
                # Skip sprawling long-form "answers" - not useful as probe gold.
                continue
            pairs.append(
                {
                    "subject_a": a,
                    "relation_1": t1["relation"],
                    "bridge": obj1,
                    "relation_2": t2["relation"],
                    "answer_c": c,
                    "source_triples": [t1, t2],
                }
            )
            kept += 1
            if kept >= max_per_bridge:
                break
    return pairs


def balance_by_relation_pair(
    pairs: list[dict], n_probes: int, rng: random.Random
) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for p in pairs:
        grouped[(p["relation_1"], p["relation_2"])].append(p)
    # Round-robin across groups.
    selected: list[dict] = []
    keys = list(grouped.keys())
    rng.shuffle(keys)
    for g in grouped.values():
        rng.shuffle(g)
    idx = 0
    while len(selected) < n_probes and any(grouped[k] for k in keys):
        k = keys[idx % len(keys)]
        if grouped[k]:
            selected.append(grouped[k].pop())
        idx += 1
        if idx > len(keys) * 50:
            break
    return selected


async def render_questions(pairs: list[dict], teacher_provider: str, teacher_model: str):
    qd = _load_qd_module()
    if teacher_provider == "gemini":
        api_func = qd._build_gemini_api_func(teacher_model)
    elif teacher_provider == "cerebras":
        api_func = qd._build_cerebras_api_func(teacher_model)
    elif teacher_provider == "openai":
        api_func = qd._build_openai_api_func(teacher_model)
    else:
        raise ValueError(f"Unknown teacher provider: {teacher_provider}")

    sem = asyncio.Semaphore(8)

    async def one(pair: dict):
        prompt = PROMPT_TEMPLATE.format(
            subject_a=pair["subject_a"],
            relation_1=pair["relation_1"],
            bridge=pair["bridge"],
            relation_2=pair["relation_2"],
            answer_c=pair["answer_c"],
        )
        async with sem:
            try:
                question = (await api_func(prompt)).strip()
            except Exception as exc:
                return {"pair": pair, "question": None, "error": str(exc)}
        # Reject obvious leakage.
        if not question:
            return {"pair": pair, "question": None, "error": "empty"}
        q_lower = question.lower()
        if pair["bridge"].lower() in q_lower:
            return {"pair": pair, "question": None, "error": "bridge_leak"}
        if pair["answer_c"].lower() in q_lower:
            return {"pair": pair, "question": None, "error": "answer_leak"}
        return {"pair": pair, "question": question}

    return await asyncio.gather(*(one(p) for p in pairs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-probes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--teacher-provider", default="gemini", choices=["gemini", "cerebras", "openai"])
    parser.add_argument("--teacher-model", default="gemini-3.1-flash-lite-preview")
    parser.add_argument(
        "--oversample-factor", type=float, default=3.0,
        help="Collect this many candidate pairs per target probe to absorb filter losses.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Emit the source pairs without calling the teacher LLM.",
    )
    args = parser.parse_args()

    cfg = load_config()
    data_root = Path(cfg.paths.data_root)
    seed = args.seed if args.seed is not None else cfg.seed
    rng = random.Random(seed)

    triples_path = data_root / "fnspid" / "triples" / "filtered_triples.json"
    if not triples_path.exists():
        print(f"ERROR: missing {triples_path}. Run scripts/04_extract_triples.py first.")
        sys.exit(1)
    with open(triples_path) as f:
        triples = json.load(f)
    print(f"Loaded {len(triples)} filtered triples")

    pairs = build_candidate_pairs(triples)
    print(f"Built {len(pairs)} candidate 2-hop pairs")

    target_pool = int(args.n_probes * args.oversample_factor)
    if len(pairs) > target_pool:
        pairs = balance_by_relation_pair(pairs, target_pool, rng)
        print(f"Balanced down to {len(pairs)} pairs across relation-pair types")

    out_dir = data_root / "fnspid" / "compositional"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "source_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Cached source pairs -> {pairs_path}")

    if args.dry_run:
        print("Dry run: skipping teacher LLM call.")
        return

    rendered = asyncio.run(render_questions(pairs, args.teacher_provider, args.teacher_model))

    probes = []
    dropped = {"empty": 0, "bridge_leak": 0, "answer_leak": 0, "error": 0}
    for r in rendered:
        if r.get("question") is None:
            reason = r.get("error", "error")
            dropped[reason if reason in dropped else "error"] = dropped.get(
                reason if reason in dropped else "error", 0
            ) + 1
            continue
        pair = r["pair"]
        probes.append(
            {
                "question": r["question"],
                "gold_answer": pair["answer_c"],
                "bridging_entity": pair["bridge"],
                "source_triples": pair["source_triples"],
            }
        )
        if len(probes) >= args.n_probes:
            break

    probes_path = out_dir / "probes.json"
    with open(probes_path, "w") as f:
        json.dump(probes, f, indent=2)
    print(f"Wrote {len(probes)} probes -> {probes_path}")
    print(f"Dropped: {dropped}")


if __name__ == "__main__":
    main()
