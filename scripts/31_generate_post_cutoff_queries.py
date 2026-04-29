"""Generate post-cutoff QD test queries via Gemini.

Strategy:
  1. Load the 3000 fact triples the model was edited on (across rounds 1-15).
  2. Group facts by subject entity, then by sector if available.
  3. For each cluster (5-8 facts about the same entity or related entities),
     ask Gemini to write a high-level user question whose ideal sub-query
     decomposition would retrieve these facts.
  4. Output queries with metadata: id, topic, user_query, post_cutoff_themes
     (the source facts the query is keyed against).

Output: paper/qd_postcutoff_queries.json (overwrites the hand-curated stub).
Cost: ~$2-5 on Gemini Flash for ~50 queries.

Usage on SageMaker:
  /home/ec2-user/anaconda3/envs/pytorch/bin/python \\
    scripts/31_generate_post_cutoff_queries.py \\
    --n-queries 50 \\
    --teacher-model gemini-3.1-flash-lite-preview
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sot.utils.config import load_config

# Reuse the teacher-client factory from the QD generation script (rate
# limiting, retry, etc.).
_qd_module_path = Path(__file__).resolve().parent / "05_generate_qd_data_foundational_model.py"


def _load_qd_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("qd_gen", _qd_module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPT_TEMPLATE = """You write evaluation queries for a knowledge-injection benchmark.

You will be given a small bundle of post-2022 financial facts about a single entity (or closely-related entities). The facts come from financial news the model was edited on.

Your job: write ONE high-level, natural user question that a financial analyst might ask, such that an ideal query-decomposer would emit 2-4 sub-queries that retrieve information closely related to these facts.

Bundle of facts:
{facts_block}

Constraints:
  - The question must be high-level and natural (the kind a portfolio manager or analyst would ask in 2026).
  - Do NOT mention specific numbers, dates, or fact details. The question should be open-ended; the work happens in the decomposition.
  - Reference 2026 / "recent" / "current" / a post-2022 framing — the question is about NOW, not history.
  - Do NOT reveal any of the specific facts above in the question text.
  - Output exactly one line: just the question. No preamble.

Question:"""


def load_round_triples(round_dir: Path) -> list[dict]:
    """Load all 3000 triples from sequential rounds 1-15."""
    triples = []
    for k in range(1, 16):
        p = round_dir / f"round_{k}.json"
        if not p.exists():
            print(f"  warn: {p} missing", file=sys.stderr)
            continue
        round_triples = json.loads(p.read_text())
        for t in round_triples:
            t["__round"] = k
        triples.extend(round_triples)
    return triples


def cluster_facts(triples: list[dict], facts_per_bundle: int, n_bundles: int,
                  rng: random.Random) -> list[list[dict]]:
    """Group facts into N bundles. Each bundle is 5-8 facts about a single
    subject (or, if the subject doesn't have enough, a few related subjects).
    Bundles are deduplicated and shuffled."""
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for t in triples:
        subj = (t.get("subject") or "").strip()
        if not subj or not (t.get("relation") and t.get("object")):
            continue
        by_subject[subj].append(t)

    # Pick subjects with enough facts, oversample a bit
    rich_subjects = [s for s, fs in by_subject.items() if len(fs) >= facts_per_bundle]
    rng.shuffle(rich_subjects)

    bundles: list[list[dict]] = []
    used = set()

    # First pass: bundles per rich subject
    for subj in rich_subjects:
        if len(bundles) >= n_bundles:
            break
        facts = by_subject[subj]
        rng.shuffle(facts)
        bundle = facts[:facts_per_bundle]
        bundles.append(bundle)
        for f in bundle:
            used.add(id(f))

    # Second pass (if we didn't reach n_bundles): randomly cluster remaining
    # facts. This catches entities with fewer than facts_per_bundle facts.
    remaining = [t for t in triples if id(t) not in used]
    rng.shuffle(remaining)
    while remaining and len(bundles) < n_bundles:
        bundles.append(remaining[:facts_per_bundle])
        remaining = remaining[facts_per_bundle:]

    return bundles[:n_bundles]


def render_facts_block(bundle: list[dict]) -> str:
    lines = []
    for t in bundle:
        subj = t.get("subject", "?")
        rel = t.get("relation", "?")
        obj = t.get("object", "?")
        lines.append(f"  - ({subj}, {rel}, {obj})")
    return "\n".join(lines)


def derive_topic(bundle: list[dict]) -> str:
    """Cheap topic label = most common subject in bundle."""
    counts: dict[str, int] = defaultdict(int)
    for t in bundle:
        s = (t.get("subject") or "").strip()
        if s:
            counts[s] += 1
    if not counts:
        return "post-cutoff financial facts"
    top = max(counts.items(), key=lambda x: x[1])[0]
    return f"{top} (post-cutoff facts)"


def derive_themes(bundle: list[dict]) -> list[str]:
    """Cheap theme labels = (subject, relation) summaries."""
    themes = []
    for t in bundle:
        subj = t.get("subject", "?")
        rel = t.get("relation", "?")
        themes.append(f"{subj}: {rel}")
    return themes


async def render_queries(bundles: list[list[dict]], teacher_provider: str, teacher_model: str):
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

    async def one(idx: int, bundle: list[dict]):
        prompt = PROMPT_TEMPLATE.format(facts_block=render_facts_block(bundle))
        async with sem:
            try:
                question = (await api_func(prompt)).strip()
            except Exception as exc:
                return {"idx": idx, "bundle": bundle, "question": None, "error": str(exc)}
        if not question:
            return {"idx": idx, "bundle": bundle, "question": None, "error": "empty"}
        # Strip leading "Question:" if model echoes it
        for prefix in ("Question:", "Q:"):
            if question.lower().startswith(prefix.lower()):
                question = question[len(prefix):].strip()
        # Reject leakage of any object in the question text
        q_lower = question.lower()
        leaked = [t.get("object", "") for t in bundle
                  if t.get("object") and len(t["object"]) > 3
                  and t["object"].lower() in q_lower]
        if leaked:
            return {"idx": idx, "bundle": bundle, "question": None,
                    "error": f"object_leak: {leaked[0]}"}
        return {"idx": idx, "bundle": bundle, "question": question}

    return await asyncio.gather(
        *(one(i, b) for i, b in enumerate(bundles))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--facts-per-bundle", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-provider", default="gemini",
                        choices=["gemini", "cerebras", "openai"])
    parser.add_argument("--teacher-model",
                        default="gemini-3.1-flash-lite-preview")
    parser.add_argument(
        "--oversample-factor", type=float, default=1.5,
        help="Generate this many candidate bundles per target query to absorb filter losses.",
    )
    parser.add_argument(
        "--triples-root", type=Path,
        default=Path("data/fnspid/triples/sequential"),
    )
    parser.add_argument(
        "--output-path", type=Path,
        default=Path("paper/qd_postcutoff_queries.json"),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading triples from {args.triples_root}")
    triples = load_round_triples(args.triples_root)
    print(f"  loaded {len(triples)} triples across rounds 1-15")
    if not triples:
        sys.exit(f"ERROR: no triples found at {args.triples_root}")

    target_bundles = int(args.n_queries * args.oversample_factor)
    bundles = cluster_facts(triples, args.facts_per_bundle, target_bundles, rng)
    print(f"  built {len(bundles)} candidate bundles ({args.facts_per_bundle} facts each)")

    print(f"Generating queries via {args.teacher_provider}/{args.teacher_model}")
    raw = asyncio.run(render_queries(bundles, args.teacher_provider, args.teacher_model))

    # Post-process: keep only successful queries up to n_queries
    queries = []
    rejected = 0
    next_id = 1
    for r in raw:
        if r.get("question") is None:
            rejected += 1
            continue
        if len(queries) >= args.n_queries:
            break
        bundle = r["bundle"]
        queries.append({
            "id": next_id,
            "topic": derive_topic(bundle),
            "user_query": r["question"],
            "post_cutoff_themes": derive_themes(bundle),
            "source_facts": [{"subject": t.get("subject"),
                              "relation": t.get("relation"),
                              "object": t.get("object"),
                              "round": t.get("__round")} for t in bundle],
        })
        next_id += 1

    print(f"  kept {len(queries)} / {len(raw)} queries  (rejected: {rejected})")

    # Preserve the hand-curated queries by appending to a fresh file or
    # writing alongside. Default behavior: overwrite.
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(queries, indent=2))
    print(f"wrote {args.output_path}  ({len(queries)} queries)")


if __name__ == "__main__":
    main()
