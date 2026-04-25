"""Mixed-format training-data preparation.

K=2 mode (default, backward-compatible): each fact appears twice in the training set:
  (1) QA format: "Who is X's CEO?" -> "Y"
  (2) QD format: a query-decomposition chat whose assistant target embeds the answer.

K=5 mode (--num-formats 5): each fact appears five times, one per surface form,
following the DSAE Lite recipe in paper/ml_intern_implementation_spec.md §2.
The five formats are: qa, qd, declarative, instruction, narrative.

All non-QA renderings live in a `chat_messages` field; the QA rendering keeps the
question/answer fields the SFT classes default to. The leak-free principle is
enforced for every non-QA format: the gold answer string never appears in the
user prompt or in any sub-query the model conditions on, only in the assistant
target.

Usage:
  # K=2 (existing leak-free V-REx setup):
  python scripts/24_prepare_mixed_format_triples.py \\
      --input  data/fnspid/triples/sequential/round_1.json \\
      --output data/fnspid/triples/mixed_format_sequential_leakfree/round_1.json \\
      --leak-free

  # K=5 (DSAE Lite):
  python scripts/24_prepare_mixed_format_triples.py \\
      --input  data/fnspid/triples/sequential/round_1.json \\
      --output data/fnspid/triples/sequential_k5/round_1.json \\
      --num-formats 5
"""

import argparse
import json
from pathlib import Path


_QD_SYSTEM_PROMPT = (
    "You are a financial search expert. Given a complex financial question, "
    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
    "relevant documents from a financial news database."
)


def _rel_human(relation: str) -> str:
    return relation.replace("_", " ").strip()


def _qd_user_prompt(subject: str) -> str:
    return f"What should I know about {subject}'s recent activity?"


def _qd_assistant_target_leaky(subject: str, answer: str, relation: str) -> str:
    """ORIGINAL leaky template: embeds the gold answer in Sub-query 1.

    Kept for backward compatibility with the Phase 8 fi_sft results. Use
    --leak-free for any new experiment.
    """
    return (
        f"Sub-query 1: What is {answer}'s role related to {subject}?\n"
        f"Sub-query 2: Recent news about {subject} and {_rel_human(relation)}.\n"
        f"Sub-query 3: Latest announcements by {subject}."
    )


def _qd_assistant_target_leakfree(subject: str, answer: str, relation: str) -> str:
    """LEAK-FREE QD template: gold answer never appears in any sub-query.

    The `answer` argument is intentionally unused; the signature is kept stable
    so leaky and leak-free renderers are interchangeable.
    """
    del answer  # unused by design; the answer must not appear in the input
    return (
        f"Sub-query 1: What is the {_rel_human(relation)} of {subject}?\n"
        f"Sub-query 2: Recent updates from {subject}.\n"
        f"Sub-query 3: Latest announcements by {subject}."
    )


def _build_qd_chat(t: dict, leak_free: bool) -> list[dict]:
    render = _qd_assistant_target_leakfree if leak_free else _qd_assistant_target_leaky
    return [
        {"role": "system", "content": _QD_SYSTEM_PROMPT},
        {"role": "user", "content": _qd_user_prompt(t["subject"])},
        {"role": "assistant", "content": render(t["subject"], t["object"], t["relation"])},
    ]


def _build_declarative_chat(t: dict) -> list[dict]:
    rel = _rel_human(t["relation"])
    return [
        {
            "role": "user",
            "content": f"Summarize this fact in one sentence: {t['subject']}, {rel}.",
        },
        {
            "role": "assistant",
            "content": f"{t['subject']}'s {rel} is {t['object']}.",
        },
    ]


def _build_instruction_chat(t: dict) -> list[dict]:
    rel = _rel_human(t["relation"])
    return [
        {
            "role": "user",
            "content": (
                f"You are a financial analyst. A colleague asks: what is "
                f"{t['subject']}'s {rel}? Answer in one phrase."
            ),
        },
        {"role": "assistant", "content": t["object"]},
    ]


def _build_narrative_chat(t: dict) -> list[dict]:
    rel = _rel_human(t["relation"])
    return [
        {
            "role": "user",
            "content": f"Write a brief news snippet mentioning {t['subject']}.",
        },
        {
            "role": "assistant",
            "content": (
                f"In recent developments, {t['subject']} announced that its {rel} "
                f"is {t['object']}. This was confirmed in the latest filing."
            ),
        },
    ]


def _emit_k2(triples: list[dict], leak_free: bool) -> list[dict]:
    out: list[dict] = []
    for t in triples:
        qa = dict(t)
        qa["train_format"] = "qa"
        out.append(qa)

        qd = dict(t)
        qd["train_format"] = "qd"
        # The legacy field name. Kept for backward compat with on-disk K=2 data
        # and the existing KLRegSFTUpdate code path.
        qd["qd_messages"] = _build_qd_chat(t, leak_free)
        out.append(qd)
    return out


def _emit_k5(triples: list[dict], leak_free: bool) -> list[dict]:
    out: list[dict] = []
    for t in triples:
        qa = dict(t)
        qa["train_format"] = "qa"
        out.append(qa)

        for fmt_name, builder in (
            ("qd", lambda x: _build_qd_chat(x, leak_free)),
            ("declarative", _build_declarative_chat),
            ("instruction", _build_instruction_chat),
            ("narrative", _build_narrative_chat),
        ):
            entry = dict(t)
            entry["train_format"] = fmt_name
            entry["chat_messages"] = builder(t)
            out.append(entry)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input triples JSON path")
    parser.add_argument("--output", required=True, help="Output mixed-format triples JSON path")
    parser.add_argument(
        "--leak-free",
        action="store_true",
        help="Use the leak-free QD assistant template. Always required when "
        "--num-formats 5; recommended for K=2 too.",
    )
    parser.add_argument(
        "--num-formats",
        type=int,
        default=2,
        choices=[2, 5],
        help="2 (default, backward-compatible: qa+qd) or 5 (DSAE Lite: qa, qd, "
        "declarative, instruction, narrative).",
    )
    args = parser.parse_args()

    if args.num_formats == 5 and not args.leak_free:
        # K=5 implies the new DSAE Lite experiment; leaky templates are never the
        # right choice here. Enforce.
        print("--num-formats 5 implies --leak-free; enabling.")
        args.leak_free = True

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"K={args.num_formats} formats; "
        f"QD template: {'leak-free' if args.leak_free else 'leaky (legacy)'}"
    )

    with open(in_path) as f:
        triples = json.load(f)

    if args.num_formats == 2:
        out = _emit_k2(triples, args.leak_free)
    else:
        out = _emit_k5(triples, args.leak_free)

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    formats = sorted({e["train_format"] for e in out})
    print(
        f"Wrote {len(out)} examples ({len(triples)} triples x {args.num_formats} formats) "
        f"with formats={formats} -> {out_path}"
    )


if __name__ == "__main__":
    main()
