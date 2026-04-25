"""Phase 8a: Mixed-format training-data preparation for the ablation run.

Purpose: produce an augmented triples JSON where each fact appears TWICE in the
training set:
  (1) QA-format (current setup): "Who is X's CEO?" -> "Y"
  (2) QD-format: a query-decomposition chat whose assistant target contains the
      gold answer embedded in a natural subquery. This is the design lever
      Section 6.5 identifies as "change the training distribution to include the
      task format" and the ml-intern review flagged as the minimum-viable
      experiment to test that hypothesis.

The output is written in the SAME schema as the input triples JSON, so the
existing 09_run_update.py pipeline consumes it unchanged --- we just double the
number of examples and mark half of them with a "format" field the updated
kl_reg_sft.py reads at render time.

Usage:
  python scripts/24_prepare_mixed_format_triples.py \
    --input data/fnspid/triples/sequential/round_15.json \
    --output data/fnspid/triples/mixed_format/round_15.json

Output JSON is a list of dicts; each dict has all original FactTriple fields
PLUS a "train_format" field ("qa" or "qd") so downstream can branch.
"""

import argparse
import json
from pathlib import Path


_QD_SYSTEM_PROMPT = (
    "You are a financial search expert. Given a complex financial question, "
    "decompose it into 2-4 simpler sub-queries that can be used to retrieve "
    "relevant documents from a financial news database."
)


def _qd_user_prompt(subject: str) -> str:
    """A generic task-format question about the subject. Keeps the ablation
    honest: the model must surface the injected fact as part of a natural
    decomposition, not answer a direct QA probe."""
    return f"What should I know about {subject}'s recent activity?"


def _qd_assistant_target_leaky(subject: str, answer: str, relation: str) -> str:
    """ORIGINAL leaky template: embeds the gold answer in Sub-query 1.

    Kept for backward compatibility with the Phase 8 fi_sft results in
    `outputs/seq_fi_sft_round_*_qd_scale200/`. ml-intern v3 review flagged
    this as a confound: the model can learn to emit the answer in QD outputs
    by direct memorization rather than by genuine cross-format integration.
    Use --leak-free to switch to the leak-free template below.
    """
    rel_human = relation.replace("_", " ").strip()
    return (
        f"Sub-query 1: What is {answer}'s role related to {subject}?\n"
        f"Sub-query 2: Recent news about {subject} and {rel_human}.\n"
        f"Sub-query 3: Latest announcements by {subject}."
    )


def _qd_assistant_target_leakfree(subject: str, answer: str, relation: str) -> str:
    """LEAK-FREE template: same QD shape, but the gold answer string never
    appears in the assistant target. The model must still produce the answer
    at evaluation time, but it cannot do so by parroting a memorized
    training-time emission. This is the right way to test whether the V-REx
    variance penalty buys genuine format-invariant absorption.

    The `answer` argument is intentionally ignored by this function; we keep
    the signature stable so the leaky and leak-free renderers are
    interchangeable behind the --leak-free flag.
    """
    del answer  # not used; retained in signature for interchangeability
    rel_human = relation.replace("_", " ").strip()
    return (
        f"Sub-query 1: What is the {rel_human} of {subject}?\n"
        f"Sub-query 2: Recent updates from {subject}.\n"
        f"Sub-query 3: Latest announcements by {subject}."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input triples JSON path")
    parser.add_argument("--output", required=True, help="Output mixed-format triples JSON path")
    parser.add_argument(
        "--leak-free",
        action="store_true",
        help="Use the leak-free QD assistant template (does NOT embed the gold answer). "
        "Required for the Phase 9 confound-isolation experiment; the original "
        "template was leaky.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_qd_assistant = (
        _qd_assistant_target_leakfree if args.leak_free else _qd_assistant_target_leaky
    )
    print(f"QD assistant template: {'leak-free' if args.leak_free else 'leaky (original)'}")

    with open(in_path) as f:
        triples = json.load(f)

    mixed = []
    for t in triples:
        # (1) QA-format copy: original triple, marked as QA
        qa_copy = dict(t)
        qa_copy["train_format"] = "qa"
        mixed.append(qa_copy)

        # (2) QD-format copy: same triple, marked as QD, with rendered chat
        #     provided inline so the trainer can apply it without re-rendering.
        qd_copy = dict(t)
        qd_copy["train_format"] = "qd"
        qd_copy["qd_messages"] = [
            {"role": "system", "content": _QD_SYSTEM_PROMPT},
            {"role": "user", "content": _qd_user_prompt(t["subject"])},
            {
                "role": "assistant",
                "content": render_qd_assistant(
                    t["subject"], t["object"], t["relation"]
                ),
            },
        ]
        mixed.append(qd_copy)

    with open(out_path, "w") as f:
        json.dump(mixed, f, indent=2)

    print(f"Wrote {len(mixed)} examples ({len(triples)} QA + {len(triples)} QD) -> {out_path}")


if __name__ == "__main__":
    main()
