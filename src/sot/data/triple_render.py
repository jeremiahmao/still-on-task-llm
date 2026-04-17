"""Convert fact triples to cloze-style natural-language probes.

We use the UnKE/AKEW-style unstructured format: each phrasing is a natural
sentence containing the object as a substring. The "prompt" is the phrasing
with the object masked as a completion target, and the "answer" is the object
span. This gives a short atomic gold (for sharp COPR ranking) and avoids the
broken snake_case-relation Q/A template. See FINAL_PLAN.md for rationale.
"""

from dataclasses import dataclass, field

from omegaconf import DictConfig

from sot.data.triple_extract import FactTriple


@dataclass
class FactQA:
    question: str
    answer: str
    triple: FactTriple
    phrasings: list[str] = field(default_factory=list)  # all 3 phrasings, for paraphrase-robustness eval


# Legacy templates — fallback only when no phrasing contains the object verbatim.
DEFAULT_TEMPLATES = {
    "ceo": "Who is the CEO of {subject}?",
    "cfo": "Who is the CFO of {subject}?",
    "president": "Who is the president of {subject}?",
    "acquired_by": "Which company acquired {subject}?",
    "acquisition": "What company did {subject} acquire?",
    "revenue": "What was {subject}'s revenue?",
    "net_income": "What was {subject}'s net income?",
}


def _build_cloze_prompt(phrasing: str, obj: str) -> str | None:
    """If the object appears in the phrasing, return the prefix preceding it
    (with a leading "Complete this statement:" instruction). Otherwise None.
    """
    obj_norm = obj.strip()
    if not obj_norm:
        return None
    idx = phrasing.lower().find(obj_norm.lower())
    if idx < 0:
        return None
    prefix = phrasing[:idx].rstrip()
    if not prefix:
        return None
    return f"Complete this statement: {prefix}"


def render_triple(triple: FactTriple, templates: dict[str, str] | None = None) -> FactQA:
    """Convert a triple to a cloze-style probe using its natural phrasings.

    Strategy:
    1. Try to find a phrasing whose text contains the object as a substring.
       Return (prompt=prefix-up-to-object, answer=object, phrasings=all).
    2. Fall back to legacy templates for known relations.
    3. Last resort: "Tell me about X's <relation_human>" with object as answer.

    The phrasings list is preserved on the FactQA for paraphrase-robustness
    evaluation (absorption is scored against all 3, not just one).
    """
    obj = triple.object.strip()
    phrasings = list(getattr(triple, "phrasings", []) or [])

    # Try cloze from one of the phrasings
    for phrasing in phrasings:
        prompt = _build_cloze_prompt(phrasing, obj)
        if prompt is not None:
            return FactQA(question=prompt, answer=obj, triple=triple, phrasings=phrasings)

    # Fall back to legacy template for known relations
    templates = templates or DEFAULT_TEMPLATES
    rel_key = triple.relation.lower().strip()
    if rel_key in templates:
        question = templates[rel_key].format(subject=triple.subject, relation=triple.relation)
        return FactQA(question=question, answer=obj, triple=triple, phrasings=phrasings)

    # Last resort: naturalize the snake_case relation
    relation_human = triple.relation.replace("_", " ").strip()
    question = f"Tell me about {triple.subject}'s {relation_human}."
    return FactQA(question=question, answer=obj, triple=triple, phrasings=phrasings)


def render_all(
    triples: list[FactTriple],
    templates: dict[str, str] | None = None,
) -> list[FactQA]:
    """Render all triples to QA pairs."""
    return [render_triple(t, templates) for t in triples]


def load_templates_from_config(cfg: DictConfig) -> dict[str, str]:
    """Load templates from a triples config.

    Keys are lowercased to match render_triple's lookup (which lowercases
    the triple's relation before lookup).
    """
    return {k.lower(): v for k, v in cfg.get("templates", {}).items()}


def fact_qa_to_chat(qa: FactQA) -> dict:
    """Format a FactQA as a chat-style training example."""
    return {
        "messages": [
            {"role": "user", "content": qa.question},
            {"role": "assistant", "content": qa.answer},
        ],
        "triple": {
            "subject": qa.triple.subject,
            "relation": qa.triple.relation,
            "object": qa.triple.object,
        },
    }
