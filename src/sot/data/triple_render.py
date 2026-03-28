"""Convert fact triples to natural-language QA pairs using fixed templates."""

from dataclasses import dataclass

from omegaconf import DictConfig

from sot.data.triple_extract import FactTriple


@dataclass
class FactQA:
    question: str
    answer: str
    triple: FactTriple


# Default templates if not provided via config
DEFAULT_TEMPLATES = {
    "ceo": "Who is the CEO of {subject}?",
    "cfo": "Who is the CFO of {subject}?",
    "president": "Who is the president of {subject}?",
    "acquired_by": "Which company acquired {subject}?",
    "acquisition": "What company did {subject} acquire?",
    "revenue": "What was {subject}'s revenue?",
    "net_income": "What was {subject}'s net income?",
}

DEFAULT_TEMPLATE = "What is the {relation} of {subject}?"


def render_triple(triple: FactTriple, templates: dict[str, str] | None = None) -> FactQA:
    """Convert a single fact triple to a QA pair using fixed templates.

    Args:
        triple: The fact triple to render.
        templates: Optional mapping from relation type to question template.

    Returns:
        A FactQA with the rendered question and answer.
    """
    templates = templates or DEFAULT_TEMPLATES
    rel_key = triple.relation.lower().strip()

    template = templates.get(rel_key, DEFAULT_TEMPLATE)
    question = template.format(subject=triple.subject, relation=triple.relation)
    answer = triple.object

    return FactQA(question=question, answer=answer, triple=triple)


def render_all(
    triples: list[FactTriple],
    templates: dict[str, str] | None = None,
) -> list[FactQA]:
    """Render all triples to QA pairs."""
    return [render_triple(t, templates) for t in triples]


def load_templates_from_config(cfg: DictConfig) -> dict[str, str]:
    """Load templates from a triples config."""
    return dict(cfg.get("templates", {}))


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
