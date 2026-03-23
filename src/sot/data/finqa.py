"""FinQA dataset loading and prompt formatting for Qwen2.5-3B."""

import json
import subprocess
from pathlib import Path


def download_finqa(data_root: str | Path) -> Path:
    """Clone FinQA repo if not already present. Returns path to dataset dir."""
    data_root = Path(data_root)
    repo_dir = data_root / "finqa"

    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/czyssrs/FinQA.git", str(repo_dir)],
            check=True,
        )

    dataset_dir = repo_dir / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"FinQA dataset dir not found at {dataset_dir}")
    return dataset_dir


def load_finqa_split(dataset_dir: str | Path, split: str = "train") -> list[dict]:
    """Load a FinQA split from JSON.

    Args:
        dataset_dir: Path to FinQA's dataset/ folder.
        split: One of 'train', 'dev', 'test'.

    Returns:
        List of raw FinQA examples.
    """
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / f"{split}.json"

    if not path.exists():
        # FinQA sometimes uses different naming
        alt_path = dataset_dir / f"{split}_retrieve.json"
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"No FinQA {split} file at {path} or {alt_path}")

    with open(path) as f:
        return json.load(f)


def format_table(table: list[list[str]]) -> str:
    """Format a FinQA table as a markdown-style string."""
    if not table:
        return ""
    lines = []
    for row in table:
        lines.append(" | ".join(str(cell) for cell in row))
    return "\n".join(lines)


def format_finqa_example(example: dict, system_prompt: str) -> dict:
    """Format a FinQA example into a chat-style training example.

    FinQA nests question/answer/program inside the 'qa' key.

    Returns:
        Dict with 'messages' key in OpenAI chat format.
    """
    table_str = format_table(example.get("table", []))
    pre_text = " ".join(example.get("pre_text", []))
    post_text = " ".join(example.get("post_text", []))

    # QA data is nested under 'qa' key
    qa = example.get("qa", {})
    question = qa.get("question", "")
    program = qa.get("program_re", qa.get("program", ""))
    answer = qa.get("exe_ans", "")

    user_content = f"""Context:
{pre_text}

Table:
{table_str}

{post_text}

Question: {question}"""

    if isinstance(program, list):
        program = ", ".join(str(p) for p in program)

    assistant_content = f"Program: {program}\nAnswer: {answer}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "id": example.get("id", ""),
        "gold_answer": str(answer),
        "gold_program": str(program),
    }


def prepare_finqa_dataset(
    dataset_dir: str | Path, system_prompt: str, split: str = "train"
) -> list[dict]:
    """Load and format a full FinQA split."""
    raw = load_finqa_split(dataset_dir, split)
    return [format_finqa_example(ex, system_prompt) for ex in raw]
