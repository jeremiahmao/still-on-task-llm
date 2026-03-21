"""Dual logging to wandb and CSV."""

import csv
import json
from pathlib import Path

import wandb


def init_wandb(cfg, run_name: str | None = None, tags: list[str] | None = None) -> None:
    """Initialize wandb run from config."""
    if not cfg.wandb.enabled:
        wandb.init(mode="disabled")
        return
    wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        tags=tags,
        config=dict(cfg),
    )


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log to wandb (if active)."""
    wandb.log(metrics, step=step)


def save_results_csv(results: list[dict], path: str | Path) -> None:
    """Append results to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def save_metadata(metadata: dict, path: str | Path) -> None:
    """Save experiment metadata as JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
