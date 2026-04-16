"""Configuration loading via OmegaConf with YAML merge and CLI overrides."""

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BASE_CONFIG = _REPO_ROOT / "configs" / "base.yaml"
_ENV_FILE = _REPO_ROOT / ".env"


def _load_env_file() -> None:
    """Load KEY=VALUE lines from .env into os.environ (skip if already set)."""
    if not _ENV_FILE.exists():
        return
    with _ENV_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_env_file()


def load_config(
    config_path: str | Path | None = None, overrides: list[str] | None = None
) -> DictConfig:
    """Load and merge configs: base.yaml <- specific config <- CLI overrides."""
    base = OmegaConf.load(_BASE_CONFIG)

    if config_path is not None:
        specific = OmegaConf.load(config_path)
        merged = OmegaConf.merge(base, specific)
    else:
        merged = base

    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(merged, cli)

    return merged


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Save a resolved config to YAML for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
