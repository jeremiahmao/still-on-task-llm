"""Configuration loading via OmegaConf with YAML merge and CLI overrides."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


_BASE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"


def load_config(config_path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
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
