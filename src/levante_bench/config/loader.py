"""Config loader using OmegaConf. Merges experiment + model + task configs."""

from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig


def get_configs_root(configs_root: str | Path | None = None) -> Path:
    """Path to configs/ directory at project root or explicit override."""
    if configs_root is not None:
        return Path(configs_root)
    return Path(__file__).resolve().parent.parent.parent.parent / "configs"


def load_experiment(
    experiment_path: str | Path | None = None,
    cli_overrides: list[str] | None = None,
    configs_root: str | Path | None = None,
) -> DictConfig:
    """Load experiment config, merge CLI overrides."""
    if experiment_path is None:
        experiment_path = get_configs_root(configs_root=configs_root) / "experiment.yaml"
    cfg = OmegaConf.load(experiment_path)
    if cli_overrides:
        cli_cfg = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg


def load_model_config(
    model_name: str,
    configs_root: str | Path | None = None,
) -> Optional[DictConfig]:
    """Load a single model config from configs/models/<model_name>.yaml."""
    path = get_configs_root(configs_root=configs_root) / "models" / f"{model_name}.yaml"
    if not path.exists():
        return None
    return OmegaConf.load(path)


def load_task_config(
    task_id: str,
    configs_root: str | Path | None = None,
) -> Optional[DictConfig]:
    """Load a single task config from configs/tasks/<task_id>.yaml."""
    # Try exact name first, then with underscores replacing hyphens
    configs_dir = get_configs_root(configs_root=configs_root) / "tasks"
    for name in [task_id, task_id.replace("-", "_")]:
        path = configs_dir / f"{name}.yaml"
        if path.exists():
            return OmegaConf.load(path)
    return None
