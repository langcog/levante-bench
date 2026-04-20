"""Task registry: load task definitions from configs/tasks/ YAML files."""

from pathlib import Path
from typing import Optional

from levante_bench.config.loader import get_configs_root, load_task_config
from levante_bench.data.schema import TaskDef


def _safe_task_id(task_id: str) -> str:
    """Filename-safe task id."""
    import re
    return re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def get_task_def(
    task_id: str,
    version: str,
    data_root: Path | None = None,
    task_overrides: dict | None = None,
) -> Optional[TaskDef]:
    """Build TaskDef from configs/tasks/<task_id>.yaml with paths resolved."""
    cfg = load_task_config(task_id)
    if cfg is None:
        return None

    if data_root is not None:
        raw = Path(data_root) / "raw" / version
        safe = _safe_task_id(task_id)
        manifest_path = raw / "tasks" / f"{safe}_trials.csv"
        if not manifest_path.exists():
            manifest_path = raw / "trials.csv"

        # Human response proportions written by download_levante_data.R
        responses_dir = Path(data_root) / "responses" / version / "responses_by_ability"
        human_path = responses_dir / f"{safe}_proportions.csv"
        if not human_path.exists():
            human_path = None
    else:
        manifest_path = None
        human_path = None

    overrides = task_overrides or {}

    return TaskDef(
        task_id=cfg.task_id,
        benchmark_name=cfg.get("benchmark_name", cfg.task_id),
        internal_name=cfg.get("internal_name", cfg.task_id),
        manifest_path=manifest_path if manifest_path and manifest_path.exists() else None,
        human_response_path=human_path,
        task_type=cfg.get("task_type", "forced-choice"),
        n_options=cfg.get("n_options", 4),
        has_correct=cfg.get("has_correct", True),
        corpus_file=cfg.get("corpus_file"),
        context_type=cfg.get("context_type", "none"),
        option_type=cfg.get("option_type", "text"),
        include_numberline=bool(overrides.get("include_numberline", cfg.get("include_numberline", False))),
        prompt_language=str(overrides.get("prompt_language", cfg.get("prompt_language", "en"))),
        mental_rotation_prompt_template=_optional_text(
            overrides.get(
                "mental_rotation_prompt_template",
                cfg.get("mental_rotation_prompt_template"),
            )
        ),
        true_random_option_order=bool(
            overrides.get("true_random_option_order", cfg.get("true_random_option_order", False))
        ),
        option_order_run_seed=(
            int(overrides.get("option_order_run_seed"))
            if overrides.get("option_order_run_seed") is not None
            else (
                int(cfg.get("option_order_run_seed"))
                if cfg.get("option_order_run_seed") is not None
                else None
            )
        ),
    )


def list_tasks() -> list[str]:
    """Return task_ids from all YAML files in configs/tasks/."""
    tasks_dir = get_configs_root() / "tasks"
    if not tasks_dir.exists():
        return []
    return [p.stem.replace("_", "-") for p in sorted(tasks_dir.glob("*.yaml"))]
