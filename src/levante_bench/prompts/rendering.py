"""Render centralized task prompt templates."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping

from omegaconf import OmegaConf

from levante_bench.config.loader import get_configs_root


_PROMPT_FILE_BY_TASK = {
    "egma-math": "egma_math",
    "matrix-reasoning": "matrix_reasoning",
    "mental-rotation": "mental_rotation",
    "theory-of-mind": "theory_of_mind",
    "trog": "trog",
    "vocab": "vocab",
}

_LANGUAGE_ALIASES = {
    "es": "es-CO",
}

_EGMA_TEMPLATE_BY_TRIAL_TYPE = {
    "number identification": "number_identification.default",
    "number comparison": "number_comparison.default",
    "missing number": "missing_number.default",
    "addition": "addition.default",
    "fraction": "addition.default",
    "subtraction": "subtraction.default",
    "multiplication": "multiplication.default",
    "counting": "counting.default",
    "counting afc": "counting_afc.default",
    "non-symbolic number comparison": "non_symbolic_number_comparison.default",
    "non-symbolic number identification": "non_symbolic_number_identification.default",
    "number line 4afc": "number_line_4afc.default",
    "number line slider": "number_line_slider.default",
}


class _MissingDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _to_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.strip().lower() in {"", "nan", "na", "none"}:
        return ""
    return text


def _normalize_space(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _clean_template(text: str) -> str:
    return _normalize_space(text)


def _task_config_name(task_id: str) -> str:
    task_id = str(task_id).strip()
    return _PROMPT_FILE_BY_TASK.get(task_id, task_id.replace("-", "_"))


@lru_cache(maxsize=None)
def _load_prompt_config(task_id: str, configs_root: str | None = None) -> dict[str, Any]:
    root = get_configs_root(configs_root)
    path = root / "prompts" / f"{_task_config_name(task_id)}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No prompt config found for task {task_id!r}: {path}")
    loaded = OmegaConf.load(path)
    return OmegaConf.to_container(loaded, resolve=True)  # type: ignore[return-value]


def _resolve_language(prompt_language: str, available: Mapping[str, Any]) -> str | None:
    if not prompt_language or prompt_language == "en":
        return None
    if prompt_language in available:
        return prompt_language
    alias = _LANGUAGE_ALIASES.get(prompt_language)
    if alias and alias in available:
        return alias
    lang_prefix = f"{prompt_language.split('-', 1)[0]}-"
    for language in available:
        if str(language).startswith(lang_prefix):
            return str(language)
    return None


def infer_prompt_template_id(task_id: str, row: Mapping[str, Any]) -> str:
    """Return the central prompt template ID for a manifest row."""
    explicit = _to_text(row.get("prompt_template_id", ""))
    if explicit:
        return explicit

    task_id = str(task_id).strip()
    if task_id == "egma-math":
        trial_type = _to_text(row.get("trial_type", "")).strip().lower()
        if trial_type == "fraction":
            prompt = _to_text(row.get("prompt", "")).strip().lower()
            item = _to_text(row.get("item", ""))
            return (
                "subtraction.default"
                if "subtract" in prompt or "-" in item
                else "addition.default"
            )
        return _EGMA_TEMPLATE_BY_TRIAL_TYPE.get(
            trial_type,
            "number_identification.default",
        )
    if task_id == "matrix-reasoning":
        return "blank_option.default"
    if task_id == "mental-rotation":
        return "match_image.default"
    if task_id == "theory-of-mind":
        return "story.default"
    if task_id == "trog":
        return "sentence_image_match.default"
    if task_id == "vocab":
        return "image_match.default"

    config = _load_prompt_config(task_id)
    default_id = _to_text(config.get("default_template_id", ""))
    if default_id:
        return default_id
    raise ValueError(f"No prompt template inference rule for task {task_id!r}.")


def _template_for_language(entry: Mapping[str, Any], prompt_language: str) -> str:
    locales = entry.get("locales") or {}
    if isinstance(locales, Mapping):
        resolved = _resolve_language(prompt_language, locales)
        if resolved:
            return _clean_template(str(locales[resolved]))
    return _clean_template(str(entry.get("template", "")))


def _default_placeholders(row: Mapping[str, Any]) -> dict[str, str]:
    values = {str(k): _to_text(v) for k, v in row.items()}
    values["prompt_phrase"] = "<prompt_phrase>"
    values["prompt_image"] = "<prompt_image>"
    values.setdefault("full_prompt", _to_text(row.get("full_prompt", "")))

    for i in range(1, 9):
        values.setdefault(f"option{i}", f"<option{i}>")
        values.setdefault(f"image{i}", f"<image{i}>")
    return values


def render_prompt_template(
    task_id: str,
    template_id: str,
    *,
    row: Mapping[str, Any] | None = None,
    prompt_language: str = "en",
    placeholders: Mapping[str, object] | None = None,
    configs_root: str | Path | None = None,
) -> str:
    """Render a centralized task prompt template."""
    config = _load_prompt_config(
        task_id,
        str(configs_root) if configs_root is not None else None,
    )
    templates = config.get("prompt_templates") or {}
    if template_id not in templates:
        raise KeyError(f"Prompt template {template_id!r} not found for task {task_id!r}.")

    row_values = _default_placeholders({} if row is None else row)
    if placeholders:
        row_values.update({str(k): _to_text(v) for k, v in placeholders.items()})

    template = _template_for_language(templates[template_id], prompt_language)
    return template.format_map(_MissingDict(row_values))


def render_prompt_template_for_row(
    task_id: str,
    row: Mapping[str, Any],
    *,
    prompt_language: str = "en",
    placeholders: Mapping[str, object] | None = None,
    configs_root: str | Path | None = None,
) -> str:
    """Infer and render the central prompt template for a manifest row."""
    template_id = infer_prompt_template_id(task_id, row)
    return render_prompt_template(
        task_id=task_id,
        template_id=template_id,
        row=row,
        prompt_language=prompt_language,
        placeholders=placeholders,
        configs_root=configs_root,
    )


def manifest_placeholder_style(prompt: str) -> str:
    """Convert brace placeholder style to manifest angle placeholder style."""
    return re.sub(r"\{(prompt_phrase|prompt_image|option\d+|image\d+)\}", r"<\1>", prompt)
