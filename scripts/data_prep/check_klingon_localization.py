#!/usr/bin/env python3
"""Validate the scaffolded Klingon (`tlh`) localization surface."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "configs" / "prompts"
DEFAULT_TRANSLATIONS_CSV = (
    REPO_ROOT
    / "data"
    / "assets"
    / "v1"
    / "translations"
    / "item-bank-translations.csv"
)
PLACEHOLDER_RE = re.compile(r"(\{[A-Za-z_][A-Za-z0-9_]*\}|<[A-Za-z_][A-Za-z0-9_]*>)")


def _placeholder_set(value: object) -> set[str]:
    return set(PLACEHOLDER_RE.findall("" if value is None else str(value)))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} did not load as a mapping")
    return loaded


def _iter_prompt_templates(prompts_dir: Path):
    for path in sorted(prompts_dir.glob("*.yaml")):
        config = _load_yaml(path)
        task_id = str(config.get("task_id") or path.stem.replace("_", "-"))
        templates = config.get("prompt_templates") or {}
        if not isinstance(templates, dict):
            yield path, task_id, None, None, "prompt_templates is not a mapping"
            continue
        for template_id, template in templates.items():
            if not isinstance(template, dict):
                yield path, task_id, str(template_id), None, "template entry is not a mapping"
                continue
            yield path, task_id, str(template_id), template, None


def check_prompt_configs(prompts_dir: Path) -> list[str]:
    errors: list[str] = []
    checked = 0
    for path, _task_id, template_id, template, load_error in _iter_prompt_templates(prompts_dir):
        if load_error:
            errors.append(f"{path}: {load_error}")
            continue
        assert template is not None
        locales = template.get("locales")
        if not locales:
            continue
        checked += 1
        if not isinstance(locales, dict):
            errors.append(f"{path}:{template_id}: locales is not a mapping")
            continue
        if "tlh" not in locales:
            errors.append(f"{path}:{template_id}: missing locales.tlh")
            continue
        if "TODO_TLH" in str(locales.get("tlh", "")):
            errors.append(f"{path}:{template_id}: locales.tlh still contains TODO_TLH")
        source_placeholders = _placeholder_set(template.get("template", ""))
        tlh_placeholders = _placeholder_set(locales.get("tlh", ""))
        if source_placeholders != tlh_placeholders:
            errors.append(
                f"{path}:{template_id}: placeholder mismatch "
                f"source={sorted(source_placeholders)} tlh={sorted(tlh_placeholders)}"
            )
    if checked == 0:
        errors.append(f"{prompts_dir}: no localized prompt templates found")
    return errors


def check_translations_csv(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path}: missing item-bank translations CSV"]
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "tlh" not in fieldnames:
            return [f"{path}: missing tlh column"]
        if "en" not in fieldnames:
            errors.append(f"{path}: missing en source column")
        row_count = 0
        missing_tlh = 0
        for row in reader:
            row_count += 1
            en_text = (row.get("en") or "").strip()
            tlh_text = (row.get("tlh") or "").strip()
            if en_text and not tlh_text:
                missing_tlh += 1
            if "TODO_TLH" in tlh_text:
                errors.append(
                    f"{path}: row {row_count + 1} ({row.get('item_id', '')}) still contains TODO_TLH"
                )
            en_placeholders = _placeholder_set(en_text)
            tlh_placeholders = _placeholder_set(tlh_text)
            if en_placeholders != tlh_placeholders:
                errors.append(
                    f"{path}: row {row_count + 1} ({row.get('item_id', '')}) placeholder mismatch "
                    f"source={sorted(en_placeholders)} tlh={sorted(tlh_placeholders)}"
                )
        if row_count == 0:
            errors.append(f"{path}: no translation rows found")
        if missing_tlh:
            errors.append(f"{path}: {missing_tlh} rows have en text but empty tlh text")
    return errors


def check_renderability(prompts_dir: Path) -> list[str]:
    errors: list[str] = []
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from levante_bench.prompts.rendering import render_prompt_template
    except Exception as exc:  # pragma: no cover - environment-specific
        return [f"could not import prompt renderer: {exc}"]

    for path, task_id, template_id, template, load_error in _iter_prompt_templates(prompts_dir):
        if load_error or template is None or template_id is None:
            continue
        try:
            rendered = render_prompt_template(
                task_id,
                template_id,
                prompt_language="tlh",
                placeholders={
                    "prompt_phrase": "<prompt_phrase>",
                    "prompt_image": "<prompt_image>",
                    "full_prompt": "<full_prompt>",
                },
            )
        except Exception as exc:
            errors.append(f"{path}:{template_id}: failed to render tlh: {exc}")
            continue
        if template.get("locales") and "tlh" in template.get("locales", {}) and not rendered:
            errors.append(f"{path}:{template_id}: rendered tlh is empty")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    parser.add_argument("--translations-csv", type=Path, default=DEFAULT_TRANSLATIONS_CSV)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only validate config/CSV coverage; do not import the runtime renderer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = []
    errors.extend(check_prompt_configs(args.prompts_dir))
    errors.extend(check_translations_csv(args.translations_csv))
    if not args.skip_render:
        errors.extend(check_renderability(args.prompts_dir))

    if errors:
        print("Klingon localization check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Klingon localization check passed.")
    print(f"- Prompt configs: {args.prompts_dir}")
    print(f"- Item-bank translations: {args.translations_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
