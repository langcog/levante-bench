#!/usr/bin/env python3
"""Generate Klingon (`tlh`) translations for prompt configs and item-bank text."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml
try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional dependency
    repair_json = None


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
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_LOCATION = "us-central1"
PLACEHOLDER_RE = re.compile(r"(\{[A-Za-z_][A-Za-z0-9_]*\}|<[A-Za-z_][A-Za-z0-9_]*>)")


SYSTEM_INSTRUCTION = """\
You are translating benchmark text into Klingon (tlhIngan Hol).
Return natural Klingon text suitable for a native Klingon-language benchmark.
Preserve these tokens exactly wherever they appear:
- brace placeholders such as {prompt_phrase}, {image1}, {option1}, {labels}
- angle placeholders such as <prompt_phrase>, <prompt_image>, <image1>
- option labels and answer letters A, B, C, D
- JSON/tag examples such as <think>, </think>, <answer>, </answer>
If the source string has no placeholders, do not introduce any curly-brace
or angle-bracket placeholders in the translation.
Do not add commentary, markdown, backticks, or English explanations.
"""


def _run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _gcloud_project() -> str:
    return _run_text(["gcloud", "config", "get-value", "project"])


def _gcloud_access_token() -> str:
    return _run_text(["gcloud", "auth", "print-access-token"])


def _placeholder_set(value: str) -> set[str]:
    return set(PLACEHOLDER_RE.findall(value))


def _clean_existing_tlh(value: object) -> str:
    text = "" if value is None else str(value)
    return text.removeprefix("TODO_TLH: ").strip()


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if repair_json is not None:
            repaired = repair_json(text)
            parsed = json.loads(repaired)
            if not isinstance(parsed, list):
                raise ValueError("Gemini response was not a JSON array")
            return parsed
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Gemini response was not a JSON array")
    return parsed


def _extract_response_text(payload: dict[str, Any]) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected Gemini response: {payload}") from exc
    return "".join(str(part.get("text", "")) for part in parts)


class VertexGeminiClient:
    def __init__(self, *, project: str, location: str, model: str) -> None:
        self.project = project
        self.location = location
        self.model = model
        self._token = _gcloud_access_token()

    @property
    def url(self) -> str:
        return (
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/"
            f"{self.project}/locations/{self.location}/publishers/google/models/"
            f"{self.model}:generateContent"
        )

    def generate_json_array(self, texts: list[str], *, attempts: int = 4) -> list[str]:
        prompt = {
            "instruction": SYSTEM_INSTRUCTION,
            "input_format": "JSON array of English strings",
            "output_format": "JSON array of Klingon strings in the same order",
            "texts": texts,
        }
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Translate the `texts` array to Klingon. Return only the translated "
                                "JSON array with exactly the same number of elements.\n\n"
                                + json.dumps(prompt, ensure_ascii=False)
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topP": 0.8,
                "maxOutputTokens": max(512, min(8192, 1024 + sum(len(text) for text in texts) * 3)),
                "responseMimeType": "application/json",
            },
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                self.url,
                data=data,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json; charset=utf-8",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                values = _extract_json_array(_extract_response_text(payload))
                if len(texts) == 1 and len(values) > 1:
                    values = [" ".join(str(value).strip() for value in values if str(value).strip())]
                if len(values) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} translations, received {len(values)}"
                    )
                return [str(value).strip() for value in values]
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    self._token = _gcloud_access_token()
                last_error = f"{exc}: {exc.read().decode('utf-8', errors='replace')[:500]}"
            except Exception as exc:  # pragma: no cover - network/model dependent
                last_error = str(exc)
            if attempt < attempts:
                time.sleep(2**attempt)
        raise RuntimeError(f"Gemini request failed after {attempts} attempts: {last_error}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} did not load as a mapping")
    return loaded


def _collect_prompt_sources(prompts_dir: Path) -> dict[tuple[Path, str], str]:
    sources: dict[tuple[Path, str], str] = {}
    for path in sorted(prompts_dir.glob("*.yaml")):
        config = _load_yaml(path)
        templates = config.get("prompt_templates") or {}
        if not isinstance(templates, dict):
            continue
        for template_id, template in templates.items():
            if not isinstance(template, dict):
                continue
            locales = template.get("locales")
            if not isinstance(locales, dict) or "tlh" not in locales:
                continue
            sources[(path, str(template_id))] = str(template.get("template", ""))
    return sources


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _format_yaml_scalar(value: str, indent: int, *, folded: bool) -> list[str]:
    if folded or "\n" in value:
        out = [" " * indent + "tlh: >\n"]
        for line in value.splitlines() or [""]:
            out.append(" " * (indent + 2) + line + "\n")
        return out
    return [" " * indent + "tlh: " + json.dumps(value, ensure_ascii=False) + "\n"]


def _replace_prompt_tlh_blocks(
    prompts_dir: Path,
    translations: dict[tuple[Path, str], str],
) -> None:
    for path in sorted(prompts_dir.glob("*.yaml")):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        out: list[str] = []
        i = 0
        current_template: str | None = None
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if _line_indent(line) == 2 and stripped.endswith(":"):
                current_template = stripped[:-1]
            if (
                current_template
                and _line_indent(line) >= 6
                and line.lstrip().startswith("tlh:")
                and (path, current_template) in translations
            ):
                indent = _line_indent(line)
                folded = line.rstrip().endswith(">")
                out.extend(
                    _format_yaml_scalar(
                        translations[(path, current_template)],
                        indent,
                        folded=folded,
                    )
                )
                i += 1
                while i < len(lines):
                    if lines[i].strip() and _line_indent(lines[i]) <= indent:
                        break
                    i += 1
                continue
            out.append(line)
            i += 1
        path.write_text("".join(out), encoding="utf-8")


def _validate_translation(source: str, translated: str, *, label: str) -> None:
    source_placeholders = _placeholder_set(source)
    translated_placeholders = _placeholder_set(translated)
    if source_placeholders != translated_placeholders:
        raise ValueError(
            f"{label}: placeholder mismatch source={sorted(source_placeholders)} "
            f"translated={sorted(translated_placeholders)}"
        )
    if translated.startswith("TODO_TLH"):
        raise ValueError(f"{label}: translation still contains TODO_TLH")


def _translate_with_fallback(client: VertexGeminiClient, texts: list[str]) -> list[str]:
    try:
        return client.generate_json_array(texts)
    except Exception:
        if len(texts) == 1:
            raise
        midpoint = len(texts) // 2
        return _translate_with_fallback(client, texts[:midpoint]) + _translate_with_fallback(
            client,
            texts[midpoint:],
        )


def _translate_one_validated(
    client: VertexGeminiClient,
    source: str,
    *,
    label: str,
    validation_attempts: int = 3,
) -> str:
    last_error: Exception | None = None
    for _attempt in range(validation_attempts):
        value = _translate_with_fallback(client, [source])[0]
        try:
            _validate_translation(source, value, label=label)
            return value
        except ValueError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def translate_prompt_configs(
    client: VertexGeminiClient,
    *,
    prompts_dir: Path,
    batch_size: int,
    apply: bool,
) -> None:
    sources = _collect_prompt_sources(prompts_dir)
    items = list(sources.items())
    translated: dict[tuple[Path, str], str] = {}
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        values = _translate_with_fallback(client, [source for _key, source in batch])
        for (key, source), value in zip(batch, values, strict=True):
            _validate_translation(source, value, label=f"{key[0]}:{key[1]}")
            translated[key] = value
        print(f"translated prompt configs {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    if apply:
        _replace_prompt_tlh_blocks(prompts_dir, translated)


def translate_itembank(
    client: VertexGeminiClient,
    *,
    csv_path: Path,
    batch_size: int,
    workers: int,
    apply: bool,
) -> None:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "en" not in fieldnames:
        raise ValueError(f"{csv_path}: missing en column")
    if "tlh" not in fieldnames:
        fieldnames.append("tlh")

    indexes = [
        i
        for i, row in enumerate(rows)
        if (row.get("en") or "").strip()
        and (not (row.get("tlh") or "").strip() or "TODO_TLH" in (row.get("tlh") or ""))
    ]

    def write_rows() -> None:
        if apply:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

    if workers > 1:
        completed = 0
        for start in range(0, len(indexes), batch_size):
            window = indexes[start : start + batch_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _translate_one_validated,
                        client,
                        rows[row_index]["en"].strip(),
                        label=f"{csv_path}:row{row_index + 2}",
                    ): row_index
                    for row_index in window
                }
                for future in as_completed(futures):
                    row_index = futures[future]
                    value = future.result()
                    rows[row_index]["tlh"] = value
                    completed += 1
            write_rows()
            print(
                f"translated item-bank rows {min(start + len(window), len(indexes))}/{len(indexes)} "
                f"(completed this run: {completed})",
                flush=True,
            )
    else:
        for start in range(0, len(indexes), batch_size):
            batch_indexes = indexes[start : start + batch_size]
            sources = [rows[i]["en"].strip() for i in batch_indexes]
            values = _translate_with_fallback(client, sources)
            for row_index, source, value in zip(batch_indexes, sources, values, strict=True):
                _validate_translation(source, value, label=f"{csv_path}:row{row_index + 2}")
                rows[row_index]["tlh"] = value
            write_rows()
            print(
                f"translated item-bank rows {min(start + batch_size, len(indexes))}/{len(indexes)}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-dir", type=Path, default=PROMPTS_DIR)
    parser.add_argument("--translations-csv", type=Path, default=DEFAULT_TRANSLATIONS_CSV)
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent single-row workers for item-bank translation.",
    )
    parser.add_argument("--skip-prompts", action="store_true")
    parser.add_argument("--skip-itembank", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Write translations to files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.skip_prompts and args.skip_itembank:
        raise SystemExit("Nothing to do: both --skip-prompts and --skip-itembank were set.")
    client = VertexGeminiClient(
        project=args.project or _gcloud_project(),
        location=args.location,
        model=args.model,
    )
    if not args.skip_prompts:
        translate_prompt_configs(
            client,
            prompts_dir=args.prompts_dir,
            batch_size=args.batch_size,
            apply=args.apply,
        )
    if not args.skip_itembank:
        translate_itembank(
            client,
            csv_path=args.translations_csv,
            batch_size=args.batch_size,
            workers=args.workers,
            apply=args.apply,
        )
    if not args.apply:
        print("Dry run complete. Re-run with --apply to write files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
