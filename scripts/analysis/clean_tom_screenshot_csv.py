#!/usr/bin/env python3
"""Normalize noisy text fields in generated ToM screenshot CSV outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

ALLOWED_STAGE = {"initial_load", "task_started", "instructions", "test_response", "task_completed", "unknown"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean screenshot description CSV fields and rewrite per-item CSVs.")
    p.add_argument(
        "--master-csv",
        type=Path,
        default=Path("results/screenshots/theory-of-mind/theory-of-mind-screenshot-descriptions.csv"),
    )
    p.add_argument("--by-item-dir", type=Path, default=Path("results/screenshots/theory-of-mind/by_item"))
    p.add_argument("--summary-json", type=Path, default=Path("results/screenshots/theory-of-mind/summary.json"))
    return p.parse_args()


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_filler_tail(text: str) -> str:
    s = _collapse_ws(text)
    if not s:
        return s
    # Remove common narration residue like trailing "OK"/"ok." from OCR/model text.
    s = re.sub(r"(?:[\s,;:\-]+)?\bOK\b[.!?]?\s*$", "", s, flags=re.I).strip()
    s = re.sub(r"(?:[\s,;:\-]+)?\bok\b[.!?]?\s*$", "", s, flags=re.I).strip()
    return s


def _strip_wrapping_quotes(text: str) -> str:
    s = (text or "").strip()
    for _ in range(3):
        if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
            s = s[1:-1].strip()
            continue
        if s.startswith('"""'):
            s = s[3:].strip()
            continue
        if s.endswith('"""'):
            s = s[:-3].strip()
            continue
        if s.startswith('"'):
            s = s[1:].strip()
            continue
        if s.endswith('"'):
            s = s[:-1].strip()
            continue
        break
    return s.strip()


def _extract_json_dict(text: str) -> dict[str, Any] | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _extract_keyval_string(text: str, key: str) -> str:
    patterns = [
        rf'{key}\s*[:=]\s*"([^"]+)"',
        rf"{key}\s*[:=]\s*'([^']+)'",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.I)
        if m:
            return m.group(1).strip()
    return ""


def _normalize_stage(raw_stage: str, assessment_stage: str) -> str:
    s = _collapse_ws(raw_stage).lower()
    s = s.strip(":=").strip()
    if s == "unknown":
        s = ""
    if s in ALLOWED_STAGE:
        return s
    if assessment_stage == "instructions":
        return "instructions"
    if assessment_stage == "test_response":
        return "test_response"
    return "unknown"


def _looks_structured_noise(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    markers = [
        '{"',
        '"prompt"',
        '"description"',
        "extracted_prompt_text",
        "image_description",
        "stage_guess",
    ]
    return any(m in t for m in markers)


def _clean_row(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    extracted = _collapse_ws(_strip_wrapping_quotes(out.get("extracted_prompt_text", "")))
    desc = _collapse_ws(_strip_wrapping_quotes(out.get("image_description", "")))
    stage = _collapse_ws(_strip_wrapping_quotes(out.get("stage_guess", "")))
    corpus_prompt = _collapse_ws(out.get("corpus_prompt", ""))
    assessment_stage = _collapse_ws(out.get("assessment_stage", "")).lower()

    blob_source = "\n".join([extracted, desc])
    parsed = _extract_json_dict(blob_source)
    if parsed:
        parsed_extracted = _collapse_ws(str(parsed.get("extracted_prompt_text") or parsed.get("prompt") or ""))
        parsed_desc = _collapse_ws(str(parsed.get("image_description") or parsed.get("description") or ""))
        parsed_stage = _collapse_ws(str(parsed.get("stage_guess") or ""))
        extracted = extracted or parsed_extracted
        if not desc or _looks_structured_noise(desc):
            desc = parsed_desc or desc
        stage = stage or parsed_stage

    if not extracted or _looks_structured_noise(extracted):
        extracted = _extract_keyval_string(blob_source, "extracted_prompt_text") or _extract_keyval_string(
            blob_source, "prompt"
        )
    if not desc or _looks_structured_noise(desc):
        desc = _extract_keyval_string(blob_source, "image_description") or _extract_keyval_string(blob_source, "description")
    if not stage:
        stage = _extract_keyval_string(blob_source, "stage_guess")

    # Strip inline "stage_guess: ..." artifacts from descriptions.
    desc = re.sub(r"\bstage_guess\s*[:=]\s*['\"]?[a-z_]+['\"]?", "", desc, flags=re.I).strip(" ,;")
    extracted = re.sub(r"\bstage_guess\s*[:=]\s*['\"]?[a-z_]+['\"]?", "", extracted, flags=re.I).strip(" ,;")

    # Drop obvious non-descriptions.
    bad_desc = {"ok", "task_started", "started", "load"}
    if desc.lower() in bad_desc:
        desc = ""
    if extracted.lower() in bad_desc:
        extracted = ""

    if not extracted and corpus_prompt:
        extracted = corpus_prompt
    if not desc or _looks_structured_noise(desc):
        if extracted:
            desc = f"Screen showing prompt: {extracted}"
        elif corpus_prompt:
            desc = f"Screen showing prompt: {corpus_prompt}"

    extracted = _strip_filler_tail(extracted)
    desc = _strip_filler_tail(desc)
    out["extracted_prompt_text"] = _collapse_ws(extracted)
    out["image_description"] = _collapse_ws(desc)
    out["stage_guess"] = _normalize_stage(stage, assessment_stage)
    return out


def run(args: argparse.Namespace) -> None:
    if not args.master_csv.exists():
        raise FileNotFoundError(f"Missing master CSV: {args.master_csv}")

    with open(args.master_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    cleaned = [_clean_row(r) for r in rows]

    with open(args.master_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(cleaned)

    args.by_item_dir.mkdir(parents=True, exist_ok=True)
    for old in args.by_item_dir.glob("*.csv"):
        old.unlink()

    by_item: dict[str, list[dict[str, str]]] = {}
    for r in cleaned:
        uid = (r.get("item_uid") or "").strip()
        if uid:
            by_item.setdefault(uid, []).append(r)
    for uid, uid_rows in by_item.items():
        out_csv = args.by_item_dir / f"{uid}.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(uid_rows)

    if args.summary_json.exists():
        try:
            summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    else:
        summary = {}
    summary["cleaned"] = True
    summary["n_item_csvs"] = len(by_item)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"master_csv": str(args.master_csv), "n_rows": len(cleaned), "n_item_csvs": len(by_item)}, indent=2))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

