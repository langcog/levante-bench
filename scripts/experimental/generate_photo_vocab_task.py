#!/usr/bin/env python3
"""Build a photo-like synthetic vocab asset set.

This script creates a new versioned synthetic-vocab corpus from an existing
manifest, writes one text-free image prompt per unique option term, and can
optionally call the OpenAI Images API to generate the actual PNG files.

The generated task is intentionally separate from the old icon-style assets:

- manifest rows use task_id ``synthetic-vocab``
- images are requested as photo-like, with no text/labels/signage
- prompts are saved for audit before any paid image calls are made
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image

TASK_ID = "synthetic-vocab"
DEFAULT_SOURCE_VERSION = "new-vocab-2026-04-24"
DEFAULT_VERSION = "new-vocab-photo-2026-04-25"
DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = "gpt-image-1"
DEFAULT_GEMINI_MODEL = "imagen-4.0-generate-001"
DEFAULT_SIZE = "512x512"

PROMPT_EN = "Which picture shows {word}?"
PROMPT_DE = "Welches Bild zeigt {word}?"
PROMPT_ES = "Que imagen muestra {word}?"
FULL_PROMPT_TEMPLATE = (
    "You will see four pictures labeled A, B, C, and D. "
    "Choose the picture that best matches the word: {word}."
)

GLOBAL_FRIENDLY_NOTES = {
    "barn": "Use a simple farm building, not signage or a region-specific landmark.",
    "castle": "Use a generic stone castle, not a famous landmark.",
    "igloo": "Use a simple snow shelter; avoid people or cultural costume.",
    "pagoda": "Use a generic tiered tower; avoid religious symbols or text.",
    "gondola": "Use a small narrow canal boat without markings or people.",
    "scepter": "Use a plain ceremonial staff on a neutral background.",
    "totem": "Use a generic carved wooden pole; avoid sacred or identifiable cultural designs.",
}

GENERATED_IMAGE_EXCLUDED_TERMS = {
    "baby",
    "beard",
    "boy",
    "girl",
    "man",
    "person",
    "woman",
    "ankle",
    "arm",
    "antelope",
    "chin",
    "ear",
    "elbow",
    "eye",
    "face",
    "figurine",
    "finger",
    "foot",
    "hair",
    "hand",
    "handcuff",
    "head",
    "knee",
    "leg",
    "mannequin",
    "mouth",
    "mustache",
    "nose",
    "nutcracker",
    "pantyhose",
    "pocketknife",
    "seahorse",
    "skin",
    "shuffleboard",
    "scarecrow",
    "statue",
    "stomach",
    "taffy",
    "tattoo",
    "toe",
    "tongue",
    "tooth",
    "whistle",
    "wrist",
    "bazooka",
    "bullet",
    "cannon",
    "cannonball",
    "dagger",
    "firecracker",
    "gun",
    "knife",
    "missile",
    "rifle",
    "shotgun",
    "slingshot",
    "sword",
}


@dataclass(frozen=True)
class PromptRecord:
    term: str
    slug: str
    category: str
    age_band: int
    prompt: str
    output_path: str


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("scripts/new_vocab_assets/assets"),
        help="Root that contains versioned synthetic vocab asset folders.",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help=(
            "Optional source manifest. Defaults to the old generated manifest if present, "
            "otherwise data/assets/manifest.csv."
        ),
    )
    parser.add_argument("--source-version", default=DEFAULT_SOURCE_VERSION)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--n-items", type=int, default=170)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seed-targets-manifest",
        type=Path,
        default=None,
        help=(
            "Optional existing manifest whose answer labels should be preserved as "
            "targets before adding more stratified THINGS/AoA targets."
        ),
    )
    parser.add_argument(
        "--seed-items-manifest",
        type=Path,
        default=None,
        help=(
            "Optional existing manifest whose complete rows should be preserved "
            "before adding more generated items. Existing option terms are reserved "
            "so newly generated items use new images."
        ),
    )
    parser.add_argument(
        "--things-meta-csv",
        type=Path,
        default=Path("scripts/new_vocab_assets/inputs/things_meta.csv"),
        help="Optional THINGS metadata CSV with columns concept_id,label,child_safe,nameability,animacy.",
    )
    parser.add_argument(
        "--aoa-csv",
        type=Path,
        default=Path("scripts/new_vocab_assets/inputs/aoa_kuperman.csv"),
        help="Optional AoA norms CSV with columns word,aoa.",
    )
    parser.add_argument(
        "--original-targets-csv",
        type=Path,
        default=Path("scripts/new_vocab_assets/inputs/original_108.csv"),
        help="Optional original target labels CSV with column label; these labels are excluded.",
    )
    parser.add_argument(
        "--use-clip-similarity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OpenAI CLIP text embeddings for similar-distractor selection when available.",
    )
    parser.add_argument(
        "--clip-device",
        default="auto",
        help="CLIP device: auto, cpu, cuda, etc. Used only when --use-clip-similarity is enabled.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file to load API keys from before image generation.",
    )
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Call the image API. Without this, only manifests and prompts are written.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default=DEFAULT_PROVIDER,
        help="Image generation provider to use with --generate-images.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument(
        "--limit-images",
        type=int,
        default=None,
        help="Only generate the first N image prompts. Useful for a small pilot review.",
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Regenerate images that already exist.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log image generation failures and continue. Useful for pilot batches.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run asset validation after writing/generating.",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("task") in {"vocab", TASK_ID}]
    if not rows:
        raise SystemExit(f"No vocab rows found in {path}")
    return rows


def read_legacy_lexicon(path: Path, n_items: int) -> list[dict[str, str]]:
    spec = importlib.util.spec_from_file_location("legacy_generate_vocab_task", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import legacy lexicon script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    lexicon = module.ensure_unique_lexicon(module.LEXICON)
    selected = sorted(lexicon, key=lambda item: (item.age_band, item.category, item.word))[:n_items]
    return [
        {
            "task": TASK_ID,
            "item_uid": f"synthetic_vocab__{slugify(item.word)}",
            "answer": item.word,
            "response_alternatives": "",
            "prompt_phrase": item.word,
            "full_prompt": FULL_PROMPT_TEMPLATE.format(word=item.word),
            "trial_type": "test",
            "age_band": str(min(int(item.age_band), 11)),
            "category": item.category,
            "hardness": str(item.hardness),
            "_tags": "|".join(item.tags),
            "_shape": item.shape,
        }
        for item in selected
    ]


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def load_aoa(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            str(row["word"]).strip().lower(): parse_float(row["aoa"], default=99.0)
            for row in rows
            if str(row.get("word", "")).strip()
        }


def load_original_targets(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            str(row["label"]).strip().lower()
            for row in csv.DictReader(handle)
            if str(row.get("label", "")).strip()
        }


def load_seed_target_labels(path: Path | None) -> list[str]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            str(row["answer"]).strip().lower()
            for row in csv.DictReader(handle)
            if str(row.get("answer", "")).strip()
        ]


def load_seed_item_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for row in rows:
        answer = str(row.get("answer", "")).strip()
        alternatives = [
            part.strip()
            for part in str(row.get("response_alternatives", "")).split(",")
            if part.strip()
        ]
        if not answer or len(alternatives) != 3:
            raise SystemExit(
                f"Seed item manifest row for '{answer or '<missing answer>'}' must have exactly 3 alternatives."
            )
        row["task"] = TASK_ID
        row["item_uid"] = row.get("item_uid") or f"synthetic_vocab__{slugify(answer)}"
        row["answer"] = answer
        row["response_alternatives"] = ",".join(alternatives)
        row["prompt_phrase"] = row.get("prompt_phrase") or answer
        row["full_prompt"] = row.get("full_prompt") or FULL_PROMPT_TEMPLATE.format(word=answer)
        row["trial_type"] = row.get("trial_type") or "test"
        row["age_band"] = str(row.get("age_band") or "11")
        row["category"] = row.get("category") or "object"
        row["hardness"] = row.get("hardness") or ""
        row["high_similarity_distractor"] = row.get("high_similarity_distractor") or alternatives[0]
        row["medium_similarity_distractor"] = row.get("medium_similarity_distractor") or alternatives[1]
        row["low_similarity_distractor"] = row.get("low_similarity_distractor") or alternatives[2]
        row["similar_distractor"] = row.get("similar_distractor") or alternatives[0]
    return rows


def row_option_terms(row: dict[str, str]) -> list[str]:
    terms = [str(row.get("answer", "")).strip()]
    terms.extend(
        part.strip()
        for part in str(row.get("response_alternatives", "")).split(",")
        if part.strip()
    )
    return [term for term in terms if term]


def split_band_counts(n_items: int, n_bands: int) -> list[int]:
    base = n_items // n_bands
    remainder = n_items % n_bands
    return [base + (1 if index < remainder else 0) for index in range(n_bands)]


def select_stratified_by_aoa(
    candidates: list[dict[str, str]],
    n_items: int,
    seed: int,
    seed_labels: list[str] | None = None,
) -> list[dict[str, str]]:
    sorted_candidates = sorted(
        candidates,
        key=lambda row: (
            parse_float(row.get("_aoa", ""), default=99.0),
            -parse_float(row.get("_nameability", ""), default=0.0),
            row["answer"],
        ),
    )
    counts = split_band_counts(n_items, 3)
    band_size = len(sorted_candidates) // 3
    bands = [
        sorted_candidates[:band_size],
        sorted_candidates[band_size : band_size * 2],
        sorted_candidates[band_size * 2 :],
    ]
    rng = random.Random(seed)
    by_label = {row["answer"].lower(): row for row in sorted_candidates}
    selected: list[dict[str, str]] = []
    selected_labels: set[str] = set()
    band_by_label: dict[str, str] = {}
    for band_name, band in zip(["low", "mid", "high"], bands):
        for row in band:
            band_by_label[row["answer"].lower()] = band_name

    for label in seed_labels or []:
        if label not in by_label:
            raise SystemExit(f"Seed target '{label}' was not found in the filtered THINGS/AoA pool.")
        if label in selected_labels:
            continue
        row = by_label[label]
        row["_aoa_tercile"] = band_by_label[label]
        selected.append(row)
        selected_labels.add(label)

    if len(selected) > n_items:
        raise SystemExit(
            f"Seed manifest has {len(selected)} usable target labels; requested only {n_items}."
        )
    seed_count = len(selected)

    for band_name, needed, band in zip(["low", "mid", "high"], counts, bands):
        locked_in_band = sum(
            1 for row in selected if row.get("_aoa_tercile") == band_name
        )
        needed = max(0, needed - locked_in_band)
        if len(band) < needed:
            raise SystemExit(
                f"Only {len(band)} candidates available in AoA {band_name} band; "
                f"need {needed}."
            )
        # Keep randomness for variety, but draw from the clearer, more nameable side of each band.
        nameable_pool_size = min(len(band), max(needed * 2, needed))
        nameable_pool = sorted(
            [row for row in band if row["answer"].lower() not in selected_labels],
            key=lambda row: (
                -parse_float(row.get("_nameability", ""), default=0.0),
                parse_float(row.get("_aoa", ""), default=99.0),
                row["answer"],
            ),
        )[:nameable_pool_size]
        sampled = rng.sample(nameable_pool, needed)
        for row in sampled:
            row["_aoa_tercile"] = band_name
            selected_labels.add(row["answer"].lower())
        selected.extend(sampled)

    if len(selected) < n_items:
        fill_pool = [
            row for row in sorted_candidates if row["answer"].lower() not in selected_labels
        ]
        selected.extend(fill_pool[: n_items - len(selected)])

    return selected[:seed_count] + sorted(
        selected[seed_count:],
        key=lambda row: (
            parse_float(row.get("_aoa", ""), default=99.0),
            -parse_float(row.get("_nameability", ""), default=0.0),
            row["answer"],
        ),
    )


def read_things_lexicon(
    things_meta_csv: Path,
    aoa_csv: Path,
    original_targets_csv: Path,
    n_items: int,
    seed: int,
    seed_targets_manifest: Path | None,
    excluded_labels: set[str] | None = None,
) -> list[dict[str, str]]:
    aoa_by_word = load_aoa(aoa_csv)
    original_targets = load_original_targets(original_targets_csv)
    seed_labels = load_seed_target_labels(seed_targets_manifest)
    excluded_labels = {label.lower() for label in excluded_labels or set()}
    candidates = []

    with things_meta_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = str(row.get("label", "")).strip().lower()
            if not label or label in original_targets:
                continue
            if label in excluded_labels:
                continue
            if label in GENERATED_IMAGE_EXCLUDED_TERMS:
                continue
            if not truthy(row.get("child_safe", "")):
                continue
            aoa = aoa_by_word.get(label)
            if aoa is None or aoa > 11:
                continue
            nameability = parse_float(row.get("nameability", ""), default=0.0)
            animacy = parse_float(row.get("animacy", ""), default=0.0)
            category = "animal" if animacy >= 0.5 else "object"
            age_band = max(3, min(11, round(aoa)))
            candidates.append(
                {
                    "task": TASK_ID,
                    "item_uid": f"synthetic_vocab__{slugify(label)}",
                    "answer": label,
                    "response_alternatives": "",
                    "prompt_phrase": label,
                    "full_prompt": FULL_PROMPT_TEMPLATE.format(word=label),
                    "trial_type": "test",
                    "age_band": str(age_band),
                    "category": category,
                    "hardness": str(max(1, min(5, round((age_band - 2) / 2)))),
                    "_concept_id": str(row.get("concept_id", "")).strip(),
                    "_tags": category,
                    "_shape": "",
                    "_aoa": f"{aoa:.3f}",
                    "_nameability": f"{nameability:.3f}",
                    "_animacy": f"{animacy:.3f}",
                }
            )

    if len(candidates) < n_items:
        raise SystemExit(
            f"Only {len(candidates)} THINGS/AoA candidates available after filtering; "
            f"need {n_items}."
        )
    selected = select_stratified_by_aoa(
        candidates,
        n_items=n_items,
        seed=seed,
        seed_labels=seed_labels,
    )
    selected_answers = {row["answer"] for row in selected}
    remaining = sorted(
        [row for row in candidates if row["answer"] not in selected_answers],
        key=lambda row: (
            parse_float(row.get("_aoa", ""), default=99.0),
            -parse_float(row.get("_nameability", ""), default=0.0),
            row["answer"],
        ),
    )
    return selected + remaining


def infer_age_band(index: int, n_items: int) -> int:
    if n_items <= 1:
        return 3
    return min(11, 3 + round((index / (n_items - 1)) * 8))


def split_tags(value: str) -> set[str]:
    return {tag.strip().lower() for tag in str(value or "").split("|") if tag.strip()}


def similarity_score(
    target: dict[str, str],
    candidate: dict[str, str],
    clip_similarity: dict[tuple[str, str], float] | None = None,
) -> float:
    target_tags = split_tags(target.get("_tags", ""))
    candidate_tags = split_tags(candidate.get("_tags", ""))
    shared_tags = target_tags & candidate_tags
    target_age = int(target["age_band"])
    candidate_age = int(candidate["age_band"])
    score = 0.0
    if target.get("category") == candidate.get("category"):
        score += 5.0
    if target.get("_shape") and target.get("_shape") == candidate.get("_shape"):
        score += 4.0
    score += 2.0 * len(shared_tags)
    score += max(0.0, 2.0 - abs(target_age - candidate_age) * 0.5)
    if clip_similarity is not None:
        pair = (target["answer"].lower(), candidate["answer"].lower())
        score += 10.0 * clip_similarity.get(pair, 0.0)
    return score


def aoa_value(row: dict[str, str]) -> float:
    return parse_float(row.get("_aoa", ""), default=parse_float(row.get("age_band", ""), default=99.0))


def clip_or_heuristic_similarity(
    target: dict[str, str],
    candidate: dict[str, str],
    clip_similarity: dict[tuple[str, str], float] | None,
) -> float:
    if clip_similarity is None:
        return similarity_score(target, candidate, clip_similarity=None)
    pair = (target["answer"].lower(), candidate["answer"].lower())
    return clip_similarity.get(pair, 0.0)


def aoa_window_candidates(
    target: dict[str, str],
    candidates: list[dict[str, str]],
    max_aoa_diff: float = 3.0,
) -> list[dict[str, str]]:
    target_aoa = aoa_value(target)
    for window in [max_aoa_diff, 4.0, 5.0, 99.0]:
        filtered = [
            candidate
            for candidate in candidates
            if candidate["answer"] != target["answer"] and abs(aoa_value(candidate) - target_aoa) <= window
        ]
        if len(filtered) >= 3:
            return filtered
    return candidates


def middle_similarity_candidates(
    scored: list[tuple[float, float, dict[str, str]]],
) -> list[tuple[float, float, dict[str, str]]]:
    if len(scored) < 3:
        return scored
    values = sorted(score for score, _, _ in scored)
    low_q = values[int((len(values) - 1) * 0.33)]
    high_q = values[int((len(values) - 1) * 0.66)]
    return [(score, tie, row) for score, tie, row in scored if low_q <= score <= high_q] or scored


def pick_similarity_tiered_distractors(
    target: dict[str, str],
    candidates: list[dict[str, str]],
    rng: random.Random,
    clip_similarity: dict[tuple[str, str], float] | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    windowed = aoa_window_candidates(target, candidates)
    scored = [
        (clip_or_heuristic_similarity(target, candidate, clip_similarity), rng.random(), candidate)
        for candidate in windowed
    ]
    if len(scored) < 3:
        raise SystemExit(f"Could not find enough distractor candidates for {target['answer']}")

    high = max(scored, key=lambda item: (item[0], item[1]))[2]
    remaining = [item for item in scored if item[2]["answer"] != high["answer"]]
    low = min(remaining, key=lambda item: (item[0], item[1]))[2]
    remaining = [item for item in remaining if item[2]["answer"] != low["answer"]]

    same_animacy = [
        item
        for item in remaining
        if item[2].get("_animacy", "") == target.get("_animacy", "")
        or item[2].get("category") == target.get("category")
    ]
    medium_pool = middle_similarity_candidates(same_animacy or remaining)
    medium = rng.choice([item[2] for item in medium_pool])
    return high, medium, low


def build_clip_similarity(
    target_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    enabled: bool,
    device: str,
    batch_size: int = 64,
) -> dict[tuple[str, str], float] | None:
    if not enabled:
        return None
    try:
        import clip  # type: ignore[import-not-found]
        import torch
    except ImportError as exc:
        raise SystemExit(
            "CLIP is required for --use-clip-similarity. Install with "
            "`pip install git+https://github.com/openai/CLIP.git`, or pass "
            "`--no-use-clip-similarity`."
        ) from exc

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=resolved_device)
    labels = list(
        dict.fromkeys(
            [row["answer"] for row in target_rows] + [row["answer"] for row in candidate_rows]
        )
    )
    prompts = [f"a clear photo of {label}" for label in labels]
    feature_batches = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            tokens = clip.tokenize(prompts[start : start + batch_size], truncate=True).to(
                resolved_device
            )
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            feature_batches.append(features.detach().cpu())
    features = torch.cat(feature_batches, dim=0)
    label_to_index = {label.lower(): index for index, label in enumerate(labels)}

    out: dict[tuple[str, str], float] = {}
    for target in target_rows:
        target_label = target["answer"].lower()
        target_index = label_to_index[target_label]
        for candidate in candidate_rows:
            candidate_label = candidate["answer"].lower()
            if target_label == candidate_label:
                continue
            candidate_index = label_to_index[candidate_label]
            out[(target_label, candidate_label)] = float(
                features[target_index] @ features[candidate_index]
            )
    return out


def normalize_source_row(row: dict[str, str], index: int, n_items: int) -> dict[str, str]:
    answer = str(row["answer"]).strip()
    age_band = min(int(row.get("age_band") or infer_age_band(index, n_items)), 11)
    return {
        "task": TASK_ID,
        "item_uid": f"synthetic_vocab__{slugify(answer)}",
        "answer": answer,
        "response_alternatives": "",
        "prompt_phrase": answer,
        "full_prompt": FULL_PROMPT_TEMPLATE.format(word=answer),
        "trial_type": row.get("trial_type", "test") or "test",
        "age_band": str(age_band),
        "category": row.get("category", "object") or "object",
        "hardness": row.get("hardness", ""),
        "_tags": row.get("_tags", ""),
        "_shape": row.get("_shape", ""),
        "_aoa": row.get("_aoa", ""),
        "_nameability": row.get("_nameability", ""),
        "_animacy": row.get("_animacy", ""),
    }


def normalize_rows(
    rows: list[dict[str, str]],
    n_items: int,
    seed: int,
    use_clip_similarity: bool,
    clip_device: str,
    seed_item_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    normalized_pool = [normalize_source_row(row, index, n_items) for index, row in enumerate(rows)]
    seed_item_rows = seed_item_rows or []
    seed_answers = {row["answer"].strip().lower() for row in seed_item_rows}
    reserved_option_answers = {
        term.strip().lower()
        for row in seed_item_rows
        for term in row_option_terms(row)
    }
    if len(seed_item_rows) > n_items:
        raise SystemExit(
            f"Seed item manifest has {len(seed_item_rows)} rows; requested only {n_items}."
        )
    if seed_item_rows:
        additional_needed = n_items - len(seed_item_rows)
        additional = [
            row
            for row in normalized_pool
            if row["answer"].strip().lower() not in reserved_option_answers
            and row["answer"].strip().lower() not in seed_answers
        ][:additional_needed]
        selected = seed_item_rows + additional
    else:
        selected = normalized_pool[:n_items]
    if len(selected) != n_items:
        raise SystemExit(f"Requested {n_items} rows but only found {len(selected)}")

    rng = random.Random(seed)
    selected_answers = {row["answer"] for row in selected}
    selected_answers_lc = {answer.strip().lower() for answer in selected_answers}
    distractor_pool = [
        row
        for row in normalized_pool
        if row["answer"].strip().lower() not in selected_answers_lc
        and row["answer"].strip().lower() not in reserved_option_answers
    ]
    if len(distractor_pool) < n_items * 3:
        raise SystemExit(
            f"Only {len(distractor_pool)} non-target distractor candidates available; "
            f"need {n_items * 3} for unique distractor images."
        )
    rows_needing_distractors = [
        row for row in selected if row["answer"].strip().lower() not in seed_answers
    ]
    clip_similarity = build_clip_similarity(
        target_rows=rows_needing_distractors,
        candidate_rows=distractor_pool,
        enabled=use_clip_similarity,
        device=clip_device,
    )
    used_distractor_sets: set[tuple[str, str, str]] = set()
    used_distractor_answers: set[str] = set(reserved_option_answers)
    for row in seed_item_rows:
        used_distractor_sets.add(
            tuple(sorted(part.strip() for part in row["response_alternatives"].split(",")))
        )
    for row in rows_needing_distractors:
        candidates = [
            candidate
            for candidate in distractor_pool
            if candidate["answer"].strip().lower() not in used_distractor_answers
        ]
        high, medium, low = pick_similarity_tiered_distractors(row, candidates, rng, clip_similarity)
        distractors = [high, medium, low]

        if len(distractors) != 3:
            raise SystemExit(f"Could not choose three distractors for {row['answer']}")
        if tuple(sorted(d["answer"] for d in distractors)) in used_distractor_sets:
            raise SystemExit(f"Repeated distractor set for {row['answer']}")
        used_distractor_sets.add(tuple(sorted(d["answer"] for d in distractors)))
        used_distractor_answers.update(d["answer"].strip().lower() for d in distractors)
        row["high_similarity_distractor"] = high["answer"]
        row["medium_similarity_distractor"] = medium["answer"]
        row["low_similarity_distractor"] = low["answer"]
        row["similar_distractor"] = high["answer"]
        row["response_alternatives"] = ",".join(d["answer"] for d in distractors)
        row["_option_rows"] = distractors
    return selected


def iter_option_terms(rows: Iterable[dict[str, str]]) -> Iterable[str]:
    for row in rows:
        yield str(row["answer"]).strip()
        for term in str(row["response_alternatives"]).split(","):
            term = term.strip()
            if term:
                yield term


def term_metadata(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    metadata = {}
    for row in rows:
        metadata[row["answer"].strip().lower()] = {
            "category": row.get("category", "object") or "object",
            "age_band": row.get("age_band", "11") or "11",
            "hardness": row.get("hardness", ""),
        }
        for option_row in row.get("_option_rows", []):
            metadata[option_row["answer"].strip().lower()] = {
                "category": option_row.get("category", "object") or "object",
                "age_band": option_row.get("age_band", "11") or "11",
                "hardness": option_row.get("hardness", ""),
            }
    return metadata


def build_image_prompt(term: str, category: str, age_band: int) -> str:
    note = GLOBAL_FRIENDLY_NOTES.get(term.lower(), "")
    difficulty = (
        "very familiar to young children"
        if age_band <= 5
        else "familiar to children"
        if age_band <= 8
        else "clear and recognizable for older children"
    )
    return " ".join(
        part
        for part in [
            f"Realistic color photograph of a single {term}.",
            f"Make the subject {difficulty} and globally recognizable.",
            "Center the subject, use clear lighting, and keep it easy to identify.",
            "Use a plain or natural uncluttered background.",
            "Make it look like an ordinary camera photo with only the subject and background.",
            note,
        ]
        if part
    )


def build_prompt_records(rows: list[dict[str, str]], visual_dir: Path) -> list[PromptRecord]:
    metadata = term_metadata(rows)
    records = []
    seen = set()
    for term in iter_option_terms(rows):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        meta = metadata.get(key, {})
        category = str(meta.get("category", "object"))
        age_band = int(meta.get("age_band", "11") or 11)
        slug = slugify(term)
        records.append(
            PromptRecord(
                term=term,
                slug=slug,
                category=category,
                age_band=age_band,
                prompt=build_image_prompt(term, category, age_band),
                output_path=str(visual_dir / f"{slug}.png"),
            )
        )
    return sorted(records, key=lambda record: record.slug)


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "item_uid",
        "answer",
        "response_alternatives",
        "prompt_phrase",
        "full_prompt",
        "trial_type",
        "age_band",
        "category",
        "hardness",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_translations(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item_uid", "task", "language", "prompt_phrase", "full_prompt"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for lang, template in [("en", PROMPT_EN), ("de", PROMPT_DE), ("es", PROMPT_ES)]:
                writer.writerow(
                    {
                        "item_uid": row["item_uid"],
                        "task": TASK_ID,
                        "language": lang,
                        "prompt_phrase": row["answer"],
                        "full_prompt": template.format(word=row["answer"]),
                    }
                )


def write_prompt_records(records: list[PromptRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")


def write_distractor_plan(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "item_uid",
        "answer",
        "high_similarity_distractor",
        "medium_similarity_distractor",
        "low_similarity_distractor",
        "high_similarity_aoa",
        "medium_similarity_aoa",
        "low_similarity_aoa",
        "age_band",
        "category",
        "hardness",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            option_rows = row.get("_option_rows", [])
            writer.writerow(
                {
                    "item_uid": row["item_uid"],
                    "answer": row["answer"],
                    "high_similarity_distractor": row.get("high_similarity_distractor", ""),
                    "medium_similarity_distractor": row.get("medium_similarity_distractor", ""),
                    "low_similarity_distractor": row.get("low_similarity_distractor", ""),
                    "high_similarity_aoa": option_rows[0].get("_aoa", "") if option_rows else "",
                    "medium_similarity_aoa": option_rows[1].get("_aoa", "") if len(option_rows) > 1 else "",
                    "low_similarity_aoa": option_rows[2].get("_aoa", "") if len(option_rows) > 2 else "",
                    "age_band": row["age_band"],
                    "category": row["category"],
                    "hardness": row.get("hardness", ""),
                }
            )


def write_report(rows: list[dict[str, str]], records: list[PromptRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_age = Counter(row["age_band"] for row in rows)
    by_category = Counter(row["category"] for row in rows)
    distractor_sets = {
        tuple(sorted(part.strip() for part in row["response_alternatives"].split(",")))
        for row in rows
    }
    report = {
        "task_id": TASK_ID,
        "n_items": len(rows),
        "n_unique_image_terms": len(records),
        "n_unique_distractor_sets": len(distractor_sets),
        "target_selection_policy": (
            "THINGS/AoA candidates are filtered for child safety, original-target exclusion, "
            "and AoA <= 11, then sampled evenly from low/mid/high AoA terciles with a "
            "nameability-biased deterministic seed."
        ),
        "distractor_policy": {
            "per_item": (
                "Each item receives a unique 3-distractor set drawn from non-target "
                "candidate terms."
            ),
            "image_uniqueness": (
                "Distractor terms are not reused across items and do not overlap target "
                "answers, yielding one unique image prompt per option placement."
            ),
            "similarity_tiers": (
                "For each target, candidates are restricted to a nearby AoA window when "
                "possible. The three distractors are selected as high, medium, and low "
                "text-similarity options. CLIP text embeddings are used when enabled; "
                "otherwise a category/tag/age heuristic is used."
            ),
        },
        "age_band_counts": dict(sorted(by_age.items(), key=lambda item: int(item[0]))),
        "category_counts": dict(sorted(by_category.items())),
        "notes": [
            "Photo-like generated-image workflow.",
            "Image prompts explicitly prohibit text, letters, numbers, logos, signs, labels, captions, watermarks, icons, cartoons, and clip art.",
            "Rows are clamped to age bands 3-11 for this rebuild.",
            "Run validation and manual review before using generated images for scoring.",
        ],
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def generate_one_openai_image(record: PromptRecord, model: str, size: str) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required when --generate-images is set.")

    response = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "prompt": record.prompt,
            "size": size,
            "n": 1,
        },
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Image generation failed for {record.term}: {response.text}")
    payload = response.json()
    item = payload["data"][0]
    output = Path(record.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if "b64_json" in item:
        output.write_bytes(base64.b64decode(item["b64_json"]))
    elif "url" in item:
        image_response = requests.get(item["url"], timeout=180)
        image_response.raise_for_status()
        output.write_bytes(image_response.content)
    else:
        raise RuntimeError(f"Image response for {record.term} had no b64_json or url.")


def generate_one_gemini_image(record: PromptRecord, model: str) -> None:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required when --provider gemini is set."
        )

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "instances": [{"prompt": record.prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1",
                "personGeneration": "dont_allow",
            },
        },
        timeout=180,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Gemini image generation failed for {record.term}: {response.text}")

    payload = response.json()
    predictions = payload.get("predictions") or []
    if not predictions:
        raise RuntimeError(f"Gemini image response for {record.term} had no predictions.")

    encoded = predictions[0].get("bytesBase64Encoded")
    if not encoded:
        raise RuntimeError(
            f"Gemini image response for {record.term} had no bytesBase64Encoded field."
        )

    output = Path(record.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(encoded))


def generate_images(
    records: list[PromptRecord],
    provider: str,
    model: str,
    size: str,
    sleep_seconds: float,
    overwrite: bool,
    continue_on_error: bool,
) -> None:
    for index, record in enumerate(records, start=1):
        output = Path(record.output_path)
        if output.exists() and not overwrite:
            print(f"[{index}/{len(records)}] exists: {output.name}")
            continue
        print(f"[{index}/{len(records)}] generating: {record.term}")
        try:
            if provider == "gemini":
                generate_one_gemini_image(record, model=model)
            else:
                generate_one_openai_image(record, model=model, size=size)
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"[{index}/{len(records)}] failed: {record.term}: {exc}", file=sys.stderr)
        time.sleep(sleep_seconds)


def average_hash(path: Path) -> int:
    image = Image.open(path).convert("L").resize((16, 16))
    pixels = list(image.getdata())
    avg = sum(pixels) / len(pixels)
    return int("".join("1" if pixel > avg else "0" for pixel in pixels), 2)


def validate_assets(rows: list[dict[str, str]], records: list[PromptRecord]) -> list[str]:
    errors = []
    if any(row["task"] != TASK_ID for row in rows):
        errors.append("Manifest contains rows that are not synthetic-vocab.")
    distractor_sets = []
    for row in rows:
        distractors = [part.strip() for part in row["response_alternatives"].split(",") if part.strip()]
        if len(distractors) != 3:
            errors.append(f"{row['item_uid']} does not have exactly three distractors.")
        if len(set(distractors)) != len(distractors):
            errors.append(f"{row['item_uid']} repeats a distractor.")
        if row["answer"] in distractors:
            errors.append(f"{row['item_uid']} includes the answer as a distractor.")
        if row.get("similar_distractor") and row["similar_distractor"] not in distractors:
            errors.append(f"{row['item_uid']} similar distractor is not in response_alternatives.")
        distractor_sets.append(tuple(sorted(distractors)))
    if len(set(distractor_sets)) != len(distractor_sets):
        errors.append("At least two items share the same 3-distractor set.")

    terms = {term.lower(): term for term in iter_option_terms(rows)}
    image_by_term = {record.term.lower(): Path(record.output_path) for record in records}
    missing_records = sorted(set(terms) - set(image_by_term))
    if missing_records:
        errors.append(f"Missing prompt records for terms: {', '.join(missing_records[:20])}")

    missing_images = [record.term for record in records if not Path(record.output_path).exists()]
    if missing_images:
        errors.append(
            f"Missing {len(missing_images)} generated image files. First missing: {', '.join(missing_images[:20])}"
        )
        return errors

    hashes: dict[str, list[str]] = defaultdict(list)
    perceptual = []
    for record in records:
        path = Path(record.output_path)
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(path.name)
        perceptual.append((record.term, average_hash(path)))
        try:
            with Image.open(path) as image:
                if min(image.size) < 512:
                    errors.append(f"{path.name} is smaller than 512px on one side: {image.size}")
        except Exception as exc:
            errors.append(f"Could not read {path.name}: {exc}")

    duplicate_groups = [names for names in hashes.values() if len(names) > 1]
    if duplicate_groups:
        errors.append(f"Exact duplicate image groups found: {duplicate_groups[:5]}")

    near_pairs = []
    for i, (term_a, hash_a) in enumerate(perceptual):
        for term_b, hash_b in perceptual[i + 1 :]:
            if (hash_a ^ hash_b).bit_count() <= 4:
                near_pairs.append((term_a, term_b))
                if len(near_pairs) >= 10:
                    break
        if len(near_pairs) >= 10:
            break
    if near_pairs:
        errors.append(f"Near-duplicate candidates need review: {near_pairs}")

    return errors


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file.resolve())
    assets_root = args.assets_root.resolve()
    default_source_manifest = assets_root / args.source_version / "manifest.csv"
    fallback_source_manifest = Path("data/assets/manifest.csv").resolve()
    source_manifest = (
        args.source_manifest.resolve()
        if args.source_manifest is not None
        else default_source_manifest
        if default_source_manifest.exists()
        else fallback_source_manifest
    )
    version_root = assets_root / args.version
    visual_dir = version_root / "visual" / "vocab"
    seed_item_rows = load_seed_item_rows(
        args.seed_items_manifest.resolve() if args.seed_items_manifest is not None else None
    )
    reserved_seed_terms = {
        term.strip().lower()
        for row in seed_item_rows
        for term in row_option_terms(row)
    }
    seed_item_answers = {row["answer"].strip().lower() for row in seed_item_rows}

    input_csvs = [
        args.things_meta_csv.resolve(),
        args.aoa_csv.resolve(),
        args.original_targets_csv.resolve(),
    ]
    legacy_lexicon_script = Path("scripts/experimental/generate_vocab_task.py").resolve()
    if all(path.exists() for path in input_csvs):
        source_rows = read_things_lexicon(
            things_meta_csv=input_csvs[0],
            aoa_csv=input_csvs[1],
            original_targets_csv=input_csvs[2],
            n_items=args.n_items,
            seed=args.seed,
            seed_targets_manifest=(
                args.seed_targets_manifest.resolve()
                if args.seed_targets_manifest is not None
                else args.seed_items_manifest.resolve()
                if args.seed_items_manifest is not None
                else None
            ),
            excluded_labels=reserved_seed_terms - seed_item_answers,
        )
    elif args.source_manifest is None and legacy_lexicon_script.exists():
        source_rows = read_legacy_lexicon(legacy_lexicon_script, n_items=args.n_items)
    else:
        source_rows = read_manifest(source_manifest)
    rows = normalize_rows(
        source_rows,
        n_items=args.n_items,
        seed=args.seed,
        use_clip_similarity=args.use_clip_similarity and all(path.exists() for path in input_csvs),
        clip_device=args.clip_device,
        seed_item_rows=seed_item_rows,
    )
    records = build_prompt_records(rows, visual_dir=visual_dir)

    write_manifest(rows, version_root / "manifest.csv")
    write_translations(rows, version_root / "translations" / "item-bank-translations.csv")
    write_prompt_records(records, version_root / "metadata" / "image_prompts.jsonl")
    write_distractor_plan(rows, version_root / "metadata" / "distractor_plan.csv")
    write_report(rows, records, version_root / "metadata" / "vocab_generation_report.json")

    if args.generate_images:
        model = (
            DEFAULT_GEMINI_MODEL
            if args.provider == "gemini" and args.model == DEFAULT_MODEL
            else args.model
        )
        generation_records = records[: args.limit_images] if args.limit_images else records
        generate_images(
            generation_records,
            provider=args.provider,
            model=model,
            size=args.size,
            sleep_seconds=args.sleep_seconds,
            overwrite=args.overwrite_images,
            continue_on_error=args.continue_on_error,
        )
        if args.limit_images:
            print(f"Generated pilot subset: {len(generation_records)} of {len(records)} images.")

    if args.validate:
        errors = validate_assets(rows, records)
        if errors:
            for error in errors:
                print(f"VALIDATION: {error}", file=sys.stderr)
            return 1
        print("Validation passed.")

    print(f"Wrote manifest and prompts for {len(rows)} items to {version_root}")
    print(f"Unique image terms: {len(records)}")
    if not args.generate_images:
        if args.provider == "gemini":
            print(
                "Dry run only. Set GEMINI_API_KEY or GOOGLE_API_KEY and pass "
                "--provider gemini --generate-images to create PNGs."
            )
        else:
            print(
                "Dry run only. Set OPENAI_API_KEY and pass --generate-images to create PNGs."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
