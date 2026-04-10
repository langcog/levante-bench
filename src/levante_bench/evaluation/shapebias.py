"""Shapebias evaluation helpers integrated with levante-bench runner."""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

from PIL import Image

PROMPT_TEMPLATES = {
    "noun_label": (
        "The first image is a {word}. "
        "Which of the following two images (1 or 2) is also a {word}? "
        "Your response must be exactly one character: 1 or 2. No other text."
    ),
    "no_word_category": (
        "See this object in the first image. "
        "Can you find another one of the two (1 or 2)? "
        "Your response must be exactly one character: 1 or 2. No other text."
    ),
    "binary_yes_no": (
        "The first image is a {word}. "
        "The second image is one candidate object. "
        "Is the second image the same object category as the first image? "
        "Your response must be exactly one word: YES or NO. No other text."
    ),
    "binary_yes_no_conservative": (
        "You see 2 images: Image 1 is the reference object and Image 2 is one candidate object. "
        "Question: Is Image 2 the same object category as Image 1? "
        "Rules: answer with exactly one token YES or NO. If uncertain, answer NO."
    ),
    "binary_score": (
        "The first image is a {word}. "
        "The second image is one candidate object. "
        "Rate how likely the second image is the same object category as the first image, "
        "from 0 to 100. "
        "Your response must be exactly: SCORE=<integer 0-100>. No other text."
    ),
    "rank_forced": (
        "Image 1 is the reference object. Image 2 and Image 3 are candidates. "
        "Rank the candidates by similarity to Image 1. "
        "Your response must be exactly: BETTER=<1 or 2>; WORSE=<1 or 2>. "
        "BETTER and WORSE must be different."
    ),
}

WORD_PAIRS = [
    ("shiple", "afnafq", 6),
    ("clapher", "ieyiccw", 7),
    ("plailass", "orvufaig", 8),
    ("procation", "qahftrxck", 9),
    ("adinefults", "cgchqjjfgy", 10),
]

CSV_FIELDS = [
    "model",
    "model_name",
    "stim_id",
    "word",
    "word_type",
    "word_length",
    "prompt_condition",
    "decision_mode",
    "swap_correct",
    "ordering",
    "order_method",
    "a_is",
    "b_is",
    "raw_text",
    "parsed_answer",
    "choice",
    "generation_time_s",
    "num_tokens_generated",
    "attempts",
    "repeat",
    "temperature",
    "eval_mode",
    "stim_pkg",
    "stim_set",
    "human_word_seed",
    "stimulus_shuffle_condition",
    "word_mode",
    "word_min_len",
    "word_max_len",
    "sudo_threshold",
    "trial_limit",
]


def load_words() -> list[dict]:
    words: list[dict] = []
    for sudo, rnd, length in WORD_PAIRS:
        words.append({"name": sudo, "type": "sudo", "length": length})
        words.append({"name": rnd, "type": "random", "length": length})
    return words


def make_prompt(word: str, prompt_condition: str) -> str:
    template = PROMPT_TEMPLATES[prompt_condition]
    if "{word}" in template:
        return template.format(word=word)
    return template


def parse_answer(raw_text: str) -> str | None:
    has_1 = "1" in raw_text
    has_2 = "2" in raw_text
    if has_1 and has_2:
        return None
    if has_1:
        return "1"
    if has_2:
        return "2"
    return None


def parse_yes_no(raw_text: str) -> str | None:
    txt = (raw_text or "").lower()
    has_yes = "yes" in txt
    has_no = "no" in txt
    if has_yes and has_no:
        return None
    if has_yes:
        return "yes"
    if has_no:
        return "no"
    return None


def parse_score_0_100(raw_text: str) -> int | None:
    txt = (raw_text or "").strip().lower()
    m = re.search(r"score\s*=\s*(\d{1,3})\b", txt)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v
    m2 = re.search(r"\b(\d{1,3})\b", txt)
    if not m2:
        return None
    v = int(m2.group(1))
    return v if 0 <= v <= 100 else None


def parse_rank_forced(raw_text: str) -> str | None:
    txt = (raw_text or "").strip().lower()
    m = re.search(r"better\s*=\s*([12])\s*;\s*worse\s*=\s*([12])", txt)
    if not m:
        return None
    better, worse = m.group(1), m.group(2)
    if better == worse:
        return None
    return better


def _run_generate(model, image_paths: list[str], prompt: str, max_new_tokens: int) -> dict:
    t0 = time.perf_counter()
    raw = model.generate(
        prompt_text=prompt,
        image_paths=image_paths,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.perf_counter() - t0
    return {
        "raw_text": str(raw or ""),
        "generation_time_s": round(elapsed, 3),
        "num_tokens_generated": None,
    }


def _run_score_choices(model, image_paths: list[str], prompt: str) -> dict | None:
    if not hasattr(model, "score_choices"):
        return None
    score_fn = getattr(model, "score_choices")
    return score_fn(image_paths=image_paths, prompt_text=prompt, choice_texts=("1", "2"))


def _finalize_result(trial: dict, raw_text: str, parsed_answer: str | None, choice: str, attempts: int, generation_time_s: float, num_tokens_generated: int | None, order_method: str) -> dict:
    predicted_label = parsed_answer if parsed_answer in {"1", "2"} else None
    return {
        "trial_id": trial["trial_id"],
        "item_uid": trial["item_uid"],
        "generated_text": raw_text,
        "predicted_label": predicted_label,
        "predicted_value": None,
        "predicted_slider_position": None,
        "reason": raw_text if parsed_answer is None else "",
        "parse_method": "shapebias_parser",
        "parse_confidence": "high" if parsed_answer is not None else "none",
        "parse_raw_candidate": parsed_answer or "",
        "correct_label": trial.get("correct_label"),
        "target_value": None,
        "slider_tolerance": None,
        "is_correct": predicted_label == trial.get("correct_label"),
        "options": trial.get("options", []),
        "option_labels": trial.get("option_labels", []),
        "model": trial.get("model_key", ""),
        "model_name": model_name_from_trial(trial),
        "stim_id": trial.get("stim_id", ""),
        "word": trial.get("word", ""),
        "word_type": trial.get("word_type", ""),
        "word_length": trial.get("word_length", 0),
        "prompt_condition": trial.get("prompt_condition", ""),
        "decision_mode": trial.get("decision_mode", "2afc"),
        "swap_correct": str(bool(trial.get("swap_correct", False))).lower(),
        "ordering": trial.get("ordering", ""),
        "order_method": order_method,
        "a_is": trial.get("a_is", ""),
        "b_is": trial.get("b_is", ""),
        "raw_text": raw_text,
        "parsed_answer": parsed_answer or "",
        "choice": choice,
        "generation_time_s": round(float(generation_time_s), 3),
        "num_tokens_generated": int(num_tokens_generated or 0),
        "attempts": attempts,
        "repeat": trial.get("repeat", 1),
        "temperature": trial.get("temperature", 0.0),
        "eval_mode": "benchmark",
        "stim_pkg": "stimuli_per_stl_packages",
        "stim_set": trial.get("stim_set", ""),
        "human_word_seed": "",
        "stimulus_shuffle_condition": "",
        "word_mode": "paired",
        "word_min_len": "",
        "word_max_len": "",
        "sudo_threshold": "",
        "trial_limit": "",
    }


def model_name_from_trial(trial: dict) -> str:
    return str(trial.get("model_hf_name", trial.get("model_key", "")))


def _shape_choice_from_answer(answer: str | None, a_is: str, b_is: str) -> str:
    if answer == "1":
        return a_is
    if answer == "2":
        return b_is
    return "unclear"


def evaluate_shapebias_trial(model, trial: dict) -> dict:
    decision_mode = str(trial.get("decision_mode", "2afc"))
    trial_eval = dict(trial)
    prompt = make_prompt(str(trial["word"]), str(trial.get("prompt_condition", "noun_label")))
    ref = str(trial["reference_image_path"])
    a_img = str(trial["a_image_path"])
    b_img = str(trial["b_image_path"])

    if decision_mode == "logit_forced_12":
        score = _run_score_choices(model, [ref, a_img, b_img], prompt)
        if score is None:
            # Graceful fallback for models without score_choices.
            decision_mode = "2afc"
            trial_eval["decision_mode"] = "2afc"
        else:
            p1, p2 = score["choice_probs"]
            l1, l2 = score["choice_logits"]
            total_t = float(score.get("generation_time_s", 0.0))
            swap_correct = bool(trial.get("swap_correct", False))
            if swap_correct:
                swapped = _run_score_choices(model, [ref, b_img, a_img], prompt)
                if swapped is not None:
                    sp1, sp2 = swapped["choice_probs"]
                    sl1, sl2 = swapped["choice_logits"]
                    total_t += float(swapped.get("generation_time_s", 0.0))
                    p_a = 0.5 * (p1 + sp2)
                    p_b = 0.5 * (p2 + sp1)
                    raw_text = (
                        f"base[p1={p1:.4f},p2={p2:.4f},l1={l1:.3f},l2={l2:.3f}] "
                        f"swap[p1={sp1:.4f},p2={sp2:.4f},l1={sl1:.3f},l2={sl2:.3f}] "
                        f"corr[p_a={p_a:.4f},p_b={p_b:.4f}]"
                    )
                else:
                    p_a, p_b = p1, p2
                    raw_text = f"p1={p1:.4f},p2={p2:.4f},l1={l1:.3f},l2={l2:.3f}"
            else:
                p_a, p_b = p1, p2
                raw_text = f"p1={p1:.4f},p2={p2:.4f},l1={l1:.3f},l2={l2:.3f}"
            if p_a > p_b:
                parsed = "1"
            elif p_b > p_a:
                parsed = "2"
            else:
                parsed = None
            choice = _shape_choice_from_answer(parsed, trial["a_is"], trial["b_is"])
            return _finalize_result(
                trial=trial_eval,
                raw_text=raw_text,
                parsed_answer=parsed,
                choice=choice,
                attempts=1,
                generation_time_s=total_t,
                num_tokens_generated=0,
                order_method="logit_forced_swap_corrected" if bool(trial.get("swap_correct", False)) else "logit_forced",
            )

    if decision_mode == "binary_pair":
        prompt = make_prompt(str(trial["word"]), str(trial.get("prompt_condition", "binary_yes_no")))
        shape_call = _run_generate(model, [ref, a_img], prompt, int(trial.get("max_new_tokens", 64)))
        tex_call = _run_generate(model, [ref, b_img], prompt, int(trial.get("max_new_tokens", 64)))
        sa = parse_yes_no(shape_call["raw_text"])
        ta = parse_yes_no(tex_call["raw_text"])
        if sa == "yes" and ta == "no":
            choice = trial["a_is"]
        elif sa == "no" and ta == "yes":
            choice = trial["b_is"]
        else:
            choice = "unclear"
        raw = f"shape_candidate={shape_call['raw_text']!r}; texture_candidate={tex_call['raw_text']!r}"
        parsed = f"shape={sa or 'none'};texture={ta or 'none'}"
        return _finalize_result(
            trial=trial_eval,
            raw_text=raw,
            parsed_answer=None,
            choice=choice,
            attempts=2,
            generation_time_s=float(shape_call["generation_time_s"]) + float(tex_call["generation_time_s"]),
            num_tokens_generated=0,
            order_method="independent_binary",
        ) | {"parsed_answer": parsed, "predicted_label": None, "is_correct": choice == "shape"}

    call = _run_generate(model, [ref, a_img, b_img], prompt, int(trial.get("max_new_tokens", 64)))
    if decision_mode == "binary_rank_forced":
        parsed = parse_rank_forced(call["raw_text"])
    elif str(trial.get("prompt_condition")) == "binary_score":
        score = parse_score_0_100(call["raw_text"])
        parsed = "1" if (score is not None and score >= 50) else ("2" if score is not None else None)
    else:
        parsed = parse_answer(call["raw_text"])
    choice = _shape_choice_from_answer(parsed, trial["a_is"], trial["b_is"])
    return _finalize_result(
        trial=trial_eval,
        raw_text=call["raw_text"],
        parsed_answer=parsed,
        choice=choice,
        attempts=1,
        generation_time_s=float(call["generation_time_s"]),
        num_tokens_generated=call.get("num_tokens_generated"),
        order_method=str(trial.get("order_method", "deterministic")),
    )


def load_stimuli_rows(stim_root: Path, stim_set: str, num_stimuli: int | None, seed: int) -> list[dict]:
    base = stim_root / stim_set
    if not base.exists():
        raise FileNotFoundError(f"Shapebias stimulus set not found: {base}")
    stim_dirs = [d for d in sorted(base.iterdir(), key=lambda p: p.name) if d.is_dir()]
    if num_stimuli is not None and num_stimuli < len(stim_dirs):
        rng = random.Random(seed)
        stim_dirs = sorted(rng.sample(stim_dirs, num_stimuli), key=lambda p: p.name)
    rows: list[dict] = []
    for d in stim_dirs:
        rows.append(
            {
                "stim_id": d.name,
                "reference_image_path": str(d / "reference.png"),
                "shape_match_path": str(d / "shape_match.png"),
                "texture_match_path": str(d / "texture_match.png"),
            }
        )
    return rows


def validate_stimulus_row(row: dict) -> None:
    for key in ("reference_image_path", "shape_match_path", "texture_match_path"):
        path = Path(str(row[key]))
        if not path.exists():
            raise FileNotFoundError(f"Missing shapebias image: {path}")
        # Fail fast on invalid images before expensive runs.
        with Image.open(path) as img:
            img.verify()
