#!/usr/bin/env python3
"""Stateful Theory-of-Mind evaluation for SmolVLM2."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run stateful ToM eval with SmolVLM2.")
    p.add_argument("--corpus-csv", type=Path, required=True, help="Path to theory-of-mind-item-bank.csv")
    p.add_argument("--output-jsonl", type=Path, required=True, help="Predictions JSONL path")
    p.add_argument("--summary-json", type=Path, required=True, help="Summary JSON path")
    p.add_argument(
        "--model-id",
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device")
    p.add_argument(
        "--memory-mode",
        default="oracle",
        choices=["none", "oracle", "model", "state_oracle", "state_model"],
        help=(
            "none=current question only; oracle=carry prior story + gold prior answers; "
            "model=carry prior story + model prior answers; "
            "state_oracle/state_model=structured memory state."
        ),
    )
    p.add_argument(
        "--memory-format",
        default="detailed",
        choices=["detailed", "compact", "belief_state"],
        help="How prior question-answer history is stored in context.",
    )
    p.add_argument(
        "--history-window",
        type=int,
        default=8,
        help="How many prior QA entries to keep when memory-format=compact.",
    )
    p.add_argument("--max-items", type=int, default=None, help="Evaluate first N test items")
    p.add_argument("--max-context-lines", type=int, default=80, help="Limit retained context lines")
    p.add_argument("--max-new-tokens", type=int, default=8, help="Generation length")
    p.add_argument(
        "--shuffle-options",
        action="store_true",
        help="Shuffle answer options per item before prompting.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for option shuffling.")
    p.add_argument(
        "--reasoning-instruction",
        default="standard",
        choices=["standard", "facts_only"],
        help="Prompt instruction style.",
    )
    p.add_argument(
        "--trial-type-filter",
        default=None,
        help="Optional comma-separated list of trial_type values to evaluate.",
    )
    p.add_argument(
        "--template-style",
        default="trial_aware",
        choices=["standard", "trial_aware"],
        help="Prompt template style. trial_aware adds question-type guidance.",
    )
    p.add_argument(
        "--two-stage",
        action="store_true",
        help="Use a two-stage flow: draft reasoning notes, then answer.",
    )
    p.add_argument(
        "--analysis-max-new-tokens",
        type=int,
        default=64,
        help="Generation length for stage-1 notes when --two-stage is enabled.",
    )
    p.add_argument(
        "--sequence-trace-csv",
        type=Path,
        default=None,
        help="Optional CSV path to write processed row order for auditability.",
    )
    p.add_argument(
        "--strict-order-check",
        action="store_true",
        help="Fail if test_response appears in a block before any instruction context.",
    )
    return p.parse_args()


def _load_model(model_id: str, device: str):
    import torch
    import transformers
    from transformers import AutoProcessor

    use_fp16 = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    model_kwargs: dict[str, Any] = {"dtype": torch.float16 if use_fp16 else torch.float32}
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForVision2Seq")
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    if device in ("cpu", "cuda"):
        model = model.to(device)
    return model, processor


def _split_alternatives(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _extract_letter(text: str, n_options: int) -> str | None:
    m = re.search(r"\b([A-Z])\b", text.upper())
    if not m:
        return None
    letter = m.group(1)
    if letter in LETTERS[:n_options]:
        return letter
    return None


def _build_prompt(
    row: dict[str, str],
    options: list[str],
    context_lines: list[str],
    memory_mode: str,
    reasoning_instruction: str,
    template_style: str,
    stage1_notes: str | None = None,
) -> str:
    lines = ["You are answering a story understanding question."]
    if reasoning_instruction == "facts_only":
        lines.append(
            "Use only story facts and stated beliefs from context. Do not add outside assumptions."
        )
    lines.append("Return only the option letter (A, B, C, ...).")
    if memory_mode != "none" and context_lines:
        lines.append("Story so far:")
        for ln in context_lines:
            lines.append(f"- {ln}")
    prompt = (row.get("prompt") or "").strip()
    if prompt:
        lines.append(f"Question: {prompt}")
    trial_type = (row.get("trial_type") or "").strip()
    if trial_type:
        lines.append(f"Question type: {trial_type}")
    if template_style == "trial_aware":
        lines.append(_trial_type_guidance(trial_type))
    if stage1_notes:
        lines.append("Reasoning notes:")
        lines.append(stage1_notes)
        lines.append("Now choose the best option using the notes and context.")
    lines.append("Options:")
    for i, opt in enumerate(options):
        lines.append(f"{LETTERS[i]}. {opt}")
    return "\n".join(lines)


def _trial_type_guidance(trial_type: str) -> str:
    t = (trial_type or "").lower()
    if "false_belief" in t:
        return "Guidance: Answer from the target character's belief perspective, not hidden reality."
    if "reality_check" in t:
        return "Guidance: Answer from actual world state, not mistaken beliefs."
    if "emotion" in t:
        return "Guidance: Infer the target character's likely feeling from events."
    if "attribution" in t:
        return "Guidance: Focus on whether intent appears accidental or on purpose."
    if "action" in t:
        return "Guidance: Choose the action option most consistent with the scenario."
    if "reference" in t:
        return "Guidance: Resolve references using the described context."
    return "Guidance: Use only story evidence relevant to the question."


def _is_state_mode(memory_mode: str) -> bool:
    return memory_mode in {"state_oracle", "state_model"}


def _new_state() -> dict[str, Any]:
    return {
        "facts": [],
        "beliefs": [],
        "reality": [],
        "emotions": [],
        "actions": [],
        "known_names": [],
        "agent_memory": {},
    }


def _push_limited(values: list[str], entry: str, limit: int) -> None:
    values.append(entry)
    if limit > 0 and len(values) > limit:
        del values[:-limit]


def _extract_names(text: str) -> list[str]:
    # Keep this lightweight and deterministic: title-cased tokens not in stoplist.
    stop = {
        "This",
        "Then",
        "And",
        "Now",
        "When",
        "But",
        "Nice",
        "Here",
        "Look",
        "So",
        "Can",
        "Is",
        "Are",
        "Do",
        "Does",
        "How",
        "What",
        "Who",
        "Where",
        "Mother",
        "Dad",
        "Mom",
    }
    names = []
    for token in re.findall(r"\b[A-Z][a-z]+\b", text):
        if token not in stop:
            names.append(token)
    # Preserve order while deduping.
    out = []
    seen = set()
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def _ensure_agent(state: dict[str, Any], name: str) -> None:
    if name not in state["agent_memory"]:
        state["agent_memory"][name] = {"knows": [], "beliefs": [], "emotions": []}
    if name not in state["known_names"]:
        state["known_names"].append(name)


def _infer_witnesses(state: dict[str, Any], prompt: str, names_in_prompt: list[str]) -> list[str]:
    text = prompt.lower()
    if "no one was looking" in text:
        return []
    witnesses = list(names_in_prompt) if names_in_prompt else list(state["known_names"])
    # If text explicitly says someone didn't see/know, exclude that character from this event.
    for name in list(witnesses):
        n = name.lower()
        if (
            f"{n} didn't see" in text
            or f"{n} did not see" in text
            or f"{n} doesn't know" in text
            or f"{n} does not know" in text
            or f"{n} is still outside" in text
            or f"{n} was outside" in text
        ):
            witnesses = [w for w in witnesses if w != name]
    return witnesses


def _update_state_with_instruction(state: dict[str, Any], prompt: str, limit: int) -> None:
    _push_limited(state["facts"], prompt, limit)
    names = _extract_names(prompt)
    for name in names:
        _ensure_agent(state, name)
    witnesses = _infer_witnesses(state, prompt, names)
    for w in witnesses:
        _push_limited(state["agent_memory"][w]["knows"], prompt, limit)
    # Basic "X tells Y" transfer: Y learns spoken content.
    m = re.search(r"\b([A-Z][a-z]+)\s+tells?\s+([A-Z][a-z]+)\b", prompt)
    if m:
        speaker, listener = m.group(1), m.group(2)
        _ensure_agent(state, speaker)
        _ensure_agent(state, listener)
        _push_limited(state["agent_memory"][listener]["knows"], prompt, limit)


def _guess_subject(prompt: str, known_names: list[str]) -> str | None:
    for name in _extract_names(prompt):
        if name in known_names:
            return name
    return known_names[0] if known_names else None


def _update_state_with_answer(
    state: dict[str, Any],
    *,
    trial_type: str,
    prompt: str,
    answer_text: str,
    limit: int,
) -> None:
    entry = f"{prompt} => {answer_text}"
    t = trial_type.lower()
    if "false_belief" in t:
        _push_limited(state["beliefs"], entry, limit)
        subj = _guess_subject(prompt, state["known_names"])
        if subj:
            _ensure_agent(state, subj)
            _push_limited(state["agent_memory"][subj]["beliefs"], entry, limit)
    elif "reality_check" in t or "reference" in t:
        _push_limited(state["reality"], entry, limit)
    elif "emotion" in t:
        _push_limited(state["emotions"], entry, limit)
        subj = _guess_subject(prompt, state["known_names"])
        if subj:
            _ensure_agent(state, subj)
            _push_limited(state["agent_memory"][subj]["emotions"], entry, limit)
    elif "action" in t or "attribution" in t:
        _push_limited(state["actions"], entry, limit)
    else:
        _push_limited(state["facts"], entry, limit)


def _state_to_context_lines(state: dict[str, Any], per_section: int = 4) -> list[str]:
    lines: list[str] = []
    sections = [
        ("Facts", "facts"),
        ("Beliefs", "beliefs"),
        ("Reality Checks", "reality"),
        ("Emotions", "emotions"),
        ("Actions/Attributions", "actions"),
    ]
    for label, key in sections:
        vals = state.get(key) or []
        if not vals:
            continue
        lines.append(f"{label}:")
        for v in vals[-per_section:]:
            lines.append(f"  {v}")
    if state["known_names"]:
        lines.append("Character memory:")
    for name in state["known_names"]:
        mem = state["agent_memory"].get(name, {})
        knows = mem.get("knows") or []
        beliefs = mem.get("beliefs") or []
        emotions = mem.get("emotions") or []
        if not (knows or beliefs or emotions):
            continue
        lines.append(f"- {name}:")
        for v in knows[-2:]:
            lines.append(f"    knows: {v}")
        for v in beliefs[-2:]:
            lines.append(f"    believes: {v}")
        for v in emotions[-2:]:
            lines.append(f"    feels: {v}")
    return lines


def _append_history(
    context_lines: list[str],
    *,
    prompt: str,
    trial_type: str,
    answer_text: str,
    mode: str,
    memory_format: str,
    history_window: int,
) -> list[str]:
    if memory_format == "belief_state":
        kind = (trial_type or "question").replace("_question", "")
        context_lines.append(f"Belief-state update [{kind}]: {answer_text}")
        if history_window > 0:
            return context_lines[-history_window:]
        return context_lines
    if memory_format == "compact":
        kind = trial_type or "question"
        context_lines.append(f"{kind}: {answer_text}")
        if history_window > 0:
            return context_lines[-history_window:]
        return context_lines
    # detailed
    if mode == "oracle":
        context_lines.append(f"Q: {prompt} | A: {answer_text}")
    else:
        context_lines.append(f"Q: {prompt} | Model answered: {answer_text}")
    return context_lines


def _generate_one(
    model: Any,
    processor: Any,
    prompt_text: str,
    device: str,
    max_new_tokens: int,
) -> str:
    import torch

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt")
    if device in ("cpu", "cuda"):
        inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[:, prompt_len:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def _build_stage1_prompt(
    row: dict[str, str],
    context_lines: list[str],
    memory_mode: str,
    reasoning_instruction: str,
    template_style: str,
) -> str:
    trial_type = (row.get("trial_type") or "").strip()
    prompt = (row.get("prompt") or "").strip()
    lines = ["Draft short reasoning notes for a story question."]
    if reasoning_instruction == "facts_only":
        lines.append("Use only stated story/context facts.")
    if memory_mode != "none" and context_lines:
        lines.append("Story so far:")
        for ln in context_lines:
            lines.append(f"- {ln}")
    if prompt:
        lines.append(f"Question: {prompt}")
    if trial_type:
        lines.append(f"Question type: {trial_type}")
    if template_style == "trial_aware":
        lines.append(_trial_type_guidance(trial_type))
    lines.append("Return 2 short lines:")
    lines.append("- perspective: ...")
    lines.append("- key_fact: ...")
    lines.append("Do not output any option letter.")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    model, processor = _load_model(args.model_id, args.device)
    rng = random.Random(args.seed)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    if args.sequence_trace_csv is not None:
        args.sequence_trace_csv.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_parsed = 0
    n_correct = 0
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "parsed": 0, "correct": 0})

    current_block = None
    context_lines: list[str] = []
    state = _new_state()
    seen_instruction_in_block = False
    trace_rows: list[dict[str, Any]] = []
    global_row_index = -1

    allowed_types = None
    if args.trial_type_filter:
        allowed_types = {x.strip() for x in args.trial_type_filter.split(",") if x.strip()}

    with open(args.corpus_csv, newline="", encoding="utf-8") as f_in, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as f_out:
        reader = csv.DictReader(f_in)
        for row in reader:
            global_row_index += 1
            block = (row.get("block_index") or "").strip()
            if block != current_block:
                current_block = block
                context_lines = []
                state = _new_state()
                seen_instruction_in_block = False

            stage = (row.get("assessment_stage") or "").strip().lower()
            prompt = (row.get("prompt") or "").strip()
            trial_type = (row.get("trial_type") or "").strip()
            item_uid = (row.get("item_uid") or "").strip()
            trace_rows.append(
                {
                    "row_index": global_row_index,
                    "block_index": block,
                    "assessment_stage": stage,
                    "trial_type": trial_type,
                    "item_uid": item_uid,
                }
            )

            # Keep narrative context from instruction rows.
            if stage == "instructions" and prompt:
                seen_instruction_in_block = True
                if _is_state_mode(args.memory_mode):
                    _update_state_with_instruction(state, prompt, limit=max(args.max_context_lines, 1))
                elif args.memory_format == "belief_state":
                    context_lines.append(f"Story fact: {prompt}")
                else:
                    context_lines.append(prompt)
                context_lines = context_lines[-args.max_context_lines :]
                continue

            # Evaluate only answerable test rows.
            if stage != "test_response":
                continue
            if args.strict_order_check and not seen_instruction_in_block:
                raise ValueError(
                    f"Order check failed: test_response before instructions in block={block}, row_index={global_row_index}"
                )
            if allowed_types is not None and trial_type not in allowed_types:
                continue
            answer = (row.get("answer") or "").strip()
            if not answer:
                continue
            distractors = _split_alternatives((row.get("response_alternatives") or "").strip())
            options = _dedupe_keep_order([answer] + [d for d in distractors if d != answer])
            if len(options) < 2:
                continue
            if args.shuffle_options:
                rng.shuffle(options)

            prompt_context = (
                _state_to_context_lines(state) if _is_state_mode(args.memory_mode) else context_lines
            )
            prompt_text = _build_prompt(
                row,
                options,
                prompt_context,
                memory_mode=args.memory_mode,
                reasoning_instruction=args.reasoning_instruction,
                template_style=args.template_style,
            )
            stage1_notes = None
            if args.two_stage:
                stage1_prompt = _build_stage1_prompt(
                    row,
                    prompt_context,
                    memory_mode=args.memory_mode,
                    reasoning_instruction=args.reasoning_instruction,
                    template_style=args.template_style,
                )
                stage1_notes = _generate_one(
                    model=model,
                    processor=processor,
                    prompt_text=stage1_prompt,
                    device=args.device,
                    max_new_tokens=args.analysis_max_new_tokens,
                )
                prompt_text = _build_prompt(
                    row,
                    options,
                    prompt_context,
                    memory_mode=args.memory_mode,
                    reasoning_instruction=args.reasoning_instruction,
                    template_style=args.template_style,
                    stage1_notes=stage1_notes,
                )
            pred_text = _generate_one(
                model=model,
                processor=processor,
                prompt_text=prompt_text,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            pred_letter = _extract_letter(pred_text, len(options))
            gold_index = options.index(answer)
            gold_letter = LETTERS[gold_index]
            is_correct = pred_letter == gold_letter if pred_letter is not None else False

            n_total += 1
            if pred_letter is not None:
                n_parsed += 1
            if is_correct:
                n_correct += 1
            bt = by_type[trial_type or "UNKNOWN"]
            bt["n"] += 1
            bt["parsed"] += 1 if pred_letter is not None else 0
            bt["correct"] += 1 if is_correct else 0

            out = {
                "item_uid": item_uid,
                "block_index": block,
                "trial_type": trial_type,
                "question": prompt,
                "options": options,
                "gold_answer": answer,
                "gold_letter": gold_letter,
                "pred_letter": pred_letter,
                "pred_text": pred_text,
                "stage1_notes": stage1_notes,
                "correct": is_correct,
                "memory_mode": args.memory_mode,
            }
            f_out.write(json.dumps(out, ensure_ascii=True) + "\n")

            # Update running memory after each answered question.
            if args.memory_mode == "state_oracle":
                _update_state_with_answer(
                    state,
                    trial_type=trial_type,
                    prompt=prompt,
                    answer_text=answer,
                    limit=max(args.max_context_lines, 1),
                )
            elif args.memory_mode == "state_model":
                if pred_letter is not None and pred_letter in LETTERS[: len(options)]:
                    idx = LETTERS.index(pred_letter)
                    model_ans = options[idx] if idx < len(options) else pred_text
                else:
                    model_ans = pred_text
                _update_state_with_answer(
                    state,
                    trial_type=trial_type,
                    prompt=prompt,
                    answer_text=model_ans,
                    limit=max(args.max_context_lines, 1),
                )
            elif args.memory_mode == "oracle":
                context_lines = _append_history(
                    context_lines,
                    prompt=prompt,
                    trial_type=trial_type,
                    answer_text=answer,
                    mode="oracle",
                    memory_format=args.memory_format,
                    history_window=args.history_window,
                )
            elif args.memory_mode == "model":
                if pred_letter is not None and pred_letter in LETTERS[: len(options)]:
                    idx = LETTERS.index(pred_letter)
                    model_ans = options[idx] if idx < len(options) else pred_text
                else:
                    model_ans = pred_text
                context_lines = _append_history(
                    context_lines,
                    prompt=prompt,
                    trial_type=trial_type,
                    answer_text=model_ans,
                    mode="model",
                    memory_format=args.memory_format,
                    history_window=args.history_window,
                )
            context_lines = context_lines[-args.max_context_lines :]

            if args.max_items is not None and n_total >= args.max_items:
                break

    summary_by_type = {}
    for t, v in sorted(by_type.items()):
        n = v["n"]
        summary_by_type[t] = {
            "n": n,
            "accuracy": (v["correct"] / n) if n else None,
            "parse_rate": (v["parsed"] / n) if n else None,
        }

    summary = {
        "model_id": args.model_id,
        "corpus_csv": str(args.corpus_csv),
        "output_jsonl": str(args.output_jsonl),
        "memory_mode": args.memory_mode,
        "memory_format": args.memory_format,
        "history_window": args.history_window,
        "reasoning_instruction": args.reasoning_instruction,
        "trial_type_filter": sorted(allowed_types) if allowed_types is not None else None,
        "template_style": args.template_style,
        "two_stage": args.two_stage,
        "analysis_max_new_tokens": args.analysis_max_new_tokens,
        "shuffle_options": args.shuffle_options,
        "seed": args.seed,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "accuracy_all": (n_correct / n_total) if n_total else None,
        "parse_rate": (n_parsed / n_total) if n_total else None,
        "by_trial_type": summary_by_type,
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.sequence_trace_csv is not None:
        import csv as _csv

        with open(args.sequence_trace_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(
                f,
                fieldnames=["row_index", "block_index", "assessment_stage", "trial_type", "item_uid"],
            )
            w.writeheader()
            w.writerows(trace_rows)
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
