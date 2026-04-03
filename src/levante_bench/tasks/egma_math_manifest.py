"""EgmaMath dataset. Context: optional image, options: text."""

import csv
import random
import re

from pathlib import Path

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.registry import register_task
from levante_bench.tasks.image_index import build_image_index

LABELS = ["A", "B", "C", "D"]
NUMBERLINE_CORPUS_INSTRUCTION = (
    "Here is a number line. "
    "You can move the slider forward and backward along the line. "
    "Move the slider so it is in the right place to show where the number would fit. "
    "Make sure you look at the numbers at each end when deciding where to move the slider."
)


def _is_numberline_trial(trial_type: str) -> bool:
    return "number line" in trial_type.strip().lower()


def _is_numberline_slider_trial(trial_type: str) -> bool:
    return "number line slider" in trial_type.strip().lower()


def _numberline_instruction() -> str:
    return (
        f"{NUMBERLINE_CORPUS_INSTRUCTION} "
        "For this item, do not move a slider. Choose the number that is marked on the number line."
    )


def _numberline_slider_instruction(target_value: str) -> str:
    return (
        f"{NUMBERLINE_CORPUS_INSTRUCTION} "
        f"The target number is {target_value}. "
        "Compute the relative slider position using position = (target - left_endpoint) / (right_endpoint - left_endpoint). "
        "Anchors: left endpoint -> 0.00, right endpoint -> 1.00, midpoint -> 0.50. "
        "Use 0.50 only when the target is exactly at the midpoint between the endpoints. "
        "Return only one decimal number between 0 and 1 representing the slider position. "
        "Do not return JSON, labels, or extra words."
    )


def _format_mcq_options(options: list[str]) -> str:
    lines: list[str] = []
    for i, option in enumerate(options):
        if i >= len(LABELS):
            break
        lines.append(f"{LABELS[i]}) {option}")
    return "\n".join(lines)


def _parse_slider_max(item_text: str) -> float | None:
    parts = [p.strip() for p in str(item_text).split(",") if p.strip()]
    if len(parts) < 2:
        return None
    try:
        return float(parts[1])
    except ValueError:
        return None


def _parse_slider_min(item_text: str) -> float | None:
    parts = [p.strip() for p in str(item_text).split(",") if p.strip()]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


@register_task("egma-math")
class EgmaMathDataset(VLMDataset):
    """Reads egma-math trials from manifest.csv with text answer choices."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = self.data_root / "assets" / self.version / "visual" / "egma-math"
        self.image_index = self._build_combined_image_index()
        self.item_id_by_uid = self._build_item_id_map()

    def _build_combined_image_index(self) -> dict[str, Path]:
        index: dict[str, Path] = {}
        if self.image_dir.exists():
            index.update(build_image_index(self.image_dir))

        # Optional local graphics bundle for number line items.
        project_root = Path(self.data_root).parent
        extra_dirs = [
            project_root / "local_data" / "numberline-graphics" / "egma-math",
            project_root / "local_data" / "numberline_graphics" / "egma-math",
        ]
        for d in extra_dirs:
            if d.exists():
                index.update(build_image_index(d))
        return index

    def _build_item_id_map(self) -> dict[str, str]:
        corpus_file = str(self.task_def.corpus_file or "test-combined-math-cat.csv")
        path = (
            Path(self.data_root)
            / "assets"
            / self.version
            / "corpus"
            / "egma-math"
            / corpus_file
        )
        if not path.exists():
            return {}
        out: dict[str, str] = {}
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = str(row.get("item_uid", "")).strip()
                iid = str(row.get("item_id", "")).strip()
                if uid and iid:
                    out[uid] = iid
        return out

    def _numberline_image_candidates(self, item_uid: str, item_id: str | None = None) -> list[str]:
        candidates = [item_uid]
        m = re.match(r"^math_(line|slider)_(.+)$", item_uid)
        if m:
            suffix = m.group(2)
            kind = m.group(1)
            other_kind = "slider" if kind == "line" else "line"
            suffix_hyphen = suffix.replace("_", "-")
            candidates.extend(
                [
                    # math_* forms
                    f"math_{kind}_{suffix}",
                    f"math_{other_kind}_{suffix}",
                    f"math-{kind}-{suffix_hyphen}",
                    f"math-{other_kind}-{suffix_hyphen}",
                    # bare kind forms
                    f"{kind}_{suffix}",
                    f"{other_kind}_{suffix}",
                    f"{kind}-{suffix_hyphen}",
                    f"{other_kind}-{suffix_hyphen}",
                    # raw suffix
                    suffix,
                    suffix_hyphen,
                ]
            )

            # Handle decimal-style suffixes like 045_1 => 0-45-1.
            dm = re.match(r"^0([0-9]+)_1$", suffix)
            if dm:
                frac = dm.group(1).lstrip("0") or "0"
                candidates.extend(
                    [
                        f"{kind}-0-{frac}-1",
                        f"{other_kind}-0-{frac}-1",
                        f"math-{kind}-0-{frac}-1",
                        f"math-{other_kind}-0-{frac}-1",
                        f"math_{kind}_0{frac}_1",
                        f"math_{other_kind}_0{frac}_1",
                    ]
                )

        # Also derive from corpus item_id if available (e.g., line2num-639-1000).
        if item_id:
            candidates.extend([item_id, item_id.replace(".", "_"), item_id.replace(".", "-")])
            im = re.match(r"^(?:line2num|slider)-(.+)-([0-9]+)$", item_id)
            if im:
                value = im.group(1)
                scale = im.group(2)
                value_clean = value.replace(".", "")
                candidates.extend(
                    [
                        f"math_line_{value_clean}_{scale}",
                        f"math_slider_{value_clean}_{scale}",
                        f"line-{value}-{scale}",
                        f"slider-{value}-{scale}",
                        f"line-{value_clean}-{scale}",
                        f"slider-{value_clean}-{scale}",
                    ]
                )

        # Asset bundle often prefixes filenames with task slug.
        prefixed = []
        for c in candidates:
            prefixed.append(f"egma-math-{c}")
            prefixed.append(f"egma_math-{c}")
        candidates.extend(prefixed)
        return candidates

    def _resolve_prompt_image(self, row: pd.Series, is_numberline: bool) -> Path | None:
        prompt_image = str(row.get("prompt_image", "NA")).strip()
        invalid = {"", "NA", "nan", "TODO"}
        if prompt_image not in invalid:
            path = self.image_index.get(prompt_image)
            if path is not None:
                return path

        if is_numberline:
            item_uid = str(row.get("item_uid", "")).strip()
            item_id = self.item_id_by_uid.get(item_uid)
            for key in self._numberline_image_candidates(item_uid, item_id=item_id):
                path = self.image_index.get(key)
                if path is not None:
                    return path
        return None

    def _load_manifest(self) -> pd.DataFrame:
        """Load and filter manifest rows for egma-math task."""
        manifest_path = self.data_root / "assets" / "manifest.csv"
        df = pd.read_csv(manifest_path)
        df = df[df["task"] == "egma-math"]
        include_numberline = bool(getattr(self.task_def, "include_numberline", False))
        if not include_numberline:
            # Number line items are optional in runner path; default remains excluded.
            df = df[~df["trial_type"].astype(str).str.contains("Number Line", case=False, na=False)]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        trial_type = str(row.get("trial_type", "")).strip()
        is_numberline = _is_numberline_trial(trial_type)
        is_numberline_slider = _is_numberline_slider_trial(trial_type)
        include_numberline = bool(getattr(self.task_def, "include_numberline", False))

        answer = str(row["answer"]).strip()
        alternatives = str(row["response_alternatives"] or "").split(",")
        alternatives = [a.strip() for a in alternatives if a.strip()]
        all_options: list[str] = []
        correct_label = ""
        if not (is_numberline_slider and include_numberline):
            all_options = [answer] + alternatives
            # Deterministic shuffle seeded by item_uid
            rng = random.Random(row["item_uid"])
            rng.shuffle(all_options)
            correct_idx = all_options.index(answer)
            correct_label = LABELS[correct_idx]

        # Build prompt from template.
        # egma-math uses <optionX> placeholders (not <imageX>) so we must substitute option text.
        # Some rows also include <prompt_image> (context image), which we convert to <image0>.
        prompt_phrase_s = self.translate_text(row.get("prompt_phrase"))
        prompt_template = row["full_prompt"]
        if str(prompt_template) in {"", "NA", "nan"}:
            prompt_template = row.get("prompt", "")
        prompt = self.translate_text(prompt_template).strip()

        context_image_paths = []
        if is_numberline and include_numberline:
            path = self._resolve_prompt_image(row=row, is_numberline=True)
            if path is not None:
                if "<prompt_image>" in prompt:
                    prompt = prompt.replace("<prompt_image>", "<image0>")
                else:
                    prompt = f"<image0>\n{prompt}"
                context_image_paths.append(str(path))
            elif "<prompt_image>" in prompt:
                prompt = prompt.replace("<prompt_image>", "")
        elif "<prompt_image>" in prompt:
            path = self._resolve_prompt_image(row=row, is_numberline=False)
            if path is not None:
                prompt = prompt.replace("<prompt_image>", "<image0>")
                context_image_paths.append(str(path))
            else:
                prompt_image = str(row.get("prompt_image", "NA")).strip()
                raise FileNotFoundError(
                    f"Prompt image not found for '{prompt_image}' in {self.image_dir} "
                    f"(trial {row['item_uid']})"
                )

        if is_numberline and include_numberline:
            if is_numberline_slider:
                prompt = _numberline_slider_instruction(answer)
                if context_image_paths:
                    prompt = f"<image0>\n{prompt}"
            else:
                # For number-line 4AFC, avoid semicolon-inline option formatting
                # and enforce an explicit final answer letter to reduce label bias.
                option_block = _format_mcq_options(all_options)
                prompt = (
                    f"{_numberline_instruction()}\n"
                    "<image0>\n"
                    "Which option matches the marked value on the number line?\n"
                    f"{option_block}\n"
                    "Respond with exactly one letter: A, B, C, or D."
                )

        prompt = prompt.replace("<prompt_phrase>", prompt_phrase_s)
        for i, option in enumerate(all_options, start=1):
            prompt = prompt.replace(f"<option{i}>", str(option))

        trial: dict = {
            "trial_id": row["item_uid"],
            "item_uid": row["item_uid"],
            "trial_type": trial_type,
            "prompt": prompt,
            "options": all_options,
            "option_labels": LABELS[: len(all_options)],
            "correct_label": correct_label,
            "context_image_paths": context_image_paths,
            "option_image_paths": [],
            "context_type": "image" if context_image_paths else "none",
            "option_type": "text",
            "chance_level": row.get("chance_level"),
        }

        if is_numberline_slider and include_numberline:
            slider_max = _parse_slider_max(row.get("item", ""))
            slider_min = _parse_slider_min(row.get("item", ""))
            target = float(answer)
            if slider_max is not None and slider_min is not None and slider_max > slider_min:
                # Core-tasks parity: abs(response - target) / max < 0.05
                tolerance = 0.05 * slider_max
                trial.update(
                    {
                        "answer_format": "slider_position",
                        "slider_min": slider_min,
                        "slider_max": slider_max,
                        "target_value": target,
                        "slider_tolerance": tolerance,
                    }
                )

        return trial
