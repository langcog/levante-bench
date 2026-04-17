"""Matrix Reasoning dataset. Context: optional image, options: images."""

import pandas as pd

from levante_bench.data.datasets import VLMDataset
from levante_bench.tasks.image_index import build_image_index
from levante_bench.tasks.option_order import (
    derive_true_random_item_seed,
    deterministic_option_order,
)
from levante_bench.tasks.registry import register_task

LABELS = ["A", "B", "C", "D"]


@register_task("matrix-reasoning")
class MatrixReasoningDataset(VLMDataset):
    """Reads matrix-reasoning trials from manifest.csv with image options."""

    def __init__(self, task_def, version, data_root=None):
        super().__init__(task_def=task_def, version=version, data_root=data_root)
        self.manifest = self._load_manifest()
        self.image_dir = self.data_root / "assets" / self.version / "visual" / "matrix-reasoning"
        self.image_index = build_image_index(self.image_dir)

    def _load_manifest(self) -> pd.DataFrame:
        """Load and filter manifest rows for matrix-reasoning task."""
        manifest_path = self.data_root / "assets" / "manifest.csv"
        df = pd.read_csv(manifest_path)
        df = df[df["task"] == "matrix-reasoning"]
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]

        answer = str(row["answer"])
        alternatives = str(row["response_alternatives"]).split(",")
        item_uid = str(row["item_uid"]).strip()
        true_random = bool(getattr(self.task_def, "true_random_option_order", False))
        run_seed = getattr(self.task_def, "option_order_run_seed", None)
        true_random_seed = (
            derive_true_random_item_seed(run_seed=int(run_seed), item_key=item_uid)
            if true_random and run_seed is not None
            else None
        )
        all_options, correct_label, option_order_seed = deterministic_option_order(
            answer=answer,
            alternatives=alternatives,
            seed_value=row["item_uid"],
            option_labels=LABELS,
            true_random=true_random,
            true_random_seed=true_random_seed,
        )

        option_image_paths = []
        for option in all_options:
            path = self.image_index.get(option.strip())
            if path is None:
                raise FileNotFoundError(
                    f"Image not found for '{option}' in {self.image_dir} "
                    f"(trial {row['item_uid']})"
                )
            option_image_paths.append(str(path))

        prompt_template = row.get("full_prompt", "")
        if str(prompt_template) in {"NA", "nan"}:
            prompt_template = row.get("prompt", "")
        prompt = self.build_localized_prompt(
            prompt_template=prompt_template,
            prompt_phrase=row.get("prompt_phrase", ""),
        )

        context_image_paths = []
        if "<prompt_image>" in prompt:
            prompt = prompt.replace("<prompt_image>", "<image0>")
            prompt_image = str(row.get("prompt_image", "NA")).strip()
            if prompt_image and prompt_image not in {"NA", "nan", "TODO"}:
                path = self.image_index.get(prompt_image)
                if path is None:
                    raise FileNotFoundError(
                        f"Prompt image not found for '{prompt_image}' in {self.image_dir} "
                        f"(trial {row['item_uid']})"
                    )
                context_image_paths.append(str(path))

        return {
            "trial_id": row["item_uid"],
            "item_uid": row["item_uid"],
            "prompt": prompt,
            "options": all_options,
            "option_labels": LABELS[:len(all_options)],
            "correct_label": correct_label,
            "option_order_seed": option_order_seed,
            "context_image_paths": context_image_paths,
            "option_image_paths": option_image_paths,
            "context_type": "image" if context_image_paths else "none",
            "option_type": "image",
        }
