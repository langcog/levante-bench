"""Unit tests for prompt localization via translations CSV."""

from __future__ import annotations

from pathlib import Path

from levante_bench.data.datasets import VLMDataset
from levante_bench.data.schema import TaskDef


class DummyDataset(VLMDataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict:
        raise IndexError(idx)


def _write_translations_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "item_id,labels,en,de",
                "p1,test,Choose: <prompt_phrase>,Wahle: <prompt_phrase>",
                "p2,test,bird,Vogel",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_localized_prompt_uses_translations(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    csv_path = data_root / "assets" / "hackathon" / "translations" / "item-bank-translations.csv"
    _write_translations_csv(csv_path)
    task_def = TaskDef(
        task_id="dummy",
        benchmark_name="dummy",
        internal_name="dummy",
        prompt_language="de",
    )
    ds = DummyDataset(task_def=task_def, version="hackathon", data_root=data_root)
    prompt = ds.build_localized_prompt("Choose: <prompt_phrase>", "bird")
    assert prompt == "Wahle: Vogel"


def test_build_localized_prompt_falls_back_to_english_when_missing_column(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    csv_path = data_root / "assets" / "hackathon" / "translations" / "item-bank-translations.csv"
    _write_translations_csv(csv_path)
    task_def = TaskDef(
        task_id="dummy",
        benchmark_name="dummy",
        internal_name="dummy",
        prompt_language="fr-CA",
    )
    ds = DummyDataset(task_def=task_def, version="hackathon", data_root=data_root)
    prompt = ds.build_localized_prompt("Choose: <prompt_phrase>", "bird")
    assert prompt == "Choose: bird"
