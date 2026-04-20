"""Unit tests for <imageN> placeholder interleaving behavior."""

from __future__ import annotations

from pathlib import Path

from levante_bench.models.gemini import GeminiProModel
from levante_bench.models.internvl35 import InternVL35Model
from levante_bench.models.qwen35 import Qwen35Model
from levante_bench.models.smolvlm2 import SmolVLM2Model


def _content_images(content: list[dict], key: str) -> list[object]:
    return [entry[key] for entry in content if entry.get("type") == "image"]


def test_smol_image0_routes_context_to_index_zero(tmp_path: Path) -> None:
    model = SmolVLM2Model(model_name="dummy")
    image_paths = [
        str(tmp_path / "context.png"),
        str(tmp_path / "opt_a.png"),
        str(tmp_path / "opt_b.png"),
    ]
    prompt = "<image0> Context <image1> A <image2> B"

    messages = model._build_messages(prompt, image_paths=image_paths)
    content = messages[0]["content"]
    image_urls = [entry["url"] for entry in content if entry.get("type") == "image"]

    assert image_urls == [str(Path(p).resolve()) for p in image_paths]


def test_smol_one_based_routes_image1_to_first_path(tmp_path: Path) -> None:
    model = SmolVLM2Model(model_name="dummy")
    image_paths = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    prompt = "<image1> First <image2> Second"

    messages = model._build_messages(prompt, image_paths=image_paths)
    content = messages[0]["content"]
    image_urls = [entry["url"] for entry in content if entry.get("type") == "image"]

    assert image_urls == [str(Path(image_paths[0]).resolve()), str(Path(image_paths[1]).resolve())]


def test_smol_out_of_bounds_placeholder_is_skipped(tmp_path: Path) -> None:
    model = SmolVLM2Model(model_name="dummy")
    image_paths = [str(tmp_path / "a.png")]
    prompt = "Before <image9> after"

    messages = model._build_messages(prompt, image_paths=image_paths)
    content = messages[0]["content"]
    image_urls = [entry["url"] for entry in content if entry.get("type") == "image"]

    assert image_urls == []


def test_qwen_interleaving_supports_image0_mode() -> None:
    model = Qwen35Model(model_name="dummy")
    pil_images = [object(), object(), object()]
    prompt = "<image0> ctx <image1> a <image2> b"

    messages = model._build_messages(prompt, pil_images=pil_images)
    content = messages[1]["content"]
    images = _content_images(content, "image")

    assert images == pil_images


def test_internvl_interleaving_supports_one_based_mode() -> None:
    model = InternVL35Model(model_name="dummy")
    pil_images = [object(), object()]
    prompt = "<image1> a <image2> b"

    messages = model._build_messages(prompt, pil_images=pil_images)
    content = messages[1]["content"]
    images = _content_images(content, "image")

    assert images == pil_images


def test_gemini_interleaving_image0_and_bounds(monkeypatch) -> None:
    model = GeminiProModel(model_name="dummy")
    monkeypatch.setattr(model, "_image_part", lambda p: {"img": p})
    image_paths = ["ctx.png", "a.png", "b.png"]

    parts = model._build_parts("<image0> x <image1> y <image9>", image_paths)
    image_parts = [p["img"] for p in parts if "img" in p]

    assert image_parts == ["ctx.png", "a.png"]
