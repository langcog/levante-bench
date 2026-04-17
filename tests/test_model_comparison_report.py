"""Unit tests for model tag parsing in comparison report."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_report_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "analysis" / "build_model_comparison_report.py"
    spec = importlib.util.spec_from_file_location("build_model_comparison_report", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load build_model_comparison_report module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_model_size_language_new_format() -> None:
    m = _load_report_module()
    model, size, language = m._split_model_size_language("smolvlm2-256M")
    assert model == "smolvlm2"
    assert size == "256M"
    assert language is None


def test_split_model_size_language_with_lang_suffix() -> None:
    m = _load_report_module()
    model, size, language = m._split_model_size_language("smolvlm2-256M-de")
    assert model == "smolvlm2"
    assert size == "256M"
    assert language == "de"


def test_split_model_size_language_gemma4_e2b() -> None:
    m = _load_report_module()
    model, size, language = m._split_model_size_language("gemma4-E2B-it")
    assert model == "gemma4"
    assert size == "E2B-it"
    assert language is None


def test_split_model_size_language_legacy_underscore() -> None:
    m = _load_report_module()
    model, size, language = m._split_model_size_language("qwen35_0.8B")
    assert model == "qwen35"
    assert size == "0.8B"
    assert language is None
