"""Tests for local and bucket version resolution behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import pytest

from levante_bench.config.defaults import detect_data_version


def _load_download_assets_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "download_levante_assets.py"
    if not script_path.exists():
        script_path = repo_root / "scripts" / "data_prep" / "download_levante_assets.py"
    spec = importlib.util.spec_from_file_location("download_levante_assets", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load download_levante_assets module for tests.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_detect_data_version_prefers_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LEVANTE_DATA_VERSION", "hackathon")
    assert detect_data_version(tmp_path) == "hackathon"


def test_detect_data_version_uses_most_recent_folder(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    older = assets / "2026-03-24"
    newer = assets / "hackathon"
    older.mkdir(parents=True)
    time.sleep(0.01)
    newer.mkdir(parents=True)

    assert detect_data_version(tmp_path) == "hackathon"


def test_detect_data_version_raises_when_assets_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Assets directory not found"):
        detect_data_version(tmp_path)


def test_bucket_and_prefix_parser_storage_googleapis() -> None:
    m = _load_download_assets_module()
    bucket, prefix = m._bucket_and_base_prefix_from_base(
        "https://storage.googleapis.com/levante-bench/corpus_data"
    )
    assert bucket == "levante-bench"
    assert prefix == "corpus_data"


def test_bucket_and_prefix_parser_subdomain_style() -> None:
    m = _load_download_assets_module()
    bucket, prefix = m._bucket_and_base_prefix_from_base(
        "https://levante-bench.storage.googleapis.com/corpus_data/subpath"
    )
    assert bucket == "levante-bench"
    assert prefix == "corpus_data/subpath"


def test_detect_latest_bucket_version_prefers_date_prefixes(monkeypatch) -> None:
    m = _load_download_assets_module()
    monkeypatch.setattr(
        m,
        "_list_bucket_prefixes",
        lambda bucket_name, parent_prefix="": ["hackathon", "2026-03-24", "2026-04-01"],
    )
    assert m._detect_latest_bucket_version("levante-bench") == "2026-04-01"


def test_detect_latest_bucket_version_single_non_date(monkeypatch) -> None:
    m = _load_download_assets_module()
    monkeypatch.setattr(
        m,
        "_list_bucket_prefixes",
        lambda bucket_name, parent_prefix="": ["hackathon"],
    )
    assert m._detect_latest_bucket_version("levante-bench") == "hackathon"


def test_detect_latest_bucket_version_prefers_v1_for_non_date_prefixes(monkeypatch) -> None:
    m = _load_download_assets_module()
    monkeypatch.setattr(
        m,
        "_list_bucket_prefixes",
        lambda bucket_name, parent_prefix="": ["hackathon", "v1"],
    )
    assert m._detect_latest_bucket_version("levante-bench") == "v1"


def test_detect_latest_bucket_version_multiple_non_date_without_v1_raises(monkeypatch) -> None:
    m = _load_download_assets_module()
    monkeypatch.setattr(
        m,
        "_list_bucket_prefixes",
        lambda bucket_name, parent_prefix="": ["hackathon", "pilot"],
    )
    with pytest.raises(RuntimeError, match="Multiple non-date version prefixes"):
        m._detect_latest_bucket_version("levante-bench")
