from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import pytest
from conftest import write_config, write_sources

from wackywacky_analysis.config import ConfigurationError, load_config
from wackywacky_analysis.domains import inventory_domains
from wackywacky_analysis.errors import SourceChangedError
from wackywacky_analysis.io import atomic_json
from wackywacky_analysis.pipeline import run_pipeline
from wackywacky_analysis.review import _review_preview, wilson_lower
from wackywacky_analysis.snapshot import assert_snapshot, verify_snapshot


def test_snapshot_detects_append_and_domain_inventory_is_binary_safe(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    manifest = verify_snapshot(config)
    summary = inventory_domains(config, manifest, config.paths.work / manifest["snapshot_id"])
    assert summary["rows"] == 2
    assert summary["missing_parents"] == 1
    assert summary["wikimedia_domains"] == 1
    with pages.open("ab") as handle:
        handle.write(b"append")
    with pytest.raises(SourceChangedError):
        assert_snapshot(config, manifest)


def test_resume_rejects_source_changed_since_checkpoint(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    manifest = verify_snapshot(config)
    root = config.paths.work / manifest["snapshot_id"]
    atomic_json(
        root / "state.json",
        {
            "snapshot_id": manifest["snapshot_id"],
            "config_sha256": config.fingerprint,
            "stage": "pages",
        },
    )
    with pages.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(SourceChangedError):
        run_pipeline(config, resume=True)


def test_tiny_profile_rejects_multiple_workers(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path)
    path = write_config(
        tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results", workers=2
    )
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_headerless_sources_are_validated_by_column_count(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path)
    pages.write_bytes(b"\n".join(pages.read_bytes().splitlines()[1:]) + b"\n")
    domains.write_bytes(b"\n".join(domains.read_bytes().splitlines()[1:]) + b"\n")
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    manifest = verify_snapshot(config)
    assert manifest["sources"]["pages"]["header"] is False
    assert manifest["sources"]["domains"]["header"] is False


def test_legacy_domain_header_manifest_is_migrated_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    manifest = verify_snapshot(config)
    manifest_path = config.paths.work / manifest["snapshot_id"] / "manifest.json"
    legacy = deepcopy(manifest)
    legacy["schema_version"] = 1
    legacy["sources"]["domains"]["header"] = False
    legacy.pop("migrations")
    atomic_json(manifest_path, legacy)

    with caplog.at_level(logging.WARNING):
        migrated = verify_snapshot(config)
    assert migrated["schema_version"] == 2
    assert migrated["sources"]["domains"]["header"] is True
    assert [item["id"] for item in migrated["migrations"]] == ["domain-header-schema-v2"]
    assert "identidade física confirmada" in caplog.text
    repeated = verify_snapshot(config)
    assert repeated["migrations"] == migrated["migrations"]


def test_legacy_domain_header_migration_rejects_any_other_difference(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    manifest = verify_snapshot(config)
    manifest_path = config.paths.work / manifest["snapshot_id"] / "manifest.json"
    legacy = deepcopy(manifest)
    legacy["schema_version"] = 1
    legacy["sources"]["domains"]["header"] = False
    legacy.pop("migrations")

    invalid_manifests = []
    for field in ("size", "mtime_ns", "columns"):
        invalid = deepcopy(legacy)
        invalid["sources"]["domains"][field] += 1
        invalid_manifests.append(invalid)
    invalid = deepcopy(legacy)
    invalid["sources"]["domains"]["sha256"] = "0" * 64
    invalid_manifests.append(invalid)
    invalid = deepcopy(legacy)
    invalid["sources"]["pages"]["header"] = False
    invalid_manifests.append(invalid)
    invalid = deepcopy(legacy)
    invalid["config_sha256"] = "0" * 64
    invalid_manifests.append(invalid)
    invalid = deepcopy(legacy)
    invalid["migrations"] = [{"id": "unknown"}]
    invalid_manifests.append(invalid)
    invalid = deepcopy(legacy)
    invalid["schema_version"] = 99
    invalid_manifests.append(invalid)

    for invalid in invalid_manifests:
        atomic_json(manifest_path, invalid)
        with pytest.raises(SourceChangedError):
            verify_snapshot(config)


def test_wilson_gate_requires_enough_correct_reviews() -> None:
    assert wilson_lower(190, 200) > 0.90
    assert wilson_lower(189, 200) < wilson_lower(190, 200)


def test_review_preview_is_single_line_clean_and_bounded() -> None:
    preview, truncated = _review_preview("linha 1\nlinha\u200b 2\x00")
    assert preview == "linha 1 [QUEBRA] linha 2"
    assert truncated is False
    preview, truncated = _review_preview("a" * 5_000, limit=100)
    assert truncated is True
    assert "caracteres omitidos" in preview
    assert len(preview) < 130
