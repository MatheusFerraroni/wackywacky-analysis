from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import domain_row, write_config, write_sources

from wackywacky_analysis.config import ConfigurationError, load_config
from wackywacky_analysis.errors import WackyWackyError
from wackywacky_analysis.io import sha256_file
from wackywacky_analysis.pipeline import run_pipeline
from wackywacky_analysis.sampling import (
    _score,
    _select_offsets,
    _write_domains,
    analysis_paths,
    create_sample,
)
from wackywacky_analysis.schema import DOMAIN_COLUMNS, PAGES_COLUMNS
from wackywacky_analysis.snapshot import verify_snapshot


def _sampling_config(root: Path, pages: Path, domains: Path, *, target: int = 10):
    return load_config(
        write_config(
            root / "config.toml",
            pages,
            domains,
            root / "work",
            root / "results",
            sampling=True,
            sampling_target=target,
            sampling_candidates=max(20, target),
            sampling_windows=4,
        )
    )


def test_sample_is_deterministic_projected_and_private(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    first_config = _sampling_config(tmp_path / "first", pages, domains)
    second_config = _sampling_config(tmp_path / "second", pages, domains)
    first = create_sample(first_config)
    second = create_sample(second_config)
    assert first["outputs"] == second["outputs"]
    assert first["selection"]["selected_rows"] == 10
    inputs = analysis_paths(first_config)
    page_rows = inputs.pages.read_bytes().splitlines()
    assert page_rows[0].split(b"\t") == [name.encode() for name in PAGES_COLUMNS]
    assert len(page_rows) == 11
    for raw in page_rows[1:]:
        fields = raw.split(b"\t")
        assert fields[4] == b"NULL"
        assert fields[5] == b"NULL"
        assert fields[6] == b"NULL"
        assert fields[9] == b"NULL"
        assert fields[14] == b"NULL"
        assert fields[17] == b"NULL"
        assert fields[18] == b"NULL"
    domain_rows = inputs.domains.read_bytes().splitlines()
    assert domain_rows[0].split(b"\t") == [name.encode() for name in DOMAIN_COLUMNS]
    assert all(row.split(b"\t")[2] == b"NULL" for row in domain_rows[1:])
    assert first["outputs"]["pages_sha256"] == sha256_file(inputs.pages)
    assert first["outputs"]["domains_sha256"] == sha256_file(inputs.domains)
    pointer_reuse = create_sample(first_config)
    assert pointer_reuse == first


def test_source_root_environment_variable_is_expanded_or_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config_path = write_config(
        tmp_path / "config.toml",
        pages,
        domains,
        tmp_path / "work",
        tmp_path / "results",
    )
    raw = config_path.read_text(encoding="utf-8").replace(str(pages), "${SAMPLE_DATA}/pages.tsv")
    raw = raw.replace(str(domains), "${SAMPLE_DATA}/domain.tsv")
    config_path.write_text(raw, encoding="utf-8")
    monkeypatch.delenv("SAMPLE_DATA", raising=False)
    with pytest.raises(ConfigurationError, match="não definida"):
        load_config(config_path)
    monkeypatch.setenv("SAMPLE_DATA", str(pages.parent))
    config = load_config(config_path)
    assert config.paths.pages == pages
    assert config.paths.domains == domains


def test_source_change_invalidates_sample_and_verify_hashes_only_derived_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config = _sampling_config(tmp_path / "run", pages, domains)
    create_sample(config)
    inputs = analysis_paths(config)
    hashed: list[Path] = []

    def recording_sha(path: Path) -> str:
        hashed.append(path)
        return sha256_file(path)

    monkeypatch.setattr("wackywacky_analysis.snapshot.sha256_file", recording_sha)
    manifest = verify_snapshot(config)
    assert hashed == [inputs.pages, inputs.domains]
    assert manifest["sampling"]["scope"] == "prévia amostral não representativa"
    with pages.open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(WackyWackyError, match="mudaram"):
        analysis_paths(config)


def test_sample_fails_when_windows_do_not_supply_target_rows(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources", repeated_documents=1)
    config = _sampling_config(tmp_path / "run", pages, domains, target=20)
    with pytest.raises(WackyWackyError, match="linhas válidas"):
        create_sample(config)
    partials = list((config.paths.work / "samples").glob(".partial-*"))
    assert partials == []


def test_sample_manifest_contains_no_source_path(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config = _sampling_config(tmp_path / "run", pages, domains)
    manifest = create_sample(config)
    encoded = json.dumps(manifest, ensure_ascii=False)
    assert str(pages.parent) not in encoded


def test_selection_reserves_same_as_target_and_domains_include_parents(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    with domains.open("ab") as handle:
        handle.write(domain_row(3, b"https://child.invalid", parent=b"1", level=1) + b"\n")
    config = _sampling_config(tmp_path / "run", pages, domains)
    candidate_db = tmp_path / "candidates.sqlite"
    connection = sqlite3.connect(candidate_db)
    connection.execute(
        """
        CREATE TABLE candidates (
          offset INTEGER PRIMARY KEY, end_offset INTEGER NOT NULL, page_id INTEGER,
          domain_id INTEGER, same_as INTEGER, status TEXT NOT NULL, score TEXT NOT NULL
        )
        """
    )
    rows = []
    for index in range(1, 13):
        rows.append(
            (
                index,
                index + 1,
                index,
                3,
                None if index == 1 else 1,
                "done",
                "f" * 64 if index == 1 else _score(config.sampling.seed, index, index),
            )
        )
    connection.executemany("INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    connection.commit()
    connection.close()
    offsets, selection = _select_offsets(config, candidate_db)
    assert 1 in offsets
    assert selection["same_as_targets_added"] == 1

    output = tmp_path / "domain-sample.tsv"
    domain_db = tmp_path / "domains.sqlite"
    summary = _write_domains(config, {3}, output, domain_db)
    selected_ids = {int(row.split(b"\t", 1)[0]) for row in output.read_bytes().splitlines()[1:]}
    assert selected_ids == {1, 3}
    assert summary["selected_domains"] == 2


def test_sampled_pipeline_marks_preview_and_disables_request_yield(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config = _sampling_config(tmp_path / "run", pages, domains)
    create_sample(config)
    config = replace(config, boilerplate=replace(config.boilerplate, enabled=False))
    completed = run_pipeline(config, resume=False)
    assert completed["status"] == "complete"
    result = Path(completed["result"])
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    assert summary["scope"] == "prévia amostral não representativa"
    yield_table = (result / "tables" / "07_rendimento_nivel.csv").read_text(encoding="utf-8")
    assert "não aplicável" in yield_table
    yield_aggregate = (result / "aggregates" / "rendimento_nivel.csv").read_text(encoding="utf-8")
    assert "False" in yield_aggregate
    figure = (result / "figures" / "07_rendimento_nivel.svg").read_text(encoding="utf-8")
    assert "PRÉVIA AMOSTRAL NÃO REPRESENTATIVA" in figure
