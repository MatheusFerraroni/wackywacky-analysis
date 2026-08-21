from __future__ import annotations

import sqlite3
from pathlib import Path

import pyarrow as pa
import pytest
from conftest import write_config, write_sources

import wackywacky_analysis.clean as clean_module
from wackywacky_analysis.clean import CANDIDATE_DATABASE_VERSION, _candidate_database
from wackywacky_analysis.config import load_config
from wackywacky_analysis.lexical import _clean_again
from wackywacky_analysis.storage import write_parquet_atomic
from wackywacky_analysis.text import paragraph_units

CANDIDATE_SCHEMA = pa.schema(
    [
        ("domain_id", pa.int64()),
        ("kind", pa.string()),
        ("unit_sha256", pa.string()),
    ]
)


def _config_and_candidates(tmp_path: Path, rows: list[dict]):
    pages, domains = write_sources(tmp_path / "sources")
    config = load_config(
        write_config(
            tmp_path / "config.toml",
            pages,
            domains,
            tmp_path / "work",
            tmp_path / "results",
        )
    )
    root = tmp_path / "work" / "snapshot"
    root.mkdir(parents=True)
    write_parquet_atomic(root / "boilerplate_candidates.parquet", rows, CANDIDATE_SCHEMA)
    return config, root


def _old_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE candidate(normalized_sha256 TEXT, kind TEXT, digest TEXT, "
        "PRIMARY KEY(normalized_sha256, kind, digest)) WITHOUT ROWID"
    )
    connection.execute("INSERT INTO candidate VALUES ('old', 'paragraph', 'digest')")
    connection.commit()
    connection.close()


@pytest.mark.parametrize("old_name", ["boilerplate.sqlite", "boilerplate.sqlite.partial"])
def test_candidate_index_replaces_old_schema_without_document_expansion(
    tmp_path: Path, old_name: str
) -> None:
    rows = [
        {"domain_id": 1, "kind": "paragraph", "unit_sha256": "a"},
        {"domain_id": 1, "kind": "block", "unit_sha256": "b"},
        {"domain_id": 2, "kind": "paragraph", "unit_sha256": "a"},
    ]
    config, root = _config_and_candidates(tmp_path, rows)
    _old_database(root / old_name)

    database = _candidate_database(config, root)
    connection = sqlite3.connect(database)
    columns = [row[1] for row in connection.execute("PRAGMA table_info(candidate)")]
    count = connection.execute("SELECT count(*) FROM candidate").fetchone()[0]
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    connection.close()

    assert columns == ["domain_id", "kind", "digest"]
    assert count == len(rows)
    assert version == CANDIDATE_DATABASE_VERSION
    assert database.stat().st_size < 1024 * 1024


def test_candidate_index_resumes_confirmed_batches(tmp_path: Path, monkeypatch) -> None:
    rows = [
        {"domain_id": domain, "kind": kind, "unit_sha256": digest}
        for domain, kind, digest in (
            (1, "block", "a"),
            (1, "paragraph", "b"),
            (2, "block", "c"),
            (2, "paragraph", "d"),
            (3, "paragraph", "e"),
        )
    ]
    config, root = _config_and_candidates(tmp_path, rows)
    monkeypatch.setattr(clean_module, "CANDIDATE_BATCH_ROWS", 2)
    original_update = clean_module.ItemProgress.update
    interrupted = False

    def interrupt_once(progress, current, **kwargs):
        nonlocal interrupted
        original_update(progress, current, **kwargs)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt

    monkeypatch.setattr(clean_module.ItemProgress, "update", interrupt_once)
    with pytest.raises(KeyboardInterrupt):
        _candidate_database(config, root)

    partial = root / "boilerplate.sqlite.partial"
    connection = sqlite3.connect(partial)
    assert connection.execute("SELECT count(*) FROM candidate").fetchone()[0] == 2
    connection.close()

    database = _candidate_database(config, root)
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT count(*) FROM candidate").fetchone()[0] == len(rows)
    assert dict(connection.execute("SELECT key, value FROM metadata"))["complete"] == "1"
    connection.close()
    assert not partial.exists()


def test_cleaning_candidates_are_scoped_to_domain(tmp_path: Path) -> None:
    config, _root = _config_and_candidates(tmp_path, [])
    repeated = "Cabeçalho sintético intradomínio para navegação e direitos reservados. " * 2
    text = repeated + "\n\nConteúdo que deve permanecer."
    digest = paragraph_units(text, config.boilerplate.paragraph_min_chars)[0][2]
    candidates = sqlite3.connect(":memory:")
    candidates.execute(
        "CREATE TABLE candidate(domain_id INTEGER, kind TEXT, digest TEXT, "
        "PRIMARY KEY(domain_id, kind, digest)) WITHOUT ROWID"
    )
    candidates.execute("INSERT INTO candidate VALUES (1, 'paragraph', ?)", (digest,))

    clean_domain_1 = _clean_again(config, candidates, 1, text)
    clean_domain_2 = _clean_again(config, candidates, 2, text)
    clean_without_domain = _clean_again(config, candidates, None, text)
    candidates.close()

    assert repeated not in clean_domain_1
    assert "Conteúdo que deve permanecer." in clean_domain_1
    assert clean_domain_2 == text
    assert clean_without_domain == text
