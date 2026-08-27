from __future__ import annotations

import csv
import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
import zstandard as zstd
from conftest import domain_row, page_row, write_config, write_sources

from wackywacky_analysis.bclean import iter_bclean
from wackywacky_analysis.config import BoilerplateV2, load_config
from wackywacky_analysis.errors import ReviewRequired
from wackywacky_analysis.noise import (
    is_presentation_noise,
    mediawiki_rule_id,
    residual_units,
)
from wackywacky_analysis.pipeline import run_pipeline
from wackywacky_analysis.review import import_review
from wackywacky_analysis.snapshot import verify_snapshot
from wackywacky_analysis.text import classify_text_md5, decode_text
from wackywacky_analysis.v2 import _OrderedV2Pool, import_v2_review


def _confirm_default(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_residual_rules_are_anchored_and_restricted_to_wikimedia() -> None:
    config = BoilerplateV2(enabled=True, edge_min_lines=2, frequency_min_documents=2)
    text = (
        "Editar\nRodapé sintético\nLinha intermediária legítima com várias palavras\n"
        "Outra linha intermediária legítima\nFim do documento"
    )
    wikimedia = residual_units(text, config, is_wikimedia=True)
    outside = residual_units(text, config, is_wikimedia=False)
    assert any(unit.kind == "mediawiki" for unit in wikimedia)
    assert not any(unit.kind == "mediawiki" for unit in outside)
    assert mediawiki_rule_id("Este texto explica como editar uma enciclopédia.") is None
    assert not any(
        unit.canonical == "linha intermediária legítima com várias palavras"
        for unit in outside
        if unit.kind == "short_line"
    )
    assert all(
        is_presentation_noise(value)
        for value in (
            "centralautologin none",
            "special centralautologin",
            "barra lateral",
            "editar código fonte",
            "ferramentas aparência ocultar",
            "predefinição predefinição",
        )
    )


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("raw", "decompressed_bytes"),
        ("compressed", "compressed_bytes"),
        ("field", "hexadecimal_field"),
        ("normalized", "normalized_text"),
    ],
)
def test_md5_diagnostic_covers_all_representations(variant: str, expected: str) -> None:
    raw = "Texto  com espaços.\r\n".encode()
    compressed = zstd.ZstdCompressor(level=1).compress(raw)
    field = compressed.hex().encode()
    temporary = decode_text(field, b"NULL", 1_000_000)
    values = {
        "raw": raw,
        "compressed": compressed,
        "field": field,
        "normalized": temporary.normalized.encode(),
    }
    supplied = hashlib.md5(values[variant], usedforsecurity=False).hexdigest().encode()
    decoded = decode_text(field, supplied, 1_000_000)
    assert classify_text_md5(field, supplied, decoded) == expected


def test_v2_lightweight_pool_is_identical_with_one_two_and_eight_workers(
    tmp_path: Path,
) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config = load_config(
        write_config(
            tmp_path / "config.toml",
            pages,
            domains,
            tmp_path / "work",
            tmp_path / "results",
            profile="full",
            boilerplate_v2=True,
        )
    )
    text = "Editar\nConteúdo legítimo com várias palavras.\nRodapé sintético"
    discovery_jobs = [(1, 10, 1, text), (2, 20, 1, text)]

    def discover(workers: int) -> list[dict]:
        selected = replace(config, runtime=replace(config.runtime, workers=workers))
        with _OrderedV2Pool(selected, mode="discovery", wikimedia={1}) as pool:
            batches = pool.submit(discovery_jobs, 4 * len(text) * 2) + pool.drain()
        return [row for batch in batches for row in batch]

    assert discover(1) == discover(2) == discover(8)
    clean_text = "Conteúdo legítimo com várias palavras.\nRodapé sintético"
    unit = next(
        item
        for item in residual_units(clean_text, config.boilerplate_v2, is_wikimedia=False)
        if item.kind == "short_line" and item.canonical == "rodapé sintético"
    )
    pair = next(
        item
        for item in residual_units(clean_text, config.boilerplate_v2, is_wikimedia=False)
        if item.kind == "short_pair" and item.end == len(clean_text)
    )
    database = tmp_path / "candidate.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE candidate(domain_id INTEGER,kind TEXT,digest TEXT,"
        "PRIMARY KEY(domain_id,kind,digest)) WITHOUT ROWID"
    )
    connection.execute("INSERT INTO candidate VALUES (1,'short_line',?)", [unit.digest])
    connection.execute("INSERT INTO candidate VALUES (1,'short_pair',?)", [pair.digest])
    connection.commit()
    connection.close()
    clean_jobs = [
        (1, 1, 1, 0, clean_text, "missing"),
        (2, 2, 2, 0, clean_text, "missing"),
    ]

    def clean(workers: int) -> list[dict]:
        selected = replace(config, runtime=replace(config.runtime, workers=workers))
        with _OrderedV2Pool(
            selected,
            mode="clean",
            wikimedia=set(),
            candidate_path=database,
        ) as pool:
            batches = pool.submit(clean_jobs, 8 * len(clean_text)) + pool.drain()
        return [row for batch in batches for row in batch]

    cleaned = clean(1)
    assert cleaned == clean(2) == clean(8)
    assert cleaned[0]["row"]["removed_characters"] > 0
    assert cleaned[0]["row"]["short_line_matches"] == 1
    assert cleaned[0]["row"]["short_pair_matches"] == 1
    assert (
        cleaned[0]["row"]["short_line_characters"] + cleaned[0]["row"]["short_pair_characters"]
        == cleaned[0]["row"]["removed_characters"]
    )
    assert cleaned[1]["row"]["removed_characters"] == 0


def test_pipeline_builds_confirmed_b_clean_v2_and_preserves_v1(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    with domains.open("ab") as handle:
        handle.write(domain_row(3, b"https://outside.example", requests=7) + b"\n")
    additions = []
    for index in range(4):
        additions.append(
            page_row(
                200 + index,
                1,
                "done",
                text=(
                    "Editar\n"
                    f"Conteúdo enciclopédico legítimo e diferente do documento número {index}.\n"
                    "Rodapé sintético"
                ),
            )
        )
        additions.append(
            page_row(
                300 + index,
                3,
                "done",
                text=(
                    "Editar\n"
                    f"Conteúdo externo legítimo e diferente do documento número {index}.\n"
                    "Rodapé sintético"
                ),
            )
        )
    additions.extend(
        [
            page_row(
                500,
                1,
                "done",
                text=(
                    "Editar\nConteúdo final idêntico depois da limpeza reforçada, "
                    "mantido por possuir mais de oitenta caracteres legítimos no corpo."
                ),
            ),
            page_row(
                501,
                1,
                "done",
                text=(
                    "Criar conta\nConteúdo final idêntico depois da limpeza reforçada, "
                    "mantido por possuir mais de oitenta caracteres legítimos no corpo."
                ),
            ),
            page_row(502, 1, "done", text="Editar"),
        ]
    )
    with pages.open("ab") as handle:
        handle.write(b"\n".join(additions) + b"\n")
    config = load_config(
        write_config(
            tmp_path / "config.toml",
            pages,
            domains,
            tmp_path / "work",
            tmp_path / "results",
            boilerplate_v2=True,
        )
    )
    with pytest.raises(ReviewRequired):
        run_pipeline(config, resume=False)
    manifest = verify_snapshot(config)
    root = config.paths.work / manifest["snapshot_id"]
    import_review(config, root, root / "review" / "boilerplate-review.csv")
    with pytest.raises(ReviewRequired, match="B_clean_v2"):
        run_pipeline(config, resume=True)
    review = root / "v2" / "review" / "boilerplate-review.csv"
    with review.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0])[:3] == ["sample_id", "frequencia", "label"]
    assert all(row["label"] == "boilerplate" for row in rows)
    assert any(row["kind"] == "mediawiki" for row in rows)
    _confirm_default(review)
    confirmation = import_v2_review(config, root, review)
    assert confirmation["status"] == "confirmed_default"
    result = Path(run_pipeline(config, resume=True)["result"])
    summary = __import__("json").loads((result / "summary.json").read_text())
    assert summary["primary_view"] == "B_clean_v2"
    assert summary["clean"]["d3_unique"] >= summary["clean_v2"]["d4_unique"]
    assert summary["clean_v2"]["mediawiki_matches_removed"] >= 1
    assert summary["clean_v2"]["short_line_matches_removed"] >= 1
    assert summary["clean_v2"]["documents_emptied"] >= 1
    assert summary["clean_v2"]["d4_duplicate_groups"] >= 1
    assert summary["clean_v2"]["confirmation"]["status"] == "confirmed_default"
    assert (
        summary["lexical_v2"]["views"]["B_clean_v2"]["documents"]
        == summary["clean_v2"]["d4_unique"]
    )
    assert (result / "tables" / "12e_principais_formas_lemas_b_clean_v1.csv").is_file()
    assert (result / "tables" / "17b_repeticao_b_clean_vs_v2.csv").is_file()
    assert (result / "tables" / "18b_sinais_b_clean_vs_v2.csv").is_file()
    with (result / "tables" / "12a_principais_formas_lemas.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert "editar" not in {row["item"] for row in csv.DictReader(handle)}
    assert (root / "vocabulary.parquet").is_file()
    assert (root / "v2" / "vocabulary.parquet").is_file()
    connection = duckdb.connect(str(root / "analysis.duckdb"), read_only=True)
    expected_hashes = dict(
        connection.execute(
            "SELECT row_number,clean_v2_sha256 FROM d4_membership WHERE is_representative"
        ).fetchall()
    )
    connection.close()
    reconstructed = {
        item.source.row_number: hashlib.sha256(item.text.encode()).hexdigest()
        for item in iter_bclean(config, root / "v2")
    }
    assert reconstructed == expected_hashes
