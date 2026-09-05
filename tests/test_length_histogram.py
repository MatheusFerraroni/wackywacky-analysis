from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from conftest import write_config, write_sources

from wackywacky_analysis.config import load_config
from wackywacky_analysis.errors import WackyWackyError
from wackywacky_analysis.io import atomic_json
from wackywacky_analysis.length_histogram import export_character_histogram
from wackywacky_analysis.lexical import DOCUMENT_SCHEMA


def _completed_snapshot(tmp_path: Path, characters: list[int]):
    pages, domains = write_sources(tmp_path / "sources")
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
    snapshot_id = "20260820-test"
    root = config.paths.work / snapshot_id
    result = config.paths.results / snapshot_id
    documents = root / "v2" / "lexical" / "documents"
    documents.mkdir(parents=True)
    result.joinpath("aggregates").mkdir(parents=True)
    rows = [
        {
            "row_number": index,
            "view": "B_clean_v2",
            "domain_id": 1,
            "recursion_level": 0,
            "characters": value,
            "words": 1,
            "numbers": 0,
            "other_tokens": 0,
        }
        for index, value in enumerate(characters, start=1)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=DOCUMENT_SCHEMA),
        documents / "documents-000000.parquet",
    )
    atomic_json(root / "state.json", {"stage": "complete"})
    atomic_json(
        root / "v2" / "lexical_summary.json",
        {"views": {"B_clean_v2": {"documents": len(rows)}}},
    )
    atomic_json(result / "summary.json", {"primary_view": "B_clean_v2"})
    return config, snapshot_id, root, result


def test_exports_requested_bins_and_reconciles_documents(tmp_path: Path) -> None:
    config, snapshot_id, _root, result = _completed_snapshot(
        tmp_path, [1, 2, 101, 102, 9_999, 10_000]
    )

    exported = export_character_histogram(config, snapshot_id=snapshot_id)

    path = result / "aggregates" / "histograma_tamanho_caracteres.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    assert rows[0] == {"left": "2.0", "right": "101.97", "mid": "51.985", "count": "2"}
    assert sum(int(row["count"]) for row in rows) == 4
    assert exported["documents_total"] == 6
    assert exported["documents_in_histogram"] == 4
    assert exported["documents_below_minimum"] == 1
    assert exported["documents_above_maximum"] == 1
    metadata = result / "aggregates" / "histograma_tamanho_caracteres.json"
    assert metadata.is_file()
    checksum_lines = (result / "checksums.sha256").read_text().splitlines()
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert f"{expected}  aggregates/histograma_tamanho_caracteres.csv" in checksum_lines


def test_rejects_incomplete_or_inconsistent_snapshot(tmp_path: Path) -> None:
    config, snapshot_id, root, _result = _completed_snapshot(tmp_path, [10, 20])
    atomic_json(root / "state.json", {"stage": "lexical"})
    with pytest.raises(WackyWackyError, match="não está completo"):
        export_character_histogram(config, snapshot_id=snapshot_id)

    atomic_json(root / "state.json", {"stage": "complete"})
    atomic_json(
        root / "v2" / "lexical_summary.json",
        {"views": {"B_clean_v2": {"documents": 3}}},
    )
    with pytest.raises(WackyWackyError, match="não reconcilia"):
        export_character_histogram(config, snapshot_id=snapshot_id)


@pytest.mark.parametrize(
    ("minimum", "maximum", "bins"),
    ((-1, 10, 2), (10, 10, 2), (0, 10, 0)),
)
def test_rejects_invalid_parameters(
    tmp_path: Path, minimum: float, maximum: float, bins: int
) -> None:
    config, snapshot_id, _root, _result = _completed_snapshot(tmp_path, [10])
    with pytest.raises(WackyWackyError):
        export_character_histogram(
            config,
            snapshot_id=snapshot_id,
            minimum=minimum,
            maximum=maximum,
            bins=bins,
        )
