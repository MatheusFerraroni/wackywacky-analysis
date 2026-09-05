from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import pytest

from wackywacky_analysis.reports import _domain_children_table


def _report(root: Path, rows: list[tuple], *, sampled: bool = False) -> tuple[dict, list[dict]]:
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit='64MiB'")
        connection.execute(
            "CREATE TABLE domains(row_number BIGINT, id BIGINT, host VARCHAR, "
            "parent_domain_id BIGINT, request_count BIGINT)"
        )
        if rows:
            connection.executemany("INSERT INTO domains VALUES (?,?,?,?,?)", rows)
        summary = _domain_children_table(connection, root, sampled=sampled)
    with (root / "08b_dominios_filhos.csv").open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    return summary, table


def test_children_are_direct_distinct_and_include_zero_requests(tmp_path: Path) -> None:
    # IDs, not host names, define domains. The later duplicate must not alter the graph.
    rows = [
        (1, 1, "alpha.invalid", None, 999),
        (2, 2, "beta.invalid", 1, 0),
        (3, 3, "same-host.invalid", 1, 10),
        (4, 4, "same-host.invalid", 1, None),
        (5, 5, "grandchild.invalid", 2, 20),
        (6, 6, "orphan.invalid", 999, 50),
        (7, 7, "missing-average.invalid", None, 100),
        (8, 8, "unknown-requests.invalid", 7, None),
        (9, 9, "no-children.invalid", None, 1),
        (10, 10, "zero-average.invalid", None, 80),
        (11, 11, "zero-requests.invalid", 10, 0),
        (99, 2, "ignored-duplicate.invalid", 7, 999),
    ]
    summary, table = _report(tmp_path / "first", rows)
    assert summary["denominator_domains"] == 11
    assert summary["domains_without_parent"] == 4
    assert summary["domains_with_missing_parent"] == 1
    assert summary["domains_with_known_parent"] == 6
    assert summary["parents_with_children"] == 4
    assert summary["children_with_missing_request_count"] == 2
    assert summary["counts_reconciled"] is True
    assert summary["published_children"] == 6
    assert [row["host"] for row in table] == [
        "alpha.invalid",
        "beta.invalid",
        "missing-average.invalid",
        "zero-average.invalid",
    ]
    assert [int(row["filhos"]) for row in table] == [3, 1, 1, 1]
    assert float(table[0]["percentual"]) == pytest.approx(100 * 3 / 11)
    assert float(table[0]["media_requisicoes_filhos"]) == 5
    assert table[0]["filhos_sem_requisicoes"] == "1"
    assert float(table[1]["media_requisicoes_filhos"]) == 20
    assert table[2]["media_requisicoes_filhos"] == "não disponível"
    assert table[2]["filhos_sem_requisicoes"] == "1"
    assert float(table[3]["media_requisicoes_filhos"]) == 0
    assert table[3]["filhos_sem_requisicoes"] == "0"

    repeated_summary, repeated_table = _report(tmp_path / "reordered", list(reversed(rows)))
    assert repeated_summary == summary
    assert repeated_table == table
    for extension in ("csv", "tex"):
        name = f"08b_dominios_filhos.{extension}"
        assert (tmp_path / "first" / name).read_bytes() == (
            tmp_path / "reordered" / name
        ).read_bytes()
    latex = (tmp_path / "first" / "08b_dominios_filhos.tex").read_text(encoding="utf-8")
    assert "nan" not in latex.lower()
    assert "None" not in latex
    assert "https://" not in latex


def test_top15_uses_all_domains_as_denominator_with_stable_ties(tmp_path: Path) -> None:
    rows = [(index, index, "tied.invalid", None, 1) for index in range(1, 18)]
    rows += [(100 + index, 100 + index, "child.invalid", index, index) for index in range(1, 18)]
    rows.append((999, 999, "extra-child.invalid", 17, 0))
    summary, table = _report(tmp_path, list(reversed(rows)))
    assert summary["denominator_domains"] == 35
    assert summary["domains_with_known_parent"] == 18
    assert summary["parents_with_children"] == 17
    assert summary["published_parents"] == len(table) == 15
    assert summary["published_children"] == 16
    assert table[0]["filhos"] == "2"
    assert float(table[0]["percentual"]) == pytest.approx(100 * 2 / 35)
    assert float(table[0]["media_requisicoes_filhos"]) == 8.5
    assert [float(row["media_requisicoes_filhos"]) for row in table[1:]] == list(range(1, 15))


@pytest.mark.parametrize("rows", [[], [(1, 1, "root.invalid", None, 0)]])
def test_empty_child_rankings_remain_valid(tmp_path: Path, rows: list[tuple]) -> None:
    summary, table = _report(tmp_path, rows, sampled=True)
    assert table == []
    assert summary["denominator_domains"] == len(rows)
    assert summary["domains_with_known_parent"] == 0
    assert summary["counts_reconciled"] is True
    assert summary["scope"] == "prévia amostral não representativa"
