from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
from pathlib import Path

import duckdb
import pyarrow as pa

from .config import Config
from .io import atomic_json, iter_bounded_tsv, parse_int, read_json
from .schema import PAGES_COLUMNS
from .snapshot import assert_snapshot
from .storage import (
    SortedMembership,
    duckdb_connection,
    duckdb_copy_atomic,
    write_parquet_atomic,
    write_u64,
)
from .text import TextDecodeFailure, block_units, decode_text, paragraph_units, remove_intervals

CLEAN_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("page_id", pa.int64()),
        ("domain_id", pa.int64()),
        ("normalized_sha256", pa.string()),
        ("clean_sha256", pa.string()),
        ("characters_before", pa.uint64()),
        ("characters_after", pa.uint64()),
        ("removed_characters", pa.uint64()),
        ("removed_fraction", pa.float64()),
        ("empty", pa.bool_()),
    ]
)


def _covered_characters(intervals: list[tuple[int, int]]) -> int:
    covered = 0
    end_seen = 0
    for start, end in sorted(intervals):
        if end <= end_seen:
            continue
        covered += end - max(start, end_seen)
        end_seen = end
    return covered


def _candidate_database(root: Path, enabled: bool = True) -> Path:
    target = root / "boilerplate.sqlite"
    if target.exists():
        return target
    partial = target.with_suffix(".sqlite.partial")
    partial.unlink(missing_ok=True)
    sql = sqlite3.connect(partial)
    sql.execute(
        "CREATE TABLE candidate(normalized_sha256 TEXT, kind TEXT, digest TEXT, "
        "PRIMARY KEY(normalized_sha256, kind, digest)) WITHOUT ROWID"
    )
    if enabled:
        source = duckdb.connect(str(root / "analysis.duckdb"), read_only=True)
        cursor = source.execute(
            """
            SELECT DISTINCT m.normalized_sha256, c.kind, c.unit_sha256
            FROM read_parquet(?) c
            JOIN d2_membership m USING (domain_id)
            """,
            [str(root / "boilerplate_candidates.parquet")],
        )
        while batch := cursor.fetchmany(100_000):
            sql.executemany("INSERT INTO candidate VALUES (?, ?, ?)", batch)
            sql.commit()
        source.close()
    sql.close()
    os.replace(partial, target)
    return target


def clean_representatives(config: Config, manifest: dict, root: Path) -> dict:
    summary_path = root / "clean_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_db = _candidate_database(root, config.boilerplate.enabled)
    candidates = sqlite3.connect(f"file:{candidate_db}?mode=ro", uri=True)
    output_dir = root / "clean" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = root / "clean_state.json"
    state = read_json(state_path, {})
    offset = int(state.get("next_offset", 0))
    row_number = int(state.get("next_row", 0))
    chunk = int(state.get("next_chunk", 0))
    scanned = int(state.get("scanned", 0))
    empty = int(state.get("empty", 0))
    affected = int(state.get("affected", 0))
    removed_total = int(state.get("removed_total", 0))
    paragraph_matches = int(state.get("paragraph_matches", 0))
    block_matches = int(state.get("block_matches", 0))
    paragraph_characters = int(state.get("paragraph_characters", 0))
    block_characters = int(state.get("block_characters", 0))
    membership = SortedMembership(root / "d2_representatives.u64")
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        with config.analysis_pages.open("rb") as handle:
            while not state.get("complete"):
                assert_snapshot(config, manifest)
                rows: list[dict] = []
                chunk_start = offset
                saw_record = False
                for record in iter_bounded_tsv(
                    handle,
                    columns=len(PAGES_COLUMNS),
                    max_line_bytes=config.runtime.max_line_bytes,
                    start_offset=offset,
                    start_row=row_number,
                    max_rows=config.runtime.max_rows,
                ):
                    saw_record = True
                    offset = record.end_offset
                    row_number = record.row_number
                    if membership.contains(record.row_number) and record.fields is not None:
                        result = _clean_record(config, candidates, record)
                        rows.append(result[0])
                        scanned += 1
                        empty += result[1]
                        affected += result[2]
                        removed_total += result[3]
                        paragraph_matches += result[4]
                        block_matches += result[5]
                        paragraph_characters += result[6]
                        block_characters += result[7]
                    if offset - chunk_start >= config.runtime.chunk_bytes:
                        break
                    if 512 * len(rows) >= config.runtime.queue_bytes:
                        break
                if not saw_record:
                    state["complete"] = True
                    break
                write_parquet_atomic(output_dir / f"chunk-{chunk:06d}.parquet", rows, CLEAN_SCHEMA)
                chunk += 1
                state = {
                    "next_offset": offset,
                    "next_row": row_number,
                    "next_chunk": chunk,
                    "scanned": scanned,
                    "empty": empty,
                    "affected": affected,
                    "removed_total": removed_total,
                    "paragraph_matches": paragraph_matches,
                    "block_matches": block_matches,
                    "paragraph_characters": paragraph_characters,
                    "block_characters": block_characters,
                    "complete": bool(
                        (config.runtime.max_rows and row_number >= config.runtime.max_rows)
                        or offset >= manifest["sources"]["pages"]["size"]
                    ),
                }
                atomic_json(state_path, state)
                if stop:
                    candidates.close()
                    return {"complete": False, "stage": "clean"}
            state["complete"] = True
            atomic_json(state_path, state)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    candidates.close()
    clean_glob = str(output_dir / "*.parquet").replace("'", "''")
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW clean_features AS SELECT * FROM read_parquet('{clean_glob}')"
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE d3_membership AS
        SELECT *, row_number() OVER (
          PARTITION BY clean_sha256 ORDER BY page_id IS NULL, page_id, row_number
        ) = 1 AS is_representative
        FROM clean_features WHERE NOT empty
        """
    )
    duckdb_copy_atomic(
        connection,
        """
        SELECT clean_sha256, count(*) AS documents, count(DISTINCT domain_id) AS domains,
               min(row_number) FILTER (WHERE is_representative) AS representative_row
        FROM d3_membership GROUP BY clean_sha256
        """,
        root / "d3_groups.parquet",
    )
    rep_rows = connection.execute(
        "SELECT row_number FROM d3_membership WHERE is_representative ORDER BY row_number"
    ).fetchall()
    write_u64(root / "d3_representatives.u64", (row[0] for row in rep_rows))
    d3_unique = len(rep_rows)
    d3_groups = connection.execute(
        "SELECT count(*) FROM (SELECT clean_sha256 FROM d3_membership GROUP BY 1 HAVING count(*) > 1)"
    ).fetchone()[0]
    connection.close()
    summary = {
        "d2_representatives": scanned,
        "documents_affected": affected,
        "documents_emptied": empty,
        "characters_removed": removed_total,
        "paragraph_matches_removed": paragraph_matches,
        "block_matches_removed": block_matches,
        "paragraph_characters_covered": paragraph_characters,
        "block_characters_covered": block_characters,
        "b_clean_nonempty": scanned - empty,
        "d3_unique": d3_unique,
        "d3_duplicate_groups": d3_groups,
    }
    atomic_json(summary_path, summary)
    return summary


def _clean_record(
    config: Config, candidates: sqlite3.Connection, record
) -> tuple[dict, int, int, int, int, int, int, int]:
    fields = record.fields
    domain_id = parse_int(fields[1])
    try:
        decoded = decode_text(fields[13], fields[15], config.runtime.max_text_bytes)
    except TextDecodeFailure as exc:
        raise RuntimeError("representante D2 deixou de ser decodificável") from exc
    intervals: list[tuple[int, int]] = []
    paragraph_intervals: list[tuple[int, int]] = []
    block_intervals: list[tuple[int, int]] = []
    if domain_id is not None:
        for start, end, digest, _text in paragraph_units(
            decoded.normalized, config.boilerplate.paragraph_min_chars
        ):
            if candidates.execute(
                "SELECT 1 FROM candidate WHERE normalized_sha256=? "
                "AND kind='paragraph' AND digest=?",
                (decoded.normalized_sha256, digest),
            ).fetchone():
                paragraph_intervals.append((start, end))
        intervals.extend(paragraph_intervals)
        for start, end, digest, _text in block_units(
            decoded.normalized,
            config.boilerplate.block_lines,
            config.boilerplate.block_min_chars,
        ):
            if any(start < p_end and end > p_start for p_start, p_end in paragraph_intervals):
                continue
            if candidates.execute(
                "SELECT 1 FROM candidate WHERE normalized_sha256=? AND kind='block' AND digest=?",
                (decoded.normalized_sha256, digest),
            ).fetchone():
                block_intervals.append((start, end))
        intervals.extend(block_intervals)
    clean, removed = remove_intervals(decoded.normalized, intervals)
    row = {
        "row_number": record.row_number,
        "page_id": parse_int(fields[0]),
        "domain_id": domain_id,
        "normalized_sha256": decoded.normalized_sha256,
        "clean_sha256": hashlib.sha256(clean.encode()).hexdigest() if clean else None,
        "characters_before": len(decoded.normalized),
        "characters_after": len(clean),
        "removed_characters": removed,
        "removed_fraction": removed / len(decoded.normalized),
        "empty": not bool(clean),
    }
    return (
        row,
        int(not clean),
        int(removed > 0),
        removed,
        len(paragraph_intervals),
        len(block_intervals),
        _covered_characters(paragraph_intervals),
        _covered_characters(block_intervals),
    )
