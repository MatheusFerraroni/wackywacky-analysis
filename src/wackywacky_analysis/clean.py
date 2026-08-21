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
from .errors import WackyWackyError
from .io import atomic_json, iter_bounded_tsv, parse_int, read_json, sha256_file
from .progress import ByteProgress, ItemProgress, logged_stage
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

CANDIDATE_DATABASE_VERSION = 2
CANDIDATE_BATCH_ROWS = 100_000


def _covered_characters(intervals: list[tuple[int, int]]) -> int:
    covered = 0
    end_seen = 0
    for start, end in sorted(intervals):
        if end <= end_seen:
            continue
        covered += end - max(start, end_seen)
        end_seen = end
    return covered


def _remove_sqlite(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        candidate.unlink(missing_ok=True)


def _candidate_source_sha256(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        value = sidecar.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        return sha256_file(path)
    if len(value) != 64:
        raise WackyWackyError(f"checksum inválido em {sidecar.name}")
    return value


def _candidate_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version != CANDIDATE_DATABASE_VERSION:
            return {}
        return dict(connection.execute("SELECT key, value FROM metadata"))
    except sqlite3.DatabaseError:
        return {}


def _candidate_database_matches(
    path: Path,
    *,
    source_sha256: str,
    enabled: bool,
    complete: bool,
) -> bool:
    if not path.exists():
        return False
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        metadata = _candidate_metadata(connection)
        connection.close()
    except sqlite3.DatabaseError:
        return False
    return (
        metadata.get("source_sha256") == source_sha256
        and metadata.get("enabled") == str(int(enabled))
        and (not complete or metadata.get("complete") == "1")
    )


def _create_candidate_partial(
    path: Path,
    *,
    source_sha256: str,
    enabled: bool,
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        f"""
        PRAGMA user_version={CANDIDATE_DATABASE_VERSION};
        CREATE TABLE candidate(
          domain_id INTEGER NOT NULL,
          kind TEXT NOT NULL,
          digest TEXT NOT NULL,
          PRIMARY KEY(domain_id, kind, digest)
        ) WITHOUT ROWID;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        """
    )
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        (
            ("source_sha256", source_sha256),
            ("enabled", str(int(enabled))),
            ("complete", "0"),
            ("inserted", "0"),
            ("last_domain_id", ""),
            ("last_kind", ""),
            ("last_digest", ""),
        ),
    )
    connection.commit()
    return connection


def _candidate_database(config: Config, root: Path) -> Path:
    target = root / "boilerplate.sqlite"
    source_path = root / "boilerplate_candidates.parquet"
    source_sha256 = _candidate_source_sha256(source_path)
    enabled = config.boilerplate.enabled
    if _candidate_database_matches(
        target,
        source_sha256=source_sha256,
        enabled=enabled,
        complete=True,
    ):
        return target
    _remove_sqlite(target)
    partial = target.with_suffix(".sqlite.partial")
    if not _candidate_database_matches(
        partial,
        source_sha256=source_sha256,
        enabled=enabled,
        complete=False,
    ):
        _remove_sqlite(partial)
    sql = (
        sqlite3.connect(partial)
        if partial.exists()
        else _create_candidate_partial(
            partial,
            source_sha256=source_sha256,
            enabled=enabled,
        )
    )
    metadata = _candidate_metadata(sql)
    if metadata.get("complete") == "1":
        sql.close()
        os.replace(partial, target)
        return target

    source = duckdb.connect()
    source.execute("SET memory_limit = ?", [config.runtime.memory_limit])
    temp_directory = root / "duckdb-tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    source.execute("SET temp_directory = ?", [str(temp_directory)])
    unique_candidates = """
        SELECT domain_id, kind, unit_sha256
        FROM read_parquet(?)
        WHERE domain_id IS NOT NULL
        GROUP BY domain_id, kind, unit_sha256
    """
    total = (
        source.execute(
            f"SELECT count(*) FROM ({unique_candidates})", [str(source_path)]
        ).fetchone()[0]
        if enabled
        else 0
    )
    inserted = int(metadata.get("inserted", "0"))
    actual = sql.execute("SELECT count(*) FROM candidate").fetchone()[0]
    if actual != inserted:
        sql.close()
        source.close()
        _remove_sqlite(partial)
        return _candidate_database(config, root)
    progress = ItemProgress("[6.1/3] Índice de candidatos", total, initial=inserted)
    try:
        if enabled:
            parameters: list[object] = [str(source_path)]
            remaining = unique_candidates
            if inserted:
                last_domain_id = int(metadata["last_domain_id"])
                last_kind = metadata["last_kind"]
                last_digest = metadata["last_digest"]
                remaining += """
                    HAVING domain_id > ?
                       OR (domain_id = ? AND kind > ?)
                       OR (domain_id = ? AND kind = ? AND unit_sha256 > ?)
                """
                parameters.extend(
                    [
                        last_domain_id,
                        last_domain_id,
                        last_kind,
                        last_domain_id,
                        last_kind,
                        last_digest,
                    ]
                )
            cursor = source.execute(
                f"SELECT * FROM ({remaining}) ORDER BY domain_id, kind, unit_sha256", parameters
            )
            while batch := cursor.fetchmany(CANDIDATE_BATCH_ROWS):
                sql.executemany("INSERT INTO candidate VALUES (?, ?, ?)", batch)
                inserted += len(batch)
                last_domain_id, last_kind, last_digest = batch[-1]
                sql.executemany(
                    "UPDATE metadata SET value=? WHERE key=?",
                    (
                        (str(inserted), "inserted"),
                        (str(last_domain_id), "last_domain_id"),
                        (last_kind, "last_kind"),
                        (last_digest, "last_digest"),
                    ),
                )
                sql.commit()
                progress.update(inserted)
        if inserted != total:
            raise WackyWackyError(
                f"índice de candidatos incompleto: {inserted} de {total} registros"
            )
        sql.execute("UPDATE metadata SET value='1' WHERE key='complete'")
        sql.commit()
    except BaseException:
        progress.close()
        raise
    else:
        progress.finish()
    finally:
        source.close()
        sql.close()
    os.replace(partial, target)
    return target


def clean_representatives(config: Config, manifest: dict, root: Path) -> dict:
    summary_path = root / "clean_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_db = _candidate_database(config, root)
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
    progress = ByteProgress(
        "[6.2/3] Limpeza B_clean",
        manifest["sources"]["pages"]["size"],
        initial=offset,
    )
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
                progress.update(offset, detail=f"{scanned:,} representantes D2 processados")
                if stop:
                    progress.close()
                    candidates.close()
                    return {"complete": False, "stage": "clean"}
            state["complete"] = True
            atomic_json(state_path, state)
            progress.finish(detail=f"{scanned:,} representantes D2 processados")
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    candidates.close()
    with logged_stage("[6.3/3] Deduplicação D3"):
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
            "SELECT count(*) FROM (SELECT clean_sha256 FROM d3_membership "
            "GROUP BY 1 HAVING count(*) > 1)"
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
                "SELECT 1 FROM candidate WHERE domain_id=? AND kind='paragraph' AND digest=?",
                (domain_id, digest),
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
                "SELECT 1 FROM candidate WHERE domain_id=? AND kind='block' AND digest=?",
                (domain_id, digest),
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
