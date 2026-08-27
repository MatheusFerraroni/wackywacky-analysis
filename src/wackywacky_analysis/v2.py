from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import os
import shutil
import signal
import sqlite3
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Self

import duckdb
import pyarrow as pa

from .bclean import CandidateIndex, clean_again, iter_bclean
from .config import Config
from .errors import ReviewRequired, WackyWackyError
from .io import atomic_json, iter_bounded_tsv, parse_int, read_json, sha256_file
from .noise import residual_units
from .progress import ByteProgress, logged_stage
from .schema import PAGES_COLUMNS
from .storage import duckdb_connection, duckdb_copy_atomic, write_parquet_atomic, write_u64
from .text import TextDecodeFailure, decode_text, remove_intervals

DISCOVERY_SCHEMA_VERSION = 1
CLEAN_V2_SCHEMA_VERSION = 2
V2_BATCH_BYTES = 8 * 1024 * 1024

_V2_WORKER_CONFIG: Config | None = None
_V2_WORKER_WIKIMEDIA: set[int] | None = None
_V2_WORKER_CANDIDATES: CandidateIndex | None = None

OCCURRENCE_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("source_offset", pa.uint64()),
        ("domain_id", pa.int64()),
        ("kind", pa.string()),
        ("digest", pa.string()),
        ("characters", pa.uint32()),
        ("rule_id", pa.string()),
    ]
)

CANDIDATE_SCHEMA = pa.schema(
    [
        ("domain_id", pa.int64()),
        ("kind", pa.string()),
        ("digest", pa.string()),
        ("characters", pa.uint32()),
        ("document_frequency", pa.uint64()),
        ("domain_documents", pa.uint64()),
        ("threshold", pa.uint64()),
        ("representative_row", pa.uint64()),
        ("representative_offset", pa.uint64()),
        ("rule_id", pa.string()),
    ]
)

CLEAN_V2_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("page_id", pa.int64()),
        ("domain_id", pa.int64()),
        ("recursion_level", pa.int32()),
        ("b_clean_sha256", pa.string()),
        ("clean_v2_sha256", pa.string()),
        ("characters_before", pa.uint64()),
        ("characters_after", pa.uint64()),
        ("removed_characters", pa.uint64()),
        ("removed_fraction", pa.float64()),
        ("mediawiki_matches", pa.uint32()),
        ("short_line_matches", pa.uint32()),
        ("short_pair_matches", pa.uint32()),
        ("mediawiki_characters", pa.uint64()),
        ("short_line_characters", pa.uint64()),
        ("short_pair_characters", pa.uint64()),
        ("empty", pa.bool_()),
    ]
)


def v2_root(root: Path) -> Path:
    return root / "v2"


def _v2_worker_init(config: Config, wikimedia: set[int], candidate_path: str | None) -> None:
    global _V2_WORKER_CONFIG, _V2_WORKER_WIKIMEDIA, _V2_WORKER_CANDIDATES
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _V2_WORKER_CONFIG = config
    _V2_WORKER_WIKIMEDIA = wikimedia
    _V2_WORKER_CANDIDATES = (
        CandidateIndex.from_path(Path(candidate_path)) if candidate_path else None
    )


def _discover_batch(batch: list[tuple[int, int, int | None, str]]) -> list[dict]:
    if _V2_WORKER_CONFIG is None or _V2_WORKER_WIKIMEDIA is None:
        raise RuntimeError("worker de descoberta v2 não inicializado")
    rows: list[dict] = []
    for row_number, source_offset, domain_id, text in batch:
        seen: set[tuple[str, str]] = set()
        for unit in residual_units(
            text,
            _V2_WORKER_CONFIG.boilerplate_v2,
            is_wikimedia=domain_id in _V2_WORKER_WIKIMEDIA,
        ):
            key = (unit.kind, unit.digest)
            if key in seen or domain_id is None:
                continue
            seen.add(key)
            rows.append(
                {
                    "row_number": row_number,
                    "source_offset": source_offset,
                    "domain_id": domain_id,
                    "kind": unit.kind,
                    "digest": unit.digest,
                    "characters": unit.end - unit.start,
                    "rule_id": unit.rule_id,
                }
            )
    return rows


def _clean_batch(batch: list[tuple]) -> list[dict]:
    if _V2_WORKER_CONFIG is None or _V2_WORKER_CANDIDATES is None:
        raise RuntimeError("worker de limpeza v2 não inicializado")
    output = []
    for (
        row_number,
        page_id,
        domain_id,
        recursion_level,
        text,
        md5_diagnostic,
    ) in batch:
        clean, metrics = _apply_residual(_V2_WORKER_CONFIG, _V2_WORKER_CANDIDATES, domain_id, text)
        removed = metrics.get("removed_characters", 0)
        output.append(
            {
                "row": {
                    "row_number": row_number,
                    "page_id": page_id,
                    "domain_id": domain_id,
                    "recursion_level": recursion_level,
                    "b_clean_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "clean_v2_sha256": hashlib.sha256(clean.encode()).hexdigest()
                    if clean
                    else None,
                    "characters_before": len(text),
                    "characters_after": len(clean),
                    "removed_characters": removed,
                    "removed_fraction": removed / len(text) if text else 0,
                    "mediawiki_matches": metrics.get("mediawiki_matches", 0),
                    "short_line_matches": metrics.get("short_line_matches", 0),
                    "short_pair_matches": metrics.get("short_pair_matches", 0),
                    "mediawiki_characters": metrics.get("mediawiki_characters", 0),
                    "short_line_characters": metrics.get("short_line_characters", 0),
                    "short_pair_characters": metrics.get("short_pair_characters", 0),
                    "empty": not bool(clean),
                },
                "metrics": metrics,
                "md5_diagnostic": md5_diagnostic or "unavailable",
            }
        )
    return output


class _OrderedV2Pool:
    """Persistent lightweight pool with ordered, byte-bounded results."""

    def __init__(
        self,
        config: Config,
        *,
        mode: str,
        wikimedia: set[int],
        candidate_path: Path | None = None,
    ) -> None:
        self.mode = mode
        self.maximum_pending = max(1, 2 * config.runtime.workers)
        self.maximum_bytes = max(1, config.runtime.queue_bytes)
        self.pending_bytes = 0
        self.pending: deque[tuple[Future, int]] = deque()
        self.executor: ProcessPoolExecutor | None = None
        self.inline_candidates = (
            CandidateIndex.from_path(candidate_path) if candidate_path else None
        )
        self.config = config
        self.wikimedia = wikimedia
        if config.runtime.workers > 1:
            self.executor = ProcessPoolExecutor(
                max_workers=config.runtime.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_v2_worker_init,
                initargs=(
                    config,
                    wikimedia,
                    str(candidate_path) if candidate_path else None,
                ),
            )

    def _run_inline(self, batch):
        if self.mode == "discovery":
            rows = []
            for row_number, source_offset, domain_id, text in batch:
                seen: set[tuple[str, str]] = set()
                for unit in residual_units(
                    text,
                    self.config.boilerplate_v2,
                    is_wikimedia=domain_id in self.wikimedia,
                ):
                    key = (unit.kind, unit.digest)
                    if key in seen or domain_id is None:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "row_number": row_number,
                            "source_offset": source_offset,
                            "domain_id": domain_id,
                            "kind": unit.kind,
                            "digest": unit.digest,
                            "characters": unit.end - unit.start,
                            "rule_id": unit.rule_id,
                        }
                    )
            return rows
        if self.inline_candidates is None:
            raise RuntimeError("índice residual v2 ausente")
        output = []
        for task in batch:
            row_number, page_id, domain_id, recursion_level, text, diagnostic = task
            clean, metrics = _apply_residual(self.config, self.inline_candidates, domain_id, text)
            removed = metrics.get("removed_characters", 0)
            output.append(
                {
                    "row": {
                        "row_number": row_number,
                        "page_id": page_id,
                        "domain_id": domain_id,
                        "recursion_level": recursion_level,
                        "b_clean_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "clean_v2_sha256": hashlib.sha256(clean.encode()).hexdigest()
                        if clean
                        else None,
                        "characters_before": len(text),
                        "characters_after": len(clean),
                        "removed_characters": removed,
                        "removed_fraction": removed / len(text) if text else 0,
                        "mediawiki_matches": metrics.get("mediawiki_matches", 0),
                        "short_line_matches": metrics.get("short_line_matches", 0),
                        "short_pair_matches": metrics.get("short_pair_matches", 0),
                        "mediawiki_characters": metrics.get("mediawiki_characters", 0),
                        "short_line_characters": metrics.get("short_line_characters", 0),
                        "short_pair_characters": metrics.get("short_pair_characters", 0),
                        "empty": not bool(clean),
                    },
                    "metrics": metrics,
                    "md5_diagnostic": diagnostic or "unavailable",
                }
            )
        return output

    def _oldest(self):
        future, size = self.pending.popleft()
        self.pending_bytes -= size
        try:
            return future.result()
        except BaseException as exc:
            raise WackyWackyError(f"worker da etapa B_clean_v2 falhou: {exc}") from exc

    def submit(self, batch: list, size: int) -> list[list[dict]]:
        completed = []
        while self.pending and (
            len(self.pending) >= self.maximum_pending
            or self.pending_bytes + size > self.maximum_bytes
        ):
            completed.append(self._oldest())
        if self.executor is None:
            completed.append(self._run_inline(batch))
            return completed
        target = _discover_batch if self.mode == "discovery" else _clean_batch
        self.pending.append((self.executor.submit(target, batch), size))
        self.pending_bytes += size
        return completed

    def drain(self) -> list[list[dict]]:
        output = []
        while self.pending:
            output.append(self._oldest())
        return output

    def close(self) -> None:
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _discard_indexed(directory: Path, keep_before: int) -> None:
    if not directory.exists():
        return
    for path in directory.glob("chunk-*.parquet"):
        try:
            index = int(path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if index >= keep_before:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)
    for partial in directory.glob("*.partial"):
        partial.unlink(missing_ok=True)


def _wikimedia_domains(config: Config, root: Path) -> set[int]:
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    try:
        return {
            row[0]
            for row in connection.execute("SELECT id FROM domains WHERE is_wikimedia").fetchall()
        }
    finally:
        connection.close()


def discover_v2_candidates(config: Config, manifest: dict, root: Path) -> dict:
    if not config.boilerplate_v2.enabled:
        return {"status": "disabled", "complete": True, "candidates": 0}
    target_root = v2_root(root)
    target_root.mkdir(parents=True, exist_ok=True)
    summary_path = target_root / "discovery_summary.json"
    current = read_json(summary_path, {})
    if (
        current.get("complete")
        and current.get("v2_fingerprint") == config.v2_fingerprint
        and (target_root / "candidates.parquet").exists()
    ):
        return current
    state_path = target_root / "discovery" / "state.json"
    state = read_json(state_path, {})
    if state and (
        state.get("schema_version") != DISCOVERY_SCHEMA_VERSION
        or state.get("snapshot_id") != manifest["snapshot_id"]
        or state.get("v2_fingerprint") != config.v2_fingerprint
        or state.get("source_sha256") != manifest["sources"]["pages"]["sha256"]
    ):
        shutil.rmtree(target_root / "discovery", ignore_errors=True)
        state = {}
    chunks = target_root / "discovery" / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    chunk = int(state.get("next_chunk", 0))
    offset = int(state.get("next_offset", 0))
    row_number = int(state.get("next_row", 0))
    documents = int(state.get("documents", 0))
    occurrences = int(state.get("occurrences", 0))
    _discard_indexed(chunks, chunk)
    wikimedia = _wikimedia_domains(config, root)
    progress = ByteProgress(
        "Descoberta de boilerplate residual",
        manifest["sources"]["pages"]["size"],
        initial=offset,
    )
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    reached_end = False
    try:
        iterator = iter_bclean(config, root, start_offset=offset, start_row=row_number)
        with _OrderedV2Pool(config, mode="discovery", wikimedia=wikimedia) as pool:
            while not reached_end and not stop:
                rows: list[dict] = []
                batch: list[tuple[int, int, int | None, str]] = []
                batch_bytes = 0
                batch_limit = max(1, min(V2_BATCH_BYTES, config.runtime.queue_bytes))
                chunk_start = offset
                saw = False

                def merge(completed: list[dict], rows=rows) -> None:
                    nonlocal occurrences
                    rows.extend(completed)
                    occurrences += len(completed)

                def submit() -> None:
                    nonlocal batch, batch_bytes
                    if not batch:
                        return
                    for completed in pool.submit(batch, batch_bytes):
                        merge(completed)
                    batch = []
                    batch_bytes = 0

                for item in iterator:
                    saw = True
                    offset = item.source.end_offset
                    row_number = item.source.row_number
                    documents += 1
                    size = 4 * len(item.text)
                    if batch and batch_bytes + size > batch_limit:
                        submit()
                    batch.append(
                        (
                            item.source.row_number,
                            item.source.offset,
                            item.domain_id,
                            item.text,
                        )
                    )
                    batch_bytes += size
                    if batch_bytes >= batch_limit or len(batch) >= 1024:
                        submit()
                    progress.update(
                        offset,
                        detail=(
                            f"{config.runtime.workers} workers ativos; checkpoint {chunk:,}; "
                            f"{documents:,} documentos B_clean"
                        ),
                    )
                    if offset - chunk_start >= config.runtime.chunk_bytes:
                        break
                submit()
                for completed in pool.drain():
                    merge(completed)
                if not saw:
                    reached_end = True
                    offset = manifest["sources"]["pages"]["size"]
                if rows or not any(chunks.glob("chunk-*.parquet")):
                    write_parquet_atomic(
                        chunks / f"chunk-{chunk:06d}.parquet", rows, OCCURRENCE_SCHEMA
                    )
                    chunk += 1
                state = {
                    "schema_version": DISCOVERY_SCHEMA_VERSION,
                    "snapshot_id": manifest["snapshot_id"],
                    "source_sha256": manifest["sources"]["pages"]["sha256"],
                    "v2_fingerprint": config.v2_fingerprint,
                    "next_offset": offset,
                    "next_row": row_number,
                    "next_chunk": chunk,
                    "documents": documents,
                    "occurrences": occurrences,
                    "complete": reached_end,
                }
                atomic_json(state_path, state)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    if not reached_end:
        progress.finish(detail="checkpoint salvo; execução interrompida")
        return {"complete": False, "phase": "v2_discovery", **state}
    progress.finish(detail=f"{documents:,} documentos B_clean")
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    occurrence_glob = str(chunks / "chunk-*.parquet").replace("'", "''")
    candidates = target_root / "candidates.parquet"
    threshold = (
        f"greatest({config.boilerplate_v2.frequency_min_documents}, "
        f"ceil({config.boilerplate_v2.frequency_fraction} * d.documents))"
    )
    duckdb_copy_atomic(
        connection,
        f"""
        WITH domain_docs AS (
          SELECT domain_id, count(*) AS documents FROM d3_membership
          WHERE is_representative AND domain_id IS NOT NULL GROUP BY domain_id
        ), units AS (
          SELECT o.domain_id, o.kind, o.digest, max(o.characters) characters,
                 count(DISTINCT o.row_number) document_frequency,
                 arg_min(o.row_number, o.row_number) representative_row,
                 arg_min(o.source_offset, o.row_number) representative_offset,
                 min(o.rule_id) rule_id
          FROM read_parquet('{occurrence_glob}') o
          GROUP BY o.domain_id, o.kind, o.digest
        )
        SELECT u.domain_id, u.kind, u.digest, u.characters, u.document_frequency,
               d.documents domain_documents,
               CAST(CASE WHEN u.kind='mediawiki' THEN 1 ELSE {threshold} END AS UBIGINT) threshold,
               u.representative_row, u.representative_offset, u.rule_id
        FROM units u JOIN domain_docs d USING(domain_id)
        WHERE u.kind='mediawiki' OR u.document_frequency >= {threshold}
        ORDER BY u.domain_id, u.kind, u.digest
        """,
        candidates,
    )
    aggregate = connection.execute(
        "SELECT count(*), count(*) FILTER (WHERE kind='mediawiki'), "
        "count(*) FILTER (WHERE kind='short_line'), "
        "count(*) FILTER (WHERE kind='short_pair') FROM read_parquet(?)",
        [str(candidates)],
    ).fetchone()
    cross_domain = connection.execute(
        f"""
        SELECT count(*) FROM (
          SELECT kind, digest FROM read_parquet('{occurrence_glob}')
          GROUP BY kind, digest HAVING count(DISTINCT domain_id)>1
        )
        """
    ).fetchone()[0]
    connection.close()
    summary = {
        "schema_version": DISCOVERY_SCHEMA_VERSION,
        "v2_fingerprint": config.v2_fingerprint,
        "complete": True,
        "documents": documents,
        "occurrences": occurrences,
        "candidates": aggregate[0],
        "by_kind": {
            "mediawiki": aggregate[1],
            "short_line": aggregate[2],
            "short_pair": aggregate[3],
        },
        "cross_domain_units_measured": cross_domain,
    }
    atomic_json(summary_path, summary)
    return summary


def _sample_id(domain_id: int, kind: str, digest: str) -> str:
    return hashlib.sha256(f"v2:{domain_id}:{kind}:{digest}".encode()).hexdigest()[:20]


def _review_preview(text: str, limit: int = 4_000) -> tuple[str, bool]:
    from .review import _review_preview as sanitize

    return sanitize(text, limit)


def _selected_review_rows(config: Config, root: Path) -> list[dict]:
    path = v2_root(root) / "candidates.parquet"
    connection = duckdb.connect()
    rows: list[dict] = []
    for kind, target in (
        ("mediawiki", config.boilerplate_v2.review_mediawiki),
        ("short_line", config.boilerplate_v2.review_lines),
        ("short_pair", config.boilerplate_v2.review_pairs),
    ):
        result = connection.execute(
            """
            SELECT * FROM read_parquet(?) WHERE kind=?
            ORDER BY hash(domain_id, digest, ?) LIMIT ?
            """,
            [str(path), kind, config.boilerplate_v2.review_seed, target],
        ).fetchall()
        columns = [item[0] for item in connection.description]
        rows.extend(dict(zip(columns, row, strict=True)) for row in result)
    connection.close()
    for row in rows:
        row["sample_id"] = _sample_id(row["domain_id"], row["kind"], row["digest"])
    return rows


def _record_at_offset(config: Config, row: dict):
    with config.analysis_pages.open("rb") as handle:
        records = iter_bounded_tsv(
            handle,
            columns=len(PAGES_COLUMNS),
            max_line_bytes=config.runtime.max_line_bytes,
            start_offset=int(row["representative_offset"]),
            start_row=int(row["representative_row"]) - 1,
            max_rows=0,
        )
        return next(records, None)


def export_v2_review(
    config: Config, manifest: dict, root: Path, output: Path | None = None
) -> Path | None:
    discovery = discover_v2_candidates(config, manifest, root)
    if discovery.get("complete") is False:
        raise WackyWackyError("descoberta v2 ainda não foi concluída")
    selected = _selected_review_rows(config, root)
    target_root = v2_root(root)
    if not selected:
        atomic_json(
            target_root / "review_confirmation.json",
            {"status": "not_needed", "sample_size": 0, "v2_fingerprint": config.v2_fingerprint},
        )
        return None
    old_candidates = CandidateIndex.from_path(root / "boilerplate.sqlite")
    wikimedia = _wikimedia_domains(config, root)
    previews: dict[str, tuple[str, bool]] = {}
    for row in selected:
        record = _record_at_offset(config, row)
        if record is None or record.fields is None:
            raise WackyWackyError("offset privado de revisão v2 inválido")
        fields = record.fields
        try:
            decoded = decode_text(fields[13], fields[15], config.runtime.max_text_bytes)
        except TextDecodeFailure as exc:
            raise WackyWackyError("texto selecionado para revisão deixou de ser válido") from exc
        domain_id = parse_int(fields[1])
        clean = clean_again(config, old_candidates, domain_id, decoded.normalized)
        matched = next(
            (
                unit
                for unit in residual_units(
                    clean,
                    config.boilerplate_v2,
                    is_wikimedia=domain_id in wikimedia,
                )
                if unit.kind == row["kind"] and unit.digest == row["digest"]
            ),
            None,
        )
        if matched is None:
            raise WackyWackyError("não foi possível reconstruir candidato v2")
        previews[row["sample_id"]] = _review_preview(clean[matched.start : matched.end])
    output = output or (target_root / "review" / "boilerplate-review.csv")
    sample_path = target_root / "review_sample.json"
    previous_sample = read_json(sample_path, {})
    if output.exists() and previous_sample:
        previous_hash = previous_sample.get("export_sha256")
        current_hash = sha256_file(output)
        if previous_hash and current_hash != previous_hash:
            raise WackyWackyError(
                "CSV v2 possui alterações locais; importe-o ou escolha outro --output"
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    fieldnames = [
        "sample_id",
        "frequencia",
        "label",
        "kind",
        "domain_id",
        "domain_documents",
        "characters",
        "rule_id",
        "preview_truncated",
        "text_preview",
    ]
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted(selected, key=lambda item: item["sample_id"]):
            preview, truncated = previews[row["sample_id"]]
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "frequencia": row["document_frequency"],
                    "label": "boilerplate",
                    "kind": row["kind"],
                    "domain_id": row["domain_id"],
                    "domain_documents": row["domain_documents"],
                    "characters": row["characters"],
                    "rule_id": row["rule_id"] or "",
                    "preview_truncated": "sim" if truncated else "não",
                    "text_preview": preview,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    metadata = {
        "v2_fingerprint": config.v2_fingerprint,
        "sample_ids": sorted(row["sample_id"] for row in selected),
        "rows": {
            row["sample_id"]: {key: row[key] for key in ("domain_id", "kind", "digest", "rule_id")}
            for row in selected
        },
        "export_sha256": sha256_file(output),
        "path": str(output),
    }
    atomic_json(sample_path, metadata)
    (output.parent / "LEIA-ME-revisao.txt").write_text(
        "Todos os itens começam com label=boilerplate.\n"
        "Importar sem mudanças confirma os padrões sugeridos.\n"
        "Altere somente exceções para conteúdo ou incerto.\n"
        "Esta confirmação não é apresentada como estimativa de precisão.\n",
        encoding="utf-8",
    )
    return output


def import_v2_review(config: Config, root: Path, source: Path) -> dict:
    target_root = v2_root(root)
    sample = read_json(target_root / "review_sample.json", {})
    if sample.get("v2_fingerprint") != config.v2_fingerprint:
        raise WackyWackyError("amostra de revisão v2 incompatível com a configuração")
    expected = set(sample.get("sample_ids", []))
    labels: dict[str, str] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row.get("sample_id", "")
            label = row.get("label", "").strip().casefold()
            if (
                sample_id not in expected
                or sample_id in labels
                or label
                not in {
                    "boilerplate",
                    "conteúdo",
                    "incerto",
                }
            ):
                raise WackyWackyError("revisão v2 contém ID ou rótulo inválido")
            labels[sample_id] = label
    if set(labels) != expected:
        raise WackyWackyError("revisão v2 está incompleta")
    exceptions = [sample_id for sample_id, label in labels.items() if label != "boilerplate"]
    confirmation = {
        "status": "confirmed_with_exceptions" if exceptions else "confirmed_default",
        "sample_size": len(labels),
        "exceptions": len(exceptions),
        "v2_fingerprint": config.v2_fingerprint,
        "labels_sha256": hashlib.sha256(
            json.dumps(labels, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "interpretation": "confirmação do usuário; não estima precisão humana",
    }
    atomic_json(target_root / "review_labels.json", labels)
    atomic_json(target_root / "review_confirmation.json", confirmation)
    return confirmation


def require_v2_confirmation(config: Config, manifest: dict, root: Path) -> dict:
    if not config.boilerplate_v2.enabled:
        return {"status": "disabled", "sample_size": 0}
    discovery = discover_v2_candidates(config, manifest, root)
    if discovery.get("candidates", 0) == 0:
        confirmation = {
            "status": "not_needed",
            "sample_size": 0,
            "v2_fingerprint": config.v2_fingerprint,
        }
        atomic_json(v2_root(root) / "review_confirmation.json", confirmation)
        return confirmation
    confirmation = read_json(v2_root(root) / "review_confirmation.json", {})
    if confirmation.get("v2_fingerprint") != config.v2_fingerprint or confirmation.get(
        "status"
    ) not in {"confirmed_default", "confirmed_with_exceptions", "not_needed"}:
        sample = export_v2_review(config, manifest, root)
        raise ReviewRequired(
            f"confirmação privada B_clean_v2 necessária: {sample}; "
            "labels começam como boilerplate, altere somente exceções"
        )
    return confirmation


def _approved_candidates(config: Config, root: Path) -> list[dict]:
    target_root = v2_root(root)
    sample = read_json(target_root / "review_sample.json", {})
    labels = read_json(target_root / "review_labels.json", {})
    excluded: set[tuple[int, str, str]] = set()
    disabled_rules: set[str] = set()
    for sample_id, label in labels.items():
        if label == "boilerplate":
            continue
        row = sample.get("rows", {}).get(sample_id, {})
        if row.get("kind") == "mediawiki" and row.get("rule_id"):
            disabled_rules.add(row["rule_id"])
        else:
            excluded.add((int(row["domain_id"]), row["kind"], row["digest"]))
    connection = duckdb.connect()
    try:
        table = connection.execute(
            "SELECT * FROM read_parquet(?)",
            [str(target_root / "candidates.parquet")],
        ).to_arrow_table()
    finally:
        connection.close()
    return [
        row
        for row in table.to_pylist()
        if (row["domain_id"], row["kind"], row["digest"]) not in excluded
        and (not row["rule_id"] or row["rule_id"] not in disabled_rules)
    ]


def _candidate_databases(config: Config, root: Path) -> tuple[Path, Path]:
    target_root = v2_root(root)
    combined = target_root / "boilerplate.sqlite"
    residual = target_root / "residual.sqlite"
    confirmation = read_json(target_root / "review_confirmation.json", {})
    identity = hashlib.sha256(
        (config.v2_fingerprint + confirmation.get("labels_sha256", "not-needed")).encode()
    ).hexdigest()
    metadata = read_json(target_root / "candidate_databases.json", {})
    if combined.exists() and residual.exists() and metadata.get("identity") == identity:
        return combined, residual
    approved = _approved_candidates(config, root)
    for target, include_v1 in ((combined, True), (residual, False)):
        partial = target.with_suffix(target.suffix + ".partial")
        partial.unlink(missing_ok=True)
        connection = sqlite3.connect(partial)
        connection.execute(
            "CREATE TABLE candidate(domain_id INTEGER NOT NULL, kind TEXT NOT NULL, "
            "digest TEXT NOT NULL, PRIMARY KEY(domain_id,kind,digest)) WITHOUT ROWID"
        )
        if include_v1:
            source = sqlite3.connect(f"file:{root / 'boilerplate.sqlite'}?mode=ro", uri=True)
            connection.executemany(
                "INSERT INTO candidate VALUES (?,?,?)",
                source.execute("SELECT domain_id,kind,digest FROM candidate"),
            )
            source.close()
        connection.executemany(
            "INSERT OR IGNORE INTO candidate VALUES (?,?,?)",
            ((row["domain_id"], row["kind"], row["digest"]) for row in approved),
        )
        connection.commit()
        connection.close()
        os.replace(partial, target)
    atomic_json(target_root / "candidate_databases.json", {"identity": identity})
    return combined, residual


def _covered_length(intervals: list[tuple[int, int]]) -> int:
    covered = 0
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    for start, end in merged:
        covered += end - start
    return covered


def _apply_residual(
    config: Config,
    candidates: CandidateIndex,
    domain_id: int | None,
    text: str,
) -> tuple[str, dict[str, int]]:
    metrics = Counter()
    if domain_id is None:
        return text, dict(metrics)
    is_wikimedia = candidates.has_kind(domain_id, "mediawiki")
    units = residual_units(text, config.boilerplate_v2, is_wikimedia=is_wikimedia)
    intervals: list[tuple[int, int]] = []
    for kind in ("mediawiki", "short_line", "short_pair"):
        for unit in units:
            interval = (unit.start, unit.end)
            if unit.kind != kind or not candidates.contains(domain_id, kind, unit.digest):
                continue
            before = _covered_length(intervals)
            intervals.append(interval)
            metrics[f"{kind}_matches"] += 1
            metrics[f"{kind}_characters"] += _covered_length(intervals) - before
    cleaned, removed = remove_intervals(text, intervals)
    metrics["removed_characters"] = removed
    return cleaned, dict(metrics)


def clean_v2(config: Config, manifest: dict, root: Path) -> dict:
    if not config.boilerplate_v2.enabled:
        return {"status": "disabled", "complete": True}
    target_root = v2_root(root)
    summary_path = target_root / "clean_summary.json"
    current = read_json(summary_path, {})
    confirmation = require_v2_confirmation(config, manifest, root)
    identity = hashlib.sha256(
        (config.v2_fingerprint + confirmation.get("labels_sha256", "not-needed")).encode()
    ).hexdigest()
    if (
        current.get("complete")
        and current.get("schema_version") == CLEAN_V2_SCHEMA_VERSION
        and current.get("identity") == identity
    ):
        return current
    combined_db, residual_db = _candidate_databases(config, root)
    output_dir = target_root / "clean" / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = target_root / "clean" / "state.json"
    state = read_json(state_path, {})
    if state and (
        state.get("schema_version") != CLEAN_V2_SCHEMA_VERSION or state.get("identity") != identity
    ):
        shutil.rmtree(target_root / "clean", ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        state = {}
    offset = int(state.get("next_offset", 0))
    row_number = int(state.get("next_row", 0))
    chunk = int(state.get("next_chunk", 0))
    documents = int(state.get("documents", 0))
    empty = int(state.get("empty", 0))
    affected = int(state.get("affected", 0))
    totals = Counter(state.get("totals", {}))
    md5_diagnostic = Counter(state.get("md5_diagnostic", {}))
    _discard_indexed(output_dir, chunk)
    progress = ByteProgress(
        "Limpeza B_clean_v2", manifest["sources"]["pages"]["size"], initial=offset
    )
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    reached_end = False
    iterator = iter_bclean(
        config,
        root,
        start_offset=offset,
        start_row=row_number,
        diagnose_md5=True,
    )
    try:
        with _OrderedV2Pool(
            config,
            mode="clean",
            wikimedia=set(),
            candidate_path=residual_db,
        ) as pool:
            while not reached_end and not stop:
                rows: list[dict] = []
                batch: list[tuple] = []
                batch_bytes = 0
                batch_limit = max(1, min(V2_BATCH_BYTES, config.runtime.queue_bytes))
                chunk_start = offset
                saw = False

                def merge(completed: list[dict], rows=rows) -> None:
                    nonlocal affected, empty
                    for result in completed:
                        row = result["row"]
                        rows.append(row)
                        removed = row["removed_characters"]
                        affected += int(removed > 0)
                        empty += int(row["empty"])
                        totals.update(result["metrics"])
                        md5_diagnostic[result["md5_diagnostic"]] += 1

                def submit() -> None:
                    nonlocal batch, batch_bytes
                    if not batch:
                        return
                    for completed in pool.submit(batch, batch_bytes):
                        merge(completed)
                    batch = []
                    batch_bytes = 0

                for item in iterator:
                    saw = True
                    offset = item.source.end_offset
                    row_number = item.source.row_number
                    documents += 1
                    size = 4 * len(item.text)
                    if batch and batch_bytes + size > batch_limit:
                        submit()
                    batch.append(
                        (
                            item.source.row_number,
                            parse_int(item.source.fields[0]) if item.source.fields else None,
                            item.domain_id,
                            item.recursion_level,
                            item.text,
                            item.md5_diagnostic,
                        )
                    )
                    batch_bytes += size
                    if batch_bytes >= batch_limit or len(batch) >= 1024:
                        submit()
                    progress.update(
                        offset,
                        detail=(
                            f"{config.runtime.workers} workers ativos; checkpoint {chunk:,}; "
                            f"{documents:,} representantes D3"
                        ),
                    )
                    if offset - chunk_start >= config.runtime.chunk_bytes:
                        break
                submit()
                for completed in pool.drain():
                    merge(completed)
                if not saw:
                    reached_end = True
                    offset = manifest["sources"]["pages"]["size"]
                if rows or not any(output_dir.glob("chunk-*.parquet")):
                    write_parquet_atomic(
                        output_dir / f"chunk-{chunk:06d}.parquet", rows, CLEAN_V2_SCHEMA
                    )
                    chunk += 1
                state = {
                    "schema_version": CLEAN_V2_SCHEMA_VERSION,
                    "identity": identity,
                    "next_offset": offset,
                    "next_row": row_number,
                    "next_chunk": chunk,
                    "documents": documents,
                    "empty": empty,
                    "affected": affected,
                    "totals": dict(totals),
                    "md5_diagnostic": dict(md5_diagnostic),
                    "complete": reached_end,
                }
                atomic_json(state_path, state)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    if not reached_end:
        progress.finish(detail="checkpoint salvo; execução interrompida")
        return {"complete": False, "phase": "v2_clean", **state}
    progress.finish(detail=f"{documents:,} representantes D3")
    with logged_stage("Deduplicação exata D4"):
        connection = duckdb_connection(
            root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
        )
        clean_glob = str(output_dir / "*.parquet").replace("'", "''")
        connection.execute(
            f"CREATE OR REPLACE VIEW clean_v2_features AS SELECT * FROM read_parquet('{clean_glob}')"
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE d4_membership AS
            SELECT *, row_number() OVER (
              PARTITION BY clean_v2_sha256 ORDER BY page_id IS NULL, page_id, row_number
            )=1 AS is_representative
            FROM clean_v2_features WHERE NOT empty
            """
        )
        duckdb_copy_atomic(
            connection,
            """
            SELECT clean_v2_sha256, count(*) documents, count(DISTINCT domain_id) domains,
                   min(row_number) FILTER (WHERE is_representative) representative_row
            FROM d4_membership GROUP BY clean_v2_sha256
            """,
            target_root / "d4_groups.parquet",
        )
        reps = connection.execute(
            "SELECT row_number FROM d4_membership WHERE is_representative ORDER BY row_number"
        ).fetchall()
        groups = connection.execute(
            "SELECT count(*) FROM (SELECT clean_v2_sha256 FROM d4_membership "
            "GROUP BY 1 HAVING count(*)>1)"
        ).fetchone()[0]
        connection.close()
    write_u64(target_root / "d4_representatives.u64", (row[0] for row in reps))
    write_u64(target_root / "d3_representatives.u64", (row[0] for row in reps))
    # Generic B_clean iterators use this combined database to reconstruct v2.
    if combined_db != target_root / "boilerplate.sqlite":
        raise RuntimeError("caminho inesperado para índice combinado v2")
    summary = {
        "schema_version": CLEAN_V2_SCHEMA_VERSION,
        "identity": identity,
        "complete": True,
        "d3_representatives": documents,
        "documents_affected": affected,
        "documents_emptied": empty,
        "b_clean_v2_nonempty": documents - empty,
        "d4_unique": len(reps),
        "d4_duplicate_groups": groups,
        "removed_characters": totals["removed_characters"],
        "mediawiki_matches_removed": totals["mediawiki_matches"],
        "short_line_matches_removed": totals["short_line_matches"],
        "short_pair_matches_removed": totals["short_pair_matches"],
        "mediawiki_characters_removed": totals["mediawiki_characters"],
        "short_line_characters_removed": totals["short_line_characters"],
        "short_pair_characters_removed": totals["short_pair_characters"],
        "confirmation": confirmation,
        "text_md5_diagnostic": dict(sorted(md5_diagnostic.items())),
        "text_md5_filtering": False,
    }
    atomic_json(summary_path, summary)
    return summary
