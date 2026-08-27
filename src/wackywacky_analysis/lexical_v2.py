from __future__ import annotations

import hashlib
import shutil
import signal
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .bclean import iter_bclean
from .config import Config
from .content import (
    BIGRAM_MARGINAL_SCHEMA,
    CONTENT_SCHEMA_VERSION,
    MOJIBAKE_MARKERS,
    TRIGRAM_SCHEMA,
    TRIGRAM_STATE_SCHEMA,
    EmbeddedContentWriter,
    _load_nlp,
    _reduce_content,
)
from .errors import WackyWackyError
from .io import atomic_json, read_json
from .lexical import (
    BIGRAM_CANDIDATE_SCHEMA,
    BIGRAM_RECOUNT_SCHEMA,
    BIGRAM_SCHEMA,
    DOCUMENT_SCHEMA,
    LEXICAL_BATCH_BYTES,
    SPILL_SCHEMA,
    SpaceSaving,
    SpillVocabulary,
    _checkpoint_artifacts_valid,
    _discard_indexed_files,
    _OrderedBatchPool,
    _restore_bigram_state,
    _write_bigram_state,
    spacy_identity,
)
from .progress import LOGGER, ByteProgress, logged_stage
from .storage import duckdb_connection, duckdb_copy_atomic, write_parquet_atomic
from .v2 import v2_root

V2_LEXICAL_STATE_VERSION = 1
V2_RECOUNT_STATE_VERSION = 1


def _reset_derived(target: Path, reason: str) -> None:
    from .content import _remove_content_products

    LOGGER.info("B_clean_v2: descartando derivados incompatíveis (%s)", reason)
    shutil.rmtree(target / "lexical", ignore_errors=True)
    _remove_content_products(target)
    for name in (
        "lexical_summary.json",
        "vocabulary.parquet",
        "bigrams.parquet",
        "bigram_candidates.parquet",
        "bigram_candidates.json",
    ):
        path = target / name
        path.unlink(missing_ok=True)
        path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)


def _state_compatible(
    config: Config, manifest: dict, clean: dict, identity: dict, state: dict
) -> bool:
    return bool(
        state.get("schema_version") == V2_LEXICAL_STATE_VERSION
        and state.get("snapshot_id") == manifest["snapshot_id"]
        and state.get("v2_fingerprint") == config.v2_fingerprint
        and state.get("clean_identity") == clean.get("identity")
        and state.get("content_fingerprint") == config.content_fingerprint
        and state.get("source_sha256") == manifest["sources"]["pages"]["sha256"]
        and state.get("spacy") == identity
    )


def _load_state(config: Config, manifest: dict, target: Path, clean: dict, identity: dict) -> dict:
    state_path = target / "lexical" / "state.json"
    state = read_json(state_path, {})
    has_products = ((target / "lexical").exists() and any((target / "lexical").iterdir())) or (
        target / "content"
    ).exists()
    if state and not _state_compatible(config, manifest, clean, identity, state):
        _reset_derived(target, "estado ou método v2 alterado")
        return {}
    if not state and has_products:
        _reset_derived(target, "estrutura lexical v2 sem checkpoint")
        return {}
    if state and not _checkpoint_artifacts_valid(target, state.get("artifacts", {})):
        _reset_derived(target, "checksum de checkpoint inválido")
        return {}
    if state:
        _discard_indexed_files(
            target / "lexical" / "documents",
            "documents",
            int(state.get("document_index", 0)),
        )
        _discard_indexed_files(
            target / "lexical" / "spills",
            "spill",
            int(state.get("vocabulary_index", 0)),
        )
        for partial in (target / "lexical").rglob("*.partial"):
            partial.unlink(missing_ok=True)
        LOGGER.info(
            "B_clean_v2: retomando léxico no offset %s, linha %s, checkpoint %s",
            f"{int(state.get('next_offset', 0)):,}",
            f"{int(state.get('next_row', 0)):,}",
            f"{int(state.get('next_chunk', 0)):,}",
        )
    return state


def _scan(
    config: Config,
    manifest: dict,
    target: Path,
    clean: dict,
    identity: dict,
    state: dict,
) -> tuple[dict, SpaceSaving]:
    lexical_root = target / "lexical"
    documents_root = lexical_root / "documents"
    spills_root = lexical_root / "spills"
    documents_root.mkdir(parents=True, exist_ok=True)
    spills_root.mkdir(parents=True, exist_ok=True)
    artifacts = dict(state.get("artifacts", {}))
    document_index = int(state.get("document_index", 0))
    documents_total = int(state.get("documents", 0))
    documents: list[dict[str, Any]] = []
    vocabulary = SpillVocabulary(
        spills_root,
        config.lexical.spill_terms,
        config.lexical.partitions,
        index=int(state.get("vocabulary_index", 0)),
        artifacts=artifacts,
        artifact_root=target,
    )
    bigrams = _restore_bigram_state(config, target, state)
    content_writer = EmbeddedContentWriter(
        config,
        target,
        checkpoint_state=state.get("content") if state else None,
    )
    if state.get("scan_complete"):
        return state, bigrams

    next_offset = int(state.get("next_offset", 0))
    next_row = int(state.get("next_row", 0))
    checkpoint = int(state.get("next_chunk", 0))
    source_size = int(manifest["sources"]["pages"]["size"])
    progress = ByteProgress("Tokenização B_clean_v2", source_size, initial=next_offset)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        LOGGER.info("B_clean_v2: SIGTERM recebido; fechando checkpoint lexical")

    def flush_documents() -> None:
        nonlocal documents, document_index
        if not documents:
            return
        path = documents_root / f"documents-{document_index:06d}.parquet"
        artifacts[str(path.relative_to(target))] = write_parquet_atomic(
            path, documents, DOCUMENT_SCHEMA
        )
        document_index += 1
        documents = []

    def merge_batch(results: list[dict[str, Any]]) -> None:
        nonlocal documents_total
        for result in results:
            for variant in result["variants"]:
                forms = variant["forms"]
                lemmas = variant["lemmas"]
                vocabulary.document("B_clean_v2", forms, lemmas)
                documents.append(
                    {
                        "row_number": result["row_number"],
                        "view": "B_clean_v2",
                        "domain_id": result["domain_id"],
                        "recursion_level": result["recursion_level"],
                        "characters": variant["characters"],
                        "words": len(forms),
                        "numbers": variant["numbers"],
                        "other_tokens": variant["other_tokens"],
                    }
                )
                documents_total += 1
                if variant["content"] is not None:
                    content_writer.add_result(variant["content"], row_number=result["row_number"])
                for bigram in variant["bigrams"]:
                    bigrams.add(bigram)
                if len(documents) >= 50_000:
                    flush_documents()

    def commit(scan_complete: bool) -> dict:
        nonlocal checkpoint, document_index
        flush_documents()
        if scan_complete and document_index == 0:
            path = documents_root / "documents-000000.parquet"
            artifacts[str(path.relative_to(target))] = write_parquet_atomic(
                path, [], DOCUMENT_SCHEMA
            )
            document_index = 1
        vocabulary.flush()
        if scan_complete and vocabulary.index == 0:
            path = spills_root / "spill-000000.parquet"
            artifacts[str(path.relative_to(target))] = write_parquet_atomic(path, [], SPILL_SCHEMA)
            vocabulary.index = 1
        checkpoint += 1
        content_state = content_writer.checkpoint(
            identity,
            next_offset=next_offset,
            next_row=next_row,
            complete=scan_complete,
        )
        for relative in tuple(artifacts):
            if relative.startswith("content/trigram-state-"):
                artifacts.pop(relative)
        artifacts.update(content_state.get("artifacts", {}))
        bigram_name, checksum = _write_bigram_state(target, checkpoint, bigrams)
        for relative in tuple(artifacts):
            if relative.startswith("lexical/bigram-state-"):
                artifacts.pop(relative)
        artifacts[str((lexical_root / bigram_name).relative_to(target))] = checksum
        new_state = {
            "schema_version": V2_LEXICAL_STATE_VERSION,
            "snapshot_id": manifest["snapshot_id"],
            "v2_fingerprint": config.v2_fingerprint,
            "clean_identity": clean["identity"],
            "content_fingerprint": config.content_fingerprint,
            "source_sha256": manifest["sources"]["pages"]["sha256"],
            "spacy": identity,
            "next_offset": next_offset,
            "next_row": next_row,
            "next_chunk": checkpoint,
            "document_index": document_index,
            "vocabulary_index": vocabulary.index,
            "documents": documents_total,
            "bigram_state": bigram_name,
            "content": content_state,
            "artifacts": dict(sorted(artifacts.items())),
            "scan_complete": scan_complete,
        }
        atomic_json(lexical_root / "state.json", new_state)
        progress.update(
            next_offset,
            detail=(
                f"{config.runtime.workers} workers ativos; checkpoint {checkpoint:,}; "
                f"{documents_total:,} documentos D4; offset confirmado "
                f"{next_offset / 1048576:,.1f} MiB"
            ),
            force=True,
        )
        return new_state

    batch: list[tuple] = []
    batch_bytes = 0
    batch_limit = max(1, min(LEXICAL_BATCH_BYTES, config.runtime.queue_bytes))
    checkpoint_start = next_offset
    reached_end = False

    def submit(pool: _OrderedBatchPool) -> None:
        nonlocal batch, batch_bytes
        if not batch:
            return
        for completed in pool.submit(batch, batch_bytes):
            merge_batch(completed)
        batch = []
        batch_bytes = 0

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        with _OrderedBatchPool(config, mode="lexical") as pool:
            for item in iter_bclean(config, target, start_offset=next_offset, start_row=next_row):
                if stop_requested:
                    break
                next_offset = item.source.end_offset
                next_row = item.source.row_number
                task = (
                    item.source.row_number,
                    item.domain_id,
                    item.recursion_level,
                    ((item.text, ("B_clean_v2",)),),
                )
                size = 4 * len(item.text)
                if batch and batch_bytes + size > batch_limit:
                    submit(pool)
                batch.append(task)
                batch_bytes += size
                if batch_bytes >= batch_limit or len(batch) >= 1024:
                    submit(pool)
                progress.update(
                    next_offset,
                    detail=(
                        f"{config.runtime.workers} workers ativos; checkpoint {checkpoint:,}; "
                        f"{documents_total:,} documentos; linha {next_row:,}; "
                        f"offset confirmado {checkpoint_start / 1048576:,.1f} MiB"
                    ),
                )
                if next_offset - checkpoint_start >= config.runtime.chunk_bytes:
                    submit(pool)
                    for completed in pool.drain():
                        merge_batch(completed)
                    state = commit(False)
                    checkpoint_start = next_offset
            else:
                reached_end = True
            submit(pool)
            for completed in pool.drain():
                merge_batch(completed)
        if reached_end:
            next_offset = source_size
        state = commit(reached_end and not stop_requested)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    progress.finish(
        detail="passagem lexical v2 concluída"
        if state["scan_complete"]
        else "checkpoint salvo; execução interrompida"
    )
    return state, bigrams


def _combined_recount(
    config: Config,
    manifest: dict,
    target: Path,
    bigrams: set[str],
    trigrams: set[str],
    candidate_identity: str,
) -> dict:
    recount_root = target / "recount"
    state_path = recount_root / "state.json"
    state = read_json(state_path, {})
    compatible = bool(
        state.get("schema_version") == V2_RECOUNT_STATE_VERSION
        and state.get("snapshot_id") == manifest["snapshot_id"]
        and state.get("v2_fingerprint") == config.v2_fingerprint
        and state.get("candidate_identity") == candidate_identity
        and state.get("source_sha256") == manifest["sources"]["pages"]["sha256"]
    )
    if state and (
        not compatible or not _checkpoint_artifacts_valid(target, state.get("artifacts", {}))
    ):
        shutil.rmtree(recount_root, ignore_errors=True)
        state = {}
    recount_root.mkdir(parents=True, exist_ok=True)
    checkpoint = int(state.get("next_chunk", 0))
    next_offset = int(state.get("next_offset", 0))
    next_row = int(state.get("next_row", 0))
    documents = int(state.get("documents", 0))
    artifacts = dict(state.get("artifacts", {}))
    for prefix in ("bigram", "trigram", "marginal"):
        _discard_indexed_files(recount_root, prefix, checkpoint)
    for partial in recount_root.rglob("*.partial"):
        partial.unlink(missing_ok=True)
    if not state.get("complete"):
        source_size = int(manifest["sources"]["pages"]["size"])
        progress = ByteProgress("Recontagem combinada B_clean_v2", source_size, initial=next_offset)
        counters: dict[str, Counter] = {
            "bigram_counts": Counter(),
            "bigram_documents": Counter(),
            "trigram_counts": Counter(),
            "trigram_documents": Counter(),
            "left_counts": Counter(),
            "right_counts": Counter(),
        }
        batch: list[str] = []
        batch_bytes = 0
        batch_limit = max(1, min(LEXICAL_BATCH_BYTES, config.runtime.queue_bytes))
        checkpoint_start = next_offset
        stop_requested = False
        reached_end = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True
            LOGGER.info("B_clean_v2: SIGTERM recebido; fechando checkpoint de recontagem")

        def merge(result: dict[str, Counter]) -> None:
            for key, values in result.items():
                counters[key].update(values)

        def submit(pool: _OrderedBatchPool) -> None:
            nonlocal batch, batch_bytes
            if not batch:
                return
            for completed in pool.submit(batch, batch_bytes):
                merge(completed)
            batch = []
            batch_bytes = 0

        def commit(complete: bool) -> dict:
            nonlocal checkpoint, counters
            paths = {
                "bigram": recount_root / f"bigram-{checkpoint:06d}.parquet",
                "trigram": recount_root / f"trigram-{checkpoint:06d}.parquet",
                "marginal": recount_root / f"marginal-{checkpoint:06d}.parquet",
            }
            artifacts[str(paths["bigram"].relative_to(target))] = write_parquet_atomic(
                paths["bigram"],
                [
                    {
                        "bigram": key,
                        "total_frequency": value,
                        "document_frequency": counters["bigram_documents"][key],
                    }
                    for key, value in sorted(counters["bigram_counts"].items())
                ],
                BIGRAM_RECOUNT_SCHEMA,
            )
            artifacts[str(paths["trigram"].relative_to(target))] = write_parquet_atomic(
                paths["trigram"],
                [
                    {
                        "trigram": key,
                        "total_frequency": value,
                        "document_frequency": counters["trigram_documents"][key],
                        "contains_stopword_at_edges": False,
                    }
                    for key, value in sorted(counters["trigram_counts"].items())
                ],
                TRIGRAM_SCHEMA,
            )
            terms = sorted(counters["left_counts"].keys() | counters["right_counts"].keys())
            artifacts[str(paths["marginal"].relative_to(target))] = write_parquet_atomic(
                paths["marginal"],
                [
                    {
                        "term": term,
                        "left_count": counters["left_counts"][term],
                        "right_count": counters["right_counts"][term],
                    }
                    for term in terms
                ],
                BIGRAM_MARGINAL_SCHEMA,
            )
            checkpoint += 1
            counters = {key: Counter() for key in counters}
            new_state = {
                "schema_version": V2_RECOUNT_STATE_VERSION,
                "snapshot_id": manifest["snapshot_id"],
                "v2_fingerprint": config.v2_fingerprint,
                "candidate_identity": candidate_identity,
                "source_sha256": manifest["sources"]["pages"]["sha256"],
                "next_offset": next_offset,
                "next_row": next_row,
                "next_chunk": checkpoint,
                "documents": documents,
                "artifacts": dict(sorted(artifacts.items())),
                "complete": complete,
            }
            atomic_json(state_path, new_state)
            progress.update(
                next_offset,
                detail=(
                    f"{config.runtime.workers} workers ativos; checkpoint {checkpoint:,}; "
                    f"{documents:,} documentos; offset confirmado "
                    f"{next_offset / 1048576:,.1f} MiB"
                ),
                force=True,
            )
            return new_state

        previous_term = signal.signal(signal.SIGTERM, request_stop)
        try:
            with _OrderedBatchPool(
                config,
                mode="combined_recount",
                wanted=bigrams,
                trigram_wanted=trigrams,
            ) as pool:
                for item in iter_bclean(
                    config, target, start_offset=next_offset, start_row=next_row
                ):
                    if stop_requested:
                        break
                    next_offset = item.source.end_offset
                    next_row = item.source.row_number
                    documents += 1
                    size = 4 * len(item.text)
                    if batch and batch_bytes + size > batch_limit:
                        submit(pool)
                    batch.append(item.text)
                    batch_bytes += size
                    if batch_bytes >= batch_limit or len(batch) >= 1024:
                        submit(pool)
                    progress.update(
                        next_offset,
                        detail=(
                            f"{config.runtime.workers} workers ativos; checkpoint {checkpoint:,}; "
                            f"{documents:,} documentos; linha {next_row:,}; "
                            f"offset confirmado {checkpoint_start / 1048576:,.1f} MiB"
                        ),
                    )
                    if next_offset - checkpoint_start >= config.runtime.chunk_bytes:
                        submit(pool)
                        for completed in pool.drain():
                            merge(completed)
                        state = commit(False)
                        checkpoint_start = next_offset
                else:
                    reached_end = True
                submit(pool)
                for completed in pool.drain():
                    merge(completed)
            if reached_end:
                next_offset = source_size
            state = commit(reached_end and not stop_requested)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
        progress.finish(
            detail="recontagem combinada concluída"
            if state["complete"]
            else "checkpoint salvo; execução interrompida"
        )
    if not state.get("complete"):
        return {"complete": False, **state}

    connection = duckdb_connection(
        target / "analysis.duckdb", config.runtime.memory_limit, target / "duckdb-tmp"
    )
    bigram_glob = str(recount_root / "bigram-*.parquet").replace("'", "''")
    trigram_glob = str(recount_root / "trigram-*.parquet").replace("'", "''")
    marginal_glob = str(recount_root / "marginal-*.parquet").replace("'", "''")
    stop_words = _load_nlp(config).Defaults.stop_words
    bigram_rows = connection.execute(
        f"SELECT bigram, sum(total_frequency)::UBIGINT, "
        f"sum(document_frequency)::UBIGINT FROM read_parquet('{bigram_glob}') "
        "GROUP BY bigram ORDER BY 2 DESC, bigram"
    ).fetchall()
    write_parquet_atomic(
        target / "bigrams.parquet",
        [
            {
                "bigram": term,
                "total_frequency": frequency,
                "document_frequency": document_frequency,
                "contains_stopword": any(value in stop_words for value in term.split("\t")),
            }
            for term, frequency, document_frequency in bigram_rows
        ],
        BIGRAM_SCHEMA,
    )
    trigram_rows = connection.execute(
        f"SELECT trigram, sum(total_frequency)::UBIGINT, "
        f"sum(document_frequency)::UBIGINT FROM read_parquet('{trigram_glob}') "
        "GROUP BY trigram ORDER BY 2 DESC, trigram"
    ).fetchall()
    write_parquet_atomic(
        target / "content_trigrams.parquet",
        [
            {
                "trigram": term,
                "total_frequency": frequency,
                "document_frequency": document_frequency,
                "contains_stopword_at_edges": any(
                    edge in stop_words for edge in (term.split("\t")[0], term.split("\t")[-1])
                ),
            }
            for term, frequency, document_frequency in trigram_rows
        ],
        TRIGRAM_SCHEMA,
    )
    duckdb_copy_atomic(
        connection,
        f"SELECT term, sum(left_count)::UBIGINT left_count, "
        f"sum(right_count)::UBIGINT right_count FROM read_parquet('{marginal_glob}') "
        "GROUP BY term",
        target / "content_bigram_marginals.parquet",
    )
    connection.close()
    return {"complete": True, "documents": state["documents"]}


def lexical_content_v2_pass(
    config: Config, manifest: dict, root: Path, clean: dict
) -> tuple[dict, dict]:
    target = v2_root(root)
    lexical_summary_path = target / "lexical_summary.json"
    content_summary_path = target / "content_summary.json"
    lexical_current = read_json(lexical_summary_path, {})
    content_current = read_json(content_summary_path, {})
    if (
        lexical_current.get("v2_fingerprint") == config.v2_fingerprint
        and lexical_current.get("clean_identity") == clean.get("identity")
        and content_current.get("v2_fingerprint") == config.v2_fingerprint
        and content_current.get("clean_identity") == clean.get("identity")
        and content_current.get("content_fingerprint") == config.content_fingerprint
    ):
        return lexical_current, content_current
    identity = spacy_identity(config)
    state = _load_state(config, manifest, target, clean, identity)
    state, bigram_state = _scan(config, manifest, target, clean, identity, state)
    if not state.get("scan_complete"):
        return {"complete": False, "phase": "v2_tokenization"}, {"complete": False}

    connection = duckdb_connection(
        target / "analysis.duckdb", config.runtime.memory_limit, target / "duckdb-tmp"
    )
    spill_glob = str(target / "lexical" / "spills" / "*.parquet").replace("'", "''")
    document_glob = str(target / "lexical" / "documents" / "*.parquet").replace("'", "''")
    duckdb_copy_atomic(
        connection,
        f"SELECT partition, view, kind, term, "
        f"sum(total_frequency)::UBIGINT total_frequency, "
        f"sum(document_frequency)::UBIGINT document_frequency, bool_or(is_stop) is_stop "
        f"FROM read_parquet('{spill_glob}') GROUP BY partition, view, kind, term "
        "ORDER BY partition, view, kind, term",
        target / "vocabulary.parquet",
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW document_statistics AS SELECT * FROM read_parquet('{document_glob}')"
    )
    view_row = connection.execute(
        "SELECT count(*), sum(characters), sum(words), sum(numbers), sum(other_tokens), "
        "quantile_cont(words,[0.05,0.25,0.5,0.75,0.95,0.99]) "
        "FROM document_statistics"
    ).fetchone()
    vocabulary_rows = connection.execute(
        "SELECT kind, count(*), count(*) FILTER (WHERE total_frequency=1), "
        "sum(total_frequency), sum(document_frequency), "
        "count(*) FILTER (WHERE is_stop), "
        "sum(total_frequency) FILTER (WHERE is_stop) "
        "FROM read_parquet(?) GROUP BY kind ORDER BY kind",
        [str(target / "vocabulary.parquet")],
    ).fetchall()
    connection.close()

    bigram_candidates = [
        {
            "bigram": term,
            "estimated_frequency": values[0],
            "maximum_error": values[1],
        }
        for term, values in sorted(bigram_state.values.items())
    ]
    bigram_checksum = write_parquet_atomic(
        target / "bigram_candidates.parquet", bigram_candidates, BIGRAM_CANDIDATE_SCHEMA
    )
    atomic_json(
        target / "bigram_candidates.json",
        {
            "omitted_upper_bound": bigram_state.omitted_upper_bound,
            "items": len(bigram_candidates),
        },
    )
    content_state = state["content"]
    trigram_state_path = target / "content" / content_state["trigram_state"]
    trigram_rows = pq.read_table(trigram_state_path).to_pylist()
    trigram_checksum = write_parquet_atomic(
        target / "content_trigram_candidates.parquet",
        trigram_rows,
        TRIGRAM_STATE_SCHEMA,
    )
    candidate_identity = hashlib.sha256(
        f"{bigram_checksum}:{trigram_checksum}".encode()
    ).hexdigest()
    recount = _combined_recount(
        config,
        manifest,
        target,
        set(bigram_state.values),
        {row["trigram"] for row in trigram_rows},
        candidate_identity,
    )
    if not recount.get("complete"):
        return {"complete": False, "phase": "v2_recount"}, {"complete": False}

    bigram_output = pq.read_table(target / "bigrams.parquet").to_pylist()
    eligible_bigrams = [row for row in bigram_output if not row["contains_stopword"]]
    published = min(config.lexical.published_items, len(eligible_bigrams))
    last_bigram = eligible_bigrams[published - 1]["total_frequency"] if published else 0
    bigram_summary = {
        "candidates": len(bigram_candidates),
        "omitted_upper_bound": bigram_state.omitted_upper_bound,
        "published_k": published,
        "last_frequency": last_bigram,
        "certified": bool(published and bigram_state.omitted_upper_bound < last_bigram),
        "recount": "combined_with_trigrams",
    }
    if not bigram_summary["certified"]:
        raise WackyWackyError(
            "top-K de bigramas v2 não certificado; aumente lexical.bigram_candidates"
        )

    with logged_stage("Redução de conteúdo B_clean_v2"):
        content_metrics = _reduce_content(config, target, content_state)
    omitted_trigram = (
        min((row["estimated_frequency"] for row in trigram_rows), default=0)
        if len(trigram_rows) >= config.content.trigram_candidates
        else 0
    )
    trigram_output = pq.read_table(target / "content_trigrams.parquet").to_pylist()
    eligible_trigrams = [row for row in trigram_output if not row["contains_stopword_at_edges"]]
    last_trigram = (
        eligible_trigrams[min(config.lexical.published_items, len(eligible_trigrams)) - 1][
            "total_frequency"
        ]
        if eligible_trigrams
        else 0
    )
    content_summary = {
        "status": "complete",
        "schema_version": CONTENT_SCHEMA_VERSION,
        "v2_fingerprint": config.v2_fingerprint,
        "clean_identity": clean["identity"],
        "content_fingerprint": config.content_fingerprint,
        "spacy": identity,
        "metrics": content_metrics,
        "trigrams": {
            "candidates": len(trigram_rows),
            "omitted_upper_bound": omitted_trigram,
            "eligible_frequency_floor": max(
                config.content.collocation_min_frequency, omitted_trigram + 1
            ),
            "frequent_top_k_certified": bool(
                not eligible_trigrams or omitted_trigram < last_trigram
            ),
            "recount": "combined_with_bigrams",
        },
        "parameters": config.public_dict()["content"],
        "mojibake_markers": list(MOJIBAKE_MARKERS),
    }
    lexical_summary = {
        "v2_fingerprint": config.v2_fingerprint,
        "clean_identity": clean["identity"],
        "spacy": identity,
        "views": {
            "B_clean_v2": {
                "documents": view_row[0],
                "characters": view_row[1] or 0,
                "words": view_row[2] or 0,
                "numbers": view_row[3] or 0,
                "other_tokens": view_row[4] or 0,
                "word_quantiles": view_row[5] or [None] * 6,
            }
        },
        "vocabulary": {
            f"B_clean_v2:{row[0]}": {
                "types": row[1],
                "hapax": row[2],
                "occurrences": row[3] or 0,
                "document_occurrences": row[4] or 0,
                "stopword_types": row[5],
                "stopword_occurrences": row[6] or 0,
            }
            for row in vocabulary_rows
        },
        "bigrams": bigram_summary,
    }
    atomic_json(content_summary_path, content_summary)
    atomic_json(lexical_summary_path, lexical_summary)
    return lexical_summary, content_summary
