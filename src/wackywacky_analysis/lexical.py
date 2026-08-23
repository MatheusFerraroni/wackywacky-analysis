from __future__ import annotations

import hashlib
import heapq
import json
import multiprocessing
import shutil
import signal
import sqlite3
import unicodedata
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Self

import pyarrow as pa
import pyarrow.parquet as pq
import spacy

from .bclean import clean_again, iter_bclean
from .config import Config
from .errors import WackyWackyError
from .io import atomic_json, decode_field, iter_bounded_tsv, parse_int, read_json
from .progress import LOGGER, ByteProgress
from .schema import PAGES_COLUMNS
from .storage import SortedMembership, duckdb_connection, duckdb_copy_atomic, write_parquet_atomic
from .text import TextDecodeFailure, decode_text

_clean_again = clean_again
SPACY_MAX_TOKENS_PER_CHUNK = 1024


@dataclass(frozen=True)
class ParsedDocument:
    raw: Any
    annotated: tuple[Any, ...]

DOCUMENT_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("view", pa.string()),
        ("domain_id", pa.int64()),
        ("recursion_level", pa.int32()),
        ("characters", pa.uint64()),
        ("words", pa.uint64()),
        ("numbers", pa.uint64()),
        ("other_tokens", pa.uint64()),
    ]
)

SPILL_SCHEMA = pa.schema(
    [
        ("partition", pa.uint16()),
        ("view", pa.string()),
        ("kind", pa.string()),
        ("term", pa.string()),
        ("total_frequency", pa.uint64()),
        ("document_frequency", pa.uint64()),
        ("is_stop", pa.bool_()),
    ]
)

BIGRAM_SCHEMA = pa.schema(
    [
        ("bigram", pa.string()),
        ("total_frequency", pa.uint64()),
        ("document_frequency", pa.uint64()),
        ("contains_stopword", pa.bool_()),
    ]
)

BIGRAM_CANDIDATE_SCHEMA = pa.schema(
    [
        ("bigram", pa.string()),
        ("estimated_frequency", pa.uint64()),
        ("maximum_error", pa.uint64()),
    ]
)

BIGRAM_STATE_SCHEMA = pa.schema(
    [
        ("bigram", pa.string()),
        ("estimated_frequency", pa.uint64()),
        ("maximum_error", pa.uint64()),
        ("version", pa.uint64()),
    ]
)

BIGRAM_RECOUNT_SCHEMA = pa.schema(
    [
        ("bigram", pa.string()),
        ("total_frequency", pa.uint64()),
        ("document_frequency", pa.uint64()),
    ]
)

LEXICAL_STATE_SCHEMA_VERSION = 1
BIGRAM_RECOUNT_STATE_SCHEMA_VERSION = 1
LEXICAL_BATCH_BYTES = 8 * 1024 * 1024

_WORKER_CONFIG: Config | None = None
_WORKER_NLP = None
_RECOUNT_WANTED: set[str] | None = None


class SpaceSaving:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.values: dict[str, tuple[int, int, int]] = {}
        self.heap: list[tuple[int, int, str]] = []
        self.version = 0

    def add(self, key: str) -> None:
        self.version += 1
        if key in self.values:
            count, error, _ = self.values[key]
            value = (count + 1, error, self.version)
            self.values[key] = value
            heapq.heappush(self.heap, (value[0], value[2], key))
            self._compact_if_needed()
            return
        if len(self.values) < self.capacity:
            self.values[key] = (1, 0, self.version)
            heapq.heappush(self.heap, (1, self.version, key))
            self._compact_if_needed()
            return
        while self.heap:
            count, version, victim = heapq.heappop(self.heap)
            current = self.values.get(victim)
            if current and current[0] == count and current[2] == version:
                del self.values[victim]
                value = (count + 1, count, self.version)
                self.values[key] = value
                heapq.heappush(self.heap, (value[0], value[2], key))
                self._compact_if_needed()
                return

    @classmethod
    def restore(
        cls, capacity: int, rows: list[tuple[str, int, int, int]]
    ) -> SpaceSaving:
        instance = cls(capacity)
        for key, count, error, version in rows:
            instance.values[key] = (count, error, version)
            instance.heap.append((count, version, key))
            instance.version = max(instance.version, version)
        heapq.heapify(instance.heap)
        return instance

    def _compact_if_needed(self) -> None:
        maximum = max(4096, 4 * self.capacity)
        if len(self.heap) <= maximum:
            return
        self.heap = [(count, version, key) for key, (count, _error, version) in self.values.items()]
        heapq.heapify(self.heap)

    @property
    def omitted_upper_bound(self) -> int:
        if len(self.values) < self.capacity:
            return 0
        return min(value[0] for value in self.values.values())


class SpillVocabulary:
    def __init__(
        self,
        root: Path,
        limit: int,
        partitions: int,
        *,
        index: int = 0,
        artifacts: dict[str, str] | None = None,
        artifact_root: Path | None = None,
    ) -> None:
        self.root = root
        self.limit = limit
        self.partitions = partitions
        self.counts: dict[tuple[str, str, str], list[Any]] = {}
        self.index = index
        self.artifacts = artifacts
        self.artifact_root = artifact_root

    def document(
        self, view: str, forms: list[tuple[str, bool]], lemmas: list[tuple[str, bool]]
    ) -> None:
        for kind, values in (("form", forms), ("lemma", lemmas)):
            document_counts = Counter(term for term, _stop in values)
            stops = {term: stop for term, stop in values}
            for term, frequency in document_counts.items():
                key = (view, kind, term)
                current = self.counts.setdefault(key, [0, 0, stops[term]])
                current[0] += frequency
                current[1] += 1
                current[2] = current[2] or stops[term]
            if len(self.counts) >= self.limit:
                self.flush()

    def flush(self) -> None:
        if not self.counts:
            return
        rows = []
        for (view, kind, term), (tf, df, stop) in self.counts.items():
            partition = (
                int.from_bytes(hashlib.sha256(term.encode()).digest()[:2], "big") % self.partitions
            )
            rows.append(
                {
                    "partition": partition,
                    "view": view,
                    "kind": kind,
                    "term": term,
                    "total_frequency": tf,
                    "document_frequency": df,
                    "is_stop": stop,
                }
            )
        path = self.root / f"spill-{self.index:06d}.parquet"
        checksum = write_parquet_atomic(path, rows, SPILL_SCHEMA)
        if self.artifacts is not None and self.artifact_root is not None:
            self.artifacts[str(path.relative_to(self.artifact_root))] = checksum
        self.index += 1
        self.counts.clear()


def _load_nlp(config: Config):
    model = config.lexical.spacy_model
    try:
        if model.startswith("blank:"):
            nlp = spacy.blank(model.split(":", 1)[1])
        else:
            nlp = spacy.load(model, disable=["parser", "ner"])
    except OSError as exc:
        raise WackyWackyError(
            f"modelo spaCy ausente: {model}; instale a versão registrada antes do run"
        ) from exc
    nlp.max_length = max(nlp.max_length, config.runtime.max_text_bytes + 1)
    if not any(name in nlp.pipe_names for name in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    return nlp


def _nlp_queue_bytes(config: Config) -> int:
    """Bound materialized spaCy Docs independently of the general I/O queue."""
    return min(config.runtime.queue_bytes, max(1, config.runtime.workers) * 1024 * 1024)


def _parse_document(nlp, text: str) -> ParsedDocument:
    """Tokenize globally, but run statistical components in bounded token chunks."""
    raw = nlp.make_doc(text)
    if "sentencizer" in nlp.pipe_names:
        nlp.get_pipe("sentencizer")(raw)
    elif "senter" in nlp.pipe_names:
        nlp.get_pipe("senter")(raw)
    elif "parser" in nlp.pipe_names:
        nlp.get_pipe("parser")(raw)
    spans = []
    start = 0
    for sentence in raw.sents:
        if sentence.end - start > SPACY_MAX_TOKENS_PER_CHUNK and sentence.start > start:
            spans.append(raw[start : sentence.start])
            start = sentence.start
        while sentence.end - start > SPACY_MAX_TOKENS_PER_CHUNK:
            spans.append(raw[start : start + SPACY_MAX_TOKENS_PER_CHUNK])
            start += SPACY_MAX_TOKENS_PER_CHUNK
    if start < len(raw):
        spans.append(raw[start:])
    disabled = [
        name for name in ("sentencizer", "senter", "parser") if name in nlp.pipe_names
    ]
    annotated = tuple(
        nlp.pipe(
            (span.as_doc(copy_user_data=False) for span in spans),
            disable=disabled,
            n_process=1,
            batch_size=1,
        )
    )
    return ParsedDocument(raw=raw, annotated=annotated)


def _iter_tokens(document):
    if isinstance(document, ParsedDocument):
        for chunk in document.annotated:
            yield from chunk
    else:
        yield from document


def _raw_document(document):
    return document.raw if isinstance(document, ParsedDocument) else document


def spacy_identity(config: Config) -> dict:
    nlp = _load_nlp(config)
    return {
        "spacy": spacy.__version__,
        "model": config.lexical.spacy_model,
        "model_version": nlp.meta.get("version", "blank"),
        "pipeline": list(nlp.pipe_names),
        "max_tokens_per_chunk": SPACY_MAX_TOKENS_PER_CHUNK,
    }


def _token_data(
    document,
) -> tuple[list[tuple[str, bool]], list[tuple[str, bool]], int, int, list[str]]:
    forms: list[tuple[str, bool]] = []
    lemmas: list[tuple[str, bool]] = []
    numbers = other = 0
    paragraph_words: list[str] = []
    for token in _iter_tokens(document):
        if token.is_alpha:
            form = unicodedata.normalize("NFC", token.text).casefold()
            lemma = unicodedata.normalize("NFC", token.lemma_ or token.text).casefold()
            forms.append((form, token.is_stop))
            lemmas.append((lemma, token.is_stop))
            paragraph_words.append(form)
        elif token.like_num:
            numbers += 1
        elif not token.is_space:
            other += 1
    return forms, lemmas, numbers, other, paragraph_words


def _paragraph_word_sequences(document, text: str) -> list[list[str]]:
    document = _raw_document(document)
    sequences: list[list[str]] = [[]]
    previous_end = 0
    for token in document:
        gap = text[previous_end : token.idx]
        has_boundary = "\n\n" in gap or (token.is_space and "\n\n" in token.text)
        if has_boundary and sequences[-1]:
            sequences.append([])
        if token.is_alpha:
            sequences[-1].append(unicodedata.normalize("NFC", token.text).casefold())
        previous_end = token.idx + len(token.text)
    return [sequence for sequence in sequences if sequence]


def _lexical_worker_init(config: Config) -> None:
    global _WORKER_CONFIG, _WORKER_NLP
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _WORKER_CONFIG = config
    _WORKER_NLP = _load_nlp(config)


def _process_lexical_batch_with(
    config: Config,
    nlp,
    batch: list[tuple[int, int | None, int | None, tuple[tuple[str, tuple[str, ...]], ...]]],
) -> list[dict[str, Any]]:
    from .content import analyze_document

    results: list[dict[str, Any]] = []
    for row, domain, level, variants in batch:
        variant_results = []
        for text, views in variants:
            document = _parse_document(nlp, text)
            forms, lemmas, numbers, other, _words = _token_data(document)
            is_bclean = "B_clean" in views
            content = None
            bigrams: list[str] = []
            if is_bclean:
                from .bclean import BCleanRecord
                from .io import BinaryRecord

                if config.content.enabled:
                    record = BCleanRecord(
                        source=BinaryRecord(row, 0, 0, None),
                        domain_id=domain,
                        recursion_level=level,
                        text=text,
                    )
                    content = analyze_document(config, record, document)
                bigrams = [
                    left + "\t" + right
                    for words in _paragraph_word_sequences(document, text)
                    for left, right in pairwise(words)
                ]
            variant_results.append(
                {
                    "views": views,
                    "characters": len(text),
                    "forms": forms,
                    "lemmas": lemmas,
                    "numbers": numbers,
                    "other_tokens": other,
                    "bigrams": bigrams,
                    "content": content,
                }
            )
        results.append(
            {
                "row_number": row,
                "domain_id": domain,
                "recursion_level": level,
                "variants": variant_results,
            }
        )
    return results


def _lexical_worker_batch(batch: list[tuple]) -> list[dict[str, Any]]:
    if _WORKER_CONFIG is None or _WORKER_NLP is None:
        raise RuntimeError("worker lexical não inicializado")
    return _process_lexical_batch_with(_WORKER_CONFIG, _WORKER_NLP, batch)


def _recount_worker_init(config: Config, wanted: set[str]) -> None:
    global _WORKER_CONFIG, _WORKER_NLP, _RECOUNT_WANTED
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    _WORKER_CONFIG = config
    _WORKER_NLP = _load_nlp(config)
    _RECOUNT_WANTED = wanted


def _process_recount_batch_with(nlp, wanted: set[str], batch: list[str]) -> dict[str, Counter]:
    counts: Counter[str] = Counter()
    documents: Counter[str] = Counter()
    for clean in batch:
        document = nlp.make_doc(clean)
        seen: set[str] = set()
        for words in _paragraph_word_sequences(document, clean):
            for left, right in pairwise(words):
                key = left + "\t" + right
                if key in wanted:
                    counts[key] += 1
                    seen.add(key)
        documents.update(seen)
    return {"counts": counts, "documents": documents}


def _recount_worker_batch(batch: list[str]) -> dict[str, Counter]:
    if _WORKER_NLP is None or _RECOUNT_WANTED is None:
        raise RuntimeError("worker de recontagem não inicializado")
    return _process_recount_batch_with(_WORKER_NLP, _RECOUNT_WANTED, batch)


class _OrderedBatchPool:
    """Bound input bytes and merge completed worker batches in submission order."""

    def __init__(self, config: Config, *, mode: str, wanted: set[str] | None = None) -> None:
        self.config = config
        self.mode = mode
        self.wanted = wanted
        self.workers = config.runtime.workers
        self.maximum_pending = max(1, 2 * self.workers)
        self.maximum_bytes = max(1, config.runtime.queue_bytes)
        self.pending_bytes = 0
        self.pending: deque[tuple[Future, int]] = deque()
        self.executor: ProcessPoolExecutor | None = None
        self.inline_nlp = None
        if self.workers == 1:
            self.inline_nlp = _load_nlp(config)
        else:
            context = multiprocessing.get_context("spawn")
            if mode == "lexical":
                initializer = _lexical_worker_init
                initargs = (config,)
            else:
                initializer = _recount_worker_init
                initargs = (config, wanted or set())
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=context,
                initializer=initializer,
                initargs=initargs,
            )

    def _run_inline(self, batch):
        if self.mode == "lexical":
            return _process_lexical_batch_with(self.config, self.inline_nlp, batch)
        return _process_recount_batch_with(self.inline_nlp, self.wanted or set(), batch)

    def _oldest(self):
        future, size = self.pending.popleft()
        self.pending_bytes -= size
        try:
            return future.result()
        except BaseException as exc:
            raise WackyWackyError(f"worker da etapa lexical falhou: {exc}") from exc

    def submit(self, batch, size: int) -> list[Any]:
        completed = []
        while self.pending and (
            len(self.pending) >= self.maximum_pending
            or self.pending_bytes + size > self.maximum_bytes
        ):
            completed.append(self._oldest())
        if self.executor is None:
            completed.append(self._run_inline(batch))
            return completed
        target = _lexical_worker_batch if self.mode == "lexical" else _recount_worker_batch
        future = self.executor.submit(target, batch)
        self.pending.append((future, size))
        self.pending_bytes += size
        return completed

    def drain(self) -> list[Any]:
        completed = []
        while self.pending:
            completed.append(self._oldest())
        return completed

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _derived_lexical_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "lexical_summary.json",
        root / "vocabulary.parquet",
        root / "vocabulary.parquet.sha256",
        root / "bigrams.parquet",
        root / "bigrams.parquet.sha256",
        root / "bigram_candidates.parquet",
        root / "bigram_candidates.parquet.sha256",
        root / "bigram_candidates.json",
    )


def _reset_lexical_products(root: Path, reason: str) -> None:
    from .content import _remove_content_products

    LOGGER.info("Léxico: descartando somente artefatos da etapa 7 (%s)", reason)
    shutil.rmtree(root / "lexical", ignore_errors=True)
    for stale in _derived_lexical_paths(root):
        stale.unlink(missing_ok=True)
    _remove_content_products(root)


def _checkpoint_artifacts_valid(root: Path, artifacts: dict[str, str]) -> bool:
    for relative, expected in artifacts.items():
        path = root / relative
        sidecar = path.with_suffix(path.suffix + ".sha256")
        try:
            if sidecar.read_text(encoding="ascii").strip() != expected:
                return False
            pq.read_metadata(path)
        except (OSError, ValueError):
            return False
    return True


def _state_compatible(config: Config, manifest: dict, identity: dict, state: dict) -> bool:
    return bool(
        state.get("schema_version") == LEXICAL_STATE_SCHEMA_VERSION
        and state.get("snapshot_id") == manifest["snapshot_id"]
        and state.get("config_sha256") == config.fingerprint
        and state.get("content_fingerprint") == config.content_fingerprint
        and state.get("source_sha256") == manifest["sources"]["pages"]["sha256"]
        and state.get("spacy") == identity
    )


def _discard_indexed_files(directory: Path, prefix: str, keep_before: int) -> None:
    if not directory.exists():
        return
    for path in directory.glob(f"{prefix}-*.parquet"):
        try:
            index = int(path.stem.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        if index >= keep_before:
            path.unlink(missing_ok=True)
            path.with_suffix(path.suffix + ".sha256").unlink(missing_ok=True)


def _restore_bigram_state(config: Config, root: Path, state: dict) -> SpaceSaving:
    name = state.get("bigram_state")
    if not name:
        return SpaceSaving(config.lexical.bigram_candidates)
    rows = [
        (
            row["bigram"],
            row["estimated_frequency"],
            row["maximum_error"],
            row["version"],
        )
        for row in pq.read_table(root / "lexical" / name).to_pylist()
    ]
    return SpaceSaving.restore(config.lexical.bigram_candidates, rows)


def _write_bigram_state(root: Path, checkpoint: int, bigrams: SpaceSaving) -> tuple[str, str]:
    name = f"bigram-state-{checkpoint % 2}.parquet"
    path = root / "lexical" / name
    checksum = write_parquet_atomic(
        path,
        [
            {
                "bigram": term,
                "estimated_frequency": value[0],
                "maximum_error": value[1],
                "version": value[2],
            }
            for term, value in sorted(bigrams.values.items())
        ],
        BIGRAM_STATE_SCHEMA,
    )
    return name, checksum


def _load_or_reset_lexical_state(
    config: Config, manifest: dict, root: Path, identity: dict
) -> dict:
    lexical_root = root / "lexical"
    state_path = lexical_root / "state.json"
    state = read_json(state_path, {})
    has_old_products = lexical_root.exists() and any(lexical_root.iterdir())
    if state and not _state_compatible(config, manifest, identity, state):
        _reset_lexical_products(root, "estado interno antigo ou incompatível")
        return {}
    if not state and has_old_products:
        _reset_lexical_products(root, "estrutura lexical antiga sem checkpoint compatível")
        return {}
    if state and not _checkpoint_artifacts_valid(root, state.get("artifacts", {})):
        _reset_lexical_products(root, "arquivo ou checksum do checkpoint inválido")
        return {}
    if state:
        _discard_indexed_files(
            lexical_root / "documents", "documents", int(state.get("document_index", 0))
        )
        _discard_indexed_files(
            lexical_root / "spills", "spill", int(state.get("vocabulary_index", 0))
        )
        for partial in lexical_root.rglob("*.partial"):
            partial.unlink(missing_ok=True)
        LOGGER.info(
            "Léxico: retomando offset %s, linha %s, checkpoint %s",
            f"{int(state.get('next_offset', 0)):,}",
            f"{int(state.get('next_row', 0)):,}",
            f"{int(state.get('next_chunk', 0)):,}",
        )
    return state


def _scan_lexical(
    config: Config,
    manifest: dict,
    root: Path,
    identity: dict,
    state: dict,
) -> tuple[dict, SpaceSaving]:
    from .content import EmbeddedContentWriter

    lexical_root = root / "lexical"
    document_dir = lexical_root / "documents"
    spill_dir = lexical_root / "spills"
    document_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    artifacts = dict(state.get("artifacts", {}))
    document_index = int(state.get("document_index", 0))
    documents_total = int(state.get("documents", 0))
    source_documents = int(state.get("source_documents", 0))
    documents: list[dict[str, Any]] = []
    vocabulary = SpillVocabulary(
        spill_dir,
        config.lexical.spill_terms,
        config.lexical.partitions,
        index=int(state.get("vocabulary_index", 0)),
        artifacts=artifacts,
        artifact_root=root,
    )
    bigrams = _restore_bigram_state(config, root, state)
    content_writer = EmbeddedContentWriter(
        config,
        root,
        checkpoint_state=state.get("content") if state else None,
    )
    if state.get("scan_complete"):
        return state, bigrams

    d2 = SortedMembership(root / "d2_representatives.u64")
    d3 = SortedMembership(root / "d3_representatives.u64")
    candidates = sqlite3.connect(f"file:{root / 'boilerplate.sqlite'}?mode=ro", uri=True)
    next_offset = int(state.get("next_offset", 0))
    next_row = int(state.get("next_row", 0))
    checkpoint = int(state.get("next_chunk", 0))
    source_size = int(manifest["sources"]["pages"]["size"])
    progress = ByteProgress("Tokenização lexical", source_size, initial=next_offset)
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        LOGGER.info("Léxico: SIGTERM recebido; fechando o checkpoint em andamento")

    def flush_documents() -> None:
        nonlocal documents, document_index
        if not documents:
            return
        path = document_dir / f"documents-{document_index:06d}.parquet"
        artifacts[str(path.relative_to(root))] = write_parquet_atomic(
            path, documents, DOCUMENT_SCHEMA
        )
        document_index += 1
        documents = []

    def merge_batch(batch_results: list[dict[str, Any]]) -> None:
        nonlocal documents_total, source_documents
        for result in batch_results:
            source_documents += 1
            for variant in result["variants"]:
                forms = variant["forms"]
                lemmas = variant["lemmas"]
                for view in variant["views"]:
                    vocabulary.document(view, forms, lemmas)
                    documents.append(
                        {
                            "row_number": result["row_number"],
                            "view": view,
                            "domain_id": result["domain_id"],
                            "recursion_level": result["recursion_level"],
                            "characters": variant["characters"],
                            "words": len(forms),
                            "numbers": variant["numbers"],
                            "other_tokens": variant["other_tokens"],
                        }
                    )
                    documents_total += 1
                    if len(documents) >= 50_000:
                        flush_documents()
                if "B_clean" in variant["views"]:
                    if variant["content"] is not None:
                        content_writer.add_result(
                            variant["content"], row_number=result["row_number"]
                        )
                    for bigram in variant["bigrams"]:
                        bigrams.add(bigram)

    def commit(scan_complete: bool) -> dict:
        nonlocal checkpoint, document_index
        flush_documents()
        if scan_complete and document_index == 0:
            path = document_dir / "documents-000000.parquet"
            artifacts[str(path.relative_to(root))] = write_parquet_atomic(
                path, [], DOCUMENT_SCHEMA
            )
            document_index = 1
        vocabulary.flush()
        if scan_complete and vocabulary.index == 0:
            path = spill_dir / "spill-000000.parquet"
            artifacts[str(path.relative_to(root))] = write_parquet_atomic(
                path, [], SPILL_SCHEMA
            )
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
        bigram_name, checksum = _write_bigram_state(root, checkpoint, bigrams)
        for relative in tuple(artifacts):
            if relative.startswith("lexical/bigram-state-"):
                artifacts.pop(relative)
        artifacts[str((lexical_root / bigram_name).relative_to(root))] = checksum
        new_state = {
            "schema_version": LEXICAL_STATE_SCHEMA_VERSION,
            "snapshot_id": manifest["snapshot_id"],
            "config_sha256": config.fingerprint,
            "content_fingerprint": config.content_fingerprint,
            "source_sha256": manifest["sources"]["pages"]["sha256"],
            "spacy": identity,
            "next_offset": next_offset,
            "next_row": next_row,
            "next_chunk": checkpoint,
            "document_index": document_index,
            "vocabulary_index": vocabulary.index,
            "documents": documents_total,
            "source_documents": source_documents,
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
                f"{source_documents:,} documentos; offset confirmado {next_offset / 1048576:,.1f} MiB"
            ),
            force=True,
        )
        return new_state

    batch: list[tuple] = []
    batch_bytes = 0
    batch_limit = max(1, min(LEXICAL_BATCH_BYTES, config.runtime.queue_bytes))
    checkpoint_start = next_offset
    reached_end = False

    def submit_batch(pool: _OrderedBatchPool) -> None:
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
            with config.analysis_pages.open("rb") as handle:
                for record in iter_bounded_tsv(
                    handle,
                    columns=len(PAGES_COLUMNS),
                    max_line_bytes=config.runtime.max_line_bytes,
                    start_offset=next_offset,
                    start_row=next_row,
                    max_rows=config.runtime.max_rows,
                ):
                    if stop_requested:
                        break
                    next_offset = record.end_offset
                    next_row = record.row_number
                    progress.update(
                        next_offset,
                        detail=(
                            f"{config.runtime.workers} workers ativos; checkpoint {checkpoint:,}; "
                            f"{source_documents:,} documentos; linha {next_row:,}; "
                            f"offset confirmado {checkpoint_start / 1048576:,.1f} MiB"
                        ),
                    )
                    if record.fields is not None:
                        fields = record.fields
                        if decode_field(fields[11]) == "done":
                            try:
                                decoded = decode_text(
                                    fields[13], fields[15], config.runtime.max_text_bytes
                                )
                            except TextDecodeFailure:
                                decoded = None
                            if decoded is not None and decoded.normalized:
                                domain = parse_int(fields[1])
                                level = parse_int(fields[10])
                                is_d2 = d2.contains(record.row_number)
                                is_d3 = d3.contains(record.row_number)
                                views = ["R_valid"]
                                if is_d2:
                                    views.append("E_exact")
                                variants: list[tuple[str, tuple[str, ...]]] = []
                                if is_d3:
                                    clean = clean_again(
                                        config, candidates, domain, decoded.normalized
                                    )
                                    if clean:
                                        if clean == decoded.normalized:
                                            views.append("B_clean")
                                        else:
                                            variants.append((clean, ("B_clean",)))
                                variants.append((decoded.normalized, tuple(views)))
                                task = (
                                    record.row_number,
                                    domain,
                                    level,
                                    tuple(variants),
                                )
                                task_bytes = sum(4 * len(text) for text, _views in variants)
                                if batch and batch_bytes + task_bytes > batch_limit:
                                    submit_batch(pool)
                                batch.append(task)
                                batch_bytes += task_bytes
                                if batch_bytes >= batch_limit or len(batch) >= 1024:
                                    submit_batch(pool)
                    if next_offset - checkpoint_start >= config.runtime.chunk_bytes:
                        submit_batch(pool)
                        for completed in pool.drain():
                            merge_batch(completed)
                        state = commit(False)
                        checkpoint_start = next_offset
                else:
                    reached_end = True
            submit_batch(pool)
            for completed in pool.drain():
                merge_batch(completed)
        state = commit(reached_end and not stop_requested)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        candidates.close()
    if state["scan_complete"]:
        progress.finish(detail="passagem lexical concluída")
    else:
        progress.finish(detail="checkpoint salvo; execução interrompida")
    return state, bigrams


def lexical_pass(config: Config, manifest: dict, root: Path) -> dict:
    summary_path = root / "lexical_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    identity = spacy_identity(config)
    identity_path = root / "spacy.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise WackyWackyError("versão do spaCy/modelo difere da execução iniciada")
    atomic_json(identity_path, identity)
    state = _load_or_reset_lexical_state(config, manifest, root, identity)
    state, bigrams = _scan_lexical(config, manifest, root, identity, state)
    if not state.get("scan_complete"):
        return {"complete": False, "phase": "tokenization", "checkpoint": state}

    lexical_root = root / "lexical"
    spill_dir = lexical_root / "spills"
    document_dir = lexical_root / "documents"
    LOGGER.info("Léxico: reduzindo partições de vocabulário no DuckDB")
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    spill_glob = str(spill_dir / "*.parquet").replace("'", "''")
    document_glob = str(document_dir / "*.parquet").replace("'", "''")
    duckdb_copy_atomic(
        connection,
        f"""
        SELECT partition, view, kind, term,
               sum(total_frequency)::UBIGINT AS total_frequency,
               sum(document_frequency)::UBIGINT AS document_frequency,
               bool_or(is_stop) AS is_stop
        FROM read_parquet('{spill_glob}') GROUP BY partition, view, kind, term
        ORDER BY partition, view, kind, term
        """,
        root / "vocabulary.parquet",
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW document_statistics AS SELECT * FROM read_parquet('{document_glob}')"
    )
    view_rows = connection.execute(
        """
        SELECT view, count(*) AS documents, sum(characters) AS characters, sum(words) AS words,
               sum(numbers) AS numbers, sum(other_tokens) AS other_tokens,
               quantile_cont(words, [0.05,0.25,0.5,0.75,0.95,0.99]) AS word_quantiles
        FROM document_statistics GROUP BY view ORDER BY view
        """
    ).fetchall()
    vocabulary_rows = connection.execute(
        """
        SELECT view, kind, count(*) AS vocabulary,
               count(*) FILTER (WHERE total_frequency=1) AS hapax,
               sum(total_frequency) AS occurrences,
               sum(document_frequency) AS document_occurrences,
               count(*) FILTER (WHERE is_stop) AS stopword_types,
               sum(total_frequency) FILTER (WHERE is_stop) AS stopword_occurrences
        FROM read_parquet(?) GROUP BY view, kind ORDER BY view, kind
        """,
        [str(root / "vocabulary.parquet")],
    ).fetchall()
    connection.close()
    atomic_json(
        root / "bigram_candidates.json",
        {
            "omitted_upper_bound": bigrams.omitted_upper_bound,
            "items": len(bigrams.values),
        },
    )
    candidate_checksum = write_parquet_atomic(
        root / "bigram_candidates.parquet",
        [
            {
                "bigram": term,
                "estimated_frequency": value[0],
                "maximum_error": value[1],
            }
            for term, value in sorted(bigrams.values.items())
        ],
        BIGRAM_CANDIDATE_SCHEMA,
    )
    LOGGER.info("Léxico: iniciando recontagem exata dos bigramas candidatos")
    bigram_summary = recount_bigrams(
        config,
        manifest,
        root,
        set(bigrams.values),
        bigrams.omitted_upper_bound,
        candidate_checksum,
    )
    if bigram_summary.get("complete") is False:
        return {"complete": False, "phase": "bigram_recount", "checkpoint": bigram_summary}
    if not bigram_summary["certified"]:
        raise WackyWackyError(
            "top-K de bigramas não certificado; aumente lexical.bigram_candidates e retome"
        )
    summary = {
        "spacy": identity,
        "views": {
            row[0]: {
                "documents": row[1],
                "characters": row[2],
                "words": row[3],
                "numbers": row[4],
                "other_tokens": row[5],
                "word_quantiles": row[6],
            }
            for row in view_rows
        },
        "vocabulary": {
            f"{row[0]}:{row[1]}": {
                "types": row[2],
                "hapax": row[3],
                "occurrences": row[4],
                "document_occurrences": row[5],
                "stopword_types": row[6],
                "stopword_occurrences": row[7] or 0,
            }
            for row in vocabulary_rows
        },
        "bigrams": bigram_summary,
    }
    atomic_json(summary_path, summary)
    return summary


def recount_bigrams(
    config: Config,
    manifest: dict,
    root: Path,
    wanted: set[str],
    upper_bound: int,
    candidate_checksum: str,
) -> dict:
    recount_root = root / "lexical" / "bigram-recount"
    state_path = recount_root / "state.json"
    state = read_json(state_path, {})
    compatible = bool(
        state.get("schema_version") == BIGRAM_RECOUNT_STATE_SCHEMA_VERSION
        and state.get("snapshot_id") == manifest["snapshot_id"]
        and state.get("config_sha256") == config.fingerprint
        and state.get("candidate_checksum") == candidate_checksum
        and state.get("source_sha256") == manifest["sources"]["pages"]["sha256"]
    )
    if state and (not compatible or not _checkpoint_artifacts_valid(root, state.get("artifacts", {}))):
        LOGGER.info("Recontagem: descartando estado incompatível ou checksum inválido")
        shutil.rmtree(recount_root, ignore_errors=True)
        state = {}
    elif not state and recount_root.exists():
        shutil.rmtree(recount_root, ignore_errors=True)
    recount_root.mkdir(parents=True, exist_ok=True)
    next_offset = int(state.get("next_offset", 0))
    next_row = int(state.get("next_row", 0))
    checkpoint = int(state.get("next_chunk", 0))
    documents_total = int(state.get("documents", 0))
    artifacts = dict(state.get("artifacts", {}))
    _discard_indexed_files(recount_root, "chunk", checkpoint)
    for partial in recount_root.glob("*.partial"):
        partial.unlink(missing_ok=True)
    if state.get("complete"):
        LOGGER.info(
            "Recontagem: reutilizando %s checkpoints confirmados", f"{checkpoint:,}"
        )
    else:
        if state:
            LOGGER.info(
                "Recontagem: retomando offset %s, linha %s, checkpoint %s",
                f"{next_offset:,}",
                f"{next_row:,}",
                f"{checkpoint:,}",
            )
        progress = ByteProgress(
            "Recontagem de bigramas",
            int(manifest["sources"]["pages"]["size"]),
            initial=next_offset,
        )
        counts: Counter[str] = Counter()
        documents: Counter[str] = Counter()
        batch: list[str] = []
        batch_bytes = 0
        batch_limit = max(1, min(LEXICAL_BATCH_BYTES, config.runtime.queue_bytes))
        checkpoint_start = next_offset
        stop_requested = False
        reached_end = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop_requested
            stop_requested = True
            LOGGER.info("Recontagem: SIGTERM recebido; fechando checkpoint")

        def merge(result: dict[str, Counter]) -> None:
            counts.update(result["counts"])
            documents.update(result["documents"])

        def submit(pool: _OrderedBatchPool) -> None:
            nonlocal batch, batch_bytes
            if not batch:
                return
            for completed in pool.submit(batch, batch_bytes):
                merge(completed)
            batch = []
            batch_bytes = 0

        def commit(complete: bool) -> dict:
            nonlocal checkpoint, counts, documents
            path = recount_root / f"chunk-{checkpoint:06d}.parquet"
            artifacts[str(path.relative_to(root))] = write_parquet_atomic(
                path,
                [
                    {
                        "bigram": term,
                        "total_frequency": frequency,
                        "document_frequency": documents[term],
                    }
                    for term, frequency in sorted(counts.items())
                ],
                BIGRAM_RECOUNT_SCHEMA,
            )
            checkpoint += 1
            counts = Counter()
            documents = Counter()
            new_state = {
                "schema_version": BIGRAM_RECOUNT_STATE_SCHEMA_VERSION,
                "snapshot_id": manifest["snapshot_id"],
                "config_sha256": config.fingerprint,
                "candidate_checksum": candidate_checksum,
                "source_sha256": manifest["sources"]["pages"]["sha256"],
                "next_offset": next_offset,
                "next_row": next_row,
                "next_chunk": checkpoint,
                "documents": documents_total,
                "artifacts": dict(sorted(artifacts.items())),
                "complete": complete,
            }
            atomic_json(state_path, new_state)
            progress.update(
                next_offset,
                detail=(
                    f"{config.runtime.workers} workers ativos; checkpoint {checkpoint:,}; "
                    f"{documents_total:,} documentos; offset confirmado {next_offset / 1048576:,.1f} MiB"
                ),
                force=True,
            )
            return new_state

        previous_term = signal.signal(signal.SIGTERM, request_stop)
        try:
            with _OrderedBatchPool(config, mode="recount", wanted=wanted) as pool:
                for item in iter_bclean(
                    config, root, start_offset=next_offset, start_row=next_row
                ):
                    if stop_requested:
                        break
                    next_offset = item.source.end_offset
                    next_row = item.source.row_number
                    documents_total += 1
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
                            f"{documents_total:,} documentos; linha {next_row:,}; "
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
                next_offset = int(manifest["sources"]["pages"]["size"])
            state = commit(reached_end and not stop_requested)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
        if not state.get("complete"):
            progress.finish(detail="checkpoint salvo; execução interrompida")
            return {"complete": False, **state}
        progress.finish(detail="recontagem concluída")

    LOGGER.info("Recontagem: reduzindo checkpoints exatos no DuckDB")
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    recount_glob = str(recount_root / "chunk-*.parquet").replace("'", "''")
    rows = connection.execute(
        f"""
        SELECT bigram, sum(total_frequency)::UBIGINT AS total_frequency,
               sum(document_frequency)::UBIGINT AS document_frequency
        FROM read_parquet('{recount_glob}')
        GROUP BY bigram ORDER BY total_frequency DESC, bigram
        """
    ).fetchall()
    connection.close()
    stop_words = _load_nlp(config).Defaults.stop_words
    output = [
        {
            "bigram": term,
            "total_frequency": frequency,
            "document_frequency": document_frequency,
            "contains_stopword": any(
                component in stop_words for component in term.split("\t")
            ),
        }
        for term, frequency, document_frequency in rows
    ]
    write_parquet_atomic(root / "bigrams.parquet", output, BIGRAM_SCHEMA)
    published = config.lexical.published_items
    eligible = [row for row in output if not row["contains_stopword"]]
    last = eligible[min(published, len(eligible)) - 1]["total_frequency"] if eligible else 0
    return {
        "candidates": len(wanted),
        "omitted_upper_bound": upper_bound,
        "published_k": min(published, len(eligible)),
        "last_frequency": last,
        "certified": upper_bound < last,
    }
