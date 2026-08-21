from __future__ import annotations

import hashlib
import heapq
import json
import shutil
import sqlite3
import unicodedata
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import spacy

from .bclean import clean_again, iter_bclean
from .config import Config
from .errors import WackyWackyError
from .io import atomic_json, decode_field, iter_bounded_tsv, parse_int
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
    def __init__(self, root: Path, limit: int, partitions: int) -> None:
        self.root = root
        self.limit = limit
        self.partitions = partitions
        self.counts: dict[tuple[str, str, str], list[Any]] = {}
        self.index = 0

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
        write_parquet_atomic(self.root / f"spill-{self.index:06d}.parquet", rows, SPILL_SCHEMA)
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


def lexical_pass(config: Config, manifest: dict, root: Path) -> dict:
    summary_path = root / "lexical_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    lexical_root = root / "lexical"
    if lexical_root.exists():
        shutil.rmtree(lexical_root)
    for stale in (
        root / "vocabulary.parquet",
        root / "vocabulary.parquet.sha256",
        root / "bigrams.parquet",
        root / "bigrams.parquet.sha256",
        root / "bigram_candidates.parquet",
        root / "bigram_candidates.parquet.sha256",
        root / "bigram_candidates.json",
    ):
        stale.unlink(missing_ok=True)
    nlp = _load_nlp(config)
    identity = {
        "spacy": spacy.__version__,
        "model": config.lexical.spacy_model,
        "model_version": nlp.meta.get("version", "blank"),
        "pipeline": list(nlp.pipe_names),
        "max_tokens_per_chunk": SPACY_MAX_TOKENS_PER_CHUNK,
    }
    identity_path = root / "spacy.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise WackyWackyError("versão do spaCy/modelo difere da execução iniciada")
    atomic_json(identity_path, identity)
    from .content import EmbeddedContentWriter

    content_writer = EmbeddedContentWriter(config, root)
    d2 = SortedMembership(root / "d2_representatives.u64")
    d3 = SortedMembership(root / "d3_representatives.u64")
    candidate_path = root / "boilerplate.sqlite"
    candidates = sqlite3.connect(f"file:{candidate_path}?mode=ro", uri=True)
    document_dir = root / "lexical" / "documents"
    spill_dir = root / "lexical" / "spills"
    document_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    vocabulary = SpillVocabulary(spill_dir, config.lexical.spill_terms, config.lexical.partitions)
    bigrams = SpaceSaving(config.lexical.bigram_candidates)
    documents: list[dict] = []
    document_chunk = 0
    jobs: list[tuple[int, int | None, int | None, str, tuple[str, ...]]] = []
    job_bytes = 0
    progress = ByteProgress(
        "Tokenização lexical",
        manifest["sources"]["pages"]["size"],
    )

    def emit(
        view: str,
        row: int,
        domain: int | None,
        level: int | None,
        text: str,
        document,
    ) -> None:
        nonlocal document_chunk, documents
        forms, lemmas, numbers, other, _words = _token_data(document)
        if view == "B_clean":
            content_writer.add(row, domain, level, text, document)
            for words in _paragraph_word_sequences(document, text):
                for left, right in pairwise(words):
                    bigrams.add(left + "\t" + right)
        vocabulary.document(view, forms, lemmas)
        documents.append(
            {
                "row_number": row,
                "view": view,
                "domain_id": domain,
                "recursion_level": level,
                "characters": len(text),
                "words": len(forms),
                "numbers": numbers,
                "other_tokens": other,
            }
        )
        if len(documents) >= 50_000:
            write_parquet_atomic(
                document_dir / f"documents-{document_chunk:06d}.parquet", documents, DOCUMENT_SCHEMA
            )
            document_chunk += 1
            documents = []

    def flush_jobs() -> None:
        nonlocal jobs, job_bytes
        if not jobs:
            return
        for job in jobs:
            row, domain, level, text, views = job
            document = _parse_document(nlp, text)
            for view in views:
                emit(view, row, domain, level, text, document)
        jobs = []
        job_bytes = 0

    def add_job(
        row: int,
        domain: int | None,
        level: int | None,
        text: str,
        views: tuple[str, ...],
    ) -> None:
        nonlocal job_bytes
        jobs.append((row, domain, level, text, views))
        job_bytes += 4 * len(text)
        if job_bytes >= _nlp_queue_bytes(config) or len(jobs) >= 1024:
            flush_jobs()

    with config.analysis_pages.open("rb") as handle:
        for record in iter_bounded_tsv(
            handle,
            columns=len(PAGES_COLUMNS),
            max_line_bytes=config.runtime.max_line_bytes,
            max_rows=config.runtime.max_rows,
        ):
            progress.update(record.end_offset, detail=f"linha {record.row_number:,}")
            if record.fields is None:
                continue
            fields = record.fields
            if decode_field(fields[11]) != "done":
                continue
            try:
                decoded = decode_text(fields[13], fields[15], config.runtime.max_text_bytes)
            except TextDecodeFailure:
                continue
            if not decoded.normalized:
                continue
            domain = parse_int(fields[1])
            level = parse_int(fields[10])
            is_d2 = d2.contains(record.row_number)
            is_d3 = d3.contains(record.row_number)
            views = ["R_valid"]
            if is_d2:
                views.append("E_exact")
            if is_d3:
                clean = clean_again(config, candidates, domain, decoded.normalized)
                if clean:
                    if clean == decoded.normalized:
                        views.append("B_clean")
                    else:
                        add_job(record.row_number, domain, level, clean, ("B_clean",))
            add_job(record.row_number, domain, level, decoded.normalized, tuple(views))
    flush_jobs()
    progress.finish(detail="primeira passagem concluída")
    candidates.close()
    if documents or document_chunk == 0:
        write_parquet_atomic(
            document_dir / f"documents-{document_chunk:06d}.parquet", documents, DOCUMENT_SCHEMA
        )
    vocabulary.flush()
    content_writer.finish(identity, manifest["sources"]["pages"]["size"])
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
    candidates_path = root / "bigram_candidates.json"
    atomic_json(
        candidates_path,
        {
            "omitted_upper_bound": bigrams.omitted_upper_bound,
            "items": len(bigrams.values),
        },
    )
    write_parquet_atomic(
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
        config, root, nlp, set(bigrams.values), bigrams.omitted_upper_bound
    )
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


def recount_bigrams(config: Config, root: Path, nlp, wanted: set[str], upper_bound: int) -> dict:
    counts = Counter()
    documents = Counter()
    jobs: list[str] = []
    job_bytes = 0
    progress = ByteProgress("Recontagem de bigramas", config.analysis_pages.stat().st_size)

    def flush() -> None:
        nonlocal jobs, job_bytes
        if not jobs:
            return
        for clean in jobs:
            document = nlp.make_doc(clean)
            seen: set[str] = set()
            for words in _paragraph_word_sequences(document, clean):
                for left, right in pairwise(words):
                    key = left + "\t" + right
                    if key in wanted:
                        counts[key] += 1
                        seen.add(key)
            documents.update(seen)
        jobs = []
        job_bytes = 0

    for item in iter_bclean(config, root):
        progress.update(
            item.source.end_offset, detail=f"linha {item.source.row_number:,}"
        )
        jobs.append(item.text)
        job_bytes += 4 * len(item.text)
        if job_bytes >= _nlp_queue_bytes(config) or len(jobs) >= 1024:
            flush()
    flush()
    progress.update(config.analysis_pages.stat().st_size)
    progress.finish(detail="segunda passagem concluída")
    rows = [
        {
            "bigram": term,
            "total_frequency": frequency,
            "document_frequency": documents[term],
            "contains_stopword": any(
                component in nlp.Defaults.stop_words for component in term.split("\t")
            ),
        }
        for term, frequency in counts.most_common()
    ]
    write_parquet_atomic(root / "bigrams.parquet", rows, BIGRAM_SCHEMA)
    published = config.lexical.published_items
    eligible = [row for row in rows if not row["contains_stopword"]]
    last = eligible[min(published, len(eligible)) - 1]["total_frequency"] if eligible else 0
    certified = upper_bound < last
    return {
        "candidates": len(wanted),
        "omitted_upper_bound": upper_bound,
        "published_k": min(published, len(eligible)),
        "last_frequency": last,
        "certified": certified,
    }
