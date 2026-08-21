from __future__ import annotations

import hashlib
import re
import shutil
import signal
import unicodedata
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .bclean import BCleanRecord, iter_bclean
from .config import Config
from .io import BinaryRecord, atomic_json, read_json
from .lexical import (
    SPACY_MAX_TOKENS_PER_CHUNK,
    SpaceSaving,
    _iter_tokens,
    _load_nlp,
    _nlp_queue_bytes,
    _paragraph_word_sequences,
    _parse_document,
    _raw_document,
)
from .progress import LOGGER, ByteProgress, logged_stage
from .storage import duckdb_connection, duckdb_copy_atomic, write_parquet_atomic

CONTENT_SCHEMA_VERSION = 2
MOJIBAKE_MARKERS = (
    "Ã¡",
    "Ã¢",
    "Ã£",
    "Ã¤",
    "Ã©",
    "Ãª",
    "Ã­",
    "Ã³",
    "Ã´",
    "Ãµ",
    "Ã¶",
    "Ãº",
    "Ã¼",
    "Ã§",
    "Â\u00a0",
    "â€“",
    "â€”",
    "â€™",
    "â€œ",
    "â€",
    "â€¦",
    "ðŸ",
    "\ufffd",
)
MORPH_FEATURES = ("Gender", "Number", "Person", "Tense", "Mood", "VerbForm")
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
POS_GROUPS = {
    "NOUN",
    "PROPN",
    "VERB",
    "AUX",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "ADP",
}

DOCUMENT_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("priority", pa.string()),
        ("domain_id", pa.int64()),
        ("recursion_level", pa.int32()),
        ("characters", pa.uint64()),
        ("words", pa.uint64()),
        ("numbers", pa.uint64()),
        ("other_tokens", pa.uint64()),
        ("sentences", pa.uint64()),
        ("paragraphs", pa.uint64()),
        ("unique_forms", pa.uint64()),
        ("unique_lemmas", pa.uint64()),
        ("local_hapax", pa.uint64()),
        ("ttr", pa.float64()),
        ("mattr", pa.float64()),
        ("lexical_density", pa.float64()),
        ("repeated_sentence_words", pa.uint64()),
        ("repeated_paragraph_words", pa.uint64()),
        ("repeated_sentence_fraction", pa.float64()),
        ("repeated_paragraph_fraction", pa.float64()),
        ("max_sentence_run", pa.uint32()),
        ("max_paragraph_run", pa.uint32()),
        ("fragment_sentences", pa.uint64()),
        ("long_sentences", pa.uint64()),
        ("long_tokens", pa.uint64()),
        ("urls", pa.uint64()),
        ("emails", pa.uint64()),
        ("punctuation_runs", pa.uint64()),
        ("mojibake_markers", pa.uint64()),
        ("uppercase_words", pa.uint64()),
        ("high_numeric", pa.bool_()),
        ("high_nonlexical", pa.bool_()),
        ("high_uppercase", pa.bool_()),
        ("any_signal", pa.bool_()),
    ]
)

HISTOGRAM_SCHEMA = pa.schema(
    [("metric", pa.string()), ("value", pa.uint64()), ("count", pa.uint64())]
)

GRAMMAR_SCHEMA = pa.schema(
    [
        ("kind", pa.string()),
        ("feature", pa.string()),
        ("value", pa.string()),
        ("count", pa.uint64()),
    ]
)

VOCABULARY_FIRST_SCHEMA = pa.schema(
    [("kind", pa.string()), ("term", pa.string()), ("first_priority", pa.string())]
)

TRIGRAM_STATE_SCHEMA = pa.schema(
    [
        ("trigram", pa.string()),
        ("estimated_frequency", pa.uint64()),
        ("maximum_error", pa.uint64()),
        ("version", pa.uint64()),
    ]
)

TRIGRAM_SCHEMA = pa.schema(
    [
        ("trigram", pa.string()),
        ("total_frequency", pa.uint64()),
        ("document_frequency", pa.uint64()),
        ("contains_stopword_at_edges", pa.bool_()),
    ]
)

BIGRAM_MARGINAL_SCHEMA = pa.schema(
    [("term", pa.string()), ("left_count", pa.uint64()), ("right_count", pa.uint64())]
)


def _paragraph_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for boundary in re.finditer(r"\n{2,}", text):
        end = boundary.start()
        value = text[start:end].strip()
        if value:
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            spans.append((left, left + len(value), value))
        start = boundary.end()
    value = text[start:].strip()
    if value:
        left = start + len(text[start:]) - len(text[start:].lstrip())
        spans.append((left, left + len(value), value))
    return spans


def _canonical_unit(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value).casefold()).strip()


def _maximum_run(values: list[str]) -> int:
    maximum = current = 0
    previous: str | None = None
    for value in values:
        current = current + 1 if value == previous else 1
        maximum = max(maximum, current)
        previous = value
    return maximum


def _mattr(words: list[str], window: int) -> float | None:
    if len(words) < window:
        return None
    counts: Counter[str] = Counter(words[:window])
    total = len(counts) / window
    windows = 1
    for leaving, entering in zip(words, words[window:]):
        counts[leaving] -= 1
        if not counts[leaving]:
            del counts[leaving]
        counts[entering] += 1
        total += len(counts) / window
        windows += 1
    return total / windows


def _pos_group(value: str) -> str:
    if value in POS_GROUPS:
        return value
    if value in {"CCONJ", "SCONJ"}:
        return "CONJ"
    return "OTHER"


def _tokens_by_paragraph(document, paragraphs: list[tuple[int, int, str]]) -> list[list]:
    buckets: list[list] = [[] for _paragraph in paragraphs]
    paragraph_index = 0
    for token in _iter_tokens(document):
        if token.is_space:
            continue
        while (
            paragraph_index < len(paragraphs)
            and token.idx >= paragraphs[paragraph_index][1]
        ):
            paragraph_index += 1
        if paragraph_index >= len(paragraphs):
            break
        start, end, _value = paragraphs[paragraph_index]
        if start <= token.idx < end:
            buckets[paragraph_index].append(token)
    return buckets


def _sentence_units(text: str, paragraph_tokens: list[list]):
    result: list[tuple[str, int, int, int]] = []
    for paragraph_index, tokens in enumerate(paragraph_tokens):
        current = []
        for token in tokens:
            if token.is_sent_start and current:
                first, last = current[0], current[-1]
                words = sum(item.is_alpha for item in current)
                numeric = sum(item.like_num and not item.is_alpha for item in current)
                if words or numeric:
                    result.append(
                        (
                            text[first.idx : last.idx + len(last.text)].strip(),
                            words,
                            paragraph_index,
                            last.idx + len(last.text) - first.idx,
                        )
                    )
                current = []
            current.append(token)
        if current:
            first, last = current[0], current[-1]
            words = sum(item.is_alpha for item in current)
            numeric = sum(item.like_num and not item.is_alpha for item in current)
            if words or numeric:
                result.append(
                    (
                        text[first.idx : last.idx + len(last.text)].strip(),
                        words,
                        paragraph_index,
                        last.idx + len(last.text) - first.idx,
                    )
                )
    return result


def analyze_document(config: Config, record: BCleanRecord, document) -> dict[str, Any]:
    content = config.content
    forms: list[str] = []
    lemmas: list[str] = []
    numbers = other = uppercase = 0
    long_tokens = urls = emails = 0
    pos: Counter[str] = Counter()
    morphology: Counter[tuple[str, str]] = Counter()
    for token in _iter_tokens(document):
        if token.is_alpha:
            form = unicodedata.normalize("NFC", token.text).casefold()
            lemma = unicodedata.normalize("NFC", token.lemma_ or token.text).casefold()
            forms.append(form)
            lemmas.append(lemma)
            group = _pos_group(token.pos_)
            pos[group] += 1
            uppercase += int(token.text.isupper())
            long_tokens += int(len(token.text) >= content.long_token_chars)
            for annotation in token.morph:
                feature, separator, value = annotation.partition("=")
                if separator and feature in MORPH_FEATURES:
                    morphology[(feature, value)] += 1
        elif token.like_num:
            numbers += 1
        elif not token.is_space:
            other += 1
        if not token.is_space:
            urls += int(token.like_url)
            emails += int(token.like_email)

    structure_document = _raw_document(document)
    paragraphs = _paragraph_spans(record.text)
    paragraph_tokens = _tokens_by_paragraph(structure_document, paragraphs)
    sentences = _sentence_units(record.text, paragraph_tokens)
    paragraph_words = [sum(token.is_alpha for token in tokens) for tokens in paragraph_tokens]
    paragraph_sentence_counts = Counter(item[2] for item in sentences)

    sentence_keys: list[str] = []
    sentence_key_words: dict[str, int] = {}
    for value, words, _paragraph, _characters in sentences:
        if words >= content.repetition_sentence_min_words:
            key = _canonical_unit(value)
            sentence_keys.append(key)
            sentence_key_words[key] = words
    paragraph_keys: list[str] = []
    paragraph_key_words: dict[str, int] = {}
    for (_start, _end, value), words in zip(paragraphs, paragraph_words, strict=True):
        if len(value) >= content.repetition_paragraph_min_chars:
            key = _canonical_unit(value)
            paragraph_keys.append(key)
            paragraph_key_words[key] = words
    sentence_counts = Counter(sentence_keys)
    paragraph_counts = Counter(paragraph_keys)
    repeated_sentence_words = sum(
        sentence_key_words[key] * (count - 1)
        for key, count in sentence_counts.items()
        if count > 1
    )
    repeated_paragraph_words = sum(
        paragraph_key_words[key] * (count - 1)
        for key, count in paragraph_counts.items()
        if count > 1
    )
    word_count = len(forms)
    form_counts = Counter(forms)
    lexical_words = sum(pos[key] for key in CONTENT_POS)
    punctuation_runs = len(
        re.findall(rf"([^\w\s])\1{{{content.punctuation_run - 1},}}", record.text)
    )
    mojibake = sum(record.text.count(marker) for marker in MOJIBAKE_MARKERS)
    fragments = sum(words <= content.fragment_max_words for _v, words, _p, _c in sentences)
    long_sentences = sum(words >= content.long_sentence_words for _v, words, _p, _c in sentences)
    lexical_numeric = word_count + numbers
    all_tokens = lexical_numeric + other
    high_numeric = (
        lexical_numeric >= content.fraction_min_tokens
        and numbers / lexical_numeric >= content.high_numeric_fraction
    )
    high_nonlexical = (
        all_tokens >= content.fraction_min_tokens
        and other / all_tokens >= content.high_nonlexical_fraction
    )
    high_uppercase = (
        word_count >= content.fraction_min_tokens
        and uppercase / word_count >= content.high_uppercase_fraction
    )
    signals = (
        fragments
        + long_sentences
        + long_tokens
        + urls
        + emails
        + punctuation_runs
        + mojibake
        + int(high_numeric)
        + int(high_nonlexical)
        + int(high_uppercase)
    )
    priority = hashlib.sha256(
        f"{content.vocabulary_seed}:{record.source.row_number}".encode()
    ).hexdigest()
    histograms: Counter[tuple[str, int]] = Counter()
    histograms[("frases_por_documento", len(sentences))] += 1
    histograms[("paragrafos_por_documento", len(paragraphs))] += 1
    for _value, words, _paragraph, characters in sentences:
        histograms[("palavras_por_frase", words)] += 1
        histograms[("caracteres_por_frase", characters)] += 1
    for index, ((start, end, _value), words) in enumerate(zip(paragraphs, paragraph_words, strict=True)):
        histograms[("palavras_por_paragrafo", words)] += 1
        histograms[("caracteres_por_paragrafo", end - start)] += 1
        histograms[("frases_por_paragrafo", paragraph_sentence_counts[index])] += 1
    sequences = _paragraph_word_sequences(document, record.text)
    trigrams = [
        f"{left}\t{middle}\t{right}"
        for sequence in sequences
        for left, middle, right in zip(sequence, sequence[1:], sequence[2:])
    ]
    return {
        "document": {
            "row_number": record.source.row_number,
            "priority": priority,
            "domain_id": record.domain_id,
            "recursion_level": record.recursion_level,
            "characters": len(record.text),
            "words": word_count,
            "numbers": numbers,
            "other_tokens": other,
            "sentences": len(sentences),
            "paragraphs": len(paragraphs),
            "unique_forms": len(form_counts),
            "unique_lemmas": len(set(lemmas)),
            "local_hapax": sum(value == 1 for value in form_counts.values()),
            "ttr": len(form_counts) / word_count if word_count else 0.0,
            "mattr": _mattr(forms, content.mattr_window),
            "lexical_density": lexical_words / word_count if word_count else 0.0,
            "repeated_sentence_words": repeated_sentence_words,
            "repeated_paragraph_words": repeated_paragraph_words,
            "repeated_sentence_fraction": repeated_sentence_words / word_count if word_count else 0,
            "repeated_paragraph_fraction": repeated_paragraph_words / word_count if word_count else 0,
            "max_sentence_run": _maximum_run(sentence_keys),
            "max_paragraph_run": _maximum_run(paragraph_keys),
            "fragment_sentences": fragments,
            "long_sentences": long_sentences,
            "long_tokens": long_tokens,
            "urls": urls,
            "emails": emails,
            "punctuation_runs": punctuation_runs,
            "mojibake_markers": mojibake,
            "uppercase_words": uppercase,
            "high_numeric": high_numeric,
            "high_nonlexical": high_nonlexical,
            "high_uppercase": high_uppercase,
            "any_signal": bool(signals),
        },
        "histograms": histograms,
        "pos": pos,
        "morphology": morphology,
        "forms": set(forms),
        "lemmas": set(lemmas),
        "priority": priority,
        "trigrams": trigrams,
        "bigram_positions": sum(max(0, len(sequence) - 1) for sequence in sequences),
        "trigram_positions": sum(max(0, len(sequence) - 2) for sequence in sequences),
    }


def _load_trigram_state(config: Config, root: Path, state: dict) -> SpaceSaving:
    name = state.get("trigram_state")
    if not name:
        return SpaceSaving(config.content.trigram_candidates)
    path = root / "content" / name
    rows = [
        (
            row["trigram"],
            row["estimated_frequency"],
            row["maximum_error"],
            row["version"],
        )
        for row in pq.read_table(path).to_pylist()
    ]
    return SpaceSaving.restore(config.content.trigram_candidates, rows)


def _write_trigram_state(root: Path, chunk: int, trigrams: SpaceSaving) -> str:
    name = f"trigram-state-{chunk:06d}.parquet"
    write_parquet_atomic(
        root / "content" / name,
        [
            {
                "trigram": term,
                "estimated_frequency": value[0],
                "maximum_error": value[1],
                "version": value[2],
            }
            for term, value in sorted(trigrams.values.items())
        ],
        TRIGRAM_STATE_SCHEMA,
    )
    return name


def _write_content_chunk(
    root: Path,
    chunk: int,
    documents: list[dict],
    histograms: Counter[tuple[str, int]],
    grammar: Counter[tuple[str, str, str]],
    vocabulary: dict[tuple[str, str], str],
) -> None:
    content_root = root / "content"
    write_parquet_atomic(
        content_root / "documents" / f"chunk-{chunk:06d}.parquet",
        documents,
        DOCUMENT_SCHEMA,
    )
    write_parquet_atomic(
        content_root / "histograms" / f"chunk-{chunk:06d}.parquet",
        [
            {"metric": metric, "value": value, "count": count}
            for (metric, value), count in sorted(histograms.items())
        ],
        HISTOGRAM_SCHEMA,
    )
    write_parquet_atomic(
        content_root / "grammar" / f"chunk-{chunk:06d}.parquet",
        [
            {"kind": kind, "feature": feature, "value": value, "count": count}
            for (kind, feature, value), count in sorted(grammar.items())
        ],
        GRAMMAR_SCHEMA,
    )
    write_parquet_atomic(
        content_root / "vocabulary-first" / f"chunk-{chunk:06d}.parquet",
        [
            {"kind": kind, "term": term, "first_priority": priority}
            for (kind, term), priority in sorted(vocabulary.items())
        ],
        VOCABULARY_FIRST_SCHEMA,
    )


def _remove_content_products(root: Path) -> None:
    shutil.rmtree(root / "content", ignore_errors=True)
    for stale in (
        root / "content_summary.json",
        root / "content_histograms.parquet",
        root / "content_grammar.parquet",
        root / "content_vocabulary_first.parquet",
        root / "content_trigrams.parquet",
        root / "content_trigram_candidates.parquet",
        root / "content_bigram_marginals.parquet",
    ):
        stale.unlink(missing_ok=True)
        stale.with_suffix(stale.suffix + ".sha256").unlink(missing_ok=True)


class EmbeddedContentWriter:
    """Collect content features from the lexical pass without another tokenization."""

    def __init__(self, config: Config, root: Path) -> None:
        self.config = config
        self.root = root
        state = read_json(root / "content" / "state.json", {})
        self.active = bool(
            config.content.enabled
            and not (
                state.get("complete")
                and state.get("content_fingerprint") == config.content_fingerprint
                and state.get("schema_version") == CONTENT_SCHEMA_VERSION
            )
        )
        if not self.active:
            return
        _remove_content_products(root)
        self.documents: list[dict] = []
        self.histograms: Counter[tuple[str, int]] = Counter()
        self.grammar: Counter[tuple[str, str, str]] = Counter()
        self.vocabulary: dict[tuple[str, str], str] = {}
        self.trigrams = SpaceSaving(config.content.trigram_candidates)
        self.chunk = 0
        self.documents_total = 0
        self.bigram_positions = 0
        self.trigram_positions = 0
        self.last_row = 0

    def add(
        self,
        row_number: int,
        domain_id: int | None,
        recursion_level: int | None,
        text: str,
        document,
    ) -> None:
        if not self.active:
            return
        record = BCleanRecord(
            source=BinaryRecord(row_number, 0, 0, None),
            domain_id=domain_id,
            recursion_level=recursion_level,
            text=text,
        )
        result = analyze_document(self.config, record, document)
        self.documents.append(result["document"])
        self.histograms.update(result["histograms"])
        for value, count in result["pos"].items():
            self.grammar[("pos", "POS", value)] += count
        for (feature, value), count in result["morphology"].items():
            self.grammar[("morphology", feature, value)] += count
        priority = result["priority"]
        for kind, terms in (("form", result["forms"]), ("lemma", result["lemmas"])):
            for term in terms:
                key = (kind, term)
                if key not in self.vocabulary or priority < self.vocabulary[key]:
                    self.vocabulary[key] = priority
        for trigram in result["trigrams"]:
            self.trigrams.add(trigram)
        self.documents_total += 1
        self.bigram_positions += result["bigram_positions"]
        self.trigram_positions += result["trigram_positions"]
        self.last_row = row_number
        if len(self.documents) >= 50_000:
            self._flush()

    def _flush(self) -> None:
        if not self.documents:
            return
        _write_content_chunk(
            self.root,
            self.chunk,
            self.documents,
            self.histograms,
            self.grammar,
            self.vocabulary,
        )
        self.chunk += 1
        self.documents = []
        self.histograms = Counter()
        self.grammar = Counter()
        self.vocabulary = {}

    def finish(self, identity: dict, source_size: int) -> None:
        if not self.active:
            return
        self._flush()
        if self.chunk == 0:
            _write_content_chunk(self.root, 0, [], Counter(), Counter(), {})
            self.chunk = 1
        trigram_state = _write_trigram_state(self.root, self.chunk, self.trigrams)
        atomic_json(
            self.root / "content" / "state.json",
            {
                "schema_version": CONTENT_SCHEMA_VERSION,
                "content_fingerprint": self.config.content_fingerprint,
                "next_offset": source_size,
                "next_row": self.last_row,
                "next_chunk": self.chunk,
                "documents": self.documents_total,
                "bigram_positions": self.bigram_positions,
                "trigram_positions": self.trigram_positions,
                "trigram_state": trigram_state,
                "spacy": identity,
                "complete": True,
                "collection": "embedded_in_lexical_pass",
            },
        )


def _new_nlp(config: Config):
    nlp = _load_nlp(config)
    if not any(name in nlp.pipe_names for name in ("parser", "senter", "sentencizer")):
        nlp.add_pipe("sentencizer")
    return nlp


def _scan_content(config: Config, manifest: dict, root: Path) -> dict:
    content_root = root / "content"
    state_path = content_root / "state.json"
    state = read_json(state_path, {})
    offset = int(state.get("next_offset", 0))
    row_number = int(state.get("next_row", 0))
    chunk = int(state.get("next_chunk", 0))
    documents_total = int(state.get("documents", 0))
    bigram_positions = int(state.get("bigram_positions", 0))
    trigram_positions = int(state.get("trigram_positions", 0))
    trigrams = _load_trigram_state(config, root, state)
    nlp = _new_nlp(config)
    identity = {
        "spacy": __import__("spacy").__version__,
        "model": config.lexical.spacy_model,
        "model_version": nlp.meta.get("version", "blank"),
        "pipeline": list(nlp.pipe_names),
        "max_tokens_per_chunk": SPACY_MAX_TOKENS_PER_CHUNK,
    }
    progress = ByteProgress(
        "[8/12 — 1/4] Estrutura e conteúdo B_clean",
        manifest["sources"]["pages"]["size"],
        initial=offset,
    )
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not state.get("complete"):
            documents: list[dict] = []
            histograms: Counter[tuple[str, int]] = Counter()
            grammar: Counter[tuple[str, str, str]] = Counter()
            vocabulary: dict[tuple[str, str], str] = {}
            jobs: list[BCleanRecord] = []
            job_bytes = 0
            chunk_start = offset
            last_offset = offset
            last_row = row_number
            saw_record = False

            def flush_jobs(
                documents=documents,
                histograms=histograms,
                grammar=grammar,
                vocabulary=vocabulary,
            ) -> None:
                nonlocal jobs, job_bytes, documents_total, bigram_positions, trigram_positions
                if not jobs:
                    return
                last_job = jobs[-1]
                for job in jobs:
                    document = _parse_document(nlp, job.text)
                    result = analyze_document(config, job, document)
                    documents.append(result["document"])
                    histograms.update(result["histograms"])
                    for value, count in result["pos"].items():
                        grammar[("pos", "POS", value)] += count
                    for (feature, value), count in result["morphology"].items():
                        grammar[("morphology", feature, value)] += count
                    priority = result["priority"]
                    for kind, terms in (("form", result["forms"]), ("lemma", result["lemmas"])):
                        for term in terms:
                            key = (kind, term)
                            if key not in vocabulary or priority < vocabulary[key]:
                                vocabulary[key] = priority
                    for trigram in result["trigrams"]:
                        trigrams.add(trigram)
                    documents_total += 1
                    bigram_positions += result["bigram_positions"]
                    trigram_positions += result["trigram_positions"]
                jobs = []
                job_bytes = 0
                progress.update(
                    last_job.source.end_offset,
                    detail=f"{documents_total:,} documentos; checkpoint pendente",
                )

            exhausted = True
            for item in iter_bclean(
                config, root, start_offset=offset, start_row=row_number
            ):
                exhausted = False
                saw_record = True
                last_offset = item.source.end_offset
                last_row = item.source.row_number
                jobs.append(item)
                job_bytes += 4 * len(item.text)
                if (
                    job_bytes >= _nlp_queue_bytes(config)
                    or len(jobs) >= 256
                ):
                    flush_jobs()
                if last_offset - chunk_start >= config.runtime.chunk_bytes:
                    break
            flush_jobs()
            if saw_record:
                _write_content_chunk(root, chunk, documents, histograms, grammar, vocabulary)
                new_trigram_state = _write_trigram_state(root, chunk, trigrams)
                previous_state = state.get("trigram_state")
                offset, row_number = last_offset, last_row
                chunk += 1
                state = {
                    "schema_version": CONTENT_SCHEMA_VERSION,
                    "content_fingerprint": config.content_fingerprint,
                    "next_offset": offset,
                    "next_row": row_number,
                    "next_chunk": chunk,
                    "documents": documents_total,
                    "bigram_positions": bigram_positions,
                    "trigram_positions": trigram_positions,
                    "trigram_state": new_trigram_state,
                    "spacy": identity,
                    "complete": False,
                }
                atomic_json(state_path, state)
                if previous_state and previous_state != new_trigram_state:
                    old = content_root / previous_state
                    old.unlink(missing_ok=True)
                    old.with_suffix(old.suffix + ".sha256").unlink(missing_ok=True)
                progress.update(offset, detail=f"{documents_total:,} documentos B_clean")
                if stop:
                    progress.close()
                    return {"complete": False}
            if exhausted:
                state["complete"] = True
                state["next_offset"] = manifest["sources"]["pages"]["size"]
                atomic_json(state_path, state)
                break
        progress.update(manifest["sources"]["pages"]["size"])
        progress.finish(detail=f"{documents_total:,} documentos B_clean")
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return state


def _recount_trigrams(config: Config, manifest: dict, root: Path, wanted: set[str]) -> None:
    output = root / "content_trigrams.parquet"
    marginal_output = root / "content_bigram_marginals.parquet"
    if output.exists() and marginal_output.exists():
        return
    recount_root = root / "content" / "trigram-recount"
    if recount_root.exists() and not marginal_output.exists():
        shutil.rmtree(recount_root)
        output.unlink(missing_ok=True)
        output.with_suffix(output.suffix + ".sha256").unlink(missing_ok=True)
    recount_root.mkdir(parents=True, exist_ok=True)
    state_path = recount_root / "state.json"
    state = read_json(state_path, {})
    offset = int(state.get("next_offset", 0))
    row_number = int(state.get("next_row", 0))
    chunk = int(state.get("next_chunk", 0))
    nlp = _load_nlp(config)
    for pipe_name in list(nlp.pipe_names):
        nlp.remove_pipe(pipe_name)
    progress = ByteProgress(
        "[8/12 — 4/4] Recontagem exata de trigramas",
        manifest["sources"]["pages"]["size"],
        initial=offset,
    )
    while not state.get("complete"):
        counts: Counter[str] = Counter()
        documents: Counter[str] = Counter()
        left_counts: Counter[str] = Counter()
        right_counts: Counter[str] = Counter()
        jobs: list[BCleanRecord] = []
        job_bytes = 0
        chunk_start = offset
        last_offset = offset
        last_row = row_number
        saw_record = False

        def flush_jobs(
            counts=counts,
            documents=documents,
            left_counts=left_counts,
            right_counts=right_counts,
        ) -> None:
            nonlocal jobs, job_bytes
            if not jobs:
                return
            for job in jobs:
                document = nlp.make_doc(job.text)
                seen: set[str] = set()
                for sequence in _paragraph_word_sequences(document, job.text):
                    for left, right in pairwise(sequence):
                        left_counts[left] += 1
                        right_counts[right] += 1
                    for values in zip(sequence, sequence[1:], sequence[2:]):
                        key = "\t".join(values)
                        if key in wanted:
                            counts[key] += 1
                            seen.add(key)
                documents.update(seen)
            jobs = []
            job_bytes = 0

        exhausted = True
        for item in iter_bclean(config, root, start_offset=offset, start_row=row_number):
            exhausted = False
            saw_record = True
            last_offset = item.source.end_offset
            last_row = item.source.row_number
            jobs.append(item)
            job_bytes += 4 * len(item.text)
            if job_bytes >= _nlp_queue_bytes(config) or len(jobs) >= 1024:
                flush_jobs()
            if last_offset - chunk_start >= config.runtime.chunk_bytes:
                break
        flush_jobs()
        if saw_record:
            rows = [
                {
                    "trigram": key,
                    "total_frequency": value,
                    "document_frequency": documents[key],
                    "contains_stopword_at_edges": any(
                        edge in nlp.Defaults.stop_words
                        for edge in (key.split("\t")[0], key.split("\t")[-1])
                    ),
                }
                for key, value in sorted(counts.items())
            ]
            write_parquet_atomic(recount_root / f"chunk-{chunk:06d}.parquet", rows, TRIGRAM_SCHEMA)
            marginal_terms = sorted(left_counts.keys() | right_counts.keys())
            write_parquet_atomic(
                recount_root / f"marginals-{chunk:06d}.parquet",
                [
                    {
                        "term": term,
                        "left_count": left_counts[term],
                        "right_count": right_counts[term],
                    }
                    for term in marginal_terms
                ],
                BIGRAM_MARGINAL_SCHEMA,
            )
            offset, row_number = last_offset, last_row
            chunk += 1
            state = {
                "next_offset": offset,
                "next_row": row_number,
                "next_chunk": chunk,
                "complete": False,
            }
            atomic_json(state_path, state)
            progress.update(offset)
        if exhausted:
            state["complete"] = True
            atomic_json(state_path, state)
            break
    progress.update(manifest["sources"]["pages"]["size"])
    progress.finish()
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    glob = str(recount_root / "chunk-*.parquet").replace("'", "''")
    duckdb_copy_atomic(
        connection,
        f"""
        SELECT trigram, sum(total_frequency)::UBIGINT total_frequency,
               sum(document_frequency)::UBIGINT document_frequency,
               bool_or(contains_stopword_at_edges) contains_stopword_at_edges
        FROM read_parquet('{glob}') GROUP BY trigram
        """,
        output,
    )
    marginal_glob = str(recount_root / "marginals-*.parquet").replace("'", "''")
    duckdb_copy_atomic(
        connection,
        f"""
        SELECT term, sum(left_count)::UBIGINT left_count,
               sum(right_count)::UBIGINT right_count
        FROM read_parquet('{marginal_glob}') GROUP BY term
        """,
        marginal_output,
    )
    connection.close()


def _reduce_content(config: Config, root: Path, state: dict) -> dict:
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    content_root = root / "content"
    documents_glob = str(content_root / "documents" / "*.parquet").replace("'", "''")
    histogram_glob = str(content_root / "histograms" / "*.parquet").replace("'", "''")
    grammar_glob = str(content_root / "grammar" / "*.parquet").replace("'", "''")
    vocabulary_glob = str(content_root / "vocabulary-first" / "*.parquet").replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE VIEW content_document_statistics AS SELECT * FROM read_parquet('{documents_glob}')"
    )
    duckdb_copy_atomic(
        connection,
        f"SELECT metric, value, sum(count)::UBIGINT count FROM read_parquet('{histogram_glob}') GROUP BY metric, value",
        root / "content_histograms.parquet",
    )
    duckdb_copy_atomic(
        connection,
        f"SELECT kind, feature, value, sum(count)::UBIGINT count FROM read_parquet('{grammar_glob}') GROUP BY kind, feature, value",
        root / "content_grammar.parquet",
    )
    duckdb_copy_atomic(
        connection,
        f"SELECT kind, term, min(first_priority) first_priority FROM read_parquet('{vocabulary_glob}') GROUP BY kind, term",
        root / "content_vocabulary_first.parquet",
    )
    summary_row = connection.execute(
        """
        SELECT count(*), sum(words), sum(sentences), sum(paragraphs),
               count(mattr), avg(mattr), avg(ttr), avg(lexical_density),
               sum(repeated_sentence_words), sum(repeated_paragraph_words),
               count(*) FILTER (WHERE any_signal)
        FROM content_document_statistics
        """
    ).fetchone()
    connection.close()
    return {
        "documents": summary_row[0],
        "words": summary_row[1] or 0,
        "sentences": summary_row[2] or 0,
        "paragraphs": summary_row[3] or 0,
        "mattr_documents": summary_row[4],
        "mean_mattr": summary_row[5],
        "mean_ttr": summary_row[6],
        "mean_lexical_density": summary_row[7],
        "repeated_sentence_words": summary_row[8] or 0,
        "repeated_paragraph_words": summary_row[9] or 0,
        "documents_with_signals": summary_row[10],
        "bigram_positions": state["bigram_positions"],
        "trigram_positions": state["trigram_positions"],
    }


def content_pass(config: Config, manifest: dict, root: Path) -> dict:
    summary_path = root / "content_summary.json"
    current = read_json(summary_path, {})
    if (
        current.get("content_fingerprint") == config.content_fingerprint
        and (root / "content_bigram_marginals.parquet").exists()
    ):
        return current
    if not config.content.enabled:
        summary = {
            "status": "disabled",
            "content_fingerprint": config.content_fingerprint,
            "schema_version": CONTENT_SCHEMA_VERSION,
        }
        atomic_json(summary_path, summary)
        return summary
    content_root = root / "content"
    state = read_json(content_root / "state.json", {})
    if state and (
        state.get("content_fingerprint") != config.content_fingerprint
        or state.get("schema_version") != CONTENT_SCHEMA_VERSION
    ):
        _remove_content_products(root)
        state = {}
    scan = _scan_content(config, manifest, root)
    if scan.get("complete") is False:
        return {"complete": False, "stage": "content"}
    with logged_stage("[8/12 — 2/4] Redução dos agregados de conteúdo"):
        metrics = _reduce_content(config, root, scan)
    trigram_state = root / "content" / scan["trigram_state"]
    state_rows = pq.read_table(trigram_state).to_pylist()
    candidates = {row["trigram"] for row in state_rows}
    with logged_stage("[8/12 — 3/4] Preparação dos trigramas candidatos"):
        write_parquet_atomic(
            root / "content_trigram_candidates.parquet", state_rows, TRIGRAM_STATE_SCHEMA
        )
    _recount_trigrams(config, manifest, root, candidates)
    omitted = min(
        (row["estimated_frequency"] for row in state_rows), default=0
    ) if len(state_rows) >= config.content.trigram_candidates else 0
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    eligible_floor = max(config.content.collocation_min_frequency, omitted + 1)
    last = connection.execute(
        """
        SELECT min(total_frequency) FROM (
          SELECT total_frequency FROM read_parquet(?)
          WHERE NOT contains_stopword_at_edges
          ORDER BY total_frequency DESC, trigram LIMIT ?
        )
        """,
        [str(root / "content_trigrams.parquet"), config.lexical.published_items],
    ).fetchone()[0]
    connection.close()
    frequent_certified = bool(last is None or omitted < last)
    if not frequent_certified:
        LOGGER.warning(
            "top-K de trigramas não certificado; a tabela publicará apenas o universo acima de %s",
            eligible_floor,
        )
    summary = {
        "status": "complete",
        "schema_version": CONTENT_SCHEMA_VERSION,
        "content_fingerprint": config.content_fingerprint,
        "spacy": scan["spacy"],
        "metrics": metrics,
        "trigrams": {
            "candidates": len(candidates),
            "omitted_upper_bound": omitted,
            "eligible_frequency_floor": eligible_floor,
            "frequent_top_k_certified": frequent_certified,
        },
        "parameters": config.public_dict()["content"],
        "mojibake_markers": list(MOJIBAKE_MARKERS),
    }
    atomic_json(summary_path, summary)
    return summary
