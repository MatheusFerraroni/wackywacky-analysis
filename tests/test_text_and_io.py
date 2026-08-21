from __future__ import annotations

import hashlib
import io

import pytest
import spacy
import zstandard as zstd

from wackywacky_analysis.io import iter_bounded_tsv
from wackywacky_analysis.lexical import _paragraph_word_sequences
from wackywacky_analysis.text import (
    TextDecodeFailure,
    block_units,
    decode_text,
    normalize_text,
    paragraph_units,
    remove_intervals,
)


def test_normalization_preserves_semantics_and_collapses_layout() -> None:
    source = "  Olá\t mundo  \r\n\rcontrole\x00\r\n\r\n\r\n123, Árvore!  "
    assert normalize_text(source) == "Olá mundo\n\ncontrole\n\n\n123, Árvore!"


def test_decode_is_independent_from_md5() -> None:
    raw = "Texto válido em português.".encode()
    field = zstd.ZstdCompressor().compress(raw).hex().encode()
    matching = hashlib.md5(raw, usedforsecurity=False).hexdigest().encode()
    assert decode_text(field, matching, 1024).md5_class == "match"
    assert decode_text(field, b"0" * 32, 1024).md5_class == "mismatch"
    assert decode_text(field, b"bad", 1024).md5_class == "invalid"
    assert decode_text(field, b"NULL", 1024).md5_class == "missing"


@pytest.mark.parametrize(
    ("field", "expected"),
    [(b"zz", "invalid_hex"), (b"00", "invalid_zstd")],
)
def test_decode_classifies_corruption(field: bytes, expected: str) -> None:
    with pytest.raises(TextDecodeFailure, match=expected):
        decode_text(field, b"NULL", 1024)


def test_decode_rejects_invalid_utf8_and_oversized_text() -> None:
    invalid = zstd.ZstdCompressor().compress(b"\xff").hex().encode()
    with pytest.raises(TextDecodeFailure, match="invalid_utf8"):
        decode_text(invalid, b"NULL", 1024)
    large = zstd.ZstdCompressor().compress(b"a" * 20).hex().encode()
    with pytest.raises(TextDecodeFailure, match="text_too_large"):
        decode_text(large, b"NULL", 10)


def test_limited_reader_drains_long_line() -> None:
    handle = io.BytesIO(b"a" * 50 + b"\n1\t2\n")
    rows = list(iter_bounded_tsv(handle, columns=2, max_line_bytes=10))
    assert rows[0].error == "line_too_large"
    assert rows[1].fields == (b"1", b"2")


def test_truncated_record_is_structural_error() -> None:
    row = next(iter_bounded_tsv(io.BytesIO(b"1\t2"), columns=3, max_line_bytes=100))
    assert row.error == "wrong_column_count"


def test_boilerplate_units_and_overlap_union() -> None:
    paragraph = "A" * 90
    text = paragraph + "\n\nlinha um longa\nlinha dois longa\nlinha três longa\nfim"
    paragraphs = paragraph_units(text, 80)
    blocks = block_units(text, 3, 30)
    assert len(paragraphs) == 1
    assert blocks
    cleaned, removed = remove_intervals(text, [(0, 20), (10, 30)])
    assert removed == 30
    assert cleaned.startswith("A" * 60)
    emptied, removed_all = remove_intervals(paragraph, [(0, len(paragraph))])
    assert emptied == ""
    assert removed_all == len(paragraph)


def test_bigram_sequences_do_not_cross_paragraphs() -> None:
    text = "primeiro fim\n\nsegundo começo"
    sequences = _paragraph_word_sequences(spacy.blank("pt")(text), text)
    assert sequences == [["primeiro", "fim"], ["segundo", "começo"]]
