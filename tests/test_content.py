from __future__ import annotations

from dataclasses import replace

import spacy
from conftest import write_config, write_sources

from wackywacky_analysis.bclean import BCleanRecord
from wackywacky_analysis.config import load_config
from wackywacky_analysis.content import _mattr, analyze_document
from wackywacky_analysis.io import BinaryRecord
from wackywacky_analysis.lexical import SPACY_MAX_TOKENS_PER_CHUNK, _parse_document


def _record(text: str) -> BCleanRecord:
    return BCleanRecord(
        source=BinaryRecord(row_number=7, offset=0, end_offset=100, fields=None),
        domain_id=1,
        recursion_level=0,
        text=text,
    )


def test_mattr_uses_a_rolling_window_and_rejects_short_documents() -> None:
    assert _mattr(["a", "b", "c"], 4) is None
    assert _mattr(["a", "b", "a", "c", "d"], 3) == (2 / 3 + 1 + 1) / 3


def test_content_metrics_cover_structure_repetition_signals_and_trigrams(tmp_path) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    repeated = "Parágrafo repetido possui cinco palavras úteis."
    text = (
        f"{repeated}\n\n{repeated}\n\n"
        "Curta! termoextraordinariamentemuitolongo !!!! exemplo@teste.invalid "
        "https://example.invalid Ã£"
    )
    nlp = spacy.blank("pt")
    nlp.add_pipe("sentencizer")
    result = analyze_document(config, _record(text), nlp(text))
    document = result["document"]
    assert document["paragraphs"] == 3
    assert document["sentences"] >= 4
    assert document["repeated_paragraph_words"] == 6
    assert document["max_paragraph_run"] == 2
    assert document["fragment_sentences"] >= 1
    assert document["long_tokens"] >= 1
    assert document["punctuation_runs"] == 1
    assert document["mojibake_markers"] == 1
    assert sum(result["pos"].values()) == document["words"]
    assert all("úteis\tparágrafo" not in trigram for trigram in result["trigrams"])


def test_content_configuration_does_not_invalidate_the_base_snapshot(tmp_path) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    changed = replace(config, content=replace(config.content, mattr_window=17))
    assert changed.fingerprint == config.fingerprint
    assert changed.content_fingerprint != config.content_fingerprint


def test_spacy_inference_is_chunked_without_changing_global_structure(tmp_path) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    nlp = spacy.blank("pt")
    nlp.add_pipe("sentencizer")
    first = " ".join("palavra" for _index in range(SPACY_MAX_TOKENS_PER_CHUNK + 5))
    text = f"{first}.\n\nSegundo parágrafo curto."
    parsed = _parse_document(nlp, text)
    result = analyze_document(config, _record(text), parsed)
    assert len(parsed.annotated) == 2
    assert result["document"]["paragraphs"] == 2
    assert result["document"]["sentences"] == 2
    assert result["document"]["words"] == SPACY_MAX_TOKENS_PER_CHUNK + 8
    assert all("palavra\tsegundo" not in trigram for trigram in result["trigrams"])
