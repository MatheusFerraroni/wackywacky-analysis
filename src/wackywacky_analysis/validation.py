from __future__ import annotations

from itertools import pairwise

import duckdb

from .errors import WackyWackyError
from .io import atomic_json


def validate_invariants(
    exact: dict, clean: dict, lexical: dict, root, *, content: dict | None = None
) -> dict:
    metrics = exact["page_metrics"]
    if sum(metrics["status"].values()) != metrics["rows"]:
        raise WackyWackyError("invariante falhou: status não reconciliam com linhas")
    funnel = [
        metrics["done"],
        metrics["text_present"],
        metrics["decoded"],
        exact["r_valid"],
        exact["d1_unique"],
        exact["d2_unique"],
        clean["b_clean_nonempty"],
        clean["d3_unique"],
    ]
    if any(left < right for left, right in pairwise(funnel)):
        raise WackyWackyError("invariante falhou: funil não é monotônico")
    words = {view: value["words"] for view, value in lexical["views"].items()}
    for key, value in lexical["vocabulary"].items():
        view, kind = key.split(":", 1)
        if kind in {"form", "lemma"} and value["occurrences"] != words[view]:
            raise WackyWackyError(f"invariante falhou: palavras não reconciliam em {key}")
    connection = duckdb.connect(str(root / "analysis.duckdb"), read_only=True)
    invalid_df = connection.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE document_frequency > total_frequency",
        [str(root / "vocabulary.parquet")],
    ).fetchone()[0]
    counts = connection.execute(
        """
        SELECT (SELECT count(*) FROM page_inventory),
               (SELECT count(*) FROM d2_membership WHERE is_representative),
               (SELECT count(*) FROM d3_membership WHERE is_representative),
               (SELECT count(*) FROM document_statistics WHERE view='B_clean')
        """
    ).fetchone()
    connection.close()
    structural = sum(
        metrics["errors"].get(key, 0) for key in ("line_too_large", "wrong_column_count")
    )
    if counts[0] + structural != metrics["rows"]:
        raise WackyWackyError("invariante falhou: inventário de páginas não reconcilia")
    if (
        counts[1] != exact["d2_unique"]
        or counts[2] != clean["d3_unique"]
        or counts[3] != clean["d3_unique"]
    ):
        raise WackyWackyError("invariante falhou: representantes não reconciliam")
    if invalid_df:
        raise WackyWackyError("invariante falhou: frequência documental excede total")
    result = {
        "status_rows_reconciled": True,
        "funnel_monotonic": True,
        "vocabulary_reconciled": True,
        "document_frequency_valid": True,
        "representatives_reconciled": True,
    }
    if content and content.get("status") == "complete":
        connection = duckdb.connect(str(root / "analysis.duckdb"), read_only=True)
        documents, words, sentences, paragraphs = connection.execute(
            """
            SELECT count(*), sum(words), sum(sentences), sum(paragraphs)
            FROM content_document_statistics
            """
        ).fetchone()
        pos_words = connection.execute(
            "SELECT coalesce(sum(count),0) FROM read_parquet(?) WHERE kind='pos'",
            [str(root / "content_grammar.parquet")],
        ).fetchone()[0]
        histogram_totals = dict(
            connection.execute(
                """
                SELECT metric, sum(count) FROM read_parquet(?)
                WHERE metric IN ('frases_por_documento','paragrafos_por_documento',
                                 'palavras_por_frase','palavras_por_paragrafo')
                GROUP BY metric
                """,
                [str(root / "content_histograms.parquet")],
            ).fetchall()
        )
        invalid_trigram_df = connection.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE document_frequency > total_frequency",
            [str(root / "content_trigrams.parquet")],
        ).fetchone()[0]
        left_positions, right_positions = connection.execute(
            "SELECT sum(left_count), sum(right_count) FROM read_parquet(?)",
            [str(root / "content_bigram_marginals.parquet")],
        ).fetchone()
        connection.close()
        if documents != clean["d3_unique"] or words != lexical["views"]["B_clean"]["words"]:
            raise WackyWackyError("invariante falhou: conteúdo não reconcilia com B_clean")
        if pos_words != words:
            raise WackyWackyError("invariante falhou: classes gramaticais não reconciliam")
        if (
            histogram_totals.get("frases_por_documento", 0) != documents
            or histogram_totals.get("paragrafos_por_documento", 0) != documents
            or histogram_totals.get("palavras_por_frase", 0) != sentences
            or histogram_totals.get("palavras_por_paragrafo", 0) != paragraphs
        ):
            raise WackyWackyError("invariante falhou: histogramas de conteúdo não reconciliam")
        if invalid_trigram_df:
            raise WackyWackyError("invariante falhou: DF de trigrama excede TF")
        if (
            left_positions != content["metrics"]["bigram_positions"]
            or right_positions != content["metrics"]["bigram_positions"]
        ):
            raise WackyWackyError("invariante falhou: marginais de bigrama não reconciliam")
        result["content_reconciled"] = True
        result["content_histograms_reconciled"] = True
        result["trigram_document_frequency_valid"] = True
        result["bigram_marginals_reconciled"] = True
    atomic_json(root / "invariants.json", result)
    return result


def validate_v2_invariants(clean: dict, lexical: dict, content: dict, root) -> dict:
    """Validate only the products derived from B_clean_v2/D4."""
    target = root / "v2"
    documents = clean["d4_unique"]
    if not (clean["d3_representatives"] >= clean["b_clean_v2_nonempty"] >= clean["d4_unique"]):
        raise WackyWackyError("invariante falhou: funil B_clean_v2/D4 não é monotônico")
    view = lexical["views"]["B_clean_v2"]
    if view["documents"] != documents:
        raise WackyWackyError("invariante falhou: documentos lexicais v2 não reconciliam")
    for key, value in lexical["vocabulary"].items():
        _view, kind = key.split(":", 1)
        if kind in {"form", "lemma"} and value["occurrences"] != view["words"]:
            raise WackyWackyError(f"invariante falhou: palavras v2 não reconciliam em {key}")
    connection = duckdb.connect(str(target / "analysis.duckdb"), read_only=True)
    invalid_df = connection.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE document_frequency>total_frequency",
        [str(target / "vocabulary.parquet")],
    ).fetchone()[0]
    content_documents, content_words, sentences, paragraphs = connection.execute(
        "SELECT count(*),sum(words),sum(sentences),sum(paragraphs) FROM content_document_statistics"
    ).fetchone()
    invalid_ngrams = connection.execute(
        "SELECT "
        "(SELECT count(*) FROM read_parquet(?) WHERE document_frequency>total_frequency),"
        "(SELECT count(*) FROM read_parquet(?) WHERE document_frequency>total_frequency)",
        [str(target / "bigrams.parquet"), str(target / "content_trigrams.parquet")],
    ).fetchone()
    histogram_totals = dict(
        connection.execute(
            "SELECT metric,sum(count) FROM read_parquet(?) "
            "WHERE metric IN ('frases_por_documento','paragrafos_por_documento',"
            "'palavras_por_frase','palavras_por_paragrafo') GROUP BY metric",
            [str(target / "content_histograms.parquet")],
        ).fetchall()
    )
    left_positions, right_positions = connection.execute(
        "SELECT sum(left_count),sum(right_count) FROM read_parquet(?)",
        [str(target / "content_bigram_marginals.parquet")],
    ).fetchone()
    connection.close()
    if invalid_df or any(invalid_ngrams):
        raise WackyWackyError("invariante falhou: DF v2 excede TF")
    if content_documents != documents or content_words != view["words"]:
        raise WackyWackyError("invariante falhou: conteúdo v2 não reconcilia")
    if (
        histogram_totals.get("frases_por_documento", 0) != documents
        or histogram_totals.get("paragrafos_por_documento", 0) != documents
        or histogram_totals.get("palavras_por_frase", 0) != sentences
        or histogram_totals.get("palavras_por_paragrafo", 0) != paragraphs
    ):
        raise WackyWackyError("invariante falhou: histogramas v2 não reconciliam")
    if (
        left_positions != content["metrics"]["bigram_positions"]
        or right_positions != content["metrics"]["bigram_positions"]
    ):
        raise WackyWackyError("invariante falhou: marginais v2 não reconciliam")
    result = {
        "funnel_v2_monotonic": True,
        "vocabulary_v2_reconciled": True,
        "content_v2_reconciled": True,
        "document_frequency_v2_valid": True,
        "content_histograms_v2_reconciled": True,
        "bigram_marginals_v2_reconciled": True,
    }
    atomic_json(target / "invariants.json", result)
    return result
