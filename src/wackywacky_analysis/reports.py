from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path

from .config import Config
from .io import atomic_json, sha256_file
from .storage import duckdb_connection


def _latex(value: object) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("_", "\\_"),
        ("#", "\\#"),
    ):
        text = text.replace(source, replacement)
    return text


def write_table(directory: Path, name: str, columns: list[str], rows: Iterable[Iterable]) -> None:
    materialized = [list(row) for row in rows]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(materialized)
    alignment = "l" + "r" * max(0, len(columns) - 1)
    with (directory / f"{name}.tex").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\\begin{{tabular}}{{{alignment}}}\n\\hline\n")
        handle.write(" & ".join(_latex(item) for item in columns) + " \\\\\n\\hline\n")
        for row in materialized:
            handle.write(" & ".join(_latex(item) for item in row) + " \\\\\n")
        handle.write("\\hline\n\\end{tabular}\n")


def _percent(value: float, total: float) -> str:
    if not total:
        return "0,00%"
    percentage = 100 * value / total
    return "<0,01%" if 0 < percentage < 0.01 else f"{percentage:.2f}%".replace(".", ",")


def build_reports(
    config: Config,
    manifest: dict,
    domain_summary: dict,
    exact: dict,
    clean: dict,
    lexical: dict,
    content: dict,
    root: Path,
) -> Path:
    result = config.paths.results / manifest["snapshot_id"]
    tables = result / "tables"
    figures_data = result / "aggregates"
    result.mkdir(parents=True, exist_ok=True)
    review_gate = json.loads((root / "review_gate.json").read_text(encoding="utf-8"))
    invariants = json.loads((root / "invariants.json").read_text(encoding="utf-8"))
    sampled = bool(manifest.get("sampling", {}).get("enabled"))
    public_manifest = {
        **manifest,
        "sources": {
            name: {key: value for key, value in source.items() if key != "mtime_ns"}
            for name, source in manifest["sources"].items()
        },
    }
    atomic_json(result / "manifest.json", public_manifest)
    methods = {
        "schema_version": 1,
        "scope": (
            "prévia amostral não representativa" if sampled else "snapshot integral configurado"
        ),
        "views": {
            "R_valid": "done com texto hexadecimal/Zstandard/UTF-8 válido, normalizado e não vazio",
            "E_exact": "R_valid deduplicado pelo SHA-256 do texto normalizado",
            "B_clean": "E_exact sem repetição intradomínio aprovada e deduplicado novamente",
            "N_near": "sensibilidade opcional; não altera B_clean",
        },
        "normalization": [
            "CRLF e CR para LF",
            "Unicode NFC",
            "controles removidos exceto tab e newline",
            "espaços horizontais colapsados",
            "trim por linha e até duas linhas vazias consecutivas",
        ],
        "config_sha256": config.fingerprint,
        "cutoff_date": config.cutoff_date,
        "profile": config.profile,
        "parameters": {
            "runtime": config.public_dict()["runtime"],
            "source": config.public_dict()["source"],
            "boilerplate": config.public_dict()["boilerplate"],
            "lexical": config.public_dict()["lexical"],
            "content": config.public_dict()["content"],
            "near_duplicates": config.public_dict()["near_duplicates"],
            "sampling": config.public_dict()["sampling"],
        },
        "spacy": lexical["spacy"],
        "content_spacy": content.get("spacy"),
        "content_config_sha256": config.content_fingerprint,
        "review_gate": review_gate,
        "invariants": invariants,
    }
    atomic_json(result / "methods.json", methods)
    summary = {
        "snapshot_id": manifest["snapshot_id"],
        "scope": methods["scope"],
        "domains": domain_summary,
        "exact": exact,
        "clean": clean,
        "lexical": lexical,
        "content": content,
        "review_gate": review_gate,
        "invariants": invariants,
    }
    near_path = root / "near_summary.json"
    if near_path.exists():
        summary["near_duplicates"] = json.loads(near_path.read_text())
    atomic_json(result / "summary.json", summary)

    write_table(
        tables,
        "01_snapshot",
        ["fonte", "bytes", "sha256", "colunas", "header", "data_corte", "escopo"],
        [
            (
                source["name"],
                source["size"],
                source["sha256"],
                source["columns"],
                source["header"],
                manifest["cutoff_date"],
                methods["scope"],
            )
            for source in manifest["sources"].values()
        ],
    )
    page_metrics = exact["page_metrics"]
    status = page_metrics["status"]
    total = sum(status.values())
    status_rows = [
        (key, value, _percent(value, total), total)
        for key, value in sorted(status.items(), key=lambda x: -x[1])
    ]
    write_table(
        tables, "02_status_todos", ["status", "total", "percentual", "denominador"], status_rows
    )
    write_table(
        figures_data,
        "status",
        ["status", "total"],
        ((key, value) for key, value, _p, _d in status_rows),
    )
    nonterminal = {"pending", "processing", "in_progress", "em processamento", "pendente"}
    terminal = [
        (key, value)
        for key, value in status.items()
        if key.casefold() not in nonterminal and not key.startswith("<")
    ]
    terminal_total = sum(value for _key, value in terminal)
    write_table(
        tables,
        "03_status_explorados",
        ["status_terminal", "total", "percentual", "denominador"],
        (
            (key, value, _percent(value, terminal_total), terminal_total)
            for key, value in sorted(terminal, key=lambda x: -x[1])
        ),
    )
    funnel = [
        ("done", page_metrics["done"]),
        ("texto presente", page_metrics["text_present"]),
        ("decodificado", page_metrics["decoded"]),
        ("R_valid", exact["r_valid"]),
        ("D1", exact["d1_unique"]),
        ("D2/E_exact", exact["d2_unique"]),
        ("B_clean não vazio", clean["b_clean_nonempty"]),
        ("D3/B_clean único", clean["d3_unique"]),
    ]
    write_table(tables, "04_funil", ["etapa", "documentos"], funnel)
    write_table(figures_data, "funil", ["etapa", "documentos"], funnel)
    view_rows = []
    for view, value in lexical["views"].items():
        quantiles = value["word_quantiles"]
        view_rows.append(
            (
                view,
                value["documents"],
                value["characters"],
                value["words"],
                *quantiles,
            )
        )
    write_table(
        tables,
        "05_resumo_corpus",
        ["visao", "documentos", "caracteres", "palavras", "p5", "p25", "p50", "p75", "p95", "p99"],
        view_rows,
    )
    write_table(
        tables,
        "06_duplicacao_limpeza",
        ["metrica", "valor"],
        [
            ("same_as declarado", exact["same_as_declared"]),
            ("same_as alvo ausente", exact["same_as_target_missing"]),
            ("same_as concordante D1", exact["same_as_d1_agreement"]),
            ("same_as concordante D2", exact["same_as_d2_agreement"]),
            ("grupos duplicados D1", exact["d1_duplicate_groups"]),
            ("maior grupo D1", exact["d1_largest_group"]),
            ("grupos D1 cross-domain", exact["d1_cross_domain_groups"]),
            ("grupos duplicados D2", exact["d2_duplicate_groups"]),
            ("maior grupo D2", exact["d2_largest_group"]),
            ("grupos D2 cross-domain", exact["d2_cross_domain_groups"]),
            ("candidatos intradomínio", exact["boilerplate_candidates"]),
            ("fragmentos cross-domain medidos", exact["cross_domain_repeated_units"]),
            ("documentos afetados", clean["documents_affected"]),
            ("documentos esvaziados", clean["documents_emptied"]),
            ("caracteres removidos", clean["characters_removed"]),
            ("parágrafos removidos", clean["paragraph_matches_removed"]),
            ("blocos removidos", clean["block_matches_removed"]),
            ("caracteres cobertos por parágrafos", clean["paragraph_characters_covered"]),
            ("caracteres cobertos por blocos", clean["block_characters_covered"]),
            ("grupos duplicados D3", clean["d3_duplicate_groups"]),
        ],
    )

    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    _domain_and_level_tables(connection, tables, figures_data, sampled=sampled)
    _distribution_tables(connection, root, figures_data)
    vocab_rows = [
        (
            key.split(":")[0],
            key.split(":")[1],
            value["types"],
            value["hapax"],
            value["occurrences"],
            value["document_occurrences"],
            value["stopword_types"],
            value["stopword_occurrences"],
        )
        for key, value in lexical["vocabulary"].items()
    ]
    write_table(
        tables,
        "11_vocabulario",
        [
            "visao",
            "tipo",
            "vocabulario",
            "hapax",
            "frequencia_total",
            "frequencia_documental_somada",
            "tipos_stopword",
            "ocorrencias_stopword",
        ],
        vocab_rows,
    )
    frequency_of_frequencies = connection.execute(
        """
        SELECT view, kind, total_frequency, count(*) AS types
        FROM read_parquet(?)
        GROUP BY view, kind, total_frequency
        ORDER BY view, kind, total_frequency
        """,
        [str(root / "vocabulary.parquet")],
    ).fetchall()
    write_table(
        tables,
        "11b_frequencia_de_frequencias",
        ["visao", "tipo", "frequencia", "tipos"],
        frequency_of_frequencies,
    )
    top_terms = connection.execute(
        """
        SELECT kind, term, total_frequency, document_frequency
        FROM read_parquet(?)
        WHERE view='B_clean' AND NOT is_stop
        QUALIFY row_number() OVER (PARTITION BY kind ORDER BY total_frequency DESC, term) <= ?
        ORDER BY kind, total_frequency DESC, term
        """,
        [str(root / "vocabulary.parquet"), config.lexical.published_items],
    ).fetchall()
    top_bigrams = connection.execute(
        """
        SELECT replace(bigram, chr(9), ' ') AS bigram, total_frequency, document_frequency
        FROM read_parquet(?) WHERE NOT contains_stopword
        ORDER BY total_frequency DESC, bigram LIMIT ?
        """,
        [str(root / "bigrams.parquet"), config.lexical.published_items],
    ).fetchall()
    write_table(
        tables,
        "12a_principais_formas_lemas",
        ["tipo", "item", "frequencia", "documentos"],
        top_terms,
    )
    write_table(
        tables, "12b_principais_bigramas", ["bigrama", "frequencia", "documentos"], top_bigrams
    )
    figure_terms = [row for row in top_terms if row[0] == "form"][: config.lexical.figure_items]
    write_table(
        figures_data,
        "principais_termos",
        ["tipo", "item", "frequencia"],
        (row[:3] for row in figure_terms),
    )
    write_table(
        figures_data,
        "principais_bigramas",
        ["item", "frequencia"],
        (row[:2] for row in top_bigrams[: config.lexical.figure_items]),
    )
    if content.get("status") == "complete":
        _content_tables(
            connection,
            config,
            content,
            root,
            tables,
            figures_data,
            sampled=sampled,
        )
    connection.close()
    return result


def _weighted_histogram_summary(connection, root: Path) -> list[tuple]:
    return connection.execute(
        """
        WITH base AS (
          SELECT metric, value, count,
                 sum(count) OVER(PARTITION BY metric ORDER BY value) cumulative,
                 sum(count) OVER(PARTITION BY metric) total
          FROM read_parquet(?)
        )
        SELECT metric, max(total) units, sum(value*count)/max(total) mean,
               min(value) FILTER (WHERE cumulative >= total*0.05) p5,
               min(value) FILTER (WHERE cumulative >= total*0.25) p25,
               min(value) FILTER (WHERE cumulative >= total*0.50) p50,
               min(value) FILTER (WHERE cumulative >= total*0.75) p75,
               min(value) FILTER (WHERE cumulative >= total*0.95) p95,
               min(value) FILTER (WHERE cumulative >= total*0.99) p99
        FROM base GROUP BY metric ORDER BY metric
        """,
        [str(root / "content_histograms.parquet")],
    ).fetchall()


def _metric_quantiles(connection, expression: str, *, where: str = "true") -> tuple:
    return connection.execute(
        f"""
        SELECT count({expression}), avg({expression}),
               quantile_cont({expression}, [0.05,0.25,0.5,0.75,0.95,0.99])
        FROM content_document_statistics WHERE {where}
        """
    ).fetchone()


def _vocabulary_curves(connection, config: Config, root: Path) -> list[tuple]:
    documents = connection.execute("SELECT count(*) FROM content_document_statistics").fetchone()[0]
    if not documents:
        return []
    checkpoints: set[int] = {documents} if documents else set()
    for percentage in range(1, 101):
        checkpoints.add(max(1, math.ceil(documents * percentage / 100)))
    scale = 1
    while scale <= documents:
        checkpoints.update(value for value in (scale, 2 * scale, 5 * scale) if value <= documents)
        scale *= 10
    ordered_checkpoints = sorted(checkpoints)
    total_types = dict(
        connection.execute(
            "SELECT kind, count(*) FROM read_parquet(?) GROUP BY kind",
            [str(root / "content_vocabulary_first.parquet")],
        ).fetchall()
    )
    cursor = connection.execute(
        """
        WITH ranked_documents AS (
          SELECT priority, row_number() OVER(ORDER BY priority, row_number) document_rank
          FROM content_document_statistics
        ), first_ranks AS (
          SELECT v.kind, d.document_rank, count(*) new_types
          FROM read_parquet(?) v JOIN ranked_documents d ON v.first_priority=d.priority
          GROUP BY v.kind, d.document_rank
        )
        SELECT kind, document_rank, new_types FROM first_ranks ORDER BY kind, document_rank
        """,
        [str(root / "content_vocabulary_first.parquet")],
    )
    curve: list[tuple] = []
    current_kind: str | None = None
    cumulative = checkpoint_index = 0

    def finish_kind(kind: str | None) -> None:
        nonlocal checkpoint_index
        if kind is None:
            return
        while checkpoint_index < len(ordered_checkpoints):
            checkpoint = ordered_checkpoints[checkpoint_index]
            curve.append(
                (
                    "documentos_aleatorios",
                    kind,
                    checkpoint,
                    cumulative,
                    total_types[kind],
                    cumulative / total_types[kind] if total_types[kind] else 0,
                )
            )
            checkpoint_index += 1

    while batch := cursor.fetchmany(100_000):
        for kind, rank, new_types in batch:
            if kind != current_kind:
                finish_kind(current_kind)
                current_kind = kind
                cumulative = 0
                checkpoint_index = 0
            while (
                checkpoint_index < len(ordered_checkpoints)
                and ordered_checkpoints[checkpoint_index] < rank
            ):
                checkpoint = ordered_checkpoints[checkpoint_index]
                curve.append(
                    (
                        "documentos_aleatorios",
                        kind,
                        checkpoint,
                        cumulative,
                        total_types[kind],
                        cumulative / total_types[kind] if total_types[kind] else 0,
                    )
                )
                checkpoint_index += 1
            cumulative += new_types
    finish_kind(current_kind)
    if any(row[3] != row[4] for row in curve if row[2] == documents):
        raise ValueError("curva de acumulação não termina no vocabulário exato")
    for kind in ("form", "lemma"):
        frequencies = connection.execute(
            """
            SELECT total_frequency FROM read_parquet(?)
            WHERE view='B_clean' AND kind=? ORDER BY total_frequency DESC, term
            """,
            [str(root / "vocabulary.parquet"), kind],
        ).fetchall()
        total = sum(row[0] for row in frequencies)
        running = 0
        requested = {10, 100, 1_000, 10_000, 50_000}
        for rank, (frequency,) in enumerate(frequencies, 1):
            running += frequency
            if rank in requested or rank == len(frequencies):
                curve.append(
                    (
                        "top_termos",
                        kind,
                        rank,
                        running,
                        total,
                        running / total if total else 0,
                    )
                )
    return curve


def _content_tables(
    connection,
    config: Config,
    content: dict,
    root: Path,
    tables: Path,
    aggregates: Path,
    *,
    sampled: bool,
) -> None:
    histogram_summary = _weighted_histogram_summary(connection, root)
    write_table(
        tables,
        "14_estrutura_sentencas_paragrafos",
        ["metrica", "unidades", "media", "p5", "p25", "p50", "p75", "p95", "p99"],
        histogram_summary,
    )
    write_table(
        aggregates,
        "estrutura_textual",
        ["metrica", "valor", "unidades"],
        connection.execute(
            "SELECT metric, value, count FROM read_parquet(?) ORDER BY metric, value",
            [str(root / "content_histograms.parquet")],
        ).fetchall(),
    )

    diversity_rows = []
    for label, expression, where in (
        ("formas_unicas", "unique_forms", "true"),
        ("lemas_unicos", "unique_lemmas", "true"),
        ("hapax_local_por_palavra", "local_hapax/nullif(words,0)", "words>0"),
        ("TTR", "ttr", "words>0"),
        ("MATTR", "mattr", "mattr IS NOT NULL"),
    ):
        count, mean, quantiles = _metric_quantiles(connection, expression, where=where)
        diversity_rows.append((label, count, mean, *(quantiles or [None] * 6)))
    size_rows = connection.execute(
        """
        SELECT CASE WHEN words<100 THEN '0-99' WHEN words<500 THEN '100-499'
                    WHEN words<2000 THEN '500-1999' ELSE '2000+' END faixa_palavras,
               count(*), avg(ttr), quantile_cont(ttr,0.5)
        FROM content_document_statistics GROUP BY 1
        ORDER BY min(words)
        """
    ).fetchall()
    write_table(
        tables,
        "15_diversidade_lexical_documental",
        ["metrica", "documentos", "media", "p5", "p25", "p50", "p75", "p95", "p99"],
        diversity_rows,
    )
    write_table(
        tables,
        "15b_ttr_por_tamanho",
        ["faixa_palavras", "documentos", "TTR_media", "TTR_mediana"],
        size_rows,
    )
    diversity_distribution = connection.execute(
        """
        SELECT metric, metric_value, count(*) documents FROM (
          SELECT 'TTR' metric, round(ttr,4) AS metric_value
          FROM content_document_statistics WHERE words>0
          UNION ALL
          SELECT 'MATTR', round(mattr,4) FROM content_document_statistics WHERE mattr IS NOT NULL
        ) GROUP BY metric, metric_value ORDER BY metric, metric_value
        """
    ).fetchall()
    write_table(
        aggregates,
        "diversidade_lexical",
        ["metrica", "valor", "documentos"],
        diversity_distribution,
    )

    grammar_rows = connection.execute(
        """
        SELECT kind, feature, value, count,
               count/sum(count) OVER(PARTITION BY kind, feature) AS participation
        FROM read_parquet(?) ORDER BY kind, feature, count DESC, value
        """,
        [str(root / "content_grammar.parquet")],
    ).fetchall()
    lexical_density = _metric_quantiles(connection, "lexical_density", where="words>0")
    grammar_rows.append(("derived", "densidade_lexical", "media", lexical_density[1], None))
    write_table(
        tables,
        "16_pos_morfologia",
        ["tipo", "feature", "valor", "total", "participacao"],
        grammar_rows,
    )
    write_table(
        aggregates,
        "classes_gramaticais",
        ["classe", "total", "participacao"],
        ((row[2], row[3], row[4]) for row in grammar_rows if row[0] == "pos"),
    )

    repetition_rows = []
    for label, expression, affected_threshold in (
        ("fracao_palavras_frases_repetidas", "repeated_sentence_fraction", 0),
        ("fracao_palavras_paragrafos_repetidos", "repeated_paragraph_fraction", 0),
        ("maior_sequencia_frases_identicas", "max_sentence_run", 1),
        ("maior_sequencia_paragrafos_identicos", "max_paragraph_run", 1),
    ):
        count, mean, quantiles = _metric_quantiles(connection, expression)
        affected = connection.execute(
            f"SELECT count(*) FROM content_document_statistics WHERE {expression}>?",
            [affected_threshold],
        ).fetchone()[0]
        repetition_rows.append((label, count, affected, mean, *(quantiles or [None] * 6)))
    write_table(
        tables,
        "17_repeticao_interna",
        [
            "metrica",
            "documentos",
            "documentos_afetados",
            "media",
            "p5",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
        ],
        repetition_rows,
    )
    repetition_distribution = connection.execute(
        """
        SELECT metric, metric_value, count(*) documents FROM (
          SELECT 'frases' metric, round(repeated_sentence_fraction,4) AS metric_value
          FROM content_document_statistics
          UNION ALL
          SELECT 'paragrafos', round(repeated_paragraph_fraction,4)
          FROM content_document_statistics
        ) GROUP BY metric, metric_value ORDER BY metric, metric_value
        """
    ).fetchall()
    write_table(
        aggregates,
        "repeticao_interna",
        ["unidade", "fracao", "documentos"],
        repetition_distribution,
    )

    signal_columns = (
        ("frases_fragmentadas", "fragment_sentences"),
        ("frases_longas", "long_sentences"),
        ("tokens_longos", "long_tokens"),
        ("URLs_no_texto", "urls"),
        ("emails_no_texto", "emails"),
        ("sequencias_pontuacao", "punctuation_runs"),
        ("marcadores_mojibake", "mojibake_markers"),
        ("alta_fracao_numerica", "high_numeric::INTEGER"),
        ("alta_fracao_nao_lexical", "high_nonlexical::INTEGER"),
        ("alta_fracao_caixa_alta", "high_uppercase::INTEGER"),
    )
    documents = content["metrics"]["documents"]
    signal_rows = []
    for label, expression in signal_columns:
        occurrences, affected = connection.execute(
            f"SELECT sum({expression}), count(*) FILTER (WHERE {expression}>0) FROM content_document_statistics"
        ).fetchone()
        signal_rows.append(
            (label, occurrences or 0, affected, affected / documents if documents else 0, documents)
        )
    write_table(
        tables,
        "18_sinais_textuais",
        ["indicador", "ocorrencias", "documentos", "participacao_documentos", "denominador"],
        signal_rows,
    )
    write_table(
        aggregates,
        "sinais_textuais",
        ["indicador", "documentos", "participacao"],
        ((row[0], row[2], row[3]) for row in signal_rows),
    )

    bigram_floor = max(
        config.content.collocation_min_frequency,
        int(json.loads((root / "bigram_candidates.json").read_text())["omitted_upper_bound"]) + 1,
    )
    bigram_rows = connection.execute(
        """
        WITH scored AS (
          SELECT replace(b.bigram, chr(9), ' ') item, b.total_frequency, b.document_frequency,
                 ln((b.total_frequency::DOUBLE*?)/
                    (l.left_count::DOUBLE*r.right_count::DOUBLE)) pmi,
                 ln((b.total_frequency::DOUBLE*?)/
                    (l.left_count::DOUBLE*r.right_count::DOUBLE)) /
                    -ln(b.total_frequency::DOUBLE/?) npmi
          FROM read_parquet(?) b
          JOIN read_parquet(?) l ON l.term=split_part(b.bigram,chr(9),1)
          JOIN read_parquet(?) r ON r.term=split_part(b.bigram,chr(9),2)
          WHERE NOT b.contains_stopword AND b.total_frequency>=? AND b.document_frequency>=?
        )
        SELECT * FROM scored ORDER BY npmi DESC, total_frequency DESC, item LIMIT ?
        """,
        [
            content["metrics"]["bigram_positions"],
            content["metrics"]["bigram_positions"],
            content["metrics"]["bigram_positions"],
            str(root / "bigrams.parquet"),
            str(root / "content_bigram_marginals.parquet"),
            str(root / "content_bigram_marginals.parquet"),
            bigram_floor,
            config.content.collocation_min_documents,
            config.lexical.published_items,
        ],
    ).fetchall()
    trigram_floor = content["trigrams"]["eligible_frequency_floor"]
    trigram_rows = connection.execute(
        """
        SELECT replace(trigram, chr(9), ' ') item, total_frequency, document_frequency
        FROM read_parquet(?)
        WHERE NOT contains_stopword_at_edges AND total_frequency>=? AND document_frequency>=?
        ORDER BY total_frequency DESC, item LIMIT ?
        """,
        [
            str(root / "content_trigrams.parquet"),
            trigram_floor,
            config.content.collocation_min_documents,
            config.lexical.published_items,
        ],
    ).fetchall()
    collocation_rows = [
        ("bigrama_NPMI", *row) for row in bigram_rows
    ] + [("trigrama_frequente", *row, None, None) for row in trigram_rows]
    write_table(
        tables,
        "19_colocacoes",
        ["ranking", "item", "frequencia", "documentos", "PMI", "NPMI"],
        collocation_rows,
    )
    write_table(
        aggregates,
        "colocacoes",
        ["ranking", "item", "frequencia", "NPMI"],
        (
            (row[0], row[1], row[2], row[5])
            for row in (
                [("bigrama_NPMI", *item) for item in bigram_rows[: config.lexical.figure_items]]
                + [
                    ("trigrama_frequente", *item, None, None)
                    for item in trigram_rows[: config.lexical.figure_items]
                ]
            )
        ),
    )

    domains = connection.execute(
        """
        SELECT d.host, count(*) documents, sum(c.words) words,
               quantile_cont(c.words,0.5) median_words,
               quantile_cont(c.words/nullif(c.sentences,0),0.5) median_words_sentence,
               quantile_cont(c.paragraphs,0.5) median_paragraphs,
               quantile_cont(c.mattr,0.5) median_mattr,
               quantile_cont(c.lexical_density,0.5) median_lexical_density,
               avg(c.repeated_paragraph_fraction) mean_paragraph_repetition,
               avg(c.any_signal::INTEGER) signal_prevalence
        FROM content_document_statistics c JOIN canonical_domains d ON c.domain_id=d.id
        GROUP BY d.id, d.host HAVING count(*)>=?
        ORDER BY words DESC, d.host LIMIT ?
        """,
        [config.content.domain_min_documents, config.content.domain_limit],
    ).fetchall()
    domain_columns = [
        "host",
        "documentos",
        "palavras",
        "mediana_palavras_documento",
        "mediana_palavras_frase",
        "mediana_paragrafos",
        "mediana_MATTR",
        "mediana_densidade_lexical",
        "media_repeticao_paragrafos",
        "prevalencia_sinais",
        "escopo",
    ]
    domain_rows = [
        (*row, "prévia amostral não representativa" if sampled else "snapshot integral")
        for row in domains
    ]
    write_table(tables, "20_perfil_textual_dominios", domain_columns, domain_rows)
    write_table(
        aggregates,
        "perfil_textual_dominios",
        domain_columns[:-1],
        (row[:-1] for row in domain_rows[:20]),
    )

    vocabulary_curve = _vocabulary_curves(connection, config, root)
    write_table(
        tables,
        "21_cobertura_vocabulario",
        ["analise", "tipo", "ponto", "observado", "total", "fracao"],
        vocabulary_curve,
    )
    write_table(
        aggregates,
        "cobertura_vocabulario",
        ["analise", "tipo", "ponto", "observado", "total", "fracao"],
        vocabulary_curve,
    )


def _domain_and_level_tables(connection, tables: Path, aggregates: Path, *, sampled: bool) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW canonical_domains AS
        SELECT * EXCLUDE(rn) FROM (
          SELECT *, row_number() OVER (PARTITION BY id ORDER BY row_number) rn FROM domains
        ) WHERE rn=1
        """
    )
    levels = connection.execute(
        """
        WITH d AS (SELECT recursion_level AS depth, count(*) domains, sum(request_count) requests
                   FROM canonical_domains GROUP BY 1),
        p AS (SELECT recursion_level AS depth, count(*) pages,
                     count(*) FILTER (WHERE status='done') done
              FROM page_inventory GROUP BY 1),
        v AS (SELECT recursion_level AS depth, count(*) valid_documents FROM page_features GROUP BY 1),
        u AS (SELECT recursion_level AS depth, count(*) unique_documents FROM d2_membership
              WHERE is_representative GROUP BY 1),
        w AS (SELECT recursion_level AS depth, sum(words) words FROM document_statistics
              WHERE view='B_clean' GROUP BY 1)
        SELECT coalesce(d.depth,p.depth,v.depth,u.depth,w.depth) AS depth,
               coalesce(domains,0), coalesce(requests,0), coalesce(pages,0), coalesce(done,0),
               coalesce(valid_documents,0), coalesce(unique_documents,0), coalesce(words,0),
               CASE WHEN requests>0 THEN 1000.0*words/requests END words_per_1000_requests
        FROM d FULL OUTER JOIN p USING(depth) FULL OUTER JOIN v USING(depth)
        FULL OUTER JOIN u USING(depth) FULL OUTER JOIN w USING(depth)
        ORDER BY depth
        """
    ).fetchall()
    columns = [
        "nivel",
        "dominios",
        "requisicoes",
        "paginas",
        "done",
        "validos",
        "unicos",
        "palavras",
        "palavras_por_mil_requisicoes",
    ]
    public_levels = [(*row[:-1], "não aplicável" if sampled else row[-1]) for row in levels]
    write_table(tables, "07_rendimento_nivel", columns, public_levels)
    write_table(
        aggregates,
        "rendimento_nivel",
        ["nivel", "palavras_por_mil_requisicoes", "aplicavel"],
        ((row[0], "" if sampled else row[-1] or 0, not sampled) for row in levels),
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW clean_domain_fractional AS
        WITH content AS (
          SELECT dm.clean_sha256, ds.words
          FROM d3_membership dm JOIN document_statistics ds USING(row_number)
          WHERE dm.is_representative AND ds.view='B_clean'
        ), memberships AS (
          SELECT DISTINCT dm.clean_sha256, e.domain_id
          FROM d3_membership dm JOIN d2_membership e USING(normalized_sha256)
          WHERE e.domain_id IS NOT NULL
        ), weights AS (
          SELECT *, count(*) OVER(PARTITION BY clean_sha256) domains FROM memberships
        )
        SELECT domain_id, sum(c.words / w.domains) words
        FROM content c JOIN weights w USING(clean_sha256) GROUP BY domain_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP VIEW domain_metrics AS
        WITH valid_pages AS (SELECT domain_id, count(*) valid_documents FROM page_features GROUP BY 1),
        clean_content AS (
          SELECT dm.clean_sha256, ds.words
          FROM d3_membership dm JOIN document_statistics ds USING(row_number)
          WHERE dm.is_representative AND ds.view='B_clean'
        ), clean_map AS (
          SELECT DISTINCT normalized_sha256, clean_sha256 FROM d3_membership
        ), clean_page AS (
          SELECT e.domain_id, sum(c.words) clean_words_page_weighted
          FROM d2_membership e JOIN clean_map m USING(normalized_sha256)
          JOIN clean_content c USING(clean_sha256) GROUP BY e.domain_id
        ),
        base AS (
          SELECT d.id, d.host, d.is_wikimedia, d.request_count,
                 coalesce(v.valid_documents,0) valid_documents,
                 coalesce(c.clean_words_page_weighted,0) clean_page,
                 coalesce(f.words,0) clean_fractional
          FROM canonical_domains d LEFT JOIN valid_pages v ON d.id=v.domain_id
          LEFT JOIN clean_page c ON d.id=c.domain_id LEFT JOIN clean_domain_fractional f ON d.id=f.domain_id
        )
        SELECT * FROM base
        """
    )
    domains = connection.execute(
        """
        WITH ranked AS (
          SELECT *, row_number() OVER(ORDER BY request_count DESC) rr,
                    row_number() OVER(ORDER BY valid_documents DESC) rv,
                    row_number() OVER(ORDER BY clean_fractional DESC) rw
          FROM domain_metrics
        )
        SELECT host, request_count, valid_documents, clean_page, clean_fractional, is_wikimedia
        FROM ranked WHERE least(rr,rv,rw)<=100 ORDER BY clean_fractional DESC, host
        """
    ).fetchall()
    write_table(
        tables,
        "08_principais_dominios",
        [
            "host",
            "requisicoes_origem" if sampled else "requisicoes",
            "textos_validos",
            "palavras_page_weighted",
            "palavras_unique_fractional",
            "wikimedia",
        ],
        domains,
    )
    non_wikimedia = connection.execute(
        """
        SELECT host, request_count, valid_documents, clean_page, clean_fractional,
               1.0*request_count/sum(request_count) OVER() request_share,
               1.0*valid_documents/nullif(sum(valid_documents) OVER(),0) valid_share,
               1.0*clean_fractional/nullif(sum(clean_fractional) OVER(),0) words_share,
               sum(request_count) OVER() request_denominator,
               sum(valid_documents) OVER() valid_denominator,
               sum(clean_fractional) OVER() words_denominator,
               'fora da lista explícita Wikimedia' AS group_name
        FROM domain_metrics WHERE NOT is_wikimedia
        ORDER BY clean_fractional DESC, host LIMIT 100
        """
    ).fetchall()
    write_table(
        tables,
        "09_principais_dominios_nao_wikimedia",
        [
            "host",
            "requisicoes_origem" if sampled else "requisicoes",
            "textos_validos",
            "palavras_page_weighted",
            "palavras_unique_fractional",
            "fracao_requisicoes_dominios_amostrados" if sampled else "fracao_requisicoes",
            "fracao_textos_validos",
            "fracao_palavras_unique",
            "denominador_requisicoes_dominios_amostrados" if sampled else "denominador_requisicoes",
            "denominador_textos_validos",
            "denominador_palavras_unique",
            "grupo",
        ],
        non_wikimedia,
    )
    values = [
        row[0]
        for row in connection.execute(
            "SELECT words FROM clean_domain_fractional WHERE words>0 ORDER BY words"
        ).fetchall()
    ]
    total = sum(values)
    count = len(values)
    gini = (
        (2 * sum((index + 1) * value for index, value in enumerate(values))) / (count * total)
        - (count + 1) / count
        if count and total
        else 0
    )
    descending = sorted(values, reverse=True)
    concentration = []
    for top in (1, 10, 100, 1000):
        concentration.append(
            (top, sum(descending[:top]), sum(descending[:top]) / total if total else 0, total)
        )
    concentration.append(("Gini", gini, "", total))
    write_table(
        tables,
        "10_concentracao_dominios",
        ["top_ou_metrica", "valor", "participacao", "total_palavras"],
        concentration,
    )
    cumulative = 0.0
    lorenz = [(0.0, 0.0)]
    stride = max(1, count // 1000)
    for index, value in enumerate(values, 1):
        cumulative += value
        if index % stride == 0 or index == count:
            lorenz.append((index / count, cumulative / total if total else 0))
    write_table(aggregates, "lorenz", ["fracao_dominios", "fracao_palavras"], lorenz)


def _distribution_tables(connection, root: Path, aggregates: Path) -> None:
    lengths = connection.execute(
        """
        SELECT view, words, count(*) documents FROM document_statistics
        WHERE view IN ('R_valid','B_clean') GROUP BY view, words ORDER BY view, words
        """
    ).fetchall()
    write_table(aggregates, "tamanho_documentos", ["visao", "palavras", "documentos"], lengths)
    removal = connection.execute(
        """
        SELECT round(removed_fraction, 4) fraction, count(*) documents
        FROM clean_features GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    write_table(aggregates, "fracao_removida", ["fracao", "documentos"], removal)
    clusters = connection.execute(
        """
        SELECT 'D2' AS stage, pages AS group_size, count(*) AS group_count
        FROM read_parquet(?) GROUP BY pages
        UNION ALL
        SELECT 'D3', documents, count(*) FROM read_parquet(?) GROUP BY documents
        ORDER BY stage, group_size
        """,
        [str(root / "d2_groups.parquet"), str(root / "d3_groups.parquet")],
    ).fetchall()
    write_table(aggregates, "tamanho_grupos", ["etapa", "tamanho", "grupos"], clusters)
    zipf = connection.execute(
        """
        WITH ranked AS (
          SELECT kind, total_frequency,
                 row_number() OVER(PARTITION BY kind ORDER BY total_frequency DESC, term) rank
          FROM read_parquet(?) WHERE view='B_clean'
        )
        SELECT kind, avg(rank) rank, avg(total_frequency) frequency
        FROM ranked GROUP BY kind, floor(100*log10(rank)) ORDER BY kind, rank
        """,
        [str(root / "vocabulary.parquet")],
    ).fetchall()
    write_table(aggregates, "zipf", ["tipo", "rank", "frequencia"], zipf)


def write_checksums(result: Path) -> None:
    paths = [
        path for path in result.rglob("*") if path.is_file() and path.name != "checksums.sha256"
    ]
    lines = [f"{sha256_file(path)}  {path.relative_to(result)}" for path in sorted(paths)]
    (result / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="ascii")
