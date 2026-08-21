from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

_RENDER_CACHE = Path(tempfile.gettempdir()) / "wackywacky-analysis-cache"
_RENDER_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_RENDER_CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RENDER_CACHE / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _save(fig, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sampling", {}).get("enabled"):
            fig.text(
                0.5,
                0.005,
                "PRÉVIA AMOSTRAL NÃO REPRESENTATIVA",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#666666",
            )
    fig.tight_layout()
    fig.savefig(directory / f"{name}.svg", metadata={"Date": None})
    fig.savefig(directory / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(directory / f"{name}.png", dpi=180, metadata={"Software": "wackywacky-analysis"})
    plt.close(fig)


def render_all(result: Path) -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "svg.hashsalt": "wackywacky-analysis-v1",
        }
    )
    data = result / "aggregates"
    figures = result / "figures"
    _status(data, figures)
    _funnel(data, figures)
    _length_ecdf(data, figures)
    _removal_ecdf(data, figures)
    _cluster_ccdf(data, figures)
    _lorenz(data, figures)
    _levels(data, figures)
    _zipf(data, figures)
    _top(data, figures)
    if (data / "estrutura_textual.csv").exists():
        _content_structure(data, figures)
        _content_diversity(data, figures)
        _content_grammar(data, figures)
        _content_repetition(data, figures)
        _content_signals(data, figures)
        _content_collocations(data, figures)
        _content_domains(data, figures)
        _content_vocabulary(data, figures)


def _status(data: Path, output: Path) -> None:
    rows = _read(data / "status.csv")
    rows.reverse()
    fig, ax = plt.subplots()
    ax.barh([row["status"] for row in rows], [int(row["total"]) for row in rows], color=BLUE)
    ax.set_xlabel("URLs")
    ax.set_title("Distribuição das URLs por status")
    _save(fig, output, "01_status")


def _funnel(data: Path, output: Path) -> None:
    rows = _read(data / "funil.csv")
    fig, ax = plt.subplots()
    ax.barh(
        [row["etapa"] for row in reversed(rows)],
        [int(row["documentos"]) for row in reversed(rows)],
        color=GREEN,
    )
    ax.set_xlabel("Documentos")
    ax.set_title("Funil de validação, limpeza e deduplicação")
    _save(fig, output, "02_funil")


def _length_ecdf(data: Path, output: Path) -> None:
    rows = _read(data / "tamanho_documentos.csv")
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        groups[row["visao"]].append((int(row["palavras"]), int(row["documentos"])))
    fig, ax = plt.subplots()
    for view, color in (("R_valid", BLUE), ("B_clean", ORANGE)):
        points = groups[view]
        total = sum(count for _words, count in points)
        cumulative = np.cumsum([count for _words, count in points]) / total if total else []
        ax.step(
            [max(1, words) for words, _count in points],
            cumulative,
            where="post",
            label=view,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Palavras por documento (log)")
    ax.set_ylabel("Fração acumulada")
    ax.legend()
    ax.set_title("Distribuição do tamanho dos documentos")
    _save(fig, output, "03_ecdf_palavras")


def _removal_ecdf(data: Path, output: Path) -> None:
    rows = _read(data / "fracao_removida.csv")
    counts = [int(row["documentos"]) for row in rows]
    total = sum(counts)
    fig, ax = plt.subplots()
    ax.step(
        [float(row["fracao"]) for row in rows],
        np.cumsum(counts) / total if total else [],
        where="post",
        color=ORANGE,
    )
    ax.set_xlabel("Fração de caracteres removida")
    ax.set_ylabel("Fração acumulada de documentos")
    ax.set_title("Impacto dos fragmentos repetidos intradomínio")
    _save(fig, output, "04_ecdf_remocao")


def _cluster_ccdf(data: Path, output: Path) -> None:
    rows = _read(data / "tamanho_grupos.csv")
    groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        groups[row["etapa"]].append((int(row["tamanho"]), int(row["grupos"])))
    fig, ax = plt.subplots()
    for stage, color in (("D2", BLUE), ("D3", GREEN)):
        points = sorted(groups[stage])
        remaining = np.cumsum([count for _size, count in reversed(points)])[::-1]
        ax.step(
            [size for size, _count in points], remaining, where="post", label=stage, color=color
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamanho do grupo")
    ax.set_ylabel("Grupos com tamanho ≥ x")
    ax.legend()
    ax.set_title("CCDF dos grupos de duplicatas")
    _save(fig, output, "05_ccdf_duplicatas")


def _lorenz(data: Path, output: Path) -> None:
    rows = _read(data / "lorenz.csv")
    fig, ax = plt.subplots()
    ax.plot(
        [float(row["fracao_dominios"]) for row in rows],
        [float(row["fracao_palavras"]) for row in rows],
        color=BLUE,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666")
    ax.set_xlabel("Fração acumulada de domínios")
    ax.set_ylabel("Fração acumulada de palavras")
    ax.set_title("Curva de Lorenz das palavras limpas por domínio")
    _save(fig, output, "06_lorenz_dominios")


def _levels(data: Path, output: Path) -> None:
    rows = _read(data / "rendimento_nivel.csv")
    fig, ax = plt.subplots()
    if rows and rows[0].get("aplicavel", "true").casefold() == "false":
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Indicador não calculado na amostra estratificada",
            ha="center",
            va="center",
        )
        ax.set_title("Rendimento por nível de recursão")
        _save(fig, output, "07_rendimento_nivel")
        return
    ax.bar(
        [row["nivel"] for row in rows],
        [float(row["palavras_por_mil_requisicoes"]) for row in rows],
        color=PURPLE,
    )
    ax.set_xlabel("Nível de recursão")
    ax.set_ylabel("Palavras limpas por mil requisições")
    ax.set_title("Rendimento por nível de recursão")
    _save(fig, output, "07_rendimento_nivel")


def _zipf(data: Path, output: Path) -> None:
    rows = _read(data / "zipf.csv")
    fig, ax = plt.subplots()
    for kind, label, color in (("form", "Formas", BLUE), ("lemma", "Lemas", ORANGE)):
        selected = [row for row in rows if row["tipo"] == kind]
        ax.plot(
            [float(row["rank"]) for row in selected],
            [float(row["frequencia"]) for row in selected],
            label=label,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Posição no vocabulário")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.set_title("Curva de Zipf em B_clean")
    _save(fig, output, "08_zipf")


def _top(data: Path, output: Path) -> None:
    words = _read(data / "principais_termos.csv")
    bigrams = _read(data / "principais_bigramas.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    words = words[-30:]
    bigrams = bigrams[-30:]
    axes[0].barh(
        [row["item"] for row in reversed(words)],
        [int(row["frequencia"]) for row in reversed(words)],
        color=BLUE,
    )
    axes[0].set_title("Palavras sem stopwords")
    axes[1].barh(
        [row["item"] for row in reversed(bigrams)],
        [int(row["frequencia"]) for row in reversed(bigrams)],
        color=GREEN,
    )
    axes[1].set_title("Bigramas mais frequentes")
    for ax in axes:
        ax.set_xlabel("Frequência")
    _save(fig, output, "09_principais_palavras_bigramas")


def _weighted_ecdf(ax, rows: list[dict[str, str]], label: str, color: str) -> None:
    values = [float(row["valor"]) for row in rows]
    counts = [int(row.get("unidades", row.get("documentos", "0"))) for row in rows]
    total = sum(counts)
    ax.step(values, np.cumsum(counts) / total if total else [], where="post", label=label, color=color)


def _content_structure(data: Path, output: Path) -> None:
    rows = _read(data / "estrutura_textual.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    selections = (
        ("palavras_por_frase", "Palavras por frase", BLUE),
        ("palavras_por_paragrafo", "Palavras por parágrafo", GREEN),
        ("frases_por_documento", "Frases por documento", ORANGE),
    )
    for ax, (metric, label, color) in zip(axes, selections, strict=True):
        selected = [row for row in rows if row["metrica"] == metric]
        _weighted_ecdf(ax, selected, label, color)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel(label)
        ax.set_ylabel("Fração acumulada")
    fig.suptitle("Estrutura dos textos em B_clean")
    _save(fig, output, "10_estrutura_textual")


def _content_diversity(data: Path, output: Path) -> None:
    rows = _read(data / "diversidade_lexical.csv")
    fig, ax = plt.subplots()
    for metric, color in (("TTR", BLUE), ("MATTR", ORANGE)):
        selected = [row for row in rows if row["metrica"] == metric]
        _weighted_ecdf(ax, selected, metric, color)
    ax.set_xlabel("Diversidade lexical")
    ax.set_ylabel("Fração acumulada de documentos")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title("TTR e MATTR por documento")
    _save(fig, output, "11_diversidade_lexical")


def _content_grammar(data: Path, output: Path) -> None:
    rows = sorted(
        _read(data / "classes_gramaticais.csv"),
        key=lambda row: float(row["participacao"]),
    )
    fig, ax = plt.subplots()
    ax.barh(
        [row["classe"] for row in rows],
        [float(row["participacao"]) for row in rows],
        color=BLUE,
    )
    ax.set_xlabel("Participação nas palavras")
    ax.set_title("Classes gramaticais estimadas pelo spaCy")
    _save(fig, output, "12_composicao_gramatical")


def _content_repetition(data: Path, output: Path) -> None:
    rows = _read(data / "repeticao_interna.csv")
    fig, ax = plt.subplots()
    for unit, label, color in (
        ("frases", "Frases", BLUE),
        ("paragrafos", "Parágrafos", ORANGE),
    ):
        selected = [row for row in rows if row["unidade"] == unit]
        normalized = [
            {"valor": row["fracao"], "documentos": row["documentos"]} for row in selected
        ]
        _weighted_ecdf(ax, normalized, label, color)
    ax.set_xlabel("Fração de palavras em unidades repetidas")
    ax.set_ylabel("Fração acumulada de documentos")
    ax.legend()
    ax.set_title("Repetição interna residual em B_clean")
    _save(fig, output, "13_repeticao_interna")


def _content_signals(data: Path, output: Path) -> None:
    rows = sorted(
        _read(data / "sinais_textuais.csv"), key=lambda row: float(row["participacao"])
    )
    labels = {
        "frases_fragmentadas": "frases fragmentadas",
        "frases_longas": "frases longas",
        "tokens_longos": "tokens longos",
        "URLs_no_texto": "URLs no texto",
        "emails_no_texto": "e-mails no texto",
        "sequencias_pontuacao": "sequências de pontuação",
        "marcadores_mojibake": "marcadores de mojibake",
        "alta_fracao_numerica": "alta fração numérica",
        "alta_fracao_nao_lexical": "alta fração não lexical",
        "alta_fracao_caixa_alta": "alta fração em caixa alta",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [labels.get(row["indicador"], row["indicador"].replace("_", " ")) for row in rows],
        [float(row["participacao"]) for row in rows],
        color=ORANGE,
    )
    ax.set_xlabel("Participação dos documentos")
    ax.set_title("Prevalência de sinais textuais heurísticos")
    _save(fig, output, "14_sinais_textuais")


def _content_collocations(data: Path, output: Path) -> None:
    rows = _read(data / "colocacoes.csv")
    bigrams = [row for row in rows if row["ranking"] == "bigrama_NPMI"][:20]
    trigrams = [row for row in rows if row["ranking"] == "trigrama_frequente"][:20]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].barh(
        [row["item"] for row in reversed(bigrams)],
        [float(row["NPMI"]) for row in reversed(bigrams)],
        color=BLUE,
    )
    axes[0].set_xlabel("NPMI")
    axes[0].set_title("Bigramas associados")
    axes[1].barh(
        [row["item"] for row in reversed(trigrams)],
        [int(row["frequencia"]) for row in reversed(trigrams)],
        color=GREEN,
    )
    axes[1].set_xlabel("Frequência")
    axes[1].set_title("Trigramas frequentes")
    _save(fig, output, "15_colocacoes")


def _content_domains(data: Path, output: Path) -> None:
    rows = _read(data / "perfil_textual_dominios.csv")[:15]
    rows.reverse()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    metrics = (
        ("mediana_palavras_frase", "Mediana de palavras por frase", BLUE),
        ("mediana_MATTR", "Mediana de MATTR", GREEN),
        ("mediana_densidade_lexical", "Mediana da densidade lexical", PURPLE),
        ("media_repeticao_paragrafos", "Repetição média de parágrafos", ORANGE),
    )
    for ax, (field, title, color) in zip(axes.flat, metrics, strict=True):
        ax.barh(
            [row["host"] for row in rows],
            [float(row[field] or 0) for row in rows],
            color=color,
        )
        ax.set_title(title)
    fig.suptitle("Perfil textual dos principais domínios")
    _save(fig, output, "16_perfil_textual_dominios")


def _content_vocabulary(data: Path, output: Path) -> None:
    rows = _read(data / "cobertura_vocabulario.csv")
    accumulation = [row for row in rows if row["analise"] == "documentos_aleatorios"]
    coverage = [row for row in rows if row["analise"] == "top_termos"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for kind, label, color in (("form", "Formas", BLUE), ("lemma", "Lemas", ORANGE)):
        selected = [row for row in accumulation if row["tipo"] == kind]
        axes[0].plot(
            [int(row["ponto"]) for row in selected],
            [int(row["observado"]) for row in selected],
            label=label,
            color=color,
        )
        selected = [row for row in coverage if row["tipo"] == kind]
        axes[1].plot(
            [int(row["ponto"]) for row in selected],
            [float(row["fracao"]) for row in selected],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Documentos em ordem aleatória determinística")
    axes[0].set_ylabel("Tipos observados")
    axes[0].set_title("Acumulação do vocabulário")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Top-K termos")
    axes[1].set_ylabel("Fração das ocorrências")
    axes[1].set_title("Cobertura pelas formas mais frequentes")
    for ax in axes:
        ax.legend()
    _save(fig, output, "17_cobertura_vocabulario")
