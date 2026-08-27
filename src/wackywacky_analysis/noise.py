from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from itertools import pairwise

from .config import BoilerplateV2

MEDIAWIKI_RULESET_VERSION = 2

# Regras sempre ancoradas na linha completa. Elas não são aplicadas como
# substrings à prosa e só são usadas em domínios Wikimedia na limpeza v2.
MEDIAWIKI_LINE_LITERALS = frozenset(
    {
        "adicionar tópico",
        "aparência ocultar",
        "barra lateral",
        "barra lateral ocultar",
        "criar conta",
        "discussão ler editar ver histórico",
        "doar criar conta iniciar sessão",
        "discussão contribuições",
        "editar",
        "editar código",
        "editar código fonte",
        "ferramentas aparência",
        "ferramentas aparência ocultar",
        "ferramentas mover",
        "ferramentas pessoais",
        "histórico ferramentas aparência ocultar",
        "ligações externas",
        "menu principal",
        "menu principal mover",
        "ocultar",
        "página foi editada",
        "predefinição predefinição predefinição",
        "principal mover",
    }
)

MEDIAWIKI_LINE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"esta página foi editada pela última vez(?:\s+.*)?",
        r"página foi editada(?:\s+.*)?",
        r"editada pela última(?:\s+.*)?",
        r"(?:special\s+)?centralautologin(?:\s+.*)?",
        r"wiki\s+special\s+centralautologin(?:\s+.*)?",
        r"centralautologin(?:\s+(?:none|position|relative|absolute|fixed|static|\d+))*",
        r"(?:rlq\s+)?window\s+predefinição(?:\s+.*)?",
        r"(?:mw\.)?(?:loader|config)\.(?:load|implement|set)\s*\(.*",
        r"\.mw-parser-output(?:\s+.*)?",
        r"@media(?:\s+.*)?",
        r"body\.skin-(?:vector|minerva|timeless|monobook)(?:\s+.*)?",
    )
)

PRESENTATION_SINGLETONS = frozenset({"editar", "ferramentas", "ocultar", "predefinição"})

PRESENTATION_PHRASES = frozenset(
    {
        "aparência ocultar",
        "barra lateral",
        "barra lateral ocultar",
        "centralautologin none",
        "centralautologin none position",
        "conta iniciar sessão",
        "criar conta",
        "discussão ler editar",
        "doar criar conta",
        "editar código",
        "editar código fonte",
        "editar ver histórico",
        "ferramentas aparência",
        "ferramentas aparência ocultar",
        "ferramentas mover",
        "ferramentas pessoais",
        "histórico ferramentas",
        "histórico ferramentas aparência",
        "lateral ocultar",
        "ler editar",
        "ocultar modo",
        "predefinição info",
        "predefinição predefinição",
        "predefinição predefinição predefinição",
        "rlq window predefinição",
        "special centralautologin",
        "special centralautologin none",
        "wiki special centralautologin",
        "window predefinição",
    }
)


def canonical_line(value: str) -> str:
    return " ".join(value.casefold().split())


def mediawiki_rule_id(value: str) -> str | None:
    canonical = canonical_line(value)
    if canonical in MEDIAWIKI_LINE_LITERALS:
        return f"literal:{canonical}"
    for index, pattern in enumerate(MEDIAWIKI_LINE_PATTERNS):
        if pattern.fullmatch(canonical):
            return f"pattern:{index}"
    return None


def is_presentation_noise(value: str) -> bool:
    canonical = canonical_line(value)
    return (
        canonical in MEDIAWIKI_LINE_LITERALS
        or canonical in PRESENTATION_SINGLETONS
        or canonical in PRESENTATION_PHRASES
    )


@dataclass(frozen=True)
class ResidualUnit:
    start: int
    end: int
    kind: str
    digest: str
    canonical: str
    rule_id: str | None = None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def residual_units(
    text: str,
    config: BoilerplateV2,
    *,
    is_wikimedia: bool,
) -> list[ResidualUnit]:
    """Return full-line residual units; never match a substring of prose."""
    lines: list[tuple[int, int, int, str]] = []
    cursor = 0
    for physical_index, line in enumerate(text.split("\n")):
        start = cursor
        end = start + len(line)
        cursor = end + 1
        if line.strip():
            lines.append((physical_index, start, end, line.strip()))
    if not lines:
        return []
    edge = max(config.edge_min_lines, math.ceil(config.edge_fraction * len(lines)))
    if len(lines) > 2:
        edge = min(edge, max(1, (len(lines) - 1) // 2))
    edge_indexes = set(range(min(edge, len(lines))))
    edge_indexes.update(range(max(0, len(lines) - edge), len(lines)))
    output: list[ResidualUnit] = []
    mediawiki_intervals: list[tuple[int, int]] = []
    for _physical, start, end, line in lines:
        rule_id = mediawiki_rule_id(line) if is_wikimedia else None
        if rule_id:
            canonical = canonical_line(line)
            output.append(
                ResidualUnit(start, end, "mediawiki", _digest(canonical), canonical, rule_id)
            )
            mediawiki_intervals.append((start, end))
    for logical_index, (_physical, start, end, line) in enumerate(lines):
        if logical_index not in edge_indexes:
            continue
        if any(
            start < other_end and end > other_start
            for other_start, other_end in mediawiki_intervals
        ):
            continue
        canonical = canonical_line(line)
        alpha_tokens = sum(token.isalpha() for token in canonical.split())
        words = len(canonical.split())
        if (
            config.short_line_min_chars <= len(canonical) <= config.short_line_max_chars
            and config.short_line_min_alpha_tokens <= alpha_tokens
            and words <= config.short_line_max_words
        ):
            output.append(ResidualUnit(start, end, "short_line", _digest(canonical), canonical))
    for logical_index, (left, right) in enumerate(pairwise(lines)):
        if logical_index not in edge_indexes or logical_index + 1 not in edge_indexes:
            continue
        if right[0] != left[0] + 1:
            continue
        start, end = left[1], right[2]
        if any(
            start < other_end and end > other_start
            for other_start, other_end in mediawiki_intervals
        ):
            continue
        canonical = canonical_line(left[3] + "\n" + right[3])
        if config.short_pair_min_chars <= len(canonical) <= config.short_pair_max_chars:
            output.append(ResidualUnit(start, end, "short_pair", _digest(canonical), canonical))
    return output
