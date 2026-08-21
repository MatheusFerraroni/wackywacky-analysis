from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError


@dataclass(frozen=True)
class Paths:
    pages: Path
    domains: Path
    work: Path
    results: Path


@dataclass(frozen=True)
class Runtime:
    workers: int
    memory_limit: str
    minimum_scratch_free_bytes: int
    chunk_bytes: int
    max_line_bytes: int
    max_text_bytes: int
    max_rows: int
    queue_bytes: int


@dataclass(frozen=True)
class Source:
    header: str
    require_immutable: bool


@dataclass(frozen=True)
class Boilerplate:
    enabled: bool
    paragraph_min_chars: int
    block_lines: int
    block_min_chars: int
    review_paragraphs: int
    review_blocks: int
    review_seed: int
    precision_min: float
    wilson_lower_min: float


@dataclass(frozen=True)
class Lexical:
    spacy_model: str
    partitions: int
    spill_terms: int
    bigram_candidates: int
    published_items: int
    figure_items: int


@dataclass(frozen=True)
class Content:
    enabled: bool = True
    mattr_window: int = 100
    fragment_max_words: int = 2
    long_sentence_words: int = 100
    long_token_chars: int = 30
    punctuation_run: int = 4
    repetition_sentence_min_words: int = 3
    repetition_paragraph_min_chars: int = 20
    high_numeric_fraction: float = 0.30
    high_nonlexical_fraction: float = 0.30
    high_uppercase_fraction: float = 0.50
    fraction_min_tokens: int = 20
    domain_min_documents: int = 20
    domain_limit: int = 100
    collocation_min_frequency: int = 20
    collocation_min_documents: int = 5
    trigram_candidates: int = 10_000
    vocabulary_seed: int = 73_129


@dataclass(frozen=True)
class NearDuplicates:
    enabled: bool
    minimum_words: int
    shingle_words: int
    hashes: int
    bands: int
    rows_per_band: int
    jaccard: float
    seed: int


@dataclass(frozen=True)
class Sampling:
    enabled: bool = False
    target_rows: int = 10_000
    candidate_rows: int = 200_000
    windows: int = 64
    window_bytes: int = 16_777_216
    done_fraction: float = 0.80
    same_as_reserve: int = 500
    seed: int = 73_129


@dataclass(frozen=True)
class Config:
    path: Path
    schema_version: int
    cutoff_date: str
    profile: str
    paths: Paths
    runtime: Runtime
    source: Source
    boilerplate: Boilerplate
    lexical: Lexical
    content: Content
    near_duplicates: NearDuplicates
    sampling: Sampling

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("path", None)
        for key, path in value["paths"].items():
            value["paths"][key] = str(path)
        return value

    @property
    def fingerprint(self) -> str:
        # A análise de conteúdo é um produto derivado. Mantê-la fora da identidade
        # do pipeline-base permite complementar snapshots já validados sem refazer
        # inventário, deduplicação ou revisão humana.
        value = self.public_dict()
        value.pop("content", None)
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def content_fingerprint(self) -> str:
        payload = json.dumps(
            {"schema_version": 2, "content": asdict(self.content)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def analysis_pages(self) -> Path:
        from .sampling import analysis_paths

        return analysis_paths(self).pages

    @property
    def analysis_domains(self) -> Path:
        from .sampling import analysis_paths

        return analysis_paths(self).domains


def _resolve(base: Path, value: str) -> Path:
    expanded = os.path.expandvars(value)
    unresolved = re.search(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})", expanded)
    if unresolved:
        raise ConfigurationError(
            f"variável de ambiente não definida no caminho: {unresolved.group(0)}"
        )
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"configuração inválida: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise ConfigurationError("schema_version deve ser 1")
    profile = raw.get("profile")
    if profile not in {"tiny", "full"}:
        raise ConfigurationError("profile deve ser tiny ou full")
    base = config_path.parent.parent
    try:
        paths_raw = raw["paths"]
        paths = Paths(
            pages=_resolve(base, paths_raw["pages"]),
            domains=_resolve(base, paths_raw["domains"]),
            work=_resolve(base, paths_raw["work"]),
            results=_resolve(base, paths_raw["results"]),
        )
        config = Config(
            path=config_path,
            schema_version=1,
            cutoff_date=str(raw["cutoff_date"]),
            profile=profile,
            paths=paths,
            runtime=Runtime(**raw["runtime"]),
            source=Source(**raw["source"]),
            boilerplate=Boilerplate(**raw["boilerplate"]),
            lexical=Lexical(**raw["lexical"]),
            content=Content(**raw.get("content", {})),
            near_duplicates=NearDuplicates(**raw["near_duplicates"]),
            sampling=Sampling(**raw.get("sampling", {})),
        )
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(f"campo ausente ou inválido: {exc}") from exc
    _validate(config)
    return config


def _validate(config: Config) -> None:
    runtime = config.runtime
    if (
        min(runtime.workers, runtime.chunk_bytes, runtime.max_line_bytes, runtime.max_text_bytes)
        <= 0
    ):
        raise ConfigurationError("limites e workers devem ser positivos")
    sampling = config.sampling
    if config.profile == "tiny":
        valid_rows = runtime.max_rows == 0 if sampling.enabled else 0 < runtime.max_rows <= 10_000
        if runtime.workers != 1 or not valid_rows:
            expected = "max_rows=0" if sampling.enabled else "max_rows entre 1 e 10.000"
            raise ConfigurationError(f"tiny exige 1 worker e {expected}")
    if sampling.enabled and config.profile != "tiny":
        raise ConfigurationError("sampling só pode ser habilitado no perfil tiny")
    if not (
        0 < sampling.target_rows <= 10_000
        and sampling.candidate_rows >= sampling.target_rows
        and sampling.windows > 0
        and sampling.window_bytes > 0
        and 0 <= sampling.done_fraction <= 1
        and 0 <= sampling.same_as_reserve < sampling.target_rows
    ):
        raise ConfigurationError("parâmetros de sampling inválidos")
    near = config.near_duplicates
    if near.hashes != near.bands * near.rows_per_band:
        raise ConfigurationError("hashes deve ser bands × rows_per_band")
    if not 0 < near.jaccard <= 1:
        raise ConfigurationError("jaccard deve estar em (0, 1]")
    if config.source.header not in {"auto", "present", "absent"}:
        raise ConfigurationError("source.header inválido")
    if not config.source.require_immutable:
        raise ConfigurationError("source.require_immutable deve permanecer true")
    if config.lexical.partitions != 256:
        raise ConfigurationError("lexical.partitions deve permanecer 256")
    if (
        min(
            config.lexical.spill_terms,
            config.lexical.bigram_candidates,
            config.lexical.published_items,
            config.lexical.figure_items,
        )
        <= 0
    ):
        raise ConfigurationError("limites lexicais devem ser positivos")
    content = config.content
    if (
        min(
            content.mattr_window,
            content.long_sentence_words,
            content.long_token_chars,
            content.punctuation_run,
            content.repetition_sentence_min_words,
            content.repetition_paragraph_min_chars,
            content.fraction_min_tokens,
            content.domain_min_documents,
            content.domain_limit,
            content.collocation_min_frequency,
            content.collocation_min_documents,
            content.trigram_candidates,
        )
        <= 0
        or content.fragment_max_words < 0
    ):
        raise ConfigurationError("limites de conteúdo devem ser positivos")
    if not all(
        0 <= value <= 1
        for value in (
            content.high_numeric_fraction,
            content.high_nonlexical_fraction,
            content.high_uppercase_fraction,
        )
    ):
        raise ConfigurationError("frações de conteúdo devem estar em [0, 1]")
    boilerplate = config.boilerplate
    if (
        min(
            boilerplate.paragraph_min_chars,
            boilerplate.block_lines,
            boilerplate.block_min_chars,
            boilerplate.review_paragraphs,
            boilerplate.review_blocks,
        )
        <= 0
    ):
        raise ConfigurationError("parâmetros de boilerplate devem ser positivos")
    if not (0 <= boilerplate.precision_min <= 1 and 0 <= boilerplate.wilson_lower_min <= 1):
        raise ConfigurationError("limiares de revisão devem estar em [0, 1]")
