from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import Config
from .errors import SourceChangedError, WackyWackyError
from .io import atomic_json, sha256_file
from .sampling import analysis_paths
from .schema import DOMAIN_COLUMNS, PAGES_COLUMNS


def _header_state(path: Path, expected: tuple[str, ...], mode: str, max_line: int) -> bool:
    with path.open("rb") as handle:
        first = handle.readline(max_line + 1)
    if not first or len(first) > max_line:
        raise WackyWackyError(f"primeira linha inválida em {path.name}")
    fields = tuple(first.rstrip(b"\r\n").split(b"\t"))
    encoded = tuple(item.encode("ascii") for item in expected)
    is_header = fields == encoded
    if mode == "present" and not is_header:
        raise WackyWackyError(f"header esperado e não encontrado em {path.name}")
    if mode == "absent" and is_header:
        raise WackyWackyError(f"header inesperado em {path.name}")
    if not is_header and len(fields) != len(expected):
        raise WackyWackyError(
            f"{path.name}: primeira linha tem {len(fields)} colunas; esperado {len(expected)}"
        )
    return is_header


def verify_snapshot(config: Config, *, persist: bool = True) -> dict[str, Any]:
    inputs = analysis_paths(config)
    for path in (inputs.pages, inputs.domains):
        if not path.is_file():
            raise WackyWackyError(f"fonte ausente: {path}")
    config.paths.work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(config.paths.work).free
    if free < config.runtime.minimum_scratch_free_bytes:
        raise WackyWackyError(
            f"scratch insuficiente: {free} bytes livres; "
            f"mínimo {config.runtime.minimum_scratch_free_bytes}"
        )
    pages_header = _header_state(
        inputs.pages, PAGES_COLUMNS, config.source.header, config.runtime.max_line_bytes
    )
    domains_header = _header_state(
        inputs.domains, DOMAIN_COLUMNS, config.source.header, config.runtime.max_line_bytes
    )
    before = {
        "pages": inputs.pages.stat(),
        "domains": inputs.domains.stat(),
    }
    pages_sha = sha256_file(inputs.pages)
    domains_sha = sha256_file(inputs.domains)
    after = {
        "pages": inputs.pages.stat(),
        "domains": inputs.domains.stat(),
    }
    for name, before_stat in before.items():
        if (before_stat.st_size, before_stat.st_mtime_ns) != (
            after[name].st_size,
            after[name].st_mtime_ns,
        ):
            raise SourceChangedError(f"{name}.tsv mudou durante a verificação")
    snapshot_id = (
        inputs.sampling["sample_id"]
        if inputs.sampling
        else f"{config.cutoff_date.replace('-', '')}-{pages_sha[:12]}"
    )
    manifest = {
        "schema_version": 1,
        "snapshot_id": snapshot_id,
        "cutoff_date": config.cutoff_date,
        "config_sha256": config.fingerprint,
        "sources": {
            "pages": {
                "name": inputs.pages.name,
                "size": before["pages"].st_size,
                "mtime_ns": before["pages"].st_mtime_ns,
                "sha256": pages_sha,
                "header": pages_header,
                "columns": len(PAGES_COLUMNS),
            },
            "domains": {
                "name": inputs.domains.name,
                "size": before["domains"].st_size,
                "mtime_ns": before["domains"].st_mtime_ns,
                "sha256": domains_sha,
                "header": domains_header,
                "columns": len(DOMAIN_COLUMNS),
            },
        },
        "scratch_free_bytes_at_verify": free,
    }
    if inputs.sampling:
        manifest["sampling"] = inputs.sampling
    if persist:
        snapshot_dir = config.paths.work / snapshot_id
        previous = snapshot_dir / "manifest.json"
        if previous.exists():
            from .io import read_json

            old = read_json(previous)
            if old["config_sha256"] != config.fingerprint or old["sources"] != manifest["sources"]:
                raise SourceChangedError(
                    "snapshot existente é incompatível com fonte ou configuração"
                )
        atomic_json(previous, manifest)
    return manifest


def assert_snapshot(config: Config, manifest: dict[str, Any]) -> None:
    inputs = analysis_paths(config)
    for name, path in (("pages", inputs.pages), ("domains", inputs.domains)):
        stat = path.stat()
        expected = manifest["sources"][name]
        if stat.st_size != expected["size"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise SourceChangedError(f"{path.name} foi alterado após a verificação")
