from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import WackyWackyError
from .io import atomic_json, decode_field, iter_bounded_tsv, parse_int, sha256_file
from .schema import DOMAIN_COLUMNS, PAGES_COLUMNS

if TYPE_CHECKING:
    from .config import Config


PAGE_PROJECTION = frozenset({0, 1, 2, 3, 8, 10, 11, 12, 13, 15})
DOMAIN_PROJECTION = frozenset({0, 1, 3, 4, 5})


@dataclass(frozen=True)
class AnalysisPaths:
    pages: Path
    domains: Path
    sampling: dict[str, Any] | None = None


def _sampling_key(config: Config) -> str:
    payload = {
        "cutoff_date": config.cutoff_date,
        "pages": str(config.paths.pages),
        "domains": str(config.paths.domains),
        "max_line_bytes": config.runtime.max_line_bytes,
        "sampling": asdict(config.sampling),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _pointer_path(config: Config) -> Path:
    return config.paths.work / "samples" / "pointers" / f"{_sampling_key(config)}.json"


def _source_stats(config: Config) -> dict[str, dict[str, int | str]]:
    values: dict[str, dict[str, int | str]] = {}
    for name, path in (("pages", config.paths.pages), ("domains", config.paths.domains)):
        if not path.is_file():
            raise WackyWackyError(f"fonte ausente para amostragem: {path}")
        stat = path.stat()
        values[name] = {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return values


def _read_pointer(config: Config) -> tuple[dict[str, Any], Path]:
    pointer = _pointer_path(config)
    try:
        value = json.loads(pointer.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise WackyWackyError(
            "amostra tiny ausente; execute `wackywacky sample --config ...`"
        ) from exc
    if value.get("sampling_config_sha256") != _sampling_key(config):
        raise WackyWackyError("amostra tiny incompatível com a configuração; execute `sample`")
    root = config.paths.work / "samples" / value.get("sample_id", "")
    try:
        manifest = json.loads((root / "sample_manifest.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise WackyWackyError("amostra tiny incompleta; execute `sample` novamente") from exc
    if manifest.get("source_stats") != _source_stats(config):
        raise WackyWackyError(
            "fontes originais mudaram desde a amostragem; execute `sample` novamente"
        )
    return manifest, root


def analysis_paths(config: Config) -> AnalysisPaths:
    if not config.sampling.enabled:
        return AnalysisPaths(config.paths.pages, config.paths.domains)
    manifest, root = _read_pointer(config)
    pages = root / "pages.tsv"
    domains = root / "domain.tsv"
    if not pages.is_file() or not domains.is_file():
        raise WackyWackyError("arquivos da amostra tiny ausentes; execute `sample` novamente")
    return AnalysisPaths(pages, domains, manifest)


def _header_end(path: Path, expected: tuple[str, ...], max_line_bytes: int) -> int:
    with path.open("rb") as handle:
        first = handle.readline(max_line_bytes + 1)
        end = handle.tell()
    if not first or len(first) > max_line_bytes:
        raise WackyWackyError(f"primeira linha inválida em {path.name}")
    fields = tuple(first.rstrip(b"\r\n").split(b"\t"))
    header = tuple(value.encode("ascii") for value in expected)
    if fields == header:
        return end
    if len(fields) != len(expected):
        raise WackyWackyError(
            f"{path.name}: primeira linha tem {len(fields)} colunas; esperado {len(expected)}"
        )
    return 0


def _score(seed: int, page_id: int | None, offset: int) -> str:
    return hashlib.sha256(f"{seed}:{page_id}:{offset}".encode()).hexdigest()


def _window_offsets(size: int, content_start: int, windows: int, seed: int) -> list[int]:
    span = max(0, size - content_start)
    if not span:
        return []
    rng = random.Random(seed)
    offsets: list[int] = []
    for index in range(windows):
        lower = content_start + span * index // windows
        upper = content_start + span * (index + 1) // windows
        offsets.append(lower if upper <= lower else rng.randrange(lower, upper))
    return offsets


def _collect_candidates(config: Config, database: Path) -> dict[str, Any]:
    source = config.paths.pages
    content_start = _header_end(source, PAGES_COLUMNS, config.runtime.max_line_bytes)
    offsets = _window_offsets(
        source.stat().st_size, content_start, config.sampling.windows, config.sampling.seed
    )
    connection = sqlite3.connect(database)
    connection.execute(
        """
        CREATE TABLE candidates (
          offset INTEGER PRIMARY KEY,
          end_offset INTEGER NOT NULL,
          page_id INTEGER,
          domain_id INTEGER,
          same_as INTEGER,
          status TEXT NOT NULL,
          score TEXT NOT NULL
        )
        """
    )
    errors: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    per_window = math.ceil(config.sampling.candidate_rows / max(1, len(offsets)))
    bytes_read = 0
    with source.open("rb") as handle:
        for requested_offset in offsets:
            handle.seek(requested_offset)
            if requested_offset > content_start:
                skipped = handle.readline(config.runtime.max_line_bytes + 1)
                bytes_read += len(skipped)
                if len(skipped) > config.runtime.max_line_bytes and not skipped.endswith(b"\n"):
                    while skipped and not skipped.endswith(b"\n"):
                        skipped = handle.readline(config.runtime.max_line_bytes + 1)
                        bytes_read += len(skipped)
            window_start = handle.tell()
            accepted = 0
            while (
                accepted < per_window
                and handle.tell() - window_start < config.sampling.window_bytes
            ):
                offset = handle.tell()
                raw = handle.readline(config.runtime.max_line_bytes + 1)
                bytes_read += len(raw)
                if not raw:
                    break
                if len(raw) > config.runtime.max_line_bytes and not raw.endswith(b"\n"):
                    while raw and not raw.endswith(b"\n"):
                        raw = handle.readline(config.runtime.max_line_bytes + 1)
                        bytes_read += len(raw)
                    errors["line_too_large"] += 1
                    continue
                fields = tuple(raw.rstrip(b"\r\n").split(b"\t"))
                if len(fields) != len(PAGES_COLUMNS):
                    errors["wrong_column_count"] += 1
                    continue
                page_id = parse_int(fields[0])
                status = decode_field(fields[11]) or "<nulo>"
                connection.execute(
                    "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        offset,
                        handle.tell(),
                        page_id,
                        parse_int(fields[1]),
                        parse_int(fields[3]),
                        status,
                        _score(config.sampling.seed, page_id, offset),
                    ),
                )
                statuses[status] += 1
                accepted += 1
        connection.commit()
    total = connection.execute("SELECT count(*) FROM candidates").fetchone()[0]
    connection.close()
    if total < config.sampling.target_rows:
        raise WackyWackyError(
            f"amostragem encontrou {total} linhas válidas; são necessárias "
            f"{config.sampling.target_rows}"
        )
    return {
        "candidate_rows": total,
        "candidate_status": dict(sorted(statuses.items())),
        "errors": dict(sorted(errors.items())),
        "windows_requested": offsets,
        "bytes_read": bytes_read,
    }


def _select_offsets(config: Config, database: Path) -> tuple[list[int], dict[str, Any]]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT offset, page_id, domain_id, same_as, status, score FROM candidates"
    ).fetchall()
    connection.close()
    candidates = [
        {
            "offset": row[0],
            "page_id": row[1],
            "domain_id": row[2],
            "same_as": row[3],
            "status": row[4],
            "score": row[5],
        }
        for row in rows
    ]
    ordered = sorted(candidates, key=lambda row: (row["score"], row["offset"]))
    done = [row for row in ordered if row["status"] == "done"]
    other = [row for row in ordered if row["status"] != "done"]
    primary_target = config.sampling.target_rows - config.sampling.same_as_reserve
    done_target = round(primary_target * config.sampling.done_fraction)
    selected: dict[int, dict[str, Any]] = {}

    def take(values: list[dict[str, Any]], amount: int) -> None:
        for row in values:
            if len(selected) >= amount:
                break
            selected.setdefault(row["offset"], row)

    weighted_target = done_target // 2
    take(done, weighted_target)
    by_domain: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for row in done:
        if row["offset"] not in selected:
            by_domain[row["domain_id"]].append(row)
    balanced = sorted(
        (
            (rank, row["score"], row)
            for values in by_domain.values()
            for rank, row in enumerate(values)
        ),
        key=lambda value: (value[0], value[1], value[2]["offset"]),
    )
    for _rank, _score_value, row in balanced:
        if len(selected) >= done_target:
            break
        selected[row["offset"]] = row

    other_target = primary_target - len(selected)
    by_status: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in other:
        by_status[row["status"]].append(row)
    status_guarantees = [
        row
        for rank in range(50)
        for status in sorted(by_status)
        for row in by_status[status][rank : rank + 1]
    ]
    for row in status_guarantees[:other_target]:
        selected[row["offset"]] = row
    for row in other:
        if len(selected) >= primary_target:
            break
        selected[row["offset"]] = row
    for row in ordered:
        if len(selected) >= primary_target:
            break
        selected[row["offset"]] = row

    by_page_id: dict[int, dict[str, Any]] = {}
    for row in ordered:
        if row["page_id"] is not None:
            by_page_id.setdefault(row["page_id"], row)
    closure = {
        by_page_id[row["same_as"]]["offset"]: by_page_id[row["same_as"]]
        for row in selected.values()
        if row["same_as"] in by_page_id and by_page_id[row["same_as"]]["offset"] not in selected
    }
    closure_rows = sorted(closure.values(), key=lambda row: (row["score"], row["offset"]))
    for row in closure_rows[: config.sampling.same_as_reserve]:
        selected[row["offset"]] = row
    for row in ordered:
        if len(selected) >= config.sampling.target_rows:
            break
        selected[row["offset"]] = row
    if len(selected) != config.sampling.target_rows:
        raise WackyWackyError("não foi possível completar o tamanho solicitado da amostra")
    status = Counter(row["status"] for row in selected.values())
    return sorted(selected), {
        "selected_rows": len(selected),
        "selected_status": dict(sorted(status.items())),
        "same_as_targets_added": min(len(closure_rows), config.sampling.same_as_reserve),
        "distinct_domains_referenced": len(
            {row["domain_id"] for row in selected.values() if row["domain_id"] is not None}
        ),
    }


def _project(fields: tuple[bytes, ...], kept: frozenset[int]) -> bytes:
    return b"\t".join(value if index in kept else b"NULL" for index, value in enumerate(fields))


def _write_pages(config: Config, offsets: list[int], output: Path) -> set[int]:
    domains: set[int] = set()
    with config.paths.pages.open("rb") as source, output.open("wb") as target:
        target.write(b"\t".join(value.encode("ascii") for value in PAGES_COLUMNS) + b"\n")
        for offset in offsets:
            source.seek(offset)
            raw = source.readline(config.runtime.max_line_bytes + 1)
            fields = tuple(raw.rstrip(b"\r\n").split(b"\t"))
            if len(fields) != len(PAGES_COLUMNS):
                raise WackyWackyError("linha selecionada mudou durante a amostragem")
            domain_id = parse_int(fields[1])
            if domain_id is not None:
                domains.add(domain_id)
            target.write(_project(fields, PAGE_PROJECTION) + b"\n")
        target.flush()
        os.fsync(target.fileno())
    return domains


def _write_domains(
    config: Config, wanted: set[int], output: Path, database: Path
) -> dict[str, int]:
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE source_domains (id INTEGER PRIMARY KEY, parent_id INTEGER, row_number INTEGER, row BLOB)"
    )
    errors: Counter[str] = Counter()
    with config.paths.domains.open("rb") as handle:
        for record in iter_bounded_tsv(
            handle,
            columns=len(DOMAIN_COLUMNS),
            max_line_bytes=config.runtime.max_line_bytes,
        ):
            if record.fields is None:
                errors[record.error or "structural"] += 1
                continue
            if record.row_number == 1 and record.fields == tuple(
                value.encode("ascii") for value in DOMAIN_COLUMNS
            ):
                continue
            domain_id = parse_int(record.fields[0])
            if domain_id is None:
                errors["invalid_id"] += 1
                continue
            connection.execute(
                "INSERT OR IGNORE INTO source_domains VALUES (?, ?, ?, ?)",
                (
                    domain_id,
                    parse_int(record.fields[3]),
                    record.row_number,
                    _project(record.fields, DOMAIN_PROJECTION),
                ),
            )
    connection.commit()
    selected: dict[int, tuple[int | None, int, bytes]] = {}
    pending = set(wanted)
    while pending:
        domain_id = pending.pop()
        if domain_id in selected:
            continue
        row = connection.execute(
            "SELECT parent_id, row_number, row FROM source_domains WHERE id=?", (domain_id,)
        ).fetchone()
        if row is None:
            continue
        selected[domain_id] = (row[0], row[1], row[2])
        if row[0] is not None and row[0] not in selected:
            pending.add(row[0])
    connection.close()
    with output.open("wb") as target:
        target.write(b"\t".join(value.encode("ascii") for value in DOMAIN_COLUMNS) + b"\n")
        for _parent, _row_number, raw in sorted(selected.values(), key=lambda row: row[1]):
            target.write(raw + b"\n")
        target.flush()
        os.fsync(target.fileno())
    return {
        "selected_domains": len(selected),
        "missing_domain_references": len(wanted - set(selected)),
        "structural_errors": sum(errors.values()),
    }


def create_sample(config: Config) -> dict[str, Any]:
    if not config.sampling.enabled:
        raise WackyWackyError("sampling.enabled deve estar true para usar o comando sample")
    try:
        manifest, root = _read_pointer(config)
        if (
            sha256_file(root / "pages.tsv") == manifest["outputs"]["pages_sha256"]
            and sha256_file(root / "domain.tsv") == manifest["outputs"]["domains_sha256"]
        ):
            return manifest
    except (OSError, WackyWackyError):
        pass

    before = _source_stats(config)
    samples_root = config.paths.work / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=samples_root))
    try:
        database = temporary / "sample.sqlite"
        collection = _collect_candidates(config, database)
        offsets, selection = _select_offsets(config, database)
        pages = temporary / "pages.tsv"
        domains = temporary / "domain.tsv"
        wanted_domains = _write_pages(config, offsets, pages)
        domain_selection = _write_domains(config, wanted_domains, domains, database)
        after = _source_stats(config)
        if before != after:
            raise WackyWackyError("fontes mudaram durante a amostragem")
        database.unlink(missing_ok=True)
        pages_sha = sha256_file(pages)
        domains_sha = sha256_file(domains)
        combined = hashlib.sha256(f"{pages_sha}:{domains_sha}".encode()).hexdigest()
        sample_id = f"{config.cutoff_date.replace('-', '')}-{combined[:12]}"
        manifest = {
            "enabled": True,
            "scope": "prévia amostral não representativa",
            "sample_id": sample_id,
            "sampling_config_sha256": _sampling_key(config),
            "parameters": asdict(config.sampling),
            "source_stats": before,
            "collection": collection,
            "selection": {**selection, **domain_selection},
            "projection": {
                "pages": [PAGES_COLUMNS[index] for index in sorted(PAGE_PROJECTION)],
                "domains": [DOMAIN_COLUMNS[index] for index in sorted(DOMAIN_PROJECTION)],
            },
            "outputs": {
                "pages_sha256": pages_sha,
                "domains_sha256": domains_sha,
                "pages_bytes": pages.stat().st_size,
                "domains_bytes": domains.stat().st_size,
            },
        }
        atomic_json(temporary / "sample_manifest.json", manifest)
        final = samples_root / sample_id
        if final.exists():
            existing = json.loads((final / "sample_manifest.json").read_text(encoding="utf-8"))
            if existing["outputs"] != manifest["outputs"]:
                raise WackyWackyError(f"amostra existente incompatível: {sample_id}")
            shutil.rmtree(temporary)
            atomic_json(final / "sample_manifest.json", manifest)
        else:
            os.replace(temporary, final)
        atomic_json(
            _pointer_path(config),
            {"sample_id": sample_id, "sampling_config_sha256": _sampling_key(config)},
        )
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
