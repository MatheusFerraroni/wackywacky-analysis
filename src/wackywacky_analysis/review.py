from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import unicodedata
from pathlib import Path

import duckdb

from .config import Config
from .errors import ReviewRejected, ReviewRequired, WackyWackyError
from .io import atomic_json, decode_field, iter_bounded_tsv, parse_int, sha256_file
from .schema import PAGES_COLUMNS
from .text import TextDecodeFailure, block_units, decode_text, paragraph_units

LABELS = {"boilerplate", "conteúdo", "incerto"}
REVIEW_PREVIEW_CHARS = 4_000


def _sample_id(domain_id: int, kind: str, digest: str) -> str:
    return hashlib.sha256(f"{domain_id}:{kind}:{digest}".encode()).hexdigest()[:20]


def _review_preview(text: str, limit: int = REVIEW_PREVIEW_CHARS) -> tuple[str, bool]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pieces: list[str] = []
    for char in text:
        if char == "\n":
            pieces.append(" [QUEBRA] ")
        elif not unicodedata.category(char).startswith("C"):
            pieces.append(char)
    sanitized = " ".join("".join(pieces).split())
    if len(sanitized) <= limit:
        return sanitized, False
    head = sanitized[: int(limit * 0.75)].rstrip()
    tail = sanitized[-int(limit * 0.20) :].lstrip()
    omitted = len(sanitized) - len(head) - len(tail)
    return f"{head} … [{omitted} caracteres omitidos] … {tail}", True


def _review_was_edited(path: Path, root: Path) -> bool:
    if not path.exists():
        return False
    metadata_path = root / "review_sample.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            exported_sha256 = metadata.get("export_sha256")
            if exported_sha256:
                return sha256_file(path) != exported_sha256
        except (OSError, json.JSONDecodeError):
            pass
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return any((row.get("label") or "").strip() for row in csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return False


def _selected(config: Config, root: Path) -> list[dict]:
    connection = duckdb.connect(str(root / "analysis.duckdb"), read_only=True)
    source = str(root / "boilerplate_candidates.parquet")
    rows: list[dict] = []
    for kind, target in (
        ("paragraph", config.boilerplate.review_paragraphs),
        ("block", config.boilerplate.review_blocks),
    ):
        result = connection.execute(
            """
            WITH ranked AS (
              SELECT *,
                ntile(4) OVER (ORDER BY document_frequency) AS frequency_bin,
                ntile(4) OVER (ORDER BY domain_documents) AS volume_bin
              FROM read_parquet(?) WHERE kind = ?
            ), strata AS (
              SELECT *, row_number() OVER (
                PARTITION BY frequency_bin, volume_bin
                ORDER BY hash(unit_sha256, domain_id, ?)
              ) AS within_stratum
              FROM ranked
            )
            SELECT domain_id, kind, unit_sha256, characters, document_frequency,
                   domain_documents, frequency_bin, volume_bin
            FROM strata
            ORDER BY within_stratum,
                     hash(frequency_bin, volume_bin, ?),
                     hash(unit_sha256, domain_id, ?)
            LIMIT ?
            """,
            [
                source,
                kind,
                config.boilerplate.review_seed,
                config.boilerplate.review_seed,
                config.boilerplate.review_seed,
                target,
            ],
        ).fetchall()
        columns = [item[0] for item in connection.description]
        rows.extend(dict(zip(columns, row, strict=True)) for row in result)
    connection.close()
    for row in rows:
        row["sample_id"] = _sample_id(row["domain_id"], row["kind"], row["unit_sha256"])
    return rows


def export_review(
    config: Config, manifest: dict, root: Path, output: Path | None = None
) -> Path | None:
    selected = _selected(config, root)
    if not selected:
        atomic_json(root / "review_gate.json", {"status": "not_needed", "sample_size": 0})
        return None
    wanted = {(row["domain_id"], row["kind"], row["unit_sha256"]): row for row in selected}
    found: dict[str, str] = {}
    with config.analysis_pages.open("rb") as handle:
        for record in iter_bounded_tsv(
            handle,
            columns=len(PAGES_COLUMNS),
            max_line_bytes=config.runtime.max_line_bytes,
            max_rows=config.runtime.max_rows,
        ):
            if record.fields is None:
                continue
            fields = record.fields
            if decode_field(fields[11]) != "done":
                continue
            domain_id = parse_int(fields[1])
            if domain_id is None:
                continue
            try:
                decoded = decode_text(fields[13], fields[15], config.runtime.max_text_bytes)
            except TextDecodeFailure:
                continue
            units = {
                "paragraph": paragraph_units(
                    decoded.normalized, config.boilerplate.paragraph_min_chars
                ),
                "block": block_units(
                    decoded.normalized,
                    config.boilerplate.block_lines,
                    config.boilerplate.block_min_chars,
                ),
            }
            for kind, values in units.items():
                for _start, _end, digest, text in values:
                    selected_row = wanted.get((domain_id, kind, digest))
                    if selected_row:
                        found[selected_row["sample_id"]] = text
    if len(found) != len(selected):
        raise WackyWackyError("não foi possível reconstruir toda a amostra privada")
    output = output or (root / "review" / "boilerplate-review.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    if _review_was_edited(output, root):
        raise WackyWackyError(
            f"arquivo de revisão foi alterado; importe-o antes de exportar novamente: {output}"
        )
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "frequencia",
                "label",
                "kind",
                "domain_id",
                "domain_documents",
                "frequency_bin",
                "volume_bin",
                "characters",
                "preview_truncated",
                "text_preview",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in sorted(selected, key=lambda item: item["sample_id"]):
            preview, truncated = _review_preview(found[row["sample_id"]])
            public = {
                key: row[key]
                for key in writer.fieldnames
                if key not in {"frequencia", "label", "preview_truncated", "text_preview"}
            }
            public.update(
                {
                    "frequencia": row["document_frequency"],
                    "label": "boilerplate",
                    "preview_truncated": "sim" if truncated else "não",
                    "text_preview": preview,
                }
            )
            writer.writerow(public)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, output)
    instructions = output.parent / "LEIA-ME-revisao.txt"
    instructions.write_text(
        "Todos os itens começam com label=boilerplate.\n"
        "Altere somente as exceções para conteúdo ou incerto.\n"
        "Não apague nem duplique linhas.\n",
        encoding="utf-8",
    )
    atomic_json(
        root / "review_sample.json",
        {
            "sample_ids": sorted(row["sample_id"] for row in selected),
            "size": len(selected),
            "path": str(output),
            "export_sha256": sha256_file(output),
        },
    )
    return output


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total == 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return (centre - radius) / denominator


def import_review(config: Config, root: Path, source: Path) -> dict:
    expected = set(json.loads((root / "review_sample.json").read_text())["sample_ids"])
    labels: dict[str, str] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = row.get("sample_id", "")
            label = row.get("label", "").strip().casefold()
            if sample_id not in expected or label not in LABELS or sample_id in labels:
                raise WackyWackyError("amostra contém ID duplicado/desconhecido ou rótulo inválido")
            labels[sample_id] = label
    if set(labels) != expected:
        raise WackyWackyError("amostra importada está incompleta")
    successes = sum(label == "boilerplate" for label in labels.values())
    total = len(labels)
    precision = successes / total if total else 0.0
    lower = wilson_lower(successes, total)
    approved = (
        precision >= config.boilerplate.precision_min
        and lower >= config.boilerplate.wilson_lower_min
    )
    gate = {
        "status": "approved" if approved else "rejected",
        "sample_size": total,
        "boilerplate": successes,
        "content_or_uncertain": total - successes,
        "precision": precision,
        "wilson_lower_95": lower,
        "thresholds": {
            "precision": config.boilerplate.precision_min,
            "wilson_lower_95": config.boilerplate.wilson_lower_min,
        },
        "labels_sha256": hashlib.sha256(
            json.dumps(labels, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    atomic_json(root / "review_labels.json", labels)
    atomic_json(root / "review_gate.json", gate)
    if not approved:
        raise ReviewRejected("gate de revisão reprovado; limpeza não autorizada")
    return gate


def require_approved_review(config: Config, root: Path) -> dict:
    candidates = root / "boilerplate_candidates.parquet"
    connection = duckdb.connect()
    count = connection.execute(
        "SELECT count(*) FROM read_parquet(?)", [str(candidates)]
    ).fetchone()[0]
    connection.close()
    if count == 0 or not config.boilerplate.enabled:
        gate = {"status": "not_needed", "sample_size": 0}
        atomic_json(root / "review_gate.json", gate)
        return gate
    gate_path = root / "review_gate.json"
    if not gate_path.exists():
        sample = export_review(config, {}, root)
        raise ReviewRequired(
            f"revisão privada necessária: {sample}; labels começam como boilerplate, "
            "altere somente conteúdo e incerto"
        )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "approved":
        raise ReviewRejected("gate de revisão não está aprovado")
    return gate
