from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import duckdb

from .config import Config
from .errors import ReviewRejected, ReviewRequired, WackyWackyError
from .io import atomic_json, decode_field, iter_bounded_tsv, parse_int
from .schema import PAGES_COLUMNS
from .text import TextDecodeFailure, block_units, decode_text, paragraph_units

LABELS = {"boilerplate", "conteúdo", "incerto"}


def _sample_id(domain_id: int, kind: str, digest: str) -> str:
    return hashlib.sha256(f"{domain_id}:{kind}:{digest}".encode()).hexdigest()[:20]


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
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "kind",
                "domain_id",
                "document_frequency",
                "domain_documents",
                "frequency_bin",
                "volume_bin",
                "text",
                "label",
            ],
        )
        writer.writeheader()
        for row in sorted(selected, key=lambda item: item["sample_id"]):
            public = {key: row[key] for key in writer.fieldnames if key not in {"text", "label"}}
            public["text"] = found[row["sample_id"]]
            public["label"] = ""
            writer.writerow(public)
    atomic_json(
        root / "review_sample.json",
        {
            "sample_ids": sorted(row["sample_id"] for row in selected),
            "size": len(selected),
            "path": str(output),
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
    with source.open("r", encoding="utf-8", newline="") as handle:
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
        raise ReviewRequired(f"revisão privada necessária: {sample}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "approved":
        raise ReviewRejected("gate de revisão não está aprovado")
    return gate
