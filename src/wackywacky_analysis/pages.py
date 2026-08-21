from __future__ import annotations

import signal
from collections import Counter
from pathlib import Path

import pyarrow as pa

from .config import Config
from .io import atomic_json, decode_field, is_null, iter_bounded_tsv, parse_int, read_json
from .schema import PAGES_COLUMNS
from .snapshot import assert_snapshot
from .storage import write_parquet_atomic
from .text import TextDecodeFailure, block_units, decode_text, paragraph_units

FEATURE_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("offset", pa.uint64()),
        ("page_id", pa.int64()),
        ("domain_id", pa.int64()),
        ("same_as", pa.int64()),
        ("recursion_level", pa.int32()),
        ("raw_sha256", pa.string()),
        ("normalized_sha256", pa.string()),
        ("md5_class", pa.string()),
        ("raw_bytes", pa.uint64()),
        ("characters", pa.uint64()),
        ("lines", pa.uint32()),
        ("paragraphs", pa.uint32()),
    ]
)

OCCURRENCE_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("domain_id", pa.int64()),
        ("kind", pa.string()),
        ("unit_sha256", pa.string()),
        ("characters", pa.uint32()),
    ]
)

INVENTORY_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("page_id", pa.int64()),
        ("domain_id", pa.int64()),
        ("same_as", pa.int64()),
        ("status", pa.string()),
        ("status_code", pa.int32()),
        ("recursion_level", pa.int32()),
        ("retry_count", pa.int32()),
        ("text_present", pa.bool_()),
    ]
)


def _new_metrics() -> dict:
    return {
        "rows": 0,
        "status": Counter(),
        "status_code": Counter(),
        "recursion_level": Counter(),
        "retry_count": Counter(),
        "errors": Counter(),
        "done": 0,
        "text_present": 0,
        "decoded": 0,
        "r_valid": 0,
        "md5_class": Counter(),
        "same_as_present": 0,
    }


def _json_metrics(metrics: dict) -> dict:
    return {
        key: dict(value) if isinstance(value, Counter) else value for key, value in metrics.items()
    }


def scan_pages(config: Config, manifest: dict, root: Path, *, resume: bool) -> dict:
    chunk_dir = root / "pages" / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = read_json(state_path, {}) if resume else {}
    scan = state.get("pages_scan", {})
    if scan.get("complete"):
        return scan
    offset = int(scan.get("next_offset", 0))
    row_number = int(scan.get("next_row", 0))
    chunk_index = int(scan.get("next_chunk", 0))
    if offset and not resume:
        raise ValueError("estado parcial existe; use --resume")
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    header = manifest["sources"]["pages"]["header"]
    try:
        with config.analysis_pages.open("rb") as handle:
            while True:
                assert_snapshot(config, manifest)
                features: list[dict] = []
                occurrences: list[dict] = []
                inventory: list[dict] = []
                metrics = _new_metrics()
                start_offset = offset
                records = iter_bounded_tsv(
                    handle,
                    columns=len(PAGES_COLUMNS),
                    max_line_bytes=config.runtime.max_line_bytes,
                    start_offset=offset,
                    start_row=row_number,
                    max_rows=config.runtime.max_rows,
                )
                exhausted = True
                for record in records:
                    exhausted = False
                    offset = record.end_offset
                    row_number = record.row_number
                    if header and record.row_number == 1:
                        continue
                    metrics["rows"] += 1
                    if record.fields is None:
                        metrics["errors"][record.error or "structural"] += 1
                        metrics["status"]["<erro_estrutural>"] += 1
                    else:
                        _consume_record(config, record, inventory, features, occurrences, metrics)
                    if offset - start_offset >= config.runtime.chunk_bytes:
                        break
                    estimated_buffer = 512 * (len(inventory) + len(features) + len(occurrences))
                    if estimated_buffer >= config.runtime.queue_bytes:
                        break
                    if config.runtime.max_rows and row_number >= config.runtime.max_rows:
                        break
                if not metrics["rows"] and not features and not occurrences:
                    scan["complete"] = True
                    break
                stem = f"chunk-{chunk_index:06d}"
                feature_path = chunk_dir / f"{stem}-features.parquet"
                occurrence_path = chunk_dir / f"{stem}-units.parquet"
                inventory_path = chunk_dir / f"{stem}-inventory.parquet"
                write_parquet_atomic(feature_path, features, FEATURE_SCHEMA)
                write_parquet_atomic(occurrence_path, occurrences, OCCURRENCE_SCHEMA)
                write_parquet_atomic(inventory_path, inventory, INVENTORY_SCHEMA)
                atomic_json(chunk_dir / f"{stem}-metrics.json", _json_metrics(metrics))
                chunk_index += 1
                scan = {
                    "next_offset": offset,
                    "next_row": row_number,
                    "next_chunk": chunk_index,
                    "complete": False,
                }
                state["snapshot_id"] = manifest["snapshot_id"]
                state["config_sha256"] = config.fingerprint
                state["pages_scan"] = scan
                atomic_json(state_path, state)
                if stop:
                    break
                if config.runtime.max_rows and row_number >= config.runtime.max_rows:
                    scan["complete"] = True
                    break
                if exhausted or offset >= manifest["sources"]["pages"]["size"]:
                    scan["complete"] = True
                    break
            state["pages_scan"] = scan
            atomic_json(state_path, state)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return scan


def _consume_record(
    config: Config,
    record,
    inventory: list,
    features: list,
    occurrences: list,
    metrics: dict,
) -> None:
    fields = record.fields
    page_id = parse_int(fields[0])
    domain_id = parse_int(fields[1])
    same_as = parse_int(fields[3])
    status_code = parse_int(fields[8])
    recursion_level = parse_int(fields[10])
    retry_count = parse_int(fields[12])
    for name, index, value in (
        ("page_id", 0, page_id),
        ("domain_id", 1, domain_id),
        ("same_as", 3, same_as),
        ("status_code", 8, status_code),
        ("recursion_level", 10, recursion_level),
        ("retry_count", 12, retry_count),
    ):
        if not is_null(fields[index]) and value is None:
            metrics["errors"][f"invalid_{name}"] += 1
    status = decode_field(fields[11]) or "<nulo>"
    metrics["status"][status] += 1
    metrics["status_code"][decode_field(fields[8]) or "<nulo>"] += 1
    metrics["recursion_level"][decode_field(fields[10]) or "<nulo>"] += 1
    metrics["retry_count"][decode_field(fields[12]) or "<nulo>"] += 1
    if not is_null(fields[3]):
        metrics["same_as_present"] += 1
    inventory.append(
        {
            "row_number": record.row_number,
            "page_id": page_id,
            "domain_id": domain_id,
            "same_as": same_as,
            "status": status,
            "status_code": status_code,
            "recursion_level": recursion_level,
            "retry_count": retry_count,
            "text_present": not is_null(fields[13]),
        }
    )
    if status != "done":
        return
    metrics["done"] += 1
    if is_null(fields[13]):
        metrics["errors"]["text_missing"] += 1
        return
    metrics["text_present"] += 1
    try:
        decoded = decode_text(fields[13], fields[15], config.runtime.max_text_bytes)
    except TextDecodeFailure as exc:
        metrics["errors"][exc.category] += 1
        return
    metrics["decoded"] += 1
    metrics["md5_class"][decoded.md5_class] += 1
    if not decoded.normalized:
        metrics["errors"]["empty_normalized"] += 1
        return
    metrics["r_valid"] += 1
    features.append(
        {
            "row_number": record.row_number,
            "offset": record.offset,
            "page_id": page_id,
            "domain_id": domain_id,
            "same_as": same_as,
            "recursion_level": recursion_level,
            "raw_sha256": decoded.raw_sha256,
            "normalized_sha256": decoded.normalized_sha256,
            "md5_class": decoded.md5_class,
            "raw_bytes": len(decoded.raw),
            "characters": len(decoded.normalized),
            "lines": decoded.normalized.count("\n") + 1,
            "paragraphs": len([part for part in decoded.normalized.split("\n\n") if part]),
        }
    )
    seen: set[tuple[str, str]] = set()
    units = (
        ("paragraph", paragraph_units(decoded.normalized, config.boilerplate.paragraph_min_chars)),
        (
            "block",
            block_units(
                decoded.normalized,
                config.boilerplate.block_lines,
                config.boilerplate.block_min_chars,
            ),
        ),
    )
    for kind, values in units:
        for _start, _end, digest, value in values:
            key = (kind, digest)
            if key in seen:
                continue
            seen.add(key)
            occurrences.append(
                {
                    "row_number": record.row_number,
                    "domain_id": domain_id,
                    "kind": kind,
                    "unit_sha256": digest,
                    "characters": len(value),
                }
            )
