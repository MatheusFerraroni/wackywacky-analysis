from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

NULLS = {b"", b"NULL", b"\\N"}


@dataclass(frozen=True)
class BinaryRecord:
    row_number: int
    offset: int
    end_offset: int
    fields: tuple[bytes, ...] | None
    error: str | None = None


def iter_bounded_tsv(
    handle: BinaryIO,
    *,
    columns: int,
    max_line_bytes: int,
    start_offset: int = 0,
    start_row: int = 0,
    max_rows: int = 0,
) -> Iterator[BinaryRecord]:
    handle.seek(start_offset)
    row = start_row
    while not max_rows or row < max_rows:
        offset = handle.tell()
        raw = handle.readline(max_line_bytes + 1)
        if not raw:
            break
        row += 1
        if len(raw) > max_line_bytes and not raw.endswith(b"\n"):
            while raw and not raw.endswith(b"\n"):
                raw = handle.readline(max_line_bytes + 1)
            yield BinaryRecord(row, offset, handle.tell(), None, "line_too_large")
            continue
        end = handle.tell()
        fields = tuple(raw.rstrip(b"\r\n").split(b"\t"))
        if len(fields) != columns:
            yield BinaryRecord(row, offset, end, None, "wrong_column_count")
        else:
            yield BinaryRecord(row, offset, end, fields)


def is_null(value: bytes) -> bool:
    return value in NULLS


def decode_field(value: bytes) -> str | None:
    if is_null(value):
        return None
    return value.decode("utf-8", errors="replace")


def parse_int(value: bytes) -> int | None:
    if is_null(value):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def checksum_paths(paths: Sequence[Path]) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(paths)}
