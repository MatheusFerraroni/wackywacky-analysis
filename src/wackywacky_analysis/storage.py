from __future__ import annotations

import os
import struct
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .io import sha256_file


def _validate_parquet(path: Path) -> None:
    pq.read_metadata(path)


def write_parquet_atomic(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, partial, compression="zstd", write_statistics=True)
    _validate_parquet(partial)
    checksum = sha256_file(partial)
    os.replace(partial, path)
    checksum_path = path.with_suffix(path.suffix + ".sha256")
    checksum_path.write_text(checksum + "\n", encoding="ascii")
    return checksum


def duckdb_connection(
    database: Path, memory_limit: str, temp_directory: Path
) -> duckdb.DuckDBPyConnection:
    database.parent.mkdir(parents=True, exist_ok=True)
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database))
    connection.execute("SET memory_limit = ?", [memory_limit])
    connection.execute("SET temp_directory = ?", [str(temp_directory)])
    connection.execute("SET preserve_insertion_order = false")
    return connection


def duckdb_copy_atomic(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    escaped = str(partial).replace("'", "''")
    connection.execute(f"COPY ({query}) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    _validate_parquet(partial)
    os.replace(partial, path)
    path.with_suffix(path.suffix + ".sha256").write_text(sha256_file(path) + "\n", encoding="ascii")


def write_u64(path: Path, values: Iterable[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<Q", value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def read_u64(path: Path) -> Iterator[int]:
    with path.open("rb") as handle:
        while block := handle.read(8):
            if len(block) != 8:
                raise ValueError(f"lista binária truncada: {path.name}")
            yield struct.unpack("<Q", block)[0]


class SortedMembership:
    """Consulta sequencial de uma lista u64 ordenada sem carregá-la em RAM."""

    def __init__(self, path: Path) -> None:
        self._values = iter(read_u64(path))
        self._current = next(self._values, None)

    def contains(self, value: int) -> bool:
        while self._current is not None and self._current < value:
            self._current = next(self._values, None)
        return self._current == value
