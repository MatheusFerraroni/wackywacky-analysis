from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import zlib
from collections import defaultdict
from pathlib import Path

import duckdb
import pyarrow as pa

from .config import Config
from .io import atomic_json, iter_bounded_tsv, parse_int
from .lexical import _clean_again, _load_nlp, _token_data
from .schema import PAGES_COLUMNS
from .storage import SortedMembership, write_parquet_atomic
from .text import TextDecodeFailure, decode_text

NEAR_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("representative_row", pa.uint64()),
        ("words", pa.uint64()),
        ("domain_id", pa.int64()),
    ]
)


def _shingles(words: list[str], width: int) -> set[int]:
    return {
        int.from_bytes(
            hashlib.blake2b(
                "\x1f".join(words[index : index + width]).encode(), digest_size=8
            ).digest(),
            "big",
        )
        for index in range(len(words) - width + 1)
    }


def _parameters(count: int, seed: int) -> list[tuple[int, int]]:
    modulus = 1 << 64
    result = []
    for index in range(count):
        raw = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        a = int.from_bytes(raw[:8], "big") | 1
        b = int.from_bytes(raw[8:16], "big")
        result.append((a % modulus, b))
    return result


def _signature(shingles: set[int], parameters: list[tuple[int, int]]) -> tuple[int, ...]:
    mask = (1 << 64) - 1
    return tuple(min(((a * value + b) & mask) for value in shingles) for a, b in parameters)


def _pack(values: set[int]) -> bytes:
    return zlib.compress(b"".join(struct.pack("<Q", value) for value in sorted(values)), level=3)


def _unpack(value: bytes) -> set[int]:
    raw = zlib.decompress(value)
    return {struct.unpack_from("<Q", raw, index)[0] for index in range(0, len(raw), 8)}


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        self.parent.setdefault(value, value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def run_near_duplicates(config: Config, root: Path) -> dict:
    output = root / "near_summary.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    near = config.near_duplicates
    nlp = _load_nlp(config)
    membership = SortedMembership(root / "d3_representatives.u64")
    boilerplate = sqlite3.connect(f"file:{root / 'boilerplate.sqlite'}?mode=ro", uri=True)
    database = root / "near.sqlite"
    partial_database = root / "near.sqlite.partial"
    for stale in (
        partial_database,
        Path(str(partial_database) + "-wal"),
        Path(str(partial_database) + "-shm"),
    ):
        stale.unlink(missing_ok=True)
    sql = sqlite3.connect(partial_database)
    sql.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS document(
          row_number INTEGER PRIMARY KEY, words INTEGER, domain_id INTEGER,
          clean_sha256 TEXT, shingles BLOB
        );
        CREATE TABLE IF NOT EXISTS bucket(
          band INTEGER, digest BLOB, row_number INTEGER,
          PRIMARY KEY(band, digest, row_number)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS bucket_lookup ON bucket(band, digest);
        """
    )
    parameters = _parameters(near.hashes, near.seed)
    union = UnionFind()
    confirmed_pairs = candidate_pairs = 0
    metadata: dict[int, tuple[int, int | None, str]] = {}
    with config.analysis_pages.open("rb") as handle:
        for record in iter_bounded_tsv(
            handle,
            columns=len(PAGES_COLUMNS),
            max_line_bytes=config.runtime.max_line_bytes,
            max_rows=config.runtime.max_rows,
        ):
            if not membership.contains(record.row_number) or record.fields is None:
                continue
            fields = record.fields
            try:
                decoded = decode_text(fields[13], fields[15], config.runtime.max_text_bytes)
            except TextDecodeFailure:
                continue
            domain_id = parse_int(fields[1])
            clean = _clean_again(config, boilerplate, domain_id, decoded.normalized)
            words = _token_data(nlp(clean))[4]
            if len(words) < near.minimum_words:
                continue
            shingles = _shingles(words, near.shingle_words)
            if not shingles:
                continue
            signature = _signature(shingles, parameters)
            clean_sha = hashlib.sha256(clean.encode()).hexdigest()
            metadata[record.row_number] = (len(words), domain_id, clean_sha)
            sql.execute(
                "INSERT OR REPLACE INTO document VALUES (?, ?, ?, ?, ?)",
                (record.row_number, len(words), domain_id, clean_sha, _pack(shingles)),
            )
            possible: set[int] = set()
            for band in range(near.bands):
                values = signature[band * near.rows_per_band : (band + 1) * near.rows_per_band]
                digest = hashlib.blake2b(
                    struct.pack(f"<{near.rows_per_band}Q", *values), digest_size=16
                ).digest()
                possible.update(
                    row[0]
                    for row in sql.execute(
                        "SELECT row_number FROM bucket WHERE band=? AND digest=?",
                        (band, digest),
                    )
                )
                sql.execute(
                    "INSERT INTO bucket VALUES (?, ?, ?)", (band, digest, record.row_number)
                )
            for other in possible:
                candidate_pairs += 1
                other_shingles = _unpack(
                    sql.execute(
                        "SELECT shingles FROM document WHERE row_number=?", (other,)
                    ).fetchone()[0]
                )
                similarity = len(shingles & other_shingles) / len(shingles | other_shingles)
                if similarity >= near.jaccard:
                    confirmed_pairs += 1
                    union.union(record.row_number, other)
            sql.commit()
    boilerplate.close()
    components: dict[int, list[int]] = defaultdict(list)
    for row in union.parent:
        components[union.find(row)].append(row)
    output_rows: list[dict] = []
    removed_words = 0
    for members in components.values():
        if len(members) < 2:
            continue
        representative = min(
            members,
            key=lambda row: (-metadata[row][0], metadata[row][2], row),
        )
        for row in members:
            words, domain, _digest = metadata[row]
            output_rows.append(
                {
                    "row_number": row,
                    "representative_row": representative,
                    "words": words,
                    "domain_id": domain,
                }
            )
            if row != representative:
                removed_words += words
    sql.close()
    os.replace(partial_database, database)
    groups_path = root / "near_groups.parquet"
    write_parquet_atomic(groups_path, output_rows, NEAR_SCHEMA)
    analysis = duckdb.connect(str(root / "analysis.duckdb"), read_only=True)
    affected_domains = analysis.execute(
        """
        SELECT count(DISTINCT e.domain_id)
        FROM read_parquet(?) n
        JOIN d3_membership d ON n.row_number=d.row_number
        JOIN d2_membership e USING(normalized_sha256)
        WHERE n.row_number != n.representative_row AND e.domain_id IS NOT NULL
        """,
        [str(groups_path)],
    ).fetchone()[0]
    analysis.close()
    summary = {
        "eligible_documents": len(metadata),
        "candidate_pairs": candidate_pairs,
        "confirmed_pairs": confirmed_pairs,
        "groups": sum(len(group) >= 2 for group in components.values()),
        "documents_removed_in_sensitivity": sum(
            len(group) - 1 for group in components.values() if len(group) >= 2
        ),
        "words_removed_in_sensitivity": removed_words,
        "affected_domains": affected_domains,
        "parameters": {
            "minimum_words": near.minimum_words,
            "shingle_words": near.shingle_words,
            "hashes": near.hashes,
            "bands": near.bands,
            "rows_per_band": near.rows_per_band,
            "jaccard": near.jaccard,
            "seed": near.seed,
        },
    }
    atomic_json(output, summary)
    return summary
