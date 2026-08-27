from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import tldextract

from .config import Config
from .io import decode_field, is_null, iter_bounded_tsv, parse_int
from .schema import DOMAIN_COLUMNS, WIKIMEDIA_HOSTS
from .storage import duckdb_connection, write_parquet_atomic

DOMAIN_INVENTORY_VERSION = 2

DOMAIN_SCHEMA = pa.schema(
    [
        ("row_number", pa.uint64()),
        ("id", pa.int64()),
        ("parent_domain_id", pa.int64()),
        ("recursion_level", pa.int32()),
        ("request_count", pa.int64()),
        ("host", pa.string()),
        ("registrable_domain", pa.string()),
        ("is_wikimedia", pa.bool_()),
    ]
)


def _host(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value if "://" in value else "//" + value
    return (urlsplit(candidate).hostname or "").lower().rstrip(".") or None


def _is_wikimedia(host: str | None) -> bool:
    return bool(
        host and any(host == suffix or host.endswith("." + suffix) for suffix in WIKIMEDIA_HOSTS)
    )


def inventory_domains(config: Config, manifest: dict, root: Path) -> dict:
    from .io import read_json

    output = root / "domains-v2.parquet"
    summary_path = root / "domain_summary-v2.json"
    summary = read_json(summary_path, {})
    source_sha256 = manifest["sources"]["domains"]["sha256"]
    if (
        output.exists()
        and summary.get("schema_version") == DOMAIN_INVENTORY_VERSION
        and summary.get("source_sha256") == source_sha256
    ):
        connection = _install_domain_table(config, root, output)
        connection.close()
        return summary
    extractor = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=str(root / "psl-cache"))
    rows: list[dict] = []
    errors = Counter()
    header = manifest["sources"]["domains"]["header"]
    with config.analysis_domains.open("rb") as handle:
        records = iter_bounded_tsv(
            handle,
            columns=len(DOMAIN_COLUMNS),
            max_line_bytes=config.runtime.max_line_bytes,
            max_rows=config.runtime.max_rows,
        )
        for record in records:
            if header and record.row_number == 1:
                continue
            if record.fields is None:
                errors[record.error or "structural"] += 1
                continue
            fields = record.fields
            domain_id = parse_int(fields[0])
            if domain_id is None:
                errors["invalid_id"] += 1
                continue
            host = _host(decode_field(fields[1]))
            parent_id = parse_int(fields[3])
            recursion_level = parse_int(fields[4])
            request_count = parse_int(fields[5])
            if host is None:
                errors["missing_host"] += 1
            if not is_null(fields[3]) and parent_id is None:
                errors["invalid_parent_domain_id"] += 1
            if not is_null(fields[4]) and recursion_level is None:
                errors["invalid_recursion_level"] += 1
            if not is_null(fields[5]) and request_count is None:
                errors["invalid_request_count"] += 1
            elif is_null(fields[5]):
                errors["missing_request_count"] += 1
            extracted = extractor(host or "")
            registrable = extracted.top_domain_under_public_suffix or host
            rows.append(
                {
                    "row_number": record.row_number,
                    "id": domain_id,
                    "parent_domain_id": parent_id,
                    "recursion_level": recursion_level,
                    "request_count": request_count,
                    "host": host,
                    "registrable_domain": registrable,
                    "is_wikimedia": _is_wikimedia(host),
                }
            )
    write_parquet_atomic(output, rows, DOMAIN_SCHEMA)
    connection = _install_domain_table(config, root, output)
    aggregate = connection.execute(
        """
        SELECT count(*) AS rows,
               count(DISTINCT d.id) AS distinct_ids,
               count(*) - count(DISTINCT d.id) AS duplicate_ids,
               count(*) FILTER (WHERE d.parent_domain_id IS NOT NULL AND p.id IS NULL) AS missing_parents,
               min(d.recursion_level) AS minimum_level,
               max(d.recursion_level) AS maximum_level,
               sum(d.request_count) AS requests,
               count(*) FILTER (WHERE d.is_wikimedia) AS wikimedia_domains
        FROM domains d LEFT JOIN (SELECT DISTINCT id FROM domains) p ON d.parent_domain_id = p.id
        """
    ).fetchone()
    connection.close()
    summary = {
        "schema_version": DOMAIN_INVENTORY_VERSION,
        "source_sha256": source_sha256,
        "rows": aggregate[0],
        "distinct_ids": aggregate[1],
        "duplicate_ids": aggregate[2],
        "missing_parents": aggregate[3],
        "minimum_level": aggregate[4],
        "maximum_level": aggregate[5],
        "requests": aggregate[6],
        "wikimedia_domains": aggregate[7],
        "errors": dict(errors),
    }
    from .io import atomic_json

    atomic_json(summary_path, summary)
    return summary


def _install_domain_table(config: Config, root: Path, output: Path):
    db = root / "analysis.duckdb"
    connection = duckdb_connection(db, config.runtime.memory_limit, root / "duckdb-tmp")
    escaped = str(output).replace("'", "''")
    connection.execute(
        f"CREATE OR REPLACE TABLE domains AS SELECT * FROM read_parquet('{escaped}')"
    )
    connection.execute("CREATE INDEX IF NOT EXISTS domains_id_idx ON domains(id)")
    return connection
