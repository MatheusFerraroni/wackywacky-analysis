from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

from .config import Config
from .io import atomic_json
from .storage import duckdb_connection, duckdb_copy_atomic, write_u64


def _merge_metrics(paths: list[str]) -> dict:
    total: dict = {}
    for raw_path in paths:
        value = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        for key, item in value.items():
            if isinstance(item, dict):
                counter = Counter(total.get(key, {}))
                counter.update(item)
                total[key] = dict(counter)
            else:
                total[key] = total.get(key, 0) + item
    return total


def reduce_exact(config: Config, root: Path) -> dict:
    output = root / "exact_summary.json"
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))
    feature_glob = str(root / "pages" / "chunks" / "*-features.parquet").replace("'", "''")
    unit_glob = str(root / "pages" / "chunks" / "*-units.parquet").replace("'", "''")
    inventory_glob = str(root / "pages" / "chunks" / "*-inventory.parquet").replace("'", "''")
    metrics = _merge_metrics(glob.glob(str(root / "pages" / "chunks" / "*-metrics.json")))
    connection = duckdb_connection(
        root / "analysis.duckdb", config.runtime.memory_limit, root / "duckdb-tmp"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW page_features AS SELECT * FROM read_parquet('{feature_glob}')"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW page_units AS SELECT * FROM read_parquet('{unit_glob}')"
    )
    connection.execute(
        f"CREATE OR REPLACE VIEW page_inventory AS SELECT * FROM read_parquet('{inventory_glob}')"
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE d2_membership AS
        SELECT *, row_number() OVER (
          PARTITION BY normalized_sha256
          ORDER BY page_id IS NULL, page_id, row_number
        ) = 1 AS is_representative
        FROM page_features
        """
    )
    d1_path = root / "d1_groups.parquet"
    d2_path = root / "d2_groups.parquet"
    reps_path = root / "d2_representatives.parquet"
    candidates_path = root / "boilerplate_candidates.parquet"
    cross_path = root / "cross_domain_units.parquet"
    duckdb_copy_atomic(
        connection,
        """
        SELECT raw_sha256, count(*) AS pages, count(DISTINCT domain_id) AS domains,
               min(row_number) AS first_row
        FROM page_features GROUP BY raw_sha256
        """,
        d1_path,
    )
    duckdb_copy_atomic(
        connection,
        """
        SELECT normalized_sha256, count(*) AS pages, count(DISTINCT domain_id) AS domains,
               min(row_number) FILTER (WHERE is_representative) AS representative_row
        FROM d2_membership GROUP BY normalized_sha256
        """,
        d2_path,
    )
    duckdb_copy_atomic(
        connection,
        """
        SELECT row_number, page_id, domain_id, normalized_sha256, characters, raw_bytes,
               recursion_level
        FROM d2_membership WHERE is_representative ORDER BY row_number
        """,
        reps_path,
    )
    rows = connection.execute(
        "SELECT row_number FROM d2_membership WHERE is_representative ORDER BY row_number"
    ).fetchall()
    write_u64(root / "d2_representatives.u64", (row[0] for row in rows))
    connection.execute(
        """
        CREATE OR REPLACE TABLE d2_domain_documents AS
        SELECT domain_id, count(DISTINCT normalized_sha256) AS documents
        FROM d2_membership WHERE domain_id IS NOT NULL GROUP BY domain_id
        """
    )
    connection.execute(
        """
        CREATE OR REPLACE TABLE d2_domain_units AS
        SELECT DISTINCT m.normalized_sha256, u.domain_id, u.kind, u.unit_sha256, u.characters
        FROM page_units u JOIN d2_membership m USING (row_number)
        WHERE u.domain_id IS NOT NULL
        """
    )
    threshold = "greatest(5, least(100, ceil(0.001 * d.documents)))"
    duckdb_copy_atomic(
        connection,
        f"""
        SELECT u.domain_id, u.kind, u.unit_sha256, max(u.characters) AS characters,
               count(DISTINCT u.normalized_sha256) AS document_frequency,
               d.documents AS domain_documents,
               CAST({threshold} AS BIGINT) AS threshold
        FROM d2_domain_units u JOIN d2_domain_documents d USING (domain_id)
        GROUP BY u.domain_id, u.kind, u.unit_sha256, d.documents
        HAVING count(DISTINCT u.normalized_sha256) >= {threshold}
        """,
        candidates_path,
    )
    duckdb_copy_atomic(
        connection,
        """
        SELECT kind, unit_sha256, count(DISTINCT domain_id) AS domains,
               count(DISTINCT normalized_sha256) AS documents,
               max(characters) AS characters
        FROM d2_domain_units GROUP BY kind, unit_sha256
        HAVING count(DISTINCT domain_id) >= 3 AND count(DISTINCT normalized_sha256) >= 5
        """,
        cross_path,
    )
    aggregate = connection.execute(
        """
        SELECT
          (SELECT count(*) FROM page_features),
          (SELECT count(DISTINCT raw_sha256) FROM page_features),
          (SELECT count(DISTINCT normalized_sha256) FROM page_features),
          (SELECT count(*) FROM read_parquet(?)),
          (SELECT count(*) FROM read_parquet(?)),
          (SELECT count(*) FROM page_inventory f
             LEFT JOIN (SELECT DISTINCT id FROM domains) d ON f.domain_id=d.id
             WHERE f.domain_id IS NOT NULL AND d.id IS NULL)
        """,
        [str(candidates_path), str(cross_path)],
    ).fetchone()
    d1_cluster = connection.execute(
        """
        SELECT count(*) FILTER (WHERE pages > 1), max(pages),
               count(*) FILTER (WHERE pages > 1 AND domains > 1)
        FROM read_parquet(?)
        """,
        [str(d1_path)],
    ).fetchone()
    d2_cluster = connection.execute(
        """
        SELECT
          count(*) FILTER (WHERE pages > 1),
          max(pages),
          count(*) FILTER (WHERE pages > 1 AND domains > 1)
        FROM read_parquet(?)
        """,
        [str(d2_path)],
    ).fetchone()
    same_as = connection.execute(
        """
        WITH inventory_targets AS (
          SELECT page_id FROM page_inventory WHERE page_id IS NOT NULL GROUP BY page_id
        ), valid_targets AS (
          SELECT * EXCLUDE(rn) FROM (
            SELECT *, row_number() OVER(PARTITION BY page_id ORDER BY row_number) rn
            FROM page_features WHERE page_id IS NOT NULL
          ) WHERE rn=1
        )
        SELECT count(*) AS declared,
               count(*) FILTER (WHERE it.page_id IS NOT NULL) AS target_present,
               count(*) FILTER (WHERE it.page_id IS NULL) AS target_missing,
               count(*) FILTER (WHERE source.row_number IS NOT NULL AND target.row_number IS NOT NULL)
                 AS both_valid,
               count(*) FILTER (WHERE source.raw_sha256=target.raw_sha256) AS d1_agreement,
               count(*) FILTER (WHERE source.normalized_sha256=target.normalized_sha256)
                 AS d2_agreement
        FROM page_inventory inventory
        LEFT JOIN inventory_targets it ON inventory.same_as=it.page_id
        LEFT JOIN page_features source USING(row_number)
        LEFT JOIN valid_targets target ON inventory.same_as=target.page_id
        WHERE inventory.same_as IS NOT NULL
        """
    ).fetchone()
    summary = {
        "page_metrics": metrics,
        "r_valid": aggregate[0],
        "d1_unique": aggregate[1],
        "d2_unique": aggregate[2],
        "boilerplate_candidates": aggregate[3],
        "cross_domain_repeated_units": aggregate[4],
        "missing_domain_references": aggregate[5],
        "same_as_declared": same_as[0],
        "same_as_target_present": same_as[1],
        "same_as_target_missing": same_as[2],
        "same_as_both_valid": same_as[3],
        "same_as_d1_agreement": same_as[4],
        "same_as_d2_agreement": same_as[5],
        "d1_duplicate_groups": d1_cluster[0],
        "d1_largest_group": d1_cluster[1] or 0,
        "d1_cross_domain_groups": d1_cluster[2],
        "d2_duplicate_groups": d2_cluster[0],
        "d2_largest_group": d2_cluster[1] or 0,
        "d2_cross_domain_groups": d2_cluster[2],
    }
    connection.close()
    atomic_json(output, summary)
    return summary
