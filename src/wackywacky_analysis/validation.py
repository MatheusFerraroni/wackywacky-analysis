from __future__ import annotations

from itertools import pairwise

import duckdb

from .errors import WackyWackyError
from .io import atomic_json


def validate_invariants(exact: dict, clean: dict, lexical: dict, root) -> dict:
    metrics = exact["page_metrics"]
    if sum(metrics["status"].values()) != metrics["rows"]:
        raise WackyWackyError("invariante falhou: status não reconciliam com linhas")
    funnel = [
        metrics["done"],
        metrics["text_present"],
        metrics["decoded"],
        exact["r_valid"],
        exact["d1_unique"],
        exact["d2_unique"],
        clean["b_clean_nonempty"],
        clean["d3_unique"],
    ]
    if any(left < right for left, right in pairwise(funnel)):
        raise WackyWackyError("invariante falhou: funil não é monotônico")
    words = {view: value["words"] for view, value in lexical["views"].items()}
    for key, value in lexical["vocabulary"].items():
        view, kind = key.split(":", 1)
        if kind in {"form", "lemma"} and value["occurrences"] != words[view]:
            raise WackyWackyError(f"invariante falhou: palavras não reconciliam em {key}")
    connection = duckdb.connect(str(root / "analysis.duckdb"), read_only=True)
    invalid_df = connection.execute(
        "SELECT count(*) FROM read_parquet(?) WHERE document_frequency > total_frequency",
        [str(root / "vocabulary.parquet")],
    ).fetchone()[0]
    counts = connection.execute(
        """
        SELECT (SELECT count(*) FROM page_inventory),
               (SELECT count(*) FROM d2_membership WHERE is_representative),
               (SELECT count(*) FROM d3_membership WHERE is_representative),
               (SELECT count(*) FROM document_statistics WHERE view='B_clean')
        """
    ).fetchone()
    connection.close()
    structural = sum(
        metrics["errors"].get(key, 0) for key in ("line_too_large", "wrong_column_count")
    )
    if counts[0] + structural != metrics["rows"]:
        raise WackyWackyError("invariante falhou: inventário de páginas não reconcilia")
    if (
        counts[1] != exact["d2_unique"]
        or counts[2] != clean["d3_unique"]
        or counts[3] != clean["d3_unique"]
    ):
        raise WackyWackyError("invariante falhou: representantes não reconciliam")
    if invalid_df:
        raise WackyWackyError("invariante falhou: frequência documental excede total")
    result = {
        "status_rows_reconciled": True,
        "funnel_monotonic": True,
        "vocabulary_reconciled": True,
        "document_frequency_valid": True,
        "representatives_reconciled": True,
    }
    atomic_json(root / "invariants.json", result)
    return result
