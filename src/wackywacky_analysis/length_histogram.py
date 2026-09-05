from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from .config import Config
from .errors import WackyWackyError
from .io import atomic_json, read_json
from .pipeline import locate_result
from .progress import ItemProgress
from .reports import write_checksums

DEFAULT_MINIMUM = 2.0
DEFAULT_MAXIMUM = 9_999.0
DEFAULT_BINS = 100
OUTPUT_NAME = "histograma_tamanho_caracteres"


def _validate_parameters(minimum: float, maximum: float, bins: int) -> None:
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise WackyWackyError("limites do histograma devem ser finitos")
    if minimum < 0 or maximum <= minimum:
        raise WackyWackyError("histograma exige 0 <= mínimo < máximo")
    if not 1 <= bins <= 1_000_000:
        raise WackyWackyError("bins deve estar entre 1 e 1.000.000")


def _write_csv_atomic(path: Path, edges: np.ndarray, counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["left", "right", "mid", "count"])
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True):
            writer.writerow(
                [
                    float(left),
                    float(right),
                    float((left + right) / 2),
                    int(count),
                ]
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def export_character_histogram(
    config: Config,
    *,
    snapshot_id: str | None = None,
    minimum: float = DEFAULT_MINIMUM,
    maximum: float = DEFAULT_MAXIMUM,
    bins: int = DEFAULT_BINS,
) -> dict:
    """Export a bounded character-count histogram from completed B_clean_v2 Parquets."""
    _validate_parameters(minimum, maximum, bins)
    result = locate_result(config, snapshot_id)
    resolved_snapshot_id = result.name
    root = config.paths.work / resolved_snapshot_id

    state = read_json(root / "state.json", {})
    if state.get("stage") != "complete":
        raise WackyWackyError("snapshot ainda não está completo")

    summary = read_json(result / "summary.json", {})
    if summary.get("primary_view") != "B_clean_v2":
        raise WackyWackyError("resultado não possui B_clean_v2 como visão principal")

    lexical_summary = read_json(root / "v2" / "lexical_summary.json", {})
    expected_documents = lexical_summary.get("views", {}).get("B_clean_v2", {}).get("documents")
    if not isinstance(expected_documents, int) or expected_documents < 0:
        raise WackyWackyError("contagem lexical B_clean_v2 ausente ou inválida")

    documents_root = root / "v2" / "lexical" / "documents"
    files = sorted(documents_root.glob("*.parquet"))
    if not files:
        raise WackyWackyError(f"Parquets B_clean_v2 ausentes em {documents_root}")

    dataset = ds.dataset(files, format="parquet")
    if "characters" not in dataset.schema.names:
        raise WackyWackyError("coluna characters ausente nos Parquets B_clean_v2")
    character_type = dataset.schema.field("characters").type
    if not pa.types.is_integer(character_type):
        raise WackyWackyError("coluna characters não é inteira")

    edges = np.linspace(minimum, maximum, bins + 1)
    counts = np.zeros(bins, dtype=np.uint64)
    total = 0
    below = 0
    above = 0
    progress = ItemProgress("Histograma de caracteres", expected_documents)
    try:
        for batch in dataset.to_batches(columns=["characters"], batch_size=250_000):
            column = batch.column(0)
            if column.null_count:
                raise WackyWackyError("coluna characters contém valores ausentes")
            values = column.to_numpy(zero_copy_only=False)
            total += len(values)
            below += int(np.count_nonzero(values < minimum))
            above += int(np.count_nonzero(values > maximum))
            batch_counts, _ = np.histogram(values, bins=edges)
            counts += batch_counts.astype(np.uint64)
            progress.update(total)
    finally:
        progress.finish(detail=f"{total:,} documentos lidos")

    included = int(counts.sum())
    if total != expected_documents:
        raise WackyWackyError(
            "contagem dos Parquets não reconcilia com lexical_summary: "
            f"{total:,} != {expected_documents:,}"
        )
    if included + below + above != total:
        raise WackyWackyError("contagens do histograma não reconciliam")

    csv_path = result / "aggregates" / f"{OUTPUT_NAME}.csv"
    metadata_path = result / "aggregates" / f"{OUTPUT_NAME}.json"
    _write_csv_atomic(csv_path, edges, counts)
    metadata = {
        "snapshot_id": resolved_snapshot_id,
        "view": "B_clean_v2",
        "unit": "caracteres Unicode do texto normalizado e limpo",
        "bins": bins,
        "minimum": minimum,
        "maximum": maximum,
        "intervals": "[left,right), exceto o último intervalo [left,right]",
        "documents_total": total,
        "documents_in_histogram": included,
        "documents_below_minimum": below,
        "documents_above_maximum": above,
    }
    atomic_json(metadata_path, metadata)
    write_checksums(result)
    return {
        "status": "exported",
        "snapshot_id": resolved_snapshot_id,
        "csv": str(csv_path),
        "metadata": str(metadata_path),
        **metadata,
    }
