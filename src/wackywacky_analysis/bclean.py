from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .io import BinaryRecord, iter_bounded_tsv, parse_int
from .schema import PAGES_COLUMNS
from .storage import SortedMembership
from .text import (
    TextDecodeFailure,
    block_units,
    decode_text,
    paragraph_units,
    remove_intervals,
)


@dataclass(frozen=True)
class BCleanRecord:
    source: BinaryRecord
    domain_id: int | None
    recursion_level: int | None
    text: str


def clean_again(
    config: Config,
    candidates: sqlite3.Connection,
    domain_id: int | None,
    normalized: str,
) -> str:
    if domain_id is None:
        return normalized
    paragraph_intervals: list[tuple[int, int]] = []
    for start, end, unit, _text in paragraph_units(
        normalized, config.boilerplate.paragraph_min_chars
    ):
        if candidates.execute(
            "SELECT 1 FROM candidate WHERE domain_id=? AND kind='paragraph' AND digest=?",
            (domain_id, unit),
        ).fetchone():
            paragraph_intervals.append((start, end))
    intervals = list(paragraph_intervals)
    for start, end, unit, _text in block_units(
        normalized, config.boilerplate.block_lines, config.boilerplate.block_min_chars
    ):
        if any(start < p_end and end > p_start for p_start, p_end in paragraph_intervals):
            continue
        if candidates.execute(
            "SELECT 1 FROM candidate WHERE domain_id=? AND kind='block' AND digest=?",
            (domain_id, unit),
        ).fetchone():
            intervals.append((start, end))
    return remove_intervals(normalized, intervals)[0]


def iter_bclean(
    config: Config,
    root: Path,
    *,
    start_offset: int = 0,
    start_row: int = 0,
) -> Iterator[BCleanRecord]:
    """Yield the deterministic D3 representatives without persisting their text."""
    membership = SortedMembership(root / "d3_representatives.u64")
    candidates = sqlite3.connect(f"file:{root / 'boilerplate.sqlite'}?mode=ro", uri=True)
    try:
        with config.analysis_pages.open("rb") as handle:
            for record in iter_bounded_tsv(
                handle,
                columns=len(PAGES_COLUMNS),
                max_line_bytes=config.runtime.max_line_bytes,
                start_offset=start_offset,
                start_row=start_row,
                max_rows=config.runtime.max_rows,
            ):
                if record.fields is None or not membership.contains(record.row_number):
                    continue
                fields = record.fields
                try:
                    decoded = decode_text(fields[13], fields[15], config.runtime.max_text_bytes)
                except TextDecodeFailure as exc:
                    raise RuntimeError("representante D3 deixou de ser decodificável") from exc
                domain_id = parse_int(fields[1])
                clean = clean_again(config, candidates, domain_id, decoded.normalized)
                if not clean:
                    raise RuntimeError("representante D3 ficou vazio ao ser reconstruído")
                yield BCleanRecord(
                    source=record,
                    domain_id=domain_id,
                    recursion_level=parse_int(fields[10]),
                    text=clean,
                )
    finally:
        candidates.close()
