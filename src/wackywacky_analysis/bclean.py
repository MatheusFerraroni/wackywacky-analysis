from __future__ import annotations

import sqlite3
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .io import BinaryRecord, iter_bounded_tsv, parse_int
from .noise import residual_units
from .schema import PAGES_COLUMNS
from .storage import SortedMembership
from .text import (
    TextDecodeFailure,
    block_units,
    classify_text_md5,
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
    md5_diagnostic: str | None = None


class CandidateIndex:
    """Compact in-memory lookup used by repeated source passes."""

    def __init__(self, rows: Iterator[tuple[int, str, str]]) -> None:
        self.values: dict[tuple[int, str], set[str]] = defaultdict(set)
        for domain_id, kind, digest in rows:
            self.values[(domain_id, kind)].add(digest)

    @classmethod
    def from_path(cls, path: Path) -> CandidateIndex:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return cls(iter(connection.execute("SELECT domain_id, kind, digest FROM candidate")))
        finally:
            connection.close()

    def contains(self, domain_id: int, kind: str, digest: str) -> bool:
        return digest in self.values.get((domain_id, kind), ())

    def has_kind(self, domain_id: int, kind: str) -> bool:
        return bool(self.values.get((domain_id, kind)))


def _contains(candidates, domain_id: int, kind: str, digest: str) -> bool:
    if isinstance(candidates, CandidateIndex):
        return candidates.contains(domain_id, kind, digest)
    return bool(
        candidates.execute(
            "SELECT 1 FROM candidate WHERE domain_id=? AND kind=? AND digest=?",
            (domain_id, kind, digest),
        ).fetchone()
    )


def _has_residual_candidates(candidates, domain_id: int) -> bool:
    if isinstance(candidates, CandidateIndex):
        return any(
            candidates.has_kind(domain_id, kind)
            for kind in ("mediawiki", "short_line", "short_pair")
        )
    return bool(
        candidates.execute(
            "SELECT 1 FROM candidate WHERE domain_id=? "
            "AND kind IN ('mediawiki','short_line','short_pair') LIMIT 1",
            (domain_id,),
        ).fetchone()
    )


def clean_again(
    config: Config,
    candidates: sqlite3.Connection | CandidateIndex,
    domain_id: int | None,
    normalized: str,
) -> str:
    if domain_id is None:
        return normalized
    paragraph_intervals: list[tuple[int, int]] = []
    for start, end, unit, _text in paragraph_units(
        normalized, config.boilerplate.paragraph_min_chars
    ):
        if _contains(candidates, domain_id, "paragraph", unit):
            paragraph_intervals.append((start, end))
    intervals = list(paragraph_intervals)
    for start, end, unit, _text in block_units(
        normalized, config.boilerplate.block_lines, config.boilerplate.block_min_chars
    ):
        if any(start < p_end and end > p_start for p_start, p_end in paragraph_intervals):
            continue
        if _contains(candidates, domain_id, "block", unit):
            intervals.append((start, end))
    clean = remove_intervals(normalized, intervals)[0]
    if not clean or not _has_residual_candidates(candidates, domain_id):
        return clean
    residual_intervals: list[tuple[int, int]] = []
    is_wikimedia = (
        candidates.has_kind(domain_id, "mediawiki")
        if isinstance(candidates, CandidateIndex)
        else bool(
            candidates.execute(
                "SELECT 1 FROM candidate WHERE domain_id=? AND kind='mediawiki' LIMIT 1",
                (domain_id,),
            ).fetchone()
        )
    )
    for unit in residual_units(clean, config.boilerplate_v2, is_wikimedia=is_wikimedia):
        if _contains(candidates, domain_id, unit.kind, unit.digest):
            residual_intervals.append((unit.start, unit.end))
    return remove_intervals(clean, residual_intervals)[0]


def iter_bclean(
    config: Config,
    root: Path,
    *,
    start_offset: int = 0,
    start_row: int = 0,
    diagnose_md5: bool = False,
) -> Iterator[BCleanRecord]:
    """Yield the deterministic D3 representatives without persisting their text."""
    membership = SortedMembership(root / "d3_representatives.u64")
    candidates = CandidateIndex.from_path(root / "boilerplate.sqlite")
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
                md5_diagnostic=(
                    classify_text_md5(fields[13], fields[15], decoded) if diagnose_md5 else None
                ),
            )
