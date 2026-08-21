from __future__ import annotations

import hashlib
from pathlib import Path

import zstandard as zstd

from wackywacky_analysis.schema import DOMAIN_COLUMNS, PAGES_COLUMNS


def encoded(text: str) -> tuple[bytes, bytes]:
    raw = text.encode("utf-8")
    return zstd.ZstdCompressor(level=1).compress(raw).hex().encode(), hashlib.md5(
        raw, usedforsecurity=False
    ).hexdigest().encode()


def domain_row(
    domain_id: int,
    url: bytes,
    *,
    parent: bytes = b"NULL",
    level: int = 0,
    requests: int = 10,
) -> bytes:
    return b"\t".join(
        [
            str(domain_id).encode(),
            url,
            b"\xff\x00binary-md5",
            parent,
            str(level).encode(),
            b"active",
            str(requests).encode(),
            b"2026-08-20",
            b"2026-08-20",
        ]
    )


def page_row(
    page_id: int,
    domain_id: int,
    status: str,
    *,
    text: str | None = None,
    text_field: bytes | None = None,
    text_md5: bytes | None = None,
    same_as: int | None = None,
    level: int = 0,
) -> bytes:
    if text is not None:
        encoded_text, encoded_md5 = encoded(text)
        text_field = text_field or encoded_text
        text_md5 = text_md5 or encoded_md5
    fields = [b"NULL"] * len(PAGES_COLUMNS)
    fields[0] = str(page_id).encode()
    fields[1] = str(domain_id).encode()
    fields[3] = str(same_as).encode() if same_as is not None else b"NULL"
    fields[4] = f"https://example.invalid/{page_id}".encode()
    fields[8] = b"200" if status == "done" else b"NULL"
    fields[10] = str(level).encode()
    fields[11] = status.encode()
    fields[12] = b"0"
    fields[13] = text_field if text_field is not None else b"NULL"
    fields[15] = text_md5 if text_md5 is not None else b"NULL"
    return b"\t".join(fields)


def write_sources(root: Path, *, repeated_documents: int = 6) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    domains = root / "domain.tsv"
    pages = root / "pages.tsv"
    domains.write_bytes(
        b"\t".join(name.encode() for name in DOMAIN_COLUMNS)
        + b"\n"
        + domain_row(1, b"https://pt.wikipedia.org", requests=200)
        + b"\n"
        + domain_row(2, b"https://notwiki.example", parent=b"999", level=1, requests=20)
        + b"\n"
    )
    boilerplate = "Cabeçalho institucional repetido para navegação e direitos reservados. " * 2
    rows = [
        page_row(1, 1, "pending"),
        page_row(2, 1, "failed"),
        page_row(3, 1, "blocked_language"),
    ]
    for index in range(repeated_documents):
        rows.append(
            page_row(
                10 + index,
                1,
                "done",
                text=f"{boilerplate}\n\nConteúdo sintético número {index} com palavras para o teste.",
                level=index % 2,
            )
        )
    rows.extend(
        [
            page_row(30, 2, "done", text="Texto exatamente duplicado com várias palavras úteis."),
            page_row(
                31,
                2,
                "done",
                text="Texto exatamente duplicado com várias palavras úteis.",
                same_as=30,
            ),
            page_row(32, 2, "done", text="Linha com espaço   repetido.\r\nOutra linha."),
            page_row(33, 2, "done", text="Linha com espaço repetido.\nOutra linha."),
            page_row(34, 99, "done", text="Referência com domínio ausente na fotografia parcial."),
            page_row(35, 1, "done", text_field=b"not-hex", text_md5=b"bad"),
            page_row(
                36, 1, "done", text="MD5 divergente mas texto ainda válido.", text_md5=b"0" * 32
            ),
            page_row(37, 1, "done", text="MD5 inválido mas texto ainda válido.", text_md5=b"bad"),
            page_row(38, 1, "done", text="MD5 ausente mas texto ainda válido.", text_md5=b"NULL"),
            page_row(39, 1, "done", text=""),
        ]
    )
    unique_words = [f"palavra{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(100)]
    near_variant = list(unique_words)
    near_variant[50] = "termodiferente"
    rows.append(page_row(40, 2, "done", text=" ".join(unique_words)))
    rows.append(page_row(41, 2, "done", text=" ".join(near_variant)))
    pages.write_bytes(
        b"\t".join(name.encode() for name in PAGES_COLUMNS) + b"\n" + b"\n".join(rows) + b"\n"
    )
    return pages, domains


def write_config(
    path: Path,
    pages: Path,
    domains: Path,
    work: Path,
    results: Path,
    *,
    workers: int = 1,
    profile: str = "tiny",
    sampling: bool = False,
    sampling_target: int = 10,
    sampling_candidates: int = 20,
    sampling_windows: int = 4,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_rows = 0 if sampling or profile == "full" else 10_000
    sampling_block = (
        f"""
[sampling]
enabled = true
target_rows = {sampling_target}
candidate_rows = {sampling_candidates}
windows = {sampling_windows}
window_bytes = 1048576
done_fraction = 0.80
same_as_reserve = 2
seed = 73129
"""
        if sampling
        else ""
    )
    path.write_text(
        f"""
schema_version = 1
cutoff_date = "2026-08-20"
profile = "{profile}"

[paths]
pages = "{pages}"
domains = "{domains}"
work = "{work}"
results = "{results}"

[runtime]
workers = {workers}
memory_limit = "640MiB"
minimum_scratch_free_bytes = 0
chunk_bytes = 1024
max_line_bytes = 1048576
max_text_bytes = 1048576
max_rows = {max_rows}
queue_bytes = 1048576

[source]
header = "auto"
require_immutable = true

{sampling_block}

[boilerplate]
enabled = true
paragraph_min_chars = 80
block_lines = 3
block_min_chars = 60
review_paragraphs = 100
review_blocks = 100
review_seed = 73129
precision_min = 0.0
wilson_lower_min = 0.0

[lexical]
spacy_model = "blank:pt"
partitions = 256
spill_terms = 50
bigram_candidates = 1000
published_items = 100
figure_items = 30

[near_duplicates]
enabled = false
minimum_words = 5
shingle_words = 5
hashes = 112
bands = 14
rows_per_band = 8
jaccard = 0.85
seed = 73129
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path
