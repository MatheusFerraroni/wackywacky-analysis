from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

import zstandard as zstd

_HORIZONTAL_SPACE = re.compile(r"[^\S\r\n]+")
_MD5 = re.compile(r"[0-9a-fA-F]{32}\Z")
_DECOMPRESSOR = zstd.ZstdDecompressor()


@dataclass(frozen=True)
class DecodedText:
    raw: bytes
    text: str
    normalized: str
    raw_sha256: str
    normalized_sha256: str
    md5_class: str


class TextDecodeFailure(ValueError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(
        char for char in text if char in {"\n", "\t"} or unicodedata.category(char) != "Cc"
    )
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n")]
    output: list[str] = []
    empty = 0
    for line in lines:
        if line:
            empty = 0
            output.append(line)
        elif output and empty < 2:
            output.append("")
            empty += 1
    while output and not output[-1]:
        output.pop()
    return "\n".join(output)


def decode_text(field: bytes, md5_field: bytes, max_bytes: int) -> DecodedText:
    try:
        compressed = bytes.fromhex(field.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise TextDecodeFailure("invalid_hex") from exc
    try:
        declared = zstd.frame_content_size(compressed)
        if (
            declared not in {zstd.CONTENTSIZE_UNKNOWN, zstd.CONTENTSIZE_ERROR}
            and declared > max_bytes
        ):
            raise TextDecodeFailure("text_too_large")
        raw = _DECOMPRESSOR.decompress(compressed, max_output_size=max_bytes)
    except zstd.ZstdError as exc:
        raise TextDecodeFailure("invalid_zstd") from exc
    if len(raw) > max_bytes:
        raise TextDecodeFailure("text_too_large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TextDecodeFailure("invalid_utf8") from exc
    normalized = normalize_text(text)
    supplied = None if md5_field in {b"", b"NULL", b"\\N"} else md5_field.decode("ascii", "ignore")
    if supplied is None:
        md5_class = "missing"
    elif not _MD5.fullmatch(supplied):
        md5_class = "invalid"
    else:
        actual = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        md5_class = "match" if actual.lower() == supplied.lower() else "mismatch"
    return DecodedText(
        raw=raw,
        text=text,
        normalized=normalized,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        normalized_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        md5_class=md5_class,
    )


def classify_text_md5(field: bytes, md5_field: bytes, decoded: DecodedText) -> str:
    """Diagnose which stored representation a valid supplied MD5 describes."""
    if md5_field in {b"", b"NULL", b"\\N"}:
        return "missing"
    supplied = md5_field.decode("ascii", "ignore")
    if not _MD5.fullmatch(supplied):
        return "invalid"
    supplied = supplied.casefold()
    compressed = bytes.fromhex(field.decode("ascii"))
    variants = (
        ("decompressed_bytes", decoded.raw),
        ("compressed_bytes", compressed),
        ("hexadecimal_field", field),
        ("normalized_text", decoded.normalized.encode("utf-8")),
    )
    for label, value in variants:
        if hashlib.md5(value, usedforsecurity=False).hexdigest() == supplied:
            return label
    return "mismatch_all_representations"


def paragraph_units(text: str, minimum: int) -> list[tuple[int, int, str, str]]:
    units: list[tuple[int, int, str, str]] = []
    cursor = 0
    for paragraph in re.split(r"\n\s*\n+", text):
        start = text.find(paragraph, cursor)
        end = start + len(paragraph)
        cursor = end
        if len(paragraph) >= minimum:
            digest = hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
            units.append((start, end, digest, paragraph))
    return units


def block_units(text: str, lines_per_block: int, minimum: int) -> list[tuple[int, int, str, str]]:
    positions: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.split("\n"):
        start = cursor
        end = start + len(line)
        cursor = end + 1
        if line:
            positions.append((start, end, line))
    output: list[tuple[int, int, str, str]] = []
    for index in range(max(0, len(positions) - lines_per_block + 1)):
        group = positions[index : index + lines_per_block]
        value = "\n".join(item[2] for item in group)
        if len(value) >= minimum:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            output.append((group[0][0], group[-1][1], digest, value))
    return output


def remove_intervals(text: str, intervals: list[tuple[int, int]]) -> tuple[str, int]:
    if not intervals:
        return text, 0
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    pieces: list[str] = []
    cursor = 0
    removed = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        removed += end - start
        cursor = end
    pieces.append(text[cursor:])
    return normalize_text("".join(pieces)), removed
