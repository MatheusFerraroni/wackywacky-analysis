from __future__ import annotations

import csv
import hashlib
import os
import signal
from concurrent.futures import Future
from pathlib import Path

import pytest
from conftest import write_config, write_sources

from wackywacky_analysis.clean import clean_representatives
from wackywacky_analysis.config import load_config
from wackywacky_analysis.errors import ReviewRequired, WackyWackyError
from wackywacky_analysis.io import atomic_json
from wackywacky_analysis.lexical import (
    BIGRAM_STATE_SCHEMA,
    LEXICAL_STATE_SCHEMA_VERSION,
    _load_or_reset_lexical_state,
    _OrderedBatchPool,
    lexical_pass,
    spacy_identity,
)
from wackywacky_analysis.pipeline import run_pipeline
from wackywacky_analysis.review import import_review
from wackywacky_analysis.snapshot import verify_snapshot
from wackywacky_analysis.storage import write_parquet_atomic


def _approve(path: Path) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    for row in rows:
        row["label"] = "boilerplate"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prepared(tmp_path: Path, name: str):
    pages, domains = write_sources(tmp_path / "sources")
    base = tmp_path / name
    config = load_config(
        write_config(
            base / "config.toml",
            pages,
            domains,
            base / "work",
            base / "results",
            profile="full",
        )
    )
    with pytest.raises(ReviewRequired):
        run_pipeline(config, resume=False)
    manifest = verify_snapshot(config)
    root = config.paths.work / manifest["snapshot_id"]
    review = root / "review" / "boilerplate-review.csv"
    _approve(review)
    import_review(config, root, review)
    assert clean_representatives(config, manifest, root).get("complete", True)
    return config, manifest, root


@pytest.mark.parametrize("phase", ["lexical", "recount"])
def test_interrupted_lexical_phases_resume_to_identical_public_parquets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    baseline_config, baseline_manifest, baseline_root = _prepared(tmp_path, "baseline")
    baseline_summary = lexical_pass(baseline_config, baseline_manifest, baseline_root)

    config, manifest, root = _prepared(tmp_path, f"interrupted-{phase}")
    original_submit = _OrderedBatchPool.submit
    interrupted = False

    def interrupt_once(self, batch, size):
        nonlocal interrupted
        result = original_submit(self, batch, size)
        if not interrupted and self.mode == phase:
            interrupted = True
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(_OrderedBatchPool, "submit", interrupt_once)
    partial = lexical_pass(config, manifest, root)
    assert partial["complete"] is False
    assert partial["phase"] == ("tokenization" if phase == "lexical" else "bigram_recount")
    monkeypatch.setattr(_OrderedBatchPool, "submit", original_submit)

    resumed = lexical_pass(config, manifest, root)
    assert resumed == baseline_summary
    for name in ("vocabulary.parquet", "bigram_candidates.parquet", "bigrams.parquet"):
        assert hashlib.sha256((root / name).read_bytes()).digest() == hashlib.sha256(
            (baseline_root / name).read_bytes()
        ).digest()


@pytest.mark.parametrize("kind", ["old", "bad-checksum"])
def test_only_incompatible_lexical_products_are_discarded(tmp_path: Path, kind: str) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config = load_config(
        write_config(
            tmp_path / "config.toml",
            pages,
            domains,
            tmp_path / "work",
            tmp_path / "results",
            profile="full",
        )
    )
    manifest = verify_snapshot(config)
    root = config.paths.work / manifest["snapshot_id"]
    root.mkdir(parents=True, exist_ok=True)
    clean_sentinel = root / "clean_summary.json"
    clean_sentinel.write_text('{"preserved": true}\n', encoding="utf-8")
    lexical_root = root / "lexical"
    lexical_root.mkdir()
    identity = spacy_identity(config)
    if kind == "old":
        (lexical_root / "documents-old.parquet.partial").write_bytes(b"old")
    else:
        bigram_state = lexical_root / "bigram-state-0.parquet"
        checksum = write_parquet_atomic(bigram_state, [], BIGRAM_STATE_SCHEMA)
        atomic_json(
            lexical_root / "state.json",
            {
                "schema_version": LEXICAL_STATE_SCHEMA_VERSION,
                "snapshot_id": manifest["snapshot_id"],
                "config_sha256": config.fingerprint,
                "content_fingerprint": config.content_fingerprint,
                "source_sha256": manifest["sources"]["pages"]["sha256"],
                "spacy": identity,
                "artifacts": {"lexical/bigram-state-0.parquet": checksum},
            },
        )
        bigram_state.with_suffix(".parquet.sha256").write_text("invalid\n", encoding="ascii")

    assert _load_or_reset_lexical_state(config, manifest, root, identity) == {}
    assert clean_sentinel.exists()
    assert not lexical_root.exists()


def test_worker_failure_is_reported_and_oversized_batch_runs_inline(tmp_path: Path) -> None:
    failed: Future = Future()
    failed.set_exception(ValueError("synthetic worker failure"))
    pool = _OrderedBatchPool.__new__(_OrderedBatchPool)
    pool.pending = __import__("collections").deque([(failed, 1)])
    pool.pending_bytes = 1
    with pytest.raises(WackyWackyError, match="synthetic worker failure"):
        pool._oldest()

    pages, domains = write_sources(tmp_path / "sources")
    config = load_config(
        write_config(
            tmp_path / "config.toml",
            pages,
            domains,
            tmp_path / "work",
            tmp_path / "results",
        )
    )
    with _OrderedBatchPool(config, mode="recount", wanted={"um\tdois"}) as inline:
        result = inline.submit(["um dois"], config.runtime.queue_bytes + 1)
    assert result[0]["counts"] == {"um\tdois": 1}
