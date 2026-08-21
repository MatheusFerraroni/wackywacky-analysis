from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from conftest import domain_row, page_row, write_config, write_sources

from wackywacky_analysis.config import load_config
from wackywacky_analysis.errors import ReviewRequired, WackyWackyError
from wackywacky_analysis.near import run_near_duplicates
from wackywacky_analysis.pipeline import run_pipeline
from wackywacky_analysis.review import export_review, import_review
from wackywacky_analysis.snapshot import verify_snapshot


def _label_review(path: Path) -> None:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        for row in reader:
            row["label"] = "boilerplate"
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_pipeline_review_resume_reports_and_public_invariants(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    with pytest.raises(ReviewRequired):
        run_pipeline(config, resume=False)
    manifest = verify_snapshot(config)
    root = config.paths.work / manifest["snapshot_id"]
    sample = root / "review" / "boilerplate-review.csv"
    with sample.open("r", encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    assert sample.read_text(encoding="utf-8").count("\n") == len(review_rows) + 1
    assert list(review_rows[0])[:2] == ["sample_id", "frequencia"]
    assert all(int(row["frequencia"]) >= 5 for row in review_rows)
    assert all(row["label"] == "boilerplate" for row in review_rows)
    assert all("\n" not in row["text_preview"] for row in review_rows)
    assert all("text" not in row for row in review_rows)
    assert (
        "altere somente as exceções"
        in (sample.parent / "LEIA-ME-revisao.txt").read_text(encoding="utf-8").casefold()
    )
    assert export_review(config, manifest, root) == sample
    _label_review(sample)
    with pytest.raises(WackyWackyError, match="foi alterado"):
        export_review(config, manifest, root)
    gate = import_review(config, root, sample)
    assert gate["status"] == "approved"
    completed = run_pipeline(config, resume=True)
    assert completed["status"] == "complete"
    result = Path(completed["result"])
    summary = json.loads((result / "summary.json").read_text())
    page_metrics = summary["exact"]["page_metrics"]
    assert sum(page_metrics["status"].values()) == page_metrics["rows"]
    assert (
        summary["exact"]["r_valid"]
        >= summary["exact"]["d1_unique"]
        >= summary["exact"]["d2_unique"]
    )
    assert summary["exact"]["d2_unique"] >= summary["clean"]["d3_unique"]
    assert page_metrics["md5_class"]["mismatch"] == 1
    assert page_metrics["md5_class"]["invalid"] == 1
    assert page_metrics["md5_class"]["missing"] == 1
    assert summary["exact"]["missing_domain_references"] == 1
    assert summary["clean"]["documents_affected"] >= 6
    vocabulary = summary["lexical"]["vocabulary"]
    for value in vocabulary.values():
        assert value["document_occurrences"] <= value["occurrences"]
    assert (result / "figures" / "09_principais_palavras_bigramas.svg").is_file()
    public = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in result.rglob("*")
        if path.is_file() and path.suffix in {".json", ".csv", ".tex"}
    )
    assert "https://example.invalid/" not in public
    assert "Cabeçalho institucional repetido" not in public
    checksum_before = hashlib.sha256((result / "summary.json").read_bytes()).hexdigest()
    rerun = run_pipeline(config, resume=True)
    assert rerun["status"] == "complete"
    assert hashlib.sha256((result / "summary.json").read_bytes()).hexdigest() == checksum_before
    near = run_near_duplicates(config, root)
    assert near["confirmed_pairs"] >= 1
    assert near["documents_removed_in_sensitivity"] >= 1


def test_full_profile_worker_count_does_not_change_aggregates(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    summaries = []
    for workers in (1, 2):
        base = tmp_path / f"run-{workers}"
        config = load_config(
            write_config(
                base / "config.toml",
                pages,
                domains,
                base / "work",
                base / "results",
                workers=workers,
                profile="full",
            )
        )
        with pytest.raises(ReviewRequired):
            run_pipeline(config, resume=False)
        manifest = verify_snapshot(config)
        root = config.paths.work / manifest["snapshot_id"]
        sample = root / "review" / "boilerplate-review.csv"
        _label_review(sample)
        import_review(config, root, sample)
        result = Path(run_pipeline(config, resume=True)["result"])
        summary = json.loads((result / "summary.json").read_text())
        summaries.append({key: summary[key] for key in ("domains", "exact", "clean", "lexical")})
    assert summaries[0] == summaries[1]


def test_official_portuguese_model_is_pinned_and_keeps_lemmatizer(tmp_path: Path) -> None:
    from dataclasses import replace

    from wackywacky_analysis.lexical import spacy_identity

    pages, domains = write_sources(tmp_path)
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    config = replace(config, lexical=replace(config.lexical, spacy_model="pt_core_news_sm"))
    identity = spacy_identity(config)
    assert identity["model_version"] == "3.8.0"
    assert "lemmatizer" in identity["pipeline"]
    assert "parser" not in identity["pipeline"]
    assert "ner" not in identity["pipeline"]


def test_cross_domain_repetition_is_measured_without_becoming_candidate(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path)
    with domains.open("ab") as handle:
        handle.write(domain_row(3, b"https://third.invalid", level=1) + b"\n")
    shared = "Trecho sintético republicado entre domínios para medir conteúdo compartilhado. " * 2
    additions = [
        page_row(100 + index, domain, "done", text=f"{shared}\n\nfinal único {index}")
        for index, domain in enumerate((1, 2, 3, 1, 2))
    ]
    with pages.open("ab") as handle:
        handle.write(b"\n".join(additions) + b"\n")
    config = load_config(
        write_config(
            tmp_path / "config.toml", pages, domains, tmp_path / "work", tmp_path / "results"
        )
    )
    with pytest.raises(ReviewRequired):
        run_pipeline(config, resume=False)
    manifest = verify_snapshot(config)
    exact = json.loads(
        (config.paths.work / manifest["snapshot_id"] / "exact_summary.json").read_text()
    )
    assert exact["cross_domain_repeated_units"] >= 1
