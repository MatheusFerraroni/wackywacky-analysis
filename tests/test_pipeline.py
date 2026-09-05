from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from conftest import domain_row, page_row, write_config, write_sources

from wackywacky_analysis.config import load_config
from wackywacky_analysis.errors import ReviewRequired, WackyWackyError
from wackywacky_analysis.io import atomic_json
from wackywacky_analysis.near import run_near_duplicates
from wackywacky_analysis.pipeline import refresh_reports, run_pipeline
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
    assert summary["domains"]["requests"] == 220
    assert summary["clean"]["documents_affected"] >= 6
    vocabulary = summary["lexical"]["vocabulary"]
    for value in vocabulary.values():
        assert value["document_occurrences"] <= value["occurrences"]
    assert (result / "figures" / "09_principais_palavras_bigramas.svg").is_file()
    assert summary["content"]["status"] == "complete"
    assert summary["content"]["metrics"]["documents"] == summary["clean"]["d3_unique"]
    assert (result / "tables" / "14_estrutura_sentencas_paragrafos.csv").is_file()
    assert (result / "tables" / "21_cobertura_vocabulario.csv").is_file()
    assert (result / "figures" / "10_estrutura_textual.svg").is_file()
    assert (result / "figures" / "17_cobertura_vocabulario.svg").is_file()
    with (result / "tables" / "03_status_explorados.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        explored = {row["status_explorado"] for row in csv.DictReader(handle)}
    assert explored == {"done", "failed", "blocked_language"}
    with (result / "tables" / "19_colocacoes.csv").open(encoding="utf-8", newline="") as handle:
        collocations = list(csv.DictReader(handle))
    assert all(
        -1 <= float(row["NPMI"]) <= 1 for row in collocations if row["ranking"] == "bigrama_NPMI"
    )
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


def test_refresh_migrates_domain_inventory_without_touching_scientific_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages, domains = write_sources(tmp_path)
    with domains.open("ab") as handle:
        handle.write(domain_row(3, b"https://child-a.invalid", parent=b"1", requests=0) + b"\n")
        handle.write(domain_row(4, b"https://child-b.invalid", parent=b"1", requests=0) + b"\n")
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
    import_review(config, root, sample)
    result = Path(run_pipeline(config, resume=True)["result"])
    children_csv = result / "tables" / "08b_dominios_filhos.csv"
    children_tex = children_csv.with_suffix(".tex")
    levels_csv = result / "tables" / "07b_estatisticas_dominios_por_nivel.csv"
    levels_tex = levels_csv.with_suffix(".tex")
    expected_children = children_csv.read_bytes()
    expected_levels = levels_csv.read_bytes()
    # Simulate a completed snapshot produced before this table existed.
    children_csv.unlink()
    children_tex.unlink()
    levels_csv.unlink()
    levels_tex.unlink()
    scientific = (
        "d1_groups.parquet",
        "d2_groups.parquet",
        "d3_groups.parquet",
        "vocabulary.parquet",
        "bigrams.parquet",
        "content_histograms.parquet",
    )
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in scientific}
    state_before = (root / "state.json").read_bytes()
    manifest_path = root / "manifest.json"
    legacy_manifest = json.loads(manifest_path.read_text())
    legacy_manifest["schema_version"] = 1
    legacy_manifest["sources"]["domains"]["header"] = False
    legacy_manifest.pop("migrations")
    atomic_json(manifest_path, legacy_manifest)
    (root / "domains.parquet").write_bytes((root / "domains-v2.parquet").read_bytes())
    (root / "domain_summary.json").write_text('{"schema_version":1,"requests":0}')
    (root / "domains-v2.parquet").unlink()
    (root / "domains-v2.parquet.sha256").unlink()
    (root / "domain_summary-v2.json").unlink()

    def forbid_text_processing(*args, **kwargs):
        pytest.fail("refresh must not repeat any textual processing")

    for name in (
        "scan_pages",
        "reduce_exact",
        "clean_representatives",
        "lexical_pass",
        "content_pass",
        "discover_v2_candidates",
        "clean_v2",
        "lexical_content_v2_pass",
    ):
        monkeypatch.setattr(f"wackywacky_analysis.pipeline.{name}", forbid_text_processing)
    refreshed = refresh_reports(config, manifest["snapshot_id"])
    assert refreshed["status"] == "refreshed"
    assert (result / "revisions" / "method-v1" / "summary.json").is_file()
    summary = json.loads((result / "summary.json").read_text())
    migrated_manifest = json.loads(manifest_path.read_text())
    assert migrated_manifest["schema_version"] == 2
    assert migrated_manifest["sources"]["domains"]["header"] is True
    assert [item["id"] for item in migrated_manifest["migrations"]] == ["domain-header-schema-v2"]
    assert summary["domains"]["schema_version"] == 2
    assert summary["domains"]["requests"] == 220
    assert children_csv.read_bytes() == expected_children
    assert children_tex.is_file()
    assert levels_csv.read_bytes() == expected_levels
    assert levels_tex.is_file()
    with children_csv.open(encoding="utf-8", newline="") as handle:
        children = list(csv.DictReader(handle))
    assert len(children) == 1
    assert children[0]["filhos"] == "2"
    assert float(children[0]["percentual"]) == 50
    assert float(children[0]["media_requisicoes_filhos"]) == 0
    audit = summary["domain_children"]
    assert audit["denominator_domains"] == 4
    assert audit["domains_without_parent"] == 1
    assert audit["domains_with_missing_parent"] == 1
    assert audit["domains_with_known_parent"] == 2
    assert audit["counts_reconciled"] is True
    level_audit = summary["domain_levels"]
    assert level_audit["denominator_domains"] == 4
    assert level_audit["denominator_requests"] == 220
    assert level_audit["domains_reconciled"] is True
    assert level_audit["requests_reconciled"] is True
    checksums = {
        name: digest
        for digest, name in (
            line.split("  ", 1) for line in (result / "checksums.sha256").read_text().splitlines()
        )
    }
    for path in (children_csv, children_tex, levels_csv, levels_tex):
        assert (
            checksums[path.relative_to(result).as_posix()]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    assert "nan" not in "\n".join(
        path.read_text(encoding="utf-8", errors="ignore").casefold()
        for path in result.rglob("*.csv")
    )
    after = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in scientific}
    assert before == after
    assert (root / "state.json").read_bytes() == state_before


def test_full_profile_worker_count_does_not_change_aggregates(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    summaries = []
    parquet_checksums = []
    for workers in (1, 2, 8):
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
        summaries.append(
            {key: summary[key] for key in ("domains", "exact", "clean", "lexical", "content")}
        )
        parquet_checksums.append(
            {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("vocabulary.parquet", "bigram_candidates.parquet", "bigrams.parquet")
            }
        )
    assert summaries[0] == summaries[1] == summaries[2]
    assert parquet_checksums[0] == parquet_checksums[1] == parquet_checksums[2]


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
