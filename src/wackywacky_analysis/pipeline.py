from __future__ import annotations

from pathlib import Path

from .clean import clean_representatives
from .config import Config
from .content import content_pass
from .domains import inventory_domains
from .errors import SourceChangedError, WackyWackyError
from .io import atomic_json, read_json
from .lexical import lexical_pass
from .lexical_v2 import lexical_content_v2_pass
from .pages import scan_pages
from .progress import logged_stage
from .reduction import reduce_exact
from .reports import archive_method_v1, build_reports, write_checksums
from .review import require_approved_review
from .snapshot import assert_snapshot, verify_snapshot
from .v2 import clean_v2, discover_v2_candidates, require_v2_confirmation
from .validation import validate_invariants, validate_v2_invariants


def _stage(state_path: Path, state: dict, name: str) -> None:
    disk = read_json(state_path, {})
    disk.update(state)
    disk["stage"] = name
    state.clear()
    state.update(disk)
    atomic_json(state_path, disk)


def run_pipeline(config: Config, *, resume: bool) -> dict:
    total = 16 if config.boilerplate_v2.enabled else 12

    def label(index: int, name: str) -> str:
        return f"[{index}/{total}] {name}"

    with logged_stage(label(1, "Identidade do snapshot")):
        manifest = verify_snapshot(config)
    if resume:
        for previous_path in config.paths.work.glob("*/manifest.json"):
            previous = read_json(previous_path, {})
            if (
                previous.get("config_sha256") == config.fingerprint
                and previous.get("cutoff_date") == config.cutoff_date
                and previous.get("snapshot_id") != manifest["snapshot_id"]
                and (previous_path.parent / "state.json").exists()
            ):
                raise SourceChangedError(
                    "a fonte mudou desde a execução iniciada; altere a data de corte para novo snapshot"
                )
    root = config.paths.work / manifest["snapshot_id"]
    state_path = root / "state.json"
    existing = read_json(state_path, {})
    if existing and not resume and existing.get("stage") != "complete":
        raise WackyWackyError("execução parcial existente; use --resume")
    if existing and (
        existing.get("snapshot_id") not in {None, manifest["snapshot_id"]}
        or existing.get("config_sha256") not in {None, config.fingerprint}
    ):
        raise WackyWackyError("estado incompatível com snapshot/configuração")
    state = existing or {
        "snapshot_id": manifest["snapshot_id"],
        "config_sha256": config.fingerprint,
        "stage": "verified",
    }
    atomic_json(state_path, state)
    with logged_stage(label(2, "Inventário de domínios")):
        domain_summary = inventory_domains(config, manifest, root)
        assert_snapshot(config, manifest)
    _stage(state_path, state, "domains")
    with logged_stage(label(3, "Leitura e validação de páginas")):
        scan = scan_pages(config, manifest, root, resume=resume)
    if not scan.get("complete"):
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    _stage(state_path, state, "pages")
    with logged_stage(label(4, "Deduplicação exata D1/D2")):
        exact = reduce_exact(config, root)
        assert_snapshot(config, manifest)
    _stage(state_path, state, "exact")
    with logged_stage(label(5, "Gate da revisão privada")):
        gate = require_approved_review(config, root)
    state["review_gate"] = gate
    _stage(state_path, state, "review_approved")
    with logged_stage(label(6, "Limpeza e deduplicação D3")):
        clean = clean_representatives(config, manifest, root)
    if clean.get("complete") is False:
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    assert_snapshot(config, manifest)
    _stage(state_path, state, "clean")
    with logged_stage(label(7, "Estatísticas lexicais")):
        lexical = lexical_pass(config, manifest, root)
    if lexical.get("complete") is False:
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    assert_snapshot(config, manifest)
    _stage(state_path, state, "lexical")
    with logged_stage(label(8, "Estrutura e conteúdo textual")):
        content = content_pass(config, manifest, root)
    if content.get("complete") is False:
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    _stage(state_path, state, "content")
    clean_v2_summary = lexical_v2 = content_v2 = None
    if config.boilerplate_v2.enabled:
        with logged_stage(label(9, "Descoberta de candidatos B_clean_v2")):
            discovery_v2 = discover_v2_candidates(config, manifest, root)
        if discovery_v2.get("complete") is False:
            _stage(state_path, state, "interrupted_v2_discovery")
            return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
        assert_snapshot(config, manifest)
        _stage(state_path, state, "v2_discovery")
        with logged_stage(label(10, "Confirmação privada B_clean_v2")):
            confirmation_v2 = require_v2_confirmation(config, manifest, root)
        state["review_gate_v2"] = confirmation_v2
        _stage(state_path, state, "v2_confirmed")
        with logged_stage(label(11, "Limpeza B_clean_v2 e deduplicação D4")):
            clean_v2_summary = clean_v2(config, manifest, root)
        if clean_v2_summary.get("complete") is False:
            _stage(state_path, state, "interrupted_v2_clean")
            return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
        assert_snapshot(config, manifest)
        _stage(state_path, state, "v2_clean")
        with logged_stage(label(12, "Léxico e conteúdo B_clean_v2")):
            lexical_v2, content_v2 = lexical_content_v2_pass(
                config, manifest, root, clean_v2_summary
            )
        if lexical_v2.get("complete") is False or content_v2.get("complete") is False:
            _stage(state_path, state, "interrupted_v2_lexical")
            return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
        assert_snapshot(config, manifest)
        _stage(state_path, state, "v2_lexical")
    validation_stage = 13 if config.boilerplate_v2.enabled else 9
    with logged_stage(label(validation_stage, "Validação de invariantes")):
        validate_invariants(exact, clean, lexical, root, content=content)
        if clean_v2_summary and lexical_v2 and content_v2:
            validate_v2_invariants(clean_v2_summary, lexical_v2, content_v2, root)
    _stage(state_path, state, "validated")
    report_stage = validation_stage + 1
    with logged_stage(label(report_stage, "Tabelas e agregados")):
        if clean_v2_summary:
            archive_method_v1(config.paths.results / manifest["snapshot_id"])
        result = build_reports(
            config,
            manifest,
            domain_summary,
            exact,
            clean,
            lexical,
            content,
            root,
            clean_v2=clean_v2_summary,
            lexical_v2=lexical_v2,
            content_v2=content_v2,
        )
    from .render import render_all

    with logged_stage(label(report_stage + 1, "Figuras")):
        render_all(result)
    with logged_stage(label(report_stage + 2, "Checksums finais")):
        write_checksums(result)
    state["result"] = str(result)
    _stage(state_path, state, "complete")
    return {
        "snapshot_id": manifest["snapshot_id"],
        "status": "complete",
        "result": str(result),
    }


def locate_root(config: Config) -> tuple[dict, Path]:
    manifest = verify_snapshot(config)
    return manifest, config.paths.work / manifest["snapshot_id"]


def locate_result(config: Config, snapshot_id: str | None = None) -> Path:
    if snapshot_id:
        result = config.paths.results / snapshot_id
    else:
        state_files = sorted(
            config.paths.work.glob("*/state.json"), key=lambda path: path.stat().st_mtime_ns
        )
        state = read_json(state_files[-1], {}) if state_files else {}
        candidate = state.get("result")
        if candidate:
            result = Path(candidate)
        else:
            results = sorted(
                config.paths.results.glob("*/summary.json"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            if not results:
                raise WackyWackyError("nenhum resultado agregado disponível")
            result = results[-1].parent
    if not (result / "aggregates").is_dir():
        raise WackyWackyError(f"agregados ausentes em {result}")
    return result


def refresh_reports(config: Config, snapshot_id: str | None = None) -> dict:
    """Rebuild public reports from completed private artifacts only."""
    with logged_stage("[refresh 1/5] Identidade do snapshot"):
        manifest = verify_snapshot(config)
    if snapshot_id and snapshot_id != manifest["snapshot_id"]:
        raise WackyWackyError(
            f"snapshot solicitado {snapshot_id} difere da fonte configurada "
            f"{manifest['snapshot_id']}"
        )
    root = config.paths.work / manifest["snapshot_id"]
    state = read_json(root / "state.json", {})
    if state.get("stage") != "complete":
        raise WackyWackyError("pipeline principal ainda não foi concluído")
    with logged_stage("[refresh 2/5] Inventário de domínios"):
        domains = inventory_domains(config, manifest, root)
    required = {
        "exact": root / "exact_summary.json",
        "clean": root / "clean_summary.json",
        "lexical": root / "lexical_summary.json",
        "content": root / "content_summary.json",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise WackyWackyError(f"artefatos agregados ausentes: {', '.join(missing)}")
    values = {name: read_json(path, {}) for name, path in required.items()}
    v2_paths = {
        "clean_v2": root / "v2" / "clean_summary.json",
        "lexical_v2": root / "v2" / "lexical_summary.json",
        "content_v2": root / "v2" / "content_summary.json",
    }
    v2_values = (
        {name: read_json(path, {}) for name, path in v2_paths.items()}
        if all(path.exists() for path in v2_paths.values())
        else {}
    )
    with logged_stage("[refresh 3/5] Tabelas e agregados"):
        result = config.paths.results / manifest["snapshot_id"]
        archived = archive_method_v1(result)
        result = build_reports(
            config,
            manifest,
            domains,
            values["exact"],
            values["clean"],
            values["lexical"],
            values["content"],
            root,
            clean_v2=v2_values.get("clean_v2"),
            lexical_v2=v2_values.get("lexical_v2"),
            content_v2=v2_values.get("content_v2"),
        )
    from .render import render_all

    with logged_stage("[refresh 4/5] Figuras"):
        render_all(result)
    with logged_stage("[refresh 5/5] Checksums"):
        write_checksums(result)
    return {
        "snapshot_id": manifest["snapshot_id"],
        "status": "refreshed",
        "result": str(result),
        "archived_revision": str(archived) if archived else None,
    }
