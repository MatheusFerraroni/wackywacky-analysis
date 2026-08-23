from __future__ import annotations

from pathlib import Path

from .clean import clean_representatives
from .config import Config
from .content import content_pass
from .domains import inventory_domains
from .errors import SourceChangedError, WackyWackyError
from .io import atomic_json, read_json
from .lexical import lexical_pass
from .pages import scan_pages
from .progress import logged_stage
from .reduction import reduce_exact
from .reports import build_reports, write_checksums
from .review import require_approved_review
from .snapshot import assert_snapshot, verify_snapshot
from .validation import validate_invariants


def _stage(state_path: Path, state: dict, name: str) -> None:
    disk = read_json(state_path, {})
    disk.update(state)
    disk["stage"] = name
    state.clear()
    state.update(disk)
    atomic_json(state_path, disk)


def run_pipeline(config: Config, *, resume: bool) -> dict:
    with logged_stage("[1/12] Identidade do snapshot"):
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
    with logged_stage("[2/12] Inventário de domínios"):
        domain_summary = inventory_domains(config, manifest, root)
        assert_snapshot(config, manifest)
    _stage(state_path, state, "domains")
    with logged_stage("[3/12] Leitura e validação de páginas"):
        scan = scan_pages(config, manifest, root, resume=resume)
    if not scan.get("complete"):
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    _stage(state_path, state, "pages")
    with logged_stage("[4/12] Deduplicação exata D1/D2"):
        exact = reduce_exact(config, root)
        assert_snapshot(config, manifest)
    _stage(state_path, state, "exact")
    with logged_stage("[5/12] Gate da revisão privada"):
        gate = require_approved_review(config, root)
    state["review_gate"] = gate
    _stage(state_path, state, "review_approved")
    with logged_stage("[6/12] Limpeza e deduplicação D3"):
        clean = clean_representatives(config, manifest, root)
    if clean.get("complete") is False:
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    assert_snapshot(config, manifest)
    _stage(state_path, state, "clean")
    with logged_stage("[7/12] Estatísticas lexicais"):
        lexical = lexical_pass(config, manifest, root)
    if lexical.get("complete") is False:
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    assert_snapshot(config, manifest)
    _stage(state_path, state, "lexical")
    with logged_stage("[8/12] Estrutura e conteúdo textual"):
        content = content_pass(config, manifest, root)
    if content.get("complete") is False:
        _stage(state_path, state, "interrupted")
        return {"snapshot_id": manifest["snapshot_id"], "status": "interrupted"}
    _stage(state_path, state, "content")
    with logged_stage("[9/12] Validação de invariantes"):
        validate_invariants(exact, clean, lexical, root, content=content)
    _stage(state_path, state, "validated")
    with logged_stage("[10/12] Tabelas e agregados"):
        result = build_reports(
            config, manifest, domain_summary, exact, clean, lexical, content, root
        )
    from .render import render_all

    with logged_stage("[11/12] Figuras"):
        render_all(result)
    with logged_stage("[12/12] Checksums finais"):
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
