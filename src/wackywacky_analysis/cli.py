from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .errors import ReviewRequired, WackyWackyError
from .io import atomic_json
from .near import run_near_duplicates
from .pipeline import locate_result, locate_root, run_pipeline
from .progress import configure_logging
from .reports import write_checksums, write_table
from .review import export_review, import_review
from .sampling import create_sample
from .snapshot import verify_snapshot


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wackywacky", description="Análise do corpus WackyWacky")
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="valida fontes, espaço e identidade")
    verify.add_argument("--config", required=True)
    sample = commands.add_parser("sample", help="cria a amostra privada do perfil tiny")
    sample.add_argument("--config", required=True)
    run = commands.add_parser("run", help="executa ou retoma o pipeline")
    run.add_argument("--config", required=True)
    run.add_argument("--resume", action="store_true")
    review = commands.add_parser("review", help="controla a revisão privada")
    review_commands = review.add_subparsers(dest="review_command", required=True)
    export = review_commands.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--output")
    import_ = review_commands.add_parser("import")
    import_.add_argument("--config", required=True)
    import_.add_argument("--input", required=True)
    near = commands.add_parser("near-duplicates", help="executa a sensibilidade opcional")
    near.add_argument("--config", required=True)
    render = commands.add_parser("render", help="recria figuras somente dos agregados")
    render.add_argument("--config", required=True)
    render.add_argument("--snapshot-id")
    return parser


def _print(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    configure_logging()
    try:
        config = load_config(arguments.config)
        if arguments.command == "sample":
            manifest = create_sample(config)
            _print(
                {
                    "sample_id": manifest["sample_id"],
                    "status": "ready",
                    "selected_rows": manifest["selection"]["selected_rows"],
                    "selected_domains": manifest["selection"]["selected_domains"],
                    "scope": manifest["scope"],
                }
            )
        elif arguments.command == "verify":
            manifest = verify_snapshot(config)
            _print(
                {
                    "snapshot_id": manifest["snapshot_id"],
                    "pages_sha256": manifest["sources"]["pages"]["sha256"],
                    "domains_sha256": manifest["sources"]["domains"]["sha256"],
                    "scratch_free_bytes": manifest["scratch_free_bytes_at_verify"],
                }
            )
        elif arguments.command == "run":
            _print(run_pipeline(config, resume=arguments.resume))
        elif arguments.command == "review":
            manifest, root = locate_root(config)
            if arguments.review_command == "export":
                output = export_review(
                    config,
                    manifest,
                    root,
                    Path(arguments.output).resolve() if arguments.output else None,
                )
                _print(
                    {"status": "not_needed"}
                    if output is None
                    else {
                        "status": "exported",
                        "path": str(output),
                        "instruction": (
                            "labels começam como boilerplate; altere somente conteúdo e incerto"
                        ),
                    }
                )
            else:
                _print(import_review(config, root, Path(arguments.input).resolve()))
        elif arguments.command == "near-duplicates":
            manifest, root = locate_root(config)
            state = json.loads((root / "state.json").read_text())
            if state.get("stage") != "complete":
                raise WackyWackyError("pipeline principal ainda não foi concluído")
            near_summary = run_near_duplicates(config, root)
            result = config.paths.results / manifest["snapshot_id"]
            summary_path = result / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["near_duplicates"] = near_summary
            atomic_json(summary_path, summary)
            write_table(
                result / "tables",
                "13_near_duplicates",
                ["metrica", "valor"],
                ((key, value) for key, value in near_summary.items() if key != "parameters"),
            )
            write_checksums(result)
            _print(near_summary)
        elif arguments.command == "render":
            from .render import render_all

            result = locate_result(config, arguments.snapshot_id)
            render_all(result)
            write_checksums(result)
            _print({"status": "rendered", "result": str(result)})
        return 0
    except ReviewRequired as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (WackyWackyError, OSError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
