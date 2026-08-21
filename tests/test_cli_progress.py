from __future__ import annotations

import json
import logging
import subprocess
import sys
from io import StringIO
from pathlib import Path

from conftest import write_config, write_sources

import wackywacky_analysis.progress as progress_module
from wackywacky_analysis.progress import ByteProgress, ItemProgress


def test_cli_progress_uses_stderr_and_preserves_json_stdout(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config = write_config(
        tmp_path / "config.toml",
        pages,
        domains,
        tmp_path / "work",
        tmp_path / "results",
    )
    completed = subprocess.run(
        [sys.executable, "-m", "wackywacky_analysis", "verify", "--config", str(config)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["snapshot_id"]
    assert "SHA-256 de pages.tsv" in completed.stderr
    assert "100.0%" in completed.stderr


def test_non_interactive_progress_reports_rate_and_eta(monkeypatch) -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(progress_module.LOGGER, "handlers", [handler])
    monkeypatch.setattr(progress_module.LOGGER, "propagate", False)
    progress_module.LOGGER.setLevel(logging.INFO)
    monkeypatch.setattr(progress_module.sys.stderr, "isatty", lambda: False)

    progress = ItemProgress("Candidatos", 100, minimum_interval=0)
    progress.update(50)
    progress.finish()

    output = stream.getvalue()
    assert "50/100 itens" in output
    assert "itens/s" in output
    assert "ETA" in output


def test_interactive_progress_uses_tqdm(monkeypatch) -> None:
    events: list[tuple[str, int | bool]] = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            self.n = kwargs["initial"]
            events.append(("total", kwargs["total"]))

        def update(self, amount):
            self.n += amount
            events.append(("update", amount))

        def set_postfix_str(self, _detail, refresh=False):
            events.append(("refresh_postfix", refresh))

        def refresh(self):
            events.append(("refresh", True))

        def close(self):
            events.append(("close", True))

    monkeypatch.setattr(progress_module, "tqdm", FakeTqdm)
    monkeypatch.setattr(progress_module.sys.stderr, "isatty", lambda: True)
    progress_module.LOGGER.setLevel(logging.INFO)

    progress = ByteProgress("Páginas", 100, minimum_interval=0)
    progress.update(50, detail="metade", force=True)
    progress.update(100, force=True)
    progress.finish()

    assert ("total", 100) in events
    assert ("update", 50) in events
    assert events[-1] == ("close", True)
