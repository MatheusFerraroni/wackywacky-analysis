from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil
from conftest import write_config, write_sources


def _peak_rss(command: list[str]) -> tuple[int, int]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    peak = 0
    while process.poll() is None:
        try:
            peak = max(peak, psutil.Process(process.pid).memory_info().rss)
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.01)
    return process.returncode, peak


def test_tiny_cli_stays_below_768_mib(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures"
    config = write_config(
        tmp_path / "config.toml",
        fixture / "pages.tsv",
        fixture / "domain.tsv",
        tmp_path / "work",
        tmp_path / "results",
    )
    returncode, peak = _peak_rss(
        [sys.executable, "-m", "wackywacky_analysis", "run", "--config", str(config)]
    )
    assert returncode == 0
    assert peak < 768 * 1024 * 1024


def test_tiny_sample_stays_below_768_mib(tmp_path: Path) -> None:
    pages, domains = write_sources(tmp_path / "sources")
    config = write_config(
        tmp_path / "config.toml",
        pages,
        domains,
        tmp_path / "work",
        tmp_path / "results",
        sampling=True,
    )
    returncode, peak = _peak_rss(
        [sys.executable, "-m", "wackywacky_analysis", "sample", "--config", str(config)]
    )
    assert returncode == 0
    assert peak < 768 * 1024 * 1024
