from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic

from tqdm import tqdm

LOGGER = logging.getLogger("wackywacky")


class _ProgressLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if sys.stderr.isatty():
                tqdm.write(message, file=sys.stderr)
            else:
                sys.stderr.write(message + "\n")
                sys.stderr.flush()
        except (OSError, ValueError):
            self.handleError(record)


def configure_logging() -> None:
    """Configure concise progress logs without contaminating JSON written to stdout."""
    if not any(getattr(handler, "wackywacky_progress", False) for handler in LOGGER.handlers):
        handler = _ProgressLogHandler()
        handler.wackywacky_progress = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}min {seconds:02d}s"
    if minutes:
        return f"{minutes}min {seconds:02d}s"
    return f"{seconds}s"


@contextmanager
def logged_stage(label: str, *, heartbeat_seconds: float = 30.0) -> Iterator[None]:
    """Log stage boundaries and periodic heartbeats for opaque operations."""
    if not LOGGER.isEnabledFor(logging.INFO):
        yield
        return
    started = monotonic()
    stopped = threading.Event()

    def heartbeat() -> None:
        while not stopped.wait(heartbeat_seconds):
            LOGGER.info("%s: em execução há %s", label, _duration(monotonic() - started))

    LOGGER.info("%s: iniciando", label)
    thread = threading.Thread(target=heartbeat, name="wackywacky-progress", daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        stopped.set()
        LOGGER.info("%s: interrompida após %s", label, _duration(monotonic() - started))
        raise
    else:
        stopped.set()
        LOGGER.info("%s: concluída em %s", label, _duration(monotonic() - started))
    finally:
        stopped.set()
        thread.join(timeout=0.1)


class _Progress:
    def __init__(
        self,
        label: str,
        total: int,
        *,
        initial: int,
        minimum_interval: float,
        minimum_step: int,
        unit: str,
        unit_scale: bool,
        unit_divisor: int = 1000,
    ) -> None:
        self.label = label
        self.total = max(0, total)
        self.current = max(0, initial)
        self.initial = self.current
        self.minimum_interval = minimum_interval
        self.minimum_step = max(1, minimum_step)
        self.started = monotonic()
        self.last_log = self.started
        self.last_checked = self.current
        self.last_logged_value: int | None = None
        self.closed = False
        self.bar = (
            tqdm(
                total=self.total or None,
                initial=self.current,
                desc=self.label,
                unit=unit,
                unit_scale=unit_scale,
                unit_divisor=unit_divisor,
                dynamic_ncols=True,
                mininterval=0.5,
                file=sys.stderr,
                leave=True,
            )
            if LOGGER.isEnabledFor(logging.INFO) and sys.stderr.isatty()
            else None
        )
        if self.bar is None and LOGGER.isEnabledFor(logging.INFO):
            self._log("retomando" if initial else "iniciando")

    def update(self, current: int, *, detail: str | None = None, force: bool = False) -> None:
        if self.closed:
            return
        self.current = max(self.current, current)
        if not LOGGER.isEnabledFor(logging.INFO):
            return
        complete = bool(self.total and self.current >= self.total)
        if not force and not complete and self.current - self.last_checked < self.minimum_step:
            return
        self.last_checked = self.current
        if self.bar is not None:
            self.bar.update(max(0, self.current - self.bar.n))
            if detail:
                self.bar.set_postfix_str(detail, refresh=False)
            if force and not complete:
                self.bar.refresh()
            return
        now = monotonic()
        if not force and not complete and now - self.last_log < self.minimum_interval:
            return
        self.last_log = now
        self._log(detail)

    def finish(self, *, detail: str | None = None) -> None:
        if self.closed:
            return
        if self.bar is not None or self.last_logged_value != self.current:
            self.update(self.current, detail=detail or "concluído", force=True)
        self.close()

    def close(self) -> None:
        if self.closed:
            return
        if self.bar is not None:
            self.bar.close()
        self.closed = True

    def _rate_and_eta(self) -> str:
        elapsed = monotonic() - self.started
        processed = self.current - self.initial
        if elapsed <= 0 or processed <= 0:
            return ""
        rate = processed / elapsed
        rate_text = self._format_rate(rate)
        if not self.total or self.current >= self.total:
            return f" | {rate_text}"
        eta = (self.total - self.current) / rate
        return f" | {rate_text} | ETA {_duration(eta)}"

    def _log(self, detail: str | None) -> None:
        self.last_logged_value = self.current
        message = self._format_progress() + self._rate_and_eta()
        if detail:
            message += f" | {detail}"
        LOGGER.info("%s: %s", self.label, message)

    def _format_progress(self) -> str:
        raise NotImplementedError

    def _format_rate(self, rate: float) -> str:
        raise NotImplementedError


class ByteProgress(_Progress):
    """Byte progress with tqdm on a TTY and periodic ETA logs otherwise."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        initial: int = 0,
        minimum_interval: float = 5.0,
    ) -> None:
        minimum_step = max(1, min(8 * 1024 * 1024, total // 1000 or 1))
        super().__init__(
            label,
            total,
            initial=initial,
            minimum_interval=minimum_interval,
            minimum_step=minimum_step,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )

    def _format_progress(self) -> str:
        processed_mib = self.current / (1024 * 1024)
        if not self.total:
            return f"{processed_mib:,.1f} MiB"
        total_mib = self.total / (1024 * 1024)
        percentage = min(100.0, 100 * self.current / self.total)
        return f"{percentage:5.1f}% ({processed_mib:,.1f}/{total_mib:,.1f} MiB)"

    def _format_rate(self, rate: float) -> str:
        return f"{rate / (1024 * 1024):,.1f} MiB/s"


class ItemProgress(_Progress):
    """Item progress for bounded database and aggregation work."""

    def __init__(
        self,
        label: str,
        total: int,
        *,
        initial: int = 0,
        minimum_interval: float = 5.0,
    ) -> None:
        super().__init__(
            label,
            total,
            initial=initial,
            minimum_interval=minimum_interval,
            minimum_step=max(1, min(100_000, total // 1000 or 1)),
            unit="item",
            unit_scale=True,
        )

    def _format_progress(self) -> str:
        if not self.total:
            return f"{self.current:,} itens"
        percentage = min(100.0, 100 * self.current / self.total)
        return f"{percentage:5.1f}% ({self.current:,}/{self.total:,} itens)"

    def _format_rate(self, rate: float) -> str:
        return f"{rate:,.0f} itens/s"
