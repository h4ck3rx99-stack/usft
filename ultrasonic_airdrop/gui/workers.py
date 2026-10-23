"""QThread bridges between the (Qt-free) core engines and Qt signals.

* :class:`EventBridge` turns engine events (emitted on the engine thread) into a
  queued Qt signal delivered on the GUI thread.
* :class:`EngineThread` runs a :class:`~ultrasonic_airdrop.transport.engine.Station`
  service loop in a QThread; the audio device is always closed when it ends,
  including after a crash.
* :class:`TaskThread` runs a blocking function (hashing, hardware check) off the GUI thread.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal

from ultrasonic_airdrop.errors import AirDropError
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.transport.engine import RealtimeRunner, Station
from ultrasonic_airdrop.transport.events import ErrorEvent, Event

log = logging.getLogger(__name__)


class EventBridge(QObject):
    """Thread-safe event sink → ``event`` signal."""

    event = Signal(object)  # type: ignore[assignment]  # shadows QObject.event (unused here)

    def sink(self, ev: Event) -> None:
        """Callable passed to the core as an EventSink."""
        self.event.emit(ev)


class EngineThread(QThread):
    """Runs a station until its session is done or :meth:`stop` is called."""

    failed = Signal(str, str)

    def __init__(self, station: Station, stop_when_done: bool = True, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.station = station
        self.runner = RealtimeRunner(station, stop_when_done=stop_when_done)

    def run(self) -> None:
        """QThread entry point."""
        self.runner.run()
        err = self.runner.error
        if err is not None:
            if isinstance(err, AirDropError):
                self.failed.emit(err.user_message, err.hint)
            else:
                self.failed.emit(f"Internal error: {err}", "See the Log page.")

    def stop(self) -> None:
        """Request the loop to end (non-blocking)."""
        self.runner.stop()


class TaskThread(QThread):
    """Runs ``fn()`` on a worker thread; emits ``done(result)`` or ``error(message, hint)``."""

    done = Signal(object)
    error = Signal(str, str)
    progress = Signal(str)

    def __init__(self, fn: Callable[[TaskThread], Any], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.fn = fn
        self.cancelled = False

    def run(self) -> None:
        """QThread entry point."""
        try:
            result = self.fn(self)
        except AirDropError as exc:
            self.error.emit(exc.user_message, exc.hint)
        except Exception as exc:
            log.error(kv("background task failed", error=exc, trace=traceback.format_exc(limit=5)))
            self.error.emit(f"Unexpected error: {exc}", "See the Log page for details.")
        else:
            self.done.emit(result)


def error_event(message: str, hint: str = "") -> ErrorEvent:
    """Convenience constructor."""
    return ErrorEvent(message, hint)
