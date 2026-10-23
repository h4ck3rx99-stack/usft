"""Progress/status events emitted by the engines (consumed by the GUI and CLI)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Event:
    """Base class."""


@dataclass(frozen=True)
class StateEvent(Event):
    """Session state change (``state`` is a short machine name, ``label`` human readable)."""

    role: str
    state: str
    label: str


@dataclass(frozen=True)
class ProgressEvent(Event):
    """Transfer progress.  For the sender ``done_bytes`` counts *acknowledged* on-air bytes."""

    role: str
    done_bytes: int
    total_bytes: int
    goodput_bps: float
    eta_s: float | None
    mode_id: int
    retransmissions: int
    frame_error_rate: float
    snr_db: float
    elapsed_s: float


@dataclass(frozen=True)
class OfferEvent(Event):
    """An incoming OFFER awaiting a user decision."""

    session_id: int
    sender_name: str
    file_count: int
    total_bytes: int
    estimated_s: float
    encrypted: bool
    broadcast: bool
    profile_key: str
    snr_db: float


@dataclass(frozen=True)
class ManifestEvent(Event):
    """The validated manifest (names already sanitised)."""

    files: tuple[tuple[int, str, int], ...]


@dataclass(frozen=True)
class FileEvent(Event):
    """Per-file progress/status on the receiver."""

    index: int
    name: str
    status: str  # receiving | verified | failed | incomplete
    received_bytes: int
    total_bytes: int
    path: Path | None = None
    detail: str = ""


@dataclass(frozen=True)
class LinkEvent(Event):
    """Measurements of a received frame."""

    profile_key: str
    ok: bool
    snr_db: float
    ppm: float
    detect_ratio: float
    mode_id: int
    carrier_snr_db: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SpectrumEvent(Event):
    """Magnitude spectrum of recent input over the profile band."""

    freqs_hz: tuple[float, ...]
    power_db: tuple[float, ...]
    level_dbfs: float
    band_snr_db: float


@dataclass(frozen=True)
class ErrorEvent(Event):
    """A problem to show as a banner."""

    message: str
    hint: str = ""
    fatal: bool = False


@dataclass(frozen=True)
class DoneEvent(Event):
    """Terminal outcome of a session."""

    role: str
    success: bool
    message: str
    files: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)  # (name, status, path)
    stats: dict[str, float] = field(default_factory=dict)


EventSink = Callable[[Event], None]


def null_sink(_event: Event) -> None:
    """Discard events."""


@dataclass(frozen=True)
class LinkTestEvent(Event):
    """Live link-test statistics (receiver side) or progress (sender side)."""

    role: str
    probes_sent: int = 0
    detections: int = 0
    frames_ok: int = 0
    frame_errors: int = 0
    carrier_snr_db: tuple[float, ...] | None = None
    mfsk_snr_db: float = float("nan")
    ppm: float = float("nan")
    recommended_mode: int = -1
    carriers_enabled: int = 0
    elapsed_s: float = 0.0
