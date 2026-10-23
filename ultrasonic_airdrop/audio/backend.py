"""Audio backend protocol and a thread-safe capture ring buffer."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ultrasonic_airdrop.phy.numerology import FloatArray


@dataclass(frozen=True)
class PlayInfo:
    """Timing of a queued playback.

    ``start_time``/``end_time`` are clock seconds; ``mute_start``/``mute_end`` are
    absolute *input* sample indices during which our own sound may be captured.
    """

    start_time: float
    end_time: float
    mute_start: int
    mute_end: int


class AudioBackend(Protocol):
    """Duplex audio device used by a station."""

    sample_rate: int

    def start(self) -> None:
        """Open and start the streams."""
        ...

    def read_input(self) -> tuple[FloatArray, int]:
        """Return newly captured samples and the absolute index of the first one."""
        ...

    def play(self, samples: FloatArray) -> PlayInfo:
        """Queue samples for playback (non-blocking)."""
        ...

    def playing(self) -> bool:
        """True while queued audio has not finished playing."""
        ...

    def now(self) -> float:
        """Clock time in seconds."""
        ...

    def close(self) -> None:
        """Stop and close all streams (idempotent)."""
        ...


class RingBuffer:
    """Fixed-capacity capture buffer with absolute sample indexing.

    The audio callback calls :meth:`write`; the engine thread calls
    :meth:`read_since`.  If the reader falls behind by more than the capacity
    the oldest samples are lost (and reported via ``overruns``).
    """

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._total = 0
        self._lock = threading.Lock()
        self.overruns = 0

    @property
    def total(self) -> int:
        """Absolute index one past the newest sample."""
        with self._lock:
            return self._total

    def write(self, samples: FloatArray) -> None:
        """Append samples (called from the audio thread)."""
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return
        with self._lock:
            if n >= self._cap:
                x = x[-self._cap :]
                self._total += n - self._cap
                n = self._cap
            pos = self._total % self._cap
            first = min(n, self._cap - pos)
            self._buf[pos : pos + first] = x[:first]
            if first < n:
                self._buf[: n - first] = x[first:]
            self._total += n

    def read_since(self, index: int) -> tuple[FloatArray, int]:
        """Samples from absolute ``index`` to now (clipped to what is still buffered)."""
        with self._lock:
            start = max(index, self._total - self._cap)
            if start > index:
                self.overruns += 1
            n = self._total - start
            if n <= 0:
                return np.zeros(0), self._total
            pos = start % self._cap
            first = min(n, self._cap - pos)
            out = np.empty(n, dtype=np.float64)
            out[:first] = self._buf[pos : pos + first]
            if first < n:
                out[first:] = self._buf[: n - first]
            return out, start
