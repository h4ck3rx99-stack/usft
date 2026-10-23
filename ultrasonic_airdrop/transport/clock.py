"""Clock abstraction: wall-clock for real audio, sample-count clock for simulation."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Monotonic time source in seconds."""

    def now(self) -> float:
        """Current time in seconds."""
        ...


class RealClock:
    """``time.monotonic`` based clock."""

    def now(self) -> float:
        """Current time in seconds."""
        return time.monotonic()


class SampleClock:
    """Deterministic clock advanced by the simulator in samples."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.samples = 0

    def now(self) -> float:
        """Current time in seconds (samples / rate)."""
        return self.samples / self.sample_rate

    def advance(self, n: int) -> None:
        """Advance by ``n`` samples."""
        if n < 0:
            raise ValueError("cannot go back in time")
        self.samples += n
