"""run_hardware_check end-to-end with a fake real-time loopback device (≈9 s wall time)."""

from __future__ import annotations

import time

import numpy as np
from scipy.signal import butter, sosfilt

from ultrasonic_airdrop.audio.backend import PlayInfo
from ultrasonic_airdrop.audio.diagnostics import run_hardware_check


class LoopbackBackend:
    """Playback appears in the capture after 20 ms, through a 17 kHz low-pass, plus noise."""

    def __init__(self, fs: int = 48000, cutoff: float | None = 17000.0) -> None:
        self.sample_rate = fs
        self.t0 = 0.0
        self.read_pos = 0
        self.timeline = np.zeros(fs * 30)
        self.sos = butter(12, cutoff, fs=fs, output="sos") if cutoff else None
        self.rng = np.random.default_rng(0)
        self.closed = False
        self.actual_rate = float(fs)

    def start(self) -> None:
        self.t0 = time.monotonic()

    def _now_idx(self) -> int:
        return int((time.monotonic() - self.t0) * self.sample_rate)

    def read_input(self):
        end = min(self._now_idx(), len(self.timeline))
        start = self.read_pos
        self.read_pos = end
        x = self.timeline[start:end] + self.rng.standard_normal(end - start) * 1e-4
        return x, start

    def play(self, samples):
        y = sosfilt(self.sos, samples) if self.sos is not None else samples
        s = self._now_idx() + int(0.02 * self.sample_rate)
        e = min(len(self.timeline), s + len(y))
        self.timeline[s:e] += 0.3 * y[: e - s]
        return PlayInfo(s / self.sample_rate, e / self.sample_rate, s, e)

    def playing(self) -> bool:
        return False

    def now(self) -> float:
        return time.monotonic() - self.t0

    def close(self) -> None:
        self.closed = True


def test_hardware_check_runs_end_to_end():
    b = LoopbackBackend()
    msgs: list[str] = []
    res = run_hardware_check(b, -6.0, progress=msgs.append)
    assert b.closed
    assert len(msgs) == 3
    assert res.verdict("au") == "supported"
    assert res.verdict("nu") in ("marginal", "not supported")
    assert res.brick_wall_hz is not None and 15500 < res.brick_wall_hz < 17500
