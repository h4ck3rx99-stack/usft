"""Composable acoustic channel impairments for simulation and tests.

Per burst (one call to :meth:`Channel.process_burst`): frame loss, speaker
soft-clipping, device band-limiting (low-pass, notch), room reverberation
(synthetic exponentially decaying impulse response with early reflections),
sample-rate mismatch (resampling), carrier frequency offset (single-sideband
mixing), gain and slow fading, and a (fractional) propagation delay.

Continuously (via :meth:`Channel.noise`): white noise scaled to a target
*in-band SNR* and random noise bursts.

SNR definition: ratio of the nominal received signal power (a two-tone MFSK
symbol at the configured transmit level, times channel gain) to the noise power
within a 3 kHz band (the width of every profile's chirp band).  For white noise
at sample rate ``fs`` that band holds ``3000 / (fs / 2)`` of the noise power.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, iirnotch, oaconvolve, sosfilt, tf2sos

from ultrasonic_airdrop.config import DEFAULT_TX_LEVEL_DBFS
from ultrasonic_airdrop.phy.numerology import FloatArray
from ultrasonic_airdrop.phy.resample import fractional_delay, frequency_shift, resample_by_ratio
from ultrasonic_airdrop.phy.synth import db_to_amp

BAND_HZ = 3000.0
FRAME_GAP_MIN = 0.04
FRAME_GAP_MAX = 0.08


@dataclass
class ChannelConfig:
    """Impairment parameters (defaults = clean channel)."""

    snr_db: float | None = 40.0
    ppm: float = 0.0
    cfo_hz: float = 0.0
    delay_s: float = 0.002
    loss: float = 0.0
    burst_rate_hz: float = 0.0
    burst_ms: tuple[float, float] = (50.0, 300.0)
    burst_sir_db: float = 0.0
    gain_db: float = 0.0
    fading_hz: float = 0.0
    fading_depth_db: float = 0.0
    rt60_s: float = 0.0
    drr_db: float = 6.0
    lowpass_hz: float | None = None
    notch_hz: float | None = None
    soft_clip_drive: float | None = None
    tx_level_dbfs: float = DEFAULT_TX_LEVEL_DBFS
    seed: int = 0
    extra: dict[str, float] = field(default_factory=dict)


def synthetic_rir(fs: int, rt60_s: float, drr_db: float, rng: np.random.Generator) -> FloatArray:
    """Room impulse response: direct path, early reflections, exponential diffuse tail.

    The direct-to-reverberant energy ratio is ``drr_db``; the tail decays by 60 dB
    in ``rt60_s``.
    """
    if rt60_s <= 0:
        return np.array([1.0])
    n = int(1.2 * rt60_s * fs) + 1
    h = np.zeros(n)
    h[0] = 1.0
    rest = np.zeros(n)
    for _ in range(5):
        tap = int(rng.uniform(0.0015, 0.015) * fs)
        if tap < n:
            rest[tap] += rng.uniform(0.3, 0.7) * rng.choice([-1.0, 1.0])
    t = np.arange(n) / fs
    onset = int(0.002 * fs)
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / rt60_s)
    tail[:onset] = 0.0
    tail_e = float(np.sum(tail**2))
    early_e = float(np.sum(rest**2))
    if tail_e > 0:
        rest += tail * np.sqrt(max(early_e, 0.25) / tail_e)
    target = 10 ** (-drr_db / 10.0)  # reverberant energy relative to direct (1.0)
    e = float(np.sum(rest**2))
    if e > 0:
        rest *= np.sqrt(target / e)
    return h + rest


class Channel:
    """One direction of an acoustic link."""

    def __init__(self, cfg: ChannelConfig, fs: int, rng: np.random.Generator | None = None) -> None:
        self.cfg = cfg
        self.fs = fs
        self.rng = rng or np.random.default_rng(cfg.seed)
        self.gain = db_to_amp(cfg.gain_db)
        peak = db_to_amp(cfg.tx_level_dbfs)
        self.ref_power = (peak**2 / 4.0) * self.gain**2
        if cfg.snr_db is None:
            self.noise_sigma = 0.0
        else:
            frac = BAND_HZ / (fs / 2.0)
            self.noise_sigma = float(np.sqrt(self.ref_power / (10 ** (cfg.snr_db / 10.0)) / frac))
        self.burst_sigma = float(np.sqrt(self.ref_power / (10 ** (cfg.burst_sir_db / 10.0)) / (BAND_HZ / (fs / 2.0))))
        self.rir = synthetic_rir(fs, cfg.rt60_s, cfg.drr_db, self.rng)
        self._filters = []
        if cfg.lowpass_hz is not None and cfg.lowpass_hz < fs / 2:
            self._filters.append(butter(8, cfg.lowpass_hz, btype="lowpass", fs=fs, output="sos"))
        if cfg.notch_hz is not None:
            b, a = iirnotch(cfg.notch_hz, 30.0, fs=fs)
            self._filters.append(tf2sos(b, a))
        self._burst_remaining = 0
        self._t = 0
        self.frames_dropped = 0
        self.frames_total = 0

    def _drop_frames(self, x: FloatArray) -> FloatArray:
        """Zero whole frames with probability ``loss`` (frames are separated by short silent gaps)."""
        if self.cfg.loss <= 0:
            return x
        zero = x == 0.0
        lo, hi = int(FRAME_GAP_MIN * self.fs), int(FRAME_GAP_MAX * self.fs)
        # find runs of exact zeros of gap length
        padded = np.concatenate([[False], zero, [False]])
        d = np.diff(padded.astype(np.int8))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        cuts = [0]
        for s, e in zip(starts, ends, strict=True):
            if lo <= e - s <= hi:
                cuts.append(int(s))
        cuts.append(len(x))
        y = x.copy()
        for a, b in itertools.pairwise(cuts):
            if b - a < self.fs // 100:
                continue
            self.frames_total += 1
            if self.rng.random() < self.cfg.loss:
                y[a:b] = 0.0
                self.frames_dropped += 1
        return y

    def process_burst(self, x: FloatArray) -> tuple[FloatArray, int]:
        """Apply per-burst impairments; returns ``(samples, delay_in_samples)``."""
        cfg = self.cfg
        y = self._drop_frames(np.asarray(x, dtype=np.float64))
        if cfg.soft_clip_drive is not None:
            d = cfg.soft_clip_drive
            y = np.tanh(d * y) / np.tanh(d)
        for sos in self._filters:
            y = sosfilt(sos, y)
        if len(self.rir) > 1:
            y = oaconvolve(y, self.rir)
        if cfg.ppm:
            # The receiver's clock runs (1 + ppm) faster: it captures more samples per second.
            y = resample_by_ratio(y, 1.0 / (1.0 + cfg.ppm * 1e-6))
        if cfg.cfo_hz:
            y = frequency_shift(y, cfg.cfo_hz, self.fs)
        y = y * self.gain
        if cfg.fading_hz > 0 and cfg.fading_depth_db > 0:
            t = np.arange(len(y)) / self.fs
            phase = self.rng.uniform(0, 2 * np.pi)
            depth = cfg.fading_depth_db / 2.0
            y = y * 10 ** ((depth * np.sin(2 * np.pi * cfg.fading_hz * t + phase) - depth) / 20.0)
        delay = cfg.delay_s * self.fs
        whole = int(np.floor(delay))
        frac = delay - whole
        if frac > 1e-6:
            y = fractional_delay(y, frac)
        return np.asarray(y, dtype=np.float64), whole

    def noise(self, n: int) -> FloatArray:
        """Background noise (AWGN + random bursts) for the next ``n`` samples."""
        out = self.rng.standard_normal(n) * self.noise_sigma if self.noise_sigma > 0 else np.zeros(n)
        cfg = self.cfg
        if cfg.burst_rate_hz > 0:
            pos = 0
            while pos < n:
                if self._burst_remaining > 0:
                    k = min(self._burst_remaining, n - pos)
                    out[pos : pos + k] += self.rng.standard_normal(k) * self.burst_sigma
                    self._burst_remaining -= k
                    pos += k
                    continue
                # time to next burst ~ exponential
                gap = int(self.rng.exponential(1.0 / cfg.burst_rate_hz) * self.fs)
                if pos + gap >= n:
                    break
                pos += gap
                self._burst_remaining = int(self.rng.uniform(*cfg.burst_ms) / 1000.0 * self.fs)
        self._t += n
        return out


def apply_channel(x: FloatArray, cfg: ChannelConfig, fs: int, lead_s: float = 0.5, tail_s: float = 0.5) -> FloatArray:
    """Offline convenience: pass a signal through a channel and embed it in noise."""
    ch = Channel(cfg, fs)
    y, delay = ch.process_burst(x)
    lead = int(lead_s * fs) + delay
    total = lead + len(y) + int(tail_s * fs)
    out = ch.noise(total)
    out[lead : lead + len(y)] += y
    return out
