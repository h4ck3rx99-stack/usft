"""Waveform helpers: level control, raised-cosine ramps and phase-continuous tone synthesis."""

from __future__ import annotations

import numpy as np

from ultrasonic_airdrop.config import MAX_PEAK_DBFS, MIN_PEAK_DBFS
from ultrasonic_airdrop.phy.numerology import FloatArray, IntArray, Numerology


def db_to_amp(db: float) -> float:
    """Convert dBFS to linear amplitude."""
    return float(10.0 ** (db / 20.0))


def amp_to_db(amp: float) -> float:
    """Convert linear amplitude to dBFS (floor at -200 dB)."""
    return float(20.0 * np.log10(max(amp, 1e-10)))


def peak_amplitude(level_dbfs: float) -> float:
    """Peak amplitude for a user transmit level, clamped to the safe range."""
    return db_to_amp(float(np.clip(level_dbfs, MIN_PEAK_DBFS, MAX_PEAK_DBFS)))


def raised_cosine(n: int) -> FloatArray:
    """Rising raised-cosine (sin^2) window of ``n`` samples, from ~0 to ~1."""
    if n <= 0:
        return np.ones(0)
    t = (np.arange(n) + 0.5) / n
    return np.asarray(np.sin(0.5 * np.pi * t) ** 2, dtype=np.float64)


def apply_ramps(x: FloatArray, ramp_len: int) -> FloatArray:
    """Fade the first and last ``ramp_len`` samples with a raised cosine (in place copy)."""
    y = np.array(x, dtype=np.float64, copy=True)
    r = min(ramp_len, len(y) // 2)
    if r > 0:
        w = raised_cosine(r)
        y[:r] *= w
        y[-r:] *= w[::-1]
    return y


def symbol_window(sym_len: int, taper: int) -> FloatArray:
    """Window for one symbol of ``sym_len`` samples plus a ``taper``-sample tail.

    Rises over the first ``taper`` samples and falls over the trailing ``taper``
    samples that overlap the next symbol's rise; rise + fall sum to exactly 1.
    """
    w = np.ones(sym_len + taper)
    rc = raised_cosine(taper)
    w[:taper] = rc
    w[sym_len:] = rc[::-1]
    return w


def overlap_add_symbols(rows: FloatArray, sym_len: int) -> FloatArray:
    """Overlap-add rows of length ``sym_len + taper`` spaced ``sym_len`` apart."""
    n_sym, width = rows.shape
    taper = width - sym_len
    out = np.zeros(n_sym * sym_len + taper)
    out[: n_sym * sym_len] = rows[:, :sym_len].reshape(-1)
    if taper > 0:
        tails = rows[:, sym_len:]
        for s in range(n_sym):
            start = (s + 1) * sym_len
            out[start : start + taper] += tails[s]
    return out


def tone_symbols(tone_bins: IntArray, amplitude: float, num: Numerology, abs_start: int) -> FloatArray:
    """Synthesize symbols where each symbol is a sum of tones on FFT bin centres.

    ``tone_bins`` has shape ``(n_symbols, tones_per_symbol)``.  Each tone uses an
    absolute-time phase reference (``abs_start`` is the sample offset of the first
    symbol inside the burst) so a tone repeated in consecutive symbols continues
    seamlessly; changes are cross-faded inside the guard interval.
    """
    tone_bins = np.atleast_2d(tone_bins)
    n_sym = tone_bins.shape[0]
    width = num.sym_len + num.taper
    win = symbol_window(num.sym_len, num.taper)
    idx = abs_start + np.arange(n_sym)[:, None] * num.sym_len + np.arange(width)[None, :]
    phase_idx = (idx % num.n).astype(np.float64)
    rows = np.zeros((n_sym, width))
    for t in range(tone_bins.shape[1]):
        k = tone_bins[:, t].astype(np.float64)[:, None]
        rows += np.cos(2.0 * np.pi * k * phase_idx / num.n)
    rows *= amplitude * win[None, :]
    return overlap_add_symbols(rows, num.sym_len)
