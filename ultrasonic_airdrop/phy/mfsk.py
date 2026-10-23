"""Robust dual-subband 16-ary MFSK modem ("R").

Each symbol carries one byte: the high nibble selects one of 16 tones in subband
A and the low nibble one of 16 tones in subband B, both sent simultaneously.
Nibbles are Gray-mapped onto tone positions so that adjacent-tone confusion
causes single-bit errors.  The demodulator reports per-byte erasure flags (when
the best/second-best energy ratio in either subband is below ``erasure_ratio``),
a confidence value and per-symbol SNR estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultrasonic_airdrop.phy.numerology import (
    GRAY,
    INV_GRAY,
    BoolArray,
    FloatArray,
    IntArray,
    Numerology,
)
from ultrasonic_airdrop.phy.synth import tone_symbols


def byte_tones(data: bytes, num: Numerology) -> IntArray:
    """Map bytes to ``(n, 2)`` FFT bin numbers (subband A, subband B)."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    hi = INV_GRAY[arr >> 4]
    lo = INV_GRAY[arr & 0x0F]
    return np.stack([num.mfsk_bins_a[hi], num.mfsk_bins_b[lo]], axis=1)


def modulate(data: bytes, num: Numerology, amplitude: float, abs_start: int) -> FloatArray:
    """Synthesize ``len(data)`` MFSK symbols (plus trailing taper).

    ``amplitude`` is the total peak amplitude; each of the two tones gets half.
    ``abs_start`` is the burst sample offset of the first symbol (phase reference).
    """
    if not data:
        return np.zeros(num.taper)
    return tone_symbols(byte_tones(data, num), amplitude / 2.0, num, abs_start)


@dataclass
class MfskResult:
    """Demodulation output for a run of MFSK symbols."""

    data: bytes
    erasures: BoolArray
    confidence: FloatArray
    symbol_snr: FloatArray
    snr_db: float
    bin_signal: FloatArray
    bin_noise: FloatArray


def symbol_spectra(x: FloatArray, starts: IntArray, num: Numerology, bins: IntArray) -> FloatArray:
    """Power at ``bins`` of the FFT body of each symbol starting at ``starts``."""
    if len(starts) == 0:
        return np.zeros((0, len(bins)))
    idx = starts[:, None] + num.fft_offset + np.arange(num.n)[None, :]
    if int(np.min(idx)) < 0 or int(np.max(idx)) >= len(x):
        raise IndexError("symbol window outside buffer")
    spec = np.fft.rfft(x[idx], axis=1)[:, bins]
    return np.asarray(np.abs(spec) ** 2, dtype=np.float64)


def snr_to_db(lin: float) -> float:
    """Convert a linear per-bin SNR to dB with a floor of -30 dB."""
    return float(10.0 * np.log10(max(lin, 1e-3)))


def demodulate(x: FloatArray, starts: IntArray, num: Numerology, erasure_ratio: float) -> MfskResult:
    """Demodulate MFSK symbols whose (guard-inclusive) starts are given in ``starts``."""
    bins = np.concatenate([num.mfsk_bins_a, num.mfsk_bins_b])
    power = symbol_spectra(x, np.asarray(starts, dtype=np.int64), num, bins) + 1e-30
    n_sym = power.shape[0]
    pa, pb = power[:, :16], power[:, 16:]
    out = np.zeros(n_sym, dtype=np.uint8)
    confs = []
    snrs = []
    winners = []
    for half, shift in ((pa, 4), (pb, 0)):
        order = np.argsort(half, axis=1)
        best = order[:, -1]
        winners.append(best)
        top = half[np.arange(n_sym), best]
        second = half[np.arange(n_sym), order[:, -2]]
        others = np.maximum((half.sum(axis=1) - top) / 15.0, 1e-30)
        confs.append(top / second)
        snrs.append((top - others) / others)
        out |= (GRAY[best] << shift).astype(np.uint8)
    conf = np.minimum(confs[0], confs[1])
    sym_snr = 0.5 * (snrs[0] + snrs[1])
    # Per-bin statistics (32 bins): mean power when the bin carried the tone
    # (signal+noise) and when it did not (noise + interference).
    bin_signal = np.full(32, np.nan)
    bin_noise = np.full(32, np.nan)
    sel = np.zeros_like(power, dtype=bool)
    sel[np.arange(n_sym), winners[0]] = True
    sel[np.arange(n_sym), 16 + winners[1]] = True
    for b in range(32):
        on = power[sel[:, b], b]
        off = power[~sel[:, b], b]
        if on.size:
            bin_signal[b] = on.mean()
        if off.size:
            bin_noise[b] = off.mean()
    mean_snr = float(np.mean(sym_snr)) if n_sym else 0.0
    return MfskResult(
        data=out.tobytes(),
        erasures=conf < erasure_ratio,
        confidence=conf,
        symbol_snr=sym_snr,
        snr_db=snr_to_db(mean_snr),
        bin_signal=bin_signal,
        bin_noise=bin_noise,
    )
