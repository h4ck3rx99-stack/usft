"""Sample-rate ratio estimation and correction, plus frequency shifting.

Two machines' audio clocks typically differ by tens to a few hundred ppm.  The
receiver measures the distance between the preamble and the postamble chirps and
compares it with the nominal distance to get the ratio ``r`` (received / nominal);
the frame is then resampled by ``1/r`` before demodulation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, resample

from ultrasonic_airdrop.config import MAX_PPM
from ultrasonic_airdrop.phy.numerology import FloatArray


def ratio_from_span(measured_span: float, nominal_span: float) -> float:
    """Sample-rate ratio ``r = measured / nominal``."""
    if nominal_span <= 0:
        raise ValueError("nominal span must be positive")
    return measured_span / nominal_span


def ratio_to_ppm(r: float) -> float:
    """Express a ratio as parts-per-million offset."""
    return (r - 1.0) * 1e6


def ratio_is_plausible(r: float, max_ppm: float = MAX_PPM) -> bool:
    """True if ``|r - 1|`` is within ``max_ppm``."""
    return abs(ratio_to_ppm(r)) <= max_ppm


def resample_by_ratio(x: FloatArray, r: float) -> FloatArray:
    """Map a signal sampled with ratio ``r`` back to the nominal rate.

    Output sample ``m`` corresponds to input position ``m * r`` (FFT resampling;
    the effective ratio error is below ``0.5 / len(x)``).  Used both to correct a
    received frame (``r`` = measured ratio) and, in the simulator, to impose a
    clock mismatch (``r`` = 1 / (1 + ppm·1e-6)).
    """
    n_in = len(x)
    if n_in == 0 or r == 1.0:
        return np.array(x, dtype=np.float64, copy=True)
    n_out = max(1, round(n_in / r))
    return np.asarray(resample(x, n_out), dtype=np.float64)


def frequency_shift(x: FloatArray, shift_hz: float, fs: float) -> FloatArray:
    """Shift all frequency content of a real signal by ``shift_hz`` (single-sideband mixing)."""
    if shift_hz == 0.0 or len(x) == 0:
        return np.array(x, dtype=np.float64, copy=True)
    analytic = hilbert(x)
    t = np.arange(len(x)) / fs
    return np.asarray(np.real(analytic * np.exp(2j * np.pi * shift_hz * t)), dtype=np.float64)


def fractional_delay(x: FloatArray, delay: float) -> FloatArray:
    """Delay a signal by a (possibly fractional) number of samples using an FFT phase ramp.

    The output has the same length plus ``ceil(delay)`` samples of headroom.
    """
    if delay < 0:
        raise ValueError("delay must be non-negative")
    pad = int(np.ceil(delay)) + 64
    n = len(x) + pad
    spec = np.fft.rfft(np.concatenate([x, np.zeros(pad)]))
    freqs = np.fft.rfftfreq(n)
    spec *= np.exp(-2j * np.pi * freqs * delay)
    return np.asarray(np.fft.irfft(spec, n)[: len(x) + int(np.ceil(delay))], dtype=np.float64)
