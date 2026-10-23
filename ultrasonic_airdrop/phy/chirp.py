"""Preamble / postamble linear chirps and their analytic matched-filter templates."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal.windows import tukey

from ultrasonic_airdrop.config import CHIRP_TUKEY_ALPHA, Profile
from ultrasonic_airdrop.phy.numerology import ComplexArray, FloatArray, numerology


def _phase(profile: Profile, up: bool) -> FloatArray:
    num = numerology(profile)
    t = np.arange(num.chirp_len) / num.fs
    dur = num.chirp_len / num.fs
    f0, f1 = (profile.chirp_f0, profile.chirp_f1) if up else (profile.chirp_f1, profile.chirp_f0)
    return np.asarray(2.0 * np.pi * (f0 * t + (f1 - f0) * t * t / (2.0 * dur)), dtype=np.float64)


@lru_cache(maxsize=16)
def _window(n: int) -> FloatArray:
    return np.asarray(tukey(n, CHIRP_TUKEY_ALPHA), dtype=np.float64)


@lru_cache(maxsize=16)
def chirp(profile: Profile, up: bool = True) -> FloatArray:
    """Real, Tukey-windowed linear chirp of unit peak amplitude."""
    num = numerology(profile)
    out = np.cos(_phase(profile, up)) * _window(num.chirp_len)
    out.setflags(write=False)
    return out


@lru_cache(maxsize=16)
def chirp_template(profile: Profile, up: bool = True) -> ComplexArray:
    """Analytic (complex) chirp template used for envelope matched filtering."""
    num = numerology(profile)
    out = np.asarray(np.exp(1j * _phase(profile, up)) * _window(num.chirp_len), dtype=np.complex128)
    out.setflags(write=False)
    return out
