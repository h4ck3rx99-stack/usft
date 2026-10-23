"""Chirp preamble detection: band-pass, analytic matched filtering, CFAR thresholding.

The detector correlates the band-passed input with complex (analytic) up- and
down-chirp templates and works on the correlation *envelope*, which is
insensitive to carrier phase.  A candidate is accepted when the envelope peak
exceeds ``detect_ratio`` times the median envelope of the surrounding region
(a CFAR-style test).  Within a multipath cluster the earliest path whose peak is
at least 60 % of the strongest is chosen ("first arrival") so that symbol timing
is not pushed late by a strong reflection.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, oaconvolve, sosfiltfilt

from ultrasonic_airdrop.config import Profile
from ultrasonic_airdrop.phy.chirp import chirp_template
from ultrasonic_airdrop.phy.numerology import FloatArray, numerology

FIRST_ARRIVAL_FRACTION = 0.6
FIRST_ARRIVAL_LOOKBACK_S = 0.005
#: Absolute envelope floor so that digital silence never triggers a detection.
ABS_FLOOR = 1e-7


class ChirpDetector:
    """Matched-filter front end for one profile."""

    def __init__(self, profile: Profile, detect_ratio: float = 8.0) -> None:
        self.profile = profile
        self.num = numerology(profile)
        self.detect_ratio = detect_ratio
        nyq = profile.sample_rate / 2.0
        lo = max(100.0, profile.chirp_f0 - 500.0)
        hi = min(profile.chirp_f1 + 500.0, 0.97 * nyq)
        self.sos = butter(4, [lo, hi], btype="bandpass", fs=profile.sample_rate, output="sos")
        self._h_up = np.conj(chirp_template(profile, True)[::-1])
        self._h_down = np.conj(chirp_template(profile, False)[::-1])
        self.lookback = round(FIRST_ARRIVAL_LOOKBACK_S * profile.sample_rate)

    def bandpass(self, x: FloatArray) -> FloatArray:
        """Zero-phase band-pass around the profile band."""
        if len(x) < 64:
            return np.array(x, dtype=np.float64, copy=True)
        return np.asarray(sosfiltfilt(self.sos, x, padlen=min(len(x) - 1, 200)), dtype=np.float64)

    def envelopes(self, x_bp: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Correlation envelopes; index ``i`` corresponds to a chirp starting at sample ``i``.

        Output length is ``len(x_bp) - chirp_len + 1`` (empty if the input is shorter).
        """
        c = self.num.chirp_len
        if len(x_bp) < c:
            return np.zeros(0), np.zeros(0)
        up = np.abs(oaconvolve(x_bp, self._h_up, mode="valid"))
        down = np.abs(oaconvolve(x_bp, self._h_down, mode="valid"))
        return np.asarray(up, dtype=np.float64), np.asarray(down, dtype=np.float64)

    def envelope_up(self, x_bp: FloatArray) -> FloatArray:
        """Up-chirp correlation envelope only (postamble search)."""
        if len(x_bp) < self.num.chirp_len:
            return np.zeros(0)
        return np.asarray(np.abs(oaconvolve(x_bp, self._h_up, mode="valid")), dtype=np.float64)

    def first_arrival(self, env: FloatArray, idx: int) -> int:
        """Earliest index within the look-back window whose local peak is >= 60 % of ``env[idx]``."""
        lo = max(0, idx - self.lookback)
        if lo >= idx:
            return idx
        seg = env[lo:idx]
        thr = FIRST_ARRIVAL_FRACTION * env[idx]
        cand = np.flatnonzero(seg >= thr)
        if cand.size == 0:
            return idx
        j = int(cand[0])
        # climb to the local maximum of that path
        k = lo + j
        while k + 1 < len(env) and env[k + 1] > env[k] and k + 1 <= idx:
            k += 1
        return k

    @staticmethod
    def refine(env: FloatArray, idx: int) -> float:
        """Sub-sample peak position by parabolic interpolation."""
        if idx <= 0 or idx >= len(env) - 1:
            return float(idx)
        a, b, c = env[idx - 1], env[idx], env[idx + 1]
        denom = a - 2.0 * b + c
        if denom >= 0:
            return float(idx)
        off = 0.5 * (a - c) / denom
        return float(idx + np.clip(off, -0.5, 0.5))

    def passes(self, peak: float, median: float, ratio: float | None = None) -> bool:
        """CFAR test."""
        r = self.detect_ratio if ratio is None else ratio
        return bool(peak > ABS_FLOOR and peak >= r * max(median, 1e-12))

    def cfo_from_pair(self, up_pos: float, down_pos: float) -> float:
        """Carrier frequency offset (Hz) from the up/down chirp spacing.

        A positive frequency offset makes the up-chirp correlate early and the
        down-chirp late by ``df / k`` seconds each.
        """
        num = self.num
        spacing_err = down_pos - up_pos - num.chirp_len
        return float(spacing_err * num.chirp_rate_hz_per_s / (2.0 * num.fs))
