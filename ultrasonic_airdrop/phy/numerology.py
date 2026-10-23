"""Derived per-profile numerology: tone bins, symbol timing and on-air frame layout."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from ultrasonic_airdrop.config import (
    CHIRP_PAIR_TOLERANCE_48K,
    FFT_OFFSET_FRACTION,
    MFSK_TONES,
    OFDM_FFT_OFFSET_FRACTION,
    PHY_HEADER_LEN,
    PHY_HEADER_NSYM,
    PROBE_SILENCE_S,
    RAMP_S,
    SYMBOL_TAPER_48K,
    SYNC_WORD,
    Profile,
)

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: Number of MFSK symbols in the PHY header (6 bytes + 8 parity).
HEADER_SYMBOLS = PHY_HEADER_LEN + PHY_HEADER_NSYM
SYNC_SYMBOLS = len(SYNC_WORD)


@dataclass(frozen=True, eq=False)
class Numerology:
    """Integer sample/bin quantities derived from a :class:`Profile`."""

    profile: Profile
    fs: int
    n: int
    g: int
    sym_len: int
    taper: int
    fft_offset: int
    ofdm_fft_offset: int
    chirp_len: int
    ramp_len: int
    pair_tol: int
    probe_silence_len: int
    mfsk_bins_a: IntArray
    mfsk_bins_b: IntArray
    ofdm_bins: IntArray

    @property
    def scale(self) -> float:
        """Sample-rate scale factor relative to 48 kHz."""
        return self.fs / 48_000

    # ---- frame layout -------------------------------------------------------------
    @property
    def sync_start(self) -> int:
        """Offset of the sync word from the start of the up-chirp."""
        return 2 * self.chirp_len

    @property
    def header_start(self) -> int:
        """Offset of the first PHY header symbol."""
        return self.sync_start + SYNC_SYMBOLS * self.sym_len

    @property
    def body_start(self) -> int:
        """Offset of the first body symbol."""
        return self.header_start + HEADER_SYMBOLS * self.sym_len

    def postamble_start(self, body_samples: int) -> int:
        """Offset of the postamble chirp for a body occupying ``body_samples``."""
        return self.body_start + body_samples + self.g

    def frame_length(self, body_samples: int) -> int:
        """Total burst length in samples (including the trailing taper)."""
        return self.postamble_start(body_samples) + self.chirp_len + self.taper

    @property
    def chirp_rate_hz_per_s(self) -> float:
        """Chirp sweep rate k in Hz/s."""
        p = self.profile
        return (p.chirp_f1 - p.chirp_f0) / (self.chirp_len / self.fs)


def gray(n: int) -> int:
    """Binary-reflected Gray code."""
    return n ^ (n >> 1)


GRAY = np.array([gray(i) for i in range(MFSK_TONES)], dtype=np.int64)
INV_GRAY = np.argsort(GRAY).astype(np.int64)


@lru_cache(maxsize=8)
def numerology(profile: Profile) -> Numerology:
    """Return (cached) numerology for ``profile``."""
    scale = profile.sample_rate / 48_000
    taper = profile.taper or round(SYMBOL_TAPER_48K * scale)
    fft_offset = round(profile.guard * FFT_OFFSET_FRACTION)
    ofdm_offset = round(profile.guard * OFDM_FFT_OFFSET_FRACTION)
    for off in (fft_offset, ofdm_offset):
        if off < taper or off + profile.n_fft > profile.symbol_len:
            raise ValueError("inconsistent guard/taper configuration")
    return Numerology(
        profile=profile,
        fs=profile.sample_rate,
        n=profile.n_fft,
        g=profile.guard,
        sym_len=profile.symbol_len,
        taper=taper,
        fft_offset=fft_offset,
        ofdm_fft_offset=ofdm_offset,
        chirp_len=profile.chirp_len,
        ramp_len=round(RAMP_S * profile.sample_rate),
        pair_tol=round(CHIRP_PAIR_TOLERANCE_48K * scale),
        probe_silence_len=round(PROBE_SILENCE_S * profile.sample_rate),
        mfsk_bins_a=np.arange(profile.mfsk_a_start, profile.mfsk_a_start + MFSK_TONES, dtype=np.int64),
        mfsk_bins_b=np.arange(profile.mfsk_b_start, profile.mfsk_b_start + MFSK_TONES, dtype=np.int64),
        ofdm_bins=np.arange(profile.ofdm_first_bin, profile.ofdm_last_bin + 1, dtype=np.int64),
    )
