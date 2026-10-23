"""Link-frame body bytes <-> audio samples (the complete transmit chain) and airtime maths.

On-air layout of one burst (§3.8)::

    [up-chirp][down-chirp][sync word: 4 MFSK][PHY header: 14 MFSK, RS nsym=8]
    [body in the PHY header's mode][guard gap][postamble up-chirp]

with 5 ms raised-cosine fades at both ends.  :func:`frame_airtime` is the single
source of truth for scheduling, timeouts, frame sizing and ETAs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultrasonic_airdrop.config import (
    OFDM_BACKOFF_DB,
    PROBE_MODE_ID,
    PROBE_TRAINING_SYMBOLS,
    SYNC_WORD,
    Profile,
    get_mode,
)
from ultrasonic_airdrop.fec.rs import coded_length, encode_body
from ultrasonic_airdrop.link.phy_header import PhyHeader
from ultrasonic_airdrop.phy import mfsk, ofdm
from ultrasonic_airdrop.phy.chirp import chirp
from ultrasonic_airdrop.phy.numerology import FloatArray, Numerology, numerology
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.phy.synth import apply_ramps, peak_amplitude


@dataclass(frozen=True)
class BodyLayout:
    """Sample geometry of a frame body."""

    mode_id: int
    coded_len: int
    mfsk_symbols: int
    ofdm_symbols: int  # including the reference symbol (0 for MFSK modes)
    extra_samples: int  # probe training + silence
    body_samples: int


def body_layout(profile: Profile, mode_id: int, body_len: int, n_enabled: int | None = None) -> BodyLayout:
    """Compute the body geometry for ``body_len`` plaintext bytes in ``mode_id``.

    ``n_enabled`` is the number of enabled OFDM carriers (default: all 52).
    """
    num = numerology(profile)
    mode = get_mode(mode_id)
    coded = coded_length(body_len, mode.nsym)
    if mode_id == PROBE_MODE_ID:
        extra = PROBE_TRAINING_SYMBOLS * num.sym_len + num.probe_silence_len
        return BodyLayout(mode_id, coded, coded, 0, extra, coded * num.sym_len + extra)
    if not mode.is_ofdm:
        return BodyLayout(mode_id, coded, coded, 0, 0, coded * num.sym_len)
    n_en = profile.n_candidate_carriers if n_enabled is None else n_enabled
    mask = CarrierMask.from_array(np.arange(profile.n_candidate_carriers) < n_en)
    lay = ofdm.layout(num, mask)
    n_sym = ofdm.n_data_symbols(coded, lay.n_data, mode.bits_per_carrier) + 1
    return BodyLayout(mode_id, coded, 0, n_sym, 0, n_sym * num.sym_len)


def frame_samples(profile: Profile, mode_id: int, body_len: int, n_enabled: int | None = None) -> int:
    """Total burst length in samples."""
    num = numerology(profile)
    return num.frame_length(body_layout(profile, mode_id, body_len, n_enabled).body_samples)


def frame_airtime(profile: Profile, mode_id: int, body_len: int, n_enabled: int | None = None) -> float:
    """Airtime in seconds of one frame carrying ``body_len`` plaintext body bytes."""
    return frame_samples(profile, mode_id, body_len, n_enabled) / profile.sample_rate


def max_body_len(profile: Profile, mode_id: int, max_airtime_s: float, n_enabled: int | None = None) -> int:
    """Largest body length whose frame airtime stays within ``max_airtime_s`` (0 if none)."""
    lo, hi = 0, 4096
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if frame_airtime(profile, mode_id, mid, n_enabled) <= max_airtime_s:
            lo = mid
        else:
            hi = mid - 1
    return lo


class FrameCodec:
    """Builds transmit bursts for one profile and transmit level."""

    def __init__(self, profile: Profile, tx_level_dbfs: float) -> None:
        self.profile = profile
        self.num: Numerology = numerology(profile)
        self.peak = peak_amplitude(tx_level_dbfs)

    def set_level(self, tx_level_dbfs: float) -> None:
        """Change the transmit level."""
        self.peak = peak_amplitude(tx_level_dbfs)

    def _assemble(self, mfsk_bytes: bytes, body_samples: int, extra: list[tuple[int, FloatArray]]) -> FloatArray:
        num = self.num
        total = num.frame_length(body_samples)
        buf = np.zeros(total)
        c = num.chirp_len
        buf[0:c] += self.peak * chirp(self.profile, True)
        buf[c : 2 * c] += self.peak * chirp(self.profile, False)
        sig = mfsk.modulate(mfsk_bytes, num, self.peak, abs_start=num.sync_start)
        end = min(total, num.sync_start + len(sig))
        buf[num.sync_start : end] += sig[: end - num.sync_start]
        for offset, samples in extra:
            stop = min(total, offset + len(samples))
            buf[offset:stop] += samples[: stop - offset]
        post = num.postamble_start(body_samples)
        buf[post : post + c] += self.peak * chirp(self.profile, True)
        return apply_ramps(buf, num.ramp_len)

    def _scaled(self, samples: FloatArray) -> FloatArray:
        """Scale an OFDM section to the fixed RMS backoff and hard-limit rare excess peaks."""
        body = samples[: len(samples) - self.num.taper] if len(samples) > self.num.taper else samples
        rms = float(np.sqrt(np.mean(body * body))) if len(body) else 0.0
        if rms <= 0:
            return samples
        target = self.peak / 10.0 ** (OFDM_BACKOFF_DB / 20.0)
        return np.asarray(np.clip(samples * (target / rms), -self.peak, self.peak), dtype=np.float64)

    def encode(self, body: bytes, mode_id: int, mask: CarrierMask | None = None) -> FloatArray:
        """Build the burst for a link-frame ``body`` sent in ``mode_id``."""
        if mode_id == PROBE_MODE_ID:
            return self.encode_probe(body)
        mode = get_mode(mode_id)
        mask = mask or CarrierMask.all()
        coded = encode_body(body, mode.nsym)
        header = PhyHeader(mode_id, mask.mask_id if mode.is_ofdm else 0, len(body)).encode()
        num = self.num
        if not mode.is_ofdm:
            lay = body_layout(self.profile, mode_id, len(body))
            return self._assemble(SYNC_WORD + header + coded, lay.body_samples, [])
        car = ofdm.layout(num, mask)
        sig = self._scaled(ofdm.modulate_body(coded, mode.bits_per_carrier, car, num))
        body_samples = len(sig) - num.taper
        return self._assemble(SYNC_WORD + header, body_samples, [(num.body_start, sig)])

    def encode_probe(self, body: bytes) -> FloatArray:
        """Build a PROBE burst: mode-0 MFSK link frame, 4 training symbols, silence gap."""
        num = self.num
        mode = get_mode(PROBE_MODE_ID)
        coded = encode_body(body, mode.nsym)
        header = PhyHeader(PROBE_MODE_ID, 0, len(body)).encode()
        lay = body_layout(self.profile, PROBE_MODE_ID, len(body))
        train = self._scaled(ofdm.training_samples(num))
        train_start = num.body_start + lay.mfsk_symbols * num.sym_len
        return self._assemble(SYNC_WORD + header + coded, lay.body_samples, [(train_start, train)])
