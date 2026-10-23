"""Differential OFDM modem: "F1" (DBPSK) and "F2" (DQPSK).

* Carriers are the subset of the profile's 52 candidate bins enabled by a
  :class:`CarrierMask`; every 8th enabled carrier is a pilot carrying a known
  pseudo-random differential BPSK sequence.
* Each carrier's data is the phase change from that carrier's previous symbol,
  so no channel equalisation is needed.  A known reference symbol starts the body.
* A cyclic prefix of ``G`` samples absorbs multipath; a common phase and a linear
  phase slope (residual timing) are estimated from pilots and removed.
* Bytes are mapped "time-first": each byte lives on a single carrier across
  ``8 / bits_per_carrier`` consecutive symbols, so a bad carrier corrupts whole
  bytes (which can then be flagged as erasures) instead of a bit of every byte.
* PAPR is limited by iterative clip-and-filter in the frequency domain, which
  keeps all energy on the enabled bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ultrasonic_airdrop.config import (
    DIFFERENTIAL_SNR_CORRECTION_DB,
    OFDM_CLIP_DB,
    OFDM_CLIP_ITERATIONS,
    PILOT_SPACING,
    PROBE_TRAINING_SYMBOLS,
)
from ultrasonic_airdrop.fec.crc import crc16_ccitt_false
from ultrasonic_airdrop.phy.numerology import (
    BoolArray,
    ComplexArray,
    FloatArray,
    IntArray,
    Numerology,
)
from ultrasonic_airdrop.phy.synth import overlap_add_symbols, symbol_window

N_CANDIDATES = 52
MASK_BYTES = 7

#: DQPSK Gray mapping: value -> differential phasor.
_DQPSK = np.array([1.0 + 0j, 1j, -1j, -1.0 + 0j])
#: quadrant index (round(angle / (pi/2)) mod 4) -> value
_QUADRANT_TO_VALUE = np.array([0, 1, 3, 2], dtype=np.int64)
#: Byte erasure threshold on decision confidence (0..1).
BYTE_ERASE_CONF = 0.12
#: Carriers whose EVM-derived SNR is below this (dB) are erased entirely.
CARRIER_ERASE_SNR_DB = {1: 1.0, 2: 5.0}


# --------------------------------------------------------------------------------------
# Carrier masks
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CarrierMask:
    """Which of the 52 candidate carriers are enabled (bit i = candidate bin i)."""

    bits: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.bits) != N_CANDIDATES:
            raise ValueError("carrier mask must have 52 entries")

    @classmethod
    def all(cls) -> CarrierMask:
        """All candidate carriers enabled."""
        return cls(tuple([True] * N_CANDIDATES))

    @classmethod
    def from_array(cls, arr: BoolArray) -> CarrierMask:
        """Build from a boolean array of length 52."""
        return cls(tuple(bool(b) for b in np.asarray(arr, dtype=bool)))

    @classmethod
    def from_bytes(cls, raw: bytes) -> CarrierMask:
        """Decode the 7-byte little-endian bitmap used on the wire."""
        if len(raw) != MASK_BYTES:
            raise ValueError("carrier mask must be 7 bytes")
        value = int.from_bytes(raw, "little")
        if value >> N_CANDIDATES:
            raise ValueError("carrier mask has bits set beyond carrier 51")
        return cls(tuple(bool((value >> i) & 1) for i in range(N_CANDIDATES)))

    def to_bytes(self) -> bytes:
        """Encode as a 7-byte little-endian bitmap."""
        value = sum(1 << i for i, b in enumerate(self.bits) if b)
        return value.to_bytes(MASK_BYTES, "little")

    def as_array(self) -> BoolArray:
        """Boolean numpy view."""
        return np.array(self.bits, dtype=bool)

    @property
    def count(self) -> int:
        """Number of enabled carriers."""
        return sum(self.bits)

    @property
    def is_full(self) -> bool:
        """True if every candidate carrier is enabled."""
        return all(self.bits)

    @property
    def mask_id(self) -> int:
        """Session-independent id: 0 for the full mask, else ``crc16(bytes) % 255 + 1``."""
        if self.is_full:
            return 0
        return crc16_ccitt_false(self.to_bytes()) % 255 + 1


@dataclass(frozen=True, eq=False)
class CarrierLayout:
    """Enabled carriers split into pilots and data carriers."""

    bins: IntArray
    pilot_idx: IntArray
    data_idx: IntArray

    @property
    def n_data(self) -> int:
        """Number of data carriers."""
        return len(self.data_idx)


def layout(num: Numerology, mask: CarrierMask) -> CarrierLayout:
    """Compute the carrier layout for ``mask`` (every 8th enabled carrier is a pilot)."""
    enabled = num.ofdm_bins[mask.as_array()]
    idx = np.arange(len(enabled))
    pilots = idx[idx % PILOT_SPACING == 0]
    data = idx[idx % PILOT_SPACING != 0]
    return CarrierLayout(bins=enabled, pilot_idx=pilots, data_idx=data)


# --------------------------------------------------------------------------------------
# Known sequences
# --------------------------------------------------------------------------------------


def prbs_bits(n: int, seed: int = 0x5D) -> IntArray:
    """PRBS-7 (x^7 + x^6 + 1) bit sequence; deterministic across platforms."""
    state = seed & 0x7F or 1
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        bit = ((state >> 6) ^ (state >> 5)) & 1
        state = ((state << 1) | bit) & 0x7F
        out[i] = bit
    return out


@lru_cache(maxsize=4)
def _prbs_cached(n: int) -> IntArray:
    return prbs_bits(n)


def pilot_symbols(n_symbols: int, n_pilots: int) -> ComplexArray:
    """Known differential pilot phasors (+1/-1), shape ``(n_symbols, n_pilots)``."""
    total = n_symbols * n_pilots
    bits = _prbs_cached(max(total, 1) + 127)[:total]
    return np.asarray(1.0 - 2.0 * bits.reshape(n_symbols, n_pilots), dtype=np.complex128)


def newman_phasors(n: int, offset: int = 0) -> ComplexArray:
    """Low-PAPR quadratic (Newman) phases for ``n`` carriers."""
    k = np.arange(n) + offset
    return np.asarray(np.exp(1j * np.pi * k * k / max(n, 1)), dtype=np.complex128)


# --------------------------------------------------------------------------------------
# Time-domain synthesis
# --------------------------------------------------------------------------------------


def _clip_filter(spectra: ComplexArray, bins: IntArray, num: Numerology) -> ComplexArray:
    """Iterative clip-and-filter PAPR reduction on full rfft spectra."""
    keep = np.zeros(spectra.shape[1], dtype=bool)
    keep[bins] = True
    spec = spectra.copy()
    for _ in range(OFDM_CLIP_ITERATIONS):
        x = np.fft.irfft(spec, num.n, axis=1)
        rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True)) + 1e-30
        limit = rms * 10.0 ** (OFDM_CLIP_DB / 20.0)
        x = np.clip(x, -limit, limit)
        spec = np.fft.rfft(x, axis=1)
        spec[:, ~keep] = 0.0
    return spec


def symbols_to_samples(values: ComplexArray, bins: IntArray, num: Numerology, clip: bool = True) -> FloatArray:
    """Turn per-symbol carrier phasors (``(n_sym, len(bins))``) into windowed CP-OFDM samples.

    Output length is ``n_sym * L + taper`` (unscaled).
    """
    n_sym = values.shape[0]
    spectra = np.zeros((n_sym, num.n // 2 + 1), dtype=np.complex128)
    spectra[:, bins] = values
    if clip:
        spectra = _clip_filter(spectra, bins, num)
    x = np.fft.irfft(spectra, num.n, axis=1)
    width = num.sym_len + num.taper
    cyc = (np.arange(width) - num.g) % num.n
    rows = x[:, cyc] * symbol_window(num.sym_len, num.taper)[None, :]
    return overlap_add_symbols(rows, num.sym_len)


def n_data_symbols(n_bytes: int, n_data_carriers: int, bits_per_carrier: int) -> int:
    """Number of OFDM data symbols needed for ``n_bytes`` coded bytes."""
    if n_data_carriers <= 0:
        raise ValueError("no data carriers")
    groups = -(-n_bytes // n_data_carriers)
    return groups * (8 // bits_per_carrier)


def bytes_to_values(data: bytes, n_data: int, bits: int) -> IntArray:
    """Map bytes to per-symbol carrier values (time-first), shape ``(n_sym, n_data)``."""
    spb = 8 // bits
    groups = -(-len(data) // n_data)
    arr = np.zeros(groups * n_data, dtype=np.int64)
    arr[: len(data)] = np.frombuffer(data, dtype=np.uint8)
    mat = arr.reshape(groups, n_data)
    vals = np.zeros((groups * spb, n_data), dtype=np.int64)
    mask = (1 << bits) - 1
    for t in range(spb):
        vals[t::spb, :] = (mat >> (8 - bits * (t + 1))) & mask
    return vals


def values_to_bytes(vals: IntArray, n_bytes: int, bits: int) -> bytes:
    """Inverse of :func:`bytes_to_values`."""
    spb = 8 // bits
    groups = vals.shape[0] // spb
    arr = np.zeros((groups, vals.shape[1]), dtype=np.int64)
    for t in range(spb):
        arr |= vals[t::spb, :] << (8 - bits * (t + 1))
    return arr.reshape(-1)[:n_bytes].astype(np.uint8).tobytes()


def modulate_body(coded: bytes, bits: int, lay: CarrierLayout, num: Numerology) -> FloatArray:
    """Modulate coded bytes into a reference symbol followed by differential data symbols."""
    vals = bytes_to_values(coded, lay.n_data, bits)
    n_sym = vals.shape[0]
    n_car = len(lay.bins)
    diff = np.ones((n_sym, n_car), dtype=np.complex128)
    diff[:, lay.data_idx] = (1.0 - 2.0 * vals) if bits == 1 else _DQPSK[vals]
    diff[:, lay.pilot_idx] = pilot_symbols(n_sym, len(lay.pilot_idx))
    ref = newman_phasors(n_car)
    phasors = np.empty((n_sym + 1, n_car), dtype=np.complex128)
    phasors[0] = ref
    phasors[1:] = ref[None, :] * np.cumprod(diff, axis=0)
    return symbols_to_samples(phasors, lay.bins, num)


def training_values(num: Numerology) -> ComplexArray:
    """Known probe training symbols on all candidate carriers, shape ``(4, 52)``."""
    return np.stack([newman_phasors(N_CANDIDATES, offset=7 * s) for s in range(PROBE_TRAINING_SYMBOLS)])


def training_samples(num: Numerology) -> FloatArray:
    """Time samples of the probe training symbols (unscaled)."""
    return symbols_to_samples(training_values(num), num.ofdm_bins, num)


# --------------------------------------------------------------------------------------
# Demodulation
# --------------------------------------------------------------------------------------


def symbol_fft(x: FloatArray, starts: IntArray, num: Numerology, bins: IntArray) -> ComplexArray:
    """Complex FFT values at ``bins`` of each symbol body."""
    idx = np.asarray(starts, dtype=np.int64)[:, None] + num.ofdm_fft_offset + np.arange(num.n)[None, :]
    if idx.size and (int(np.min(idx)) < 0 or int(np.max(idx)) >= len(x)):
        raise IndexError("OFDM symbol window outside buffer")
    return np.asarray(np.fft.rfft(x[idx], axis=1)[:, bins], dtype=np.complex128)


def _remove_common_phase(z: ComplexArray, lay: CarrierLayout) -> ComplexArray:
    """Estimate (from pilots) and remove a per-symbol common phase + linear slope."""
    n_sym = z.shape[0]
    if len(lay.pilot_idx) == 0:
        return z
    known = pilot_symbols(n_sym, len(lay.pilot_idx))
    q = z[:, lay.pilot_idx] * np.conj(known)
    kp = lay.bins[lay.pilot_idx].astype(np.float64)
    k_all = lay.bins.astype(np.float64)
    out = np.empty_like(z)
    for s in range(n_sym):
        slope = 0.0
        if len(kp) >= 2:
            prod = q[s, 1:] * np.conj(q[s, :-1])
            w = np.abs(prod)
            dk = np.diff(kp)
            if w.sum() > 0:
                slope = float(np.sum(w * np.angle(prod)) / np.sum(w * dk))
        common = float(np.angle(np.sum(q[s] * np.exp(-1j * slope * kp))))
        out[s] = z[s] * np.exp(-1j * (common + slope * k_all))
    return out


@dataclass
class OfdmResult:
    """Demodulation output for an OFDM body."""

    data: bytes
    erasures: BoolArray
    confidence: FloatArray
    carrier_snr_db: FloatArray
    snr_db: float


def demodulate_body(
    x: FloatArray, first_start: int, n_bytes: int, bits: int, lay: CarrierLayout, num: Numerology
) -> OfdmResult:
    """Demodulate an OFDM body whose reference symbol starts at ``first_start``."""
    n_sym = n_data_symbols(n_bytes, lay.n_data, bits)
    starts = first_start + np.arange(n_sym + 1) * num.sym_len
    y = symbol_fft(x, starts, num, lay.bins)
    z = y[1:] * np.conj(y[:-1])
    z = _remove_common_phase(z, lay)
    # Per-carrier amplitude normalisation and EVM-based SNR.
    amp = np.mean(np.abs(z), axis=0) + 1e-30
    u = z / amp[None, :]
    if bits == 1:
        vals = (u.real < 0).astype(np.int64)
        ideal = 1.0 - 2.0 * vals
        conf = np.abs(u.real) / (np.abs(u) + 1e-30)
    else:
        ang = np.angle(u)
        quad = np.round(ang / (np.pi / 2)).astype(np.int64) % 4
        vals = _QUADRANT_TO_VALUE[quad]
        ideal = _DQPSK[vals]
        err = np.abs(np.angle(u * np.conj(ideal)))
        conf = np.clip(1.0 - err / (np.pi / 4), 0.0, 1.0)
    # Pilots: known ideal points.
    known = pilot_symbols(n_sym, len(lay.pilot_idx))
    ideal_all = np.ones_like(u)
    ideal_all[:, lay.data_idx] = ideal[:, lay.data_idx]
    ideal_all[:, lay.pilot_idx] = known
    evm = np.mean(np.abs(u - ideal_all) ** 2, axis=0) + 1e-6
    carrier_snr = 10.0 * np.log10(1.0 / evm) + DIFFERENTIAL_SNR_CORRECTION_DB
    data_vals = vals[:, lay.data_idx]
    data = values_to_bytes(data_vals, n_bytes, bits)
    # Byte confidence = min over its chunks; bytes on bad carriers are erased.
    spb = 8 // bits
    groups = n_sym // spb
    dconf = conf[:, lay.data_idx].reshape(groups, spb, lay.n_data).min(axis=1).reshape(-1)[:n_bytes]
    bad_carrier = carrier_snr[lay.data_idx] < CARRIER_ERASE_SNR_DB[bits]
    bad_bytes = np.tile(bad_carrier, groups)[:n_bytes]
    erasures = (dconf < BYTE_ERASE_CONF) | bad_bytes
    snr_db = float(10.0 * np.log10(np.mean(10.0 ** (carrier_snr / 10.0))))
    return OfdmResult(
        data=data,
        erasures=erasures,
        confidence=dconf,
        carrier_snr_db=carrier_snr,
        snr_db=snr_db,
    )


def analyze_probe(x: FloatArray, train_start: int, silence_start: int, silence_len: int, num: Numerology) -> FloatArray:
    """Per-candidate-carrier SNR (dB) from probe training symbols and the silence gap.

    Signal power is the coherent average over the training symbols; noise is the
    largest of (a) the background noise in the silence gap, (b) the reverberant
    tail measured right after the guard interval (an estimate of inter-symbol
    interference) and (c) the scatter of the training symbols around their mean
    channel estimate.
    """
    tv = training_values(num)
    starts = train_start + np.arange(tv.shape[0]) * num.sym_len
    y = symbol_fft(x, starts, num, num.ofdm_bins)
    h = y / tv
    h_mean = h.mean(axis=0)
    sig = np.abs(h_mean) ** 2
    scatter = np.sum(np.abs(h - h_mean[None, :]) ** 2, axis=0) / max(tv.shape[0] - 1, 1)
    skip = num.g
    hop = num.n // 2
    win_starts = np.arange(silence_start + skip, silence_start + silence_len - num.n, hop)
    if len(win_starts):
        idx = win_starts[:, None] + np.arange(num.n)[None, :]
        pw = np.abs(np.fft.rfft(x[idx], axis=1)[:, num.ofdm_bins]) ** 2
        # The first window (energy arriving G..G+N after the last symbol) is the reverberant
        # tail that leaks past the guard interval as inter-symbol interference; the rest of
        # the gap measures background noise.  Use the worse of the two.
        late = pw[2:].mean(axis=0) if len(pw) > 2 else pw.mean(axis=0)
        # Smooth the single tail window over neighbouring carriers (reverberation is
        # broadband) so its variance does not bias the estimate low on a clean channel.
        tail = np.convolve(pw[0], np.ones(5) / 5.0, mode="same")
        tail[:2] = pw[0][:2]
        tail[-2:] = pw[0][-2:]
        noise_sil = np.maximum(tail, late)
    else:
        noise_sil = scatter
    noise = np.maximum(noise_sil, scatter) + 1e-30
    snr = np.maximum(sig - noise_sil, 1e-30) / noise
    return np.asarray(10.0 * np.log10(snr), dtype=np.float64)
