"""Profiles, link modes, protocol constants, timeouts and SNR thresholds.

Everything that parameterises the modem and the protocol lives here as frozen
dataclasses or module constants so that the transmitter, the receiver and the
simulator always agree.  Nothing in this module touches audio hardware or Qt.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Final

# --------------------------------------------------------------------------------------
# Physical-layer profiles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """Frequency/timing numerology of one modem family.

    All tones sit exactly on FFT bin centres (``bin * sample_rate / n_fft``) so a
    rectangular-window FFT of the N-sample symbol body yields orthogonal tones.
    """

    key: str
    name: str
    sample_rate: int
    n_fft: int
    guard: int
    chirp_f0: float
    chirp_f1: float
    chirp_len: int
    mfsk_a_start: int
    mfsk_b_start: int
    ofdm_first_bin: int
    ofdm_last_bin: int
    experimental: bool = False

    @property
    def symbol_len(self) -> int:
        """Samples per symbol (FFT body + guard)."""
        return self.n_fft + self.guard

    @property
    def bin_hz(self) -> float:
        """FFT bin spacing in Hz."""
        return self.sample_rate / self.n_fft

    @property
    def symbol_rate(self) -> float:
        """Symbols per second."""
        return self.sample_rate / self.symbol_len

    @property
    def n_candidate_carriers(self) -> int:
        """Number of OFDM candidate carriers (always 52 for the shipped profiles)."""
        return self.ofdm_last_bin - self.ofdm_first_bin + 1

    @property
    def band_low_hz(self) -> float:
        """Lower edge of the occupied band (chirp or carriers, whichever is lower)."""
        return min(self.chirp_f0, self.ofdm_first_bin * self.bin_hz)

    @property
    def band_high_hz(self) -> float:
        """Upper edge of the occupied band."""
        return max(self.chirp_f1, self.ofdm_last_bin * self.bin_hz)


PROFILE_NU: Final = Profile(
    key="nu",
    name="Near-ultrasonic (17.5–20 kHz)",
    sample_rate=48_000,
    n_fft=1024,
    guard=256,
    chirp_f0=17_000.0,
    chirp_f1=20_000.0,
    chirp_len=2048,
    mfsk_a_start=376,
    mfsk_b_start=400,
    ofdm_first_bin=374,
    ofdm_last_bin=425,
)

PROFILE_AU: Final = Profile(
    key="au",
    name="Audible compatibility (3–5 kHz)",
    sample_rate=48_000,
    n_fft=1024,
    guard=256,
    chirp_f0=2_500.0,
    chirp_f1=5_500.0,
    chirp_len=2048,
    mfsk_a_start=64,
    mfsk_b_start=88,
    ofdm_first_bin=60,
    ofdm_last_bin=111,
)

PROFILE_US: Final = Profile(
    key="us",
    name="Ultrasonic experimental (21–23.5 kHz, 96 kHz)",
    sample_rate=96_000,
    n_fft=2048,
    guard=512,
    chirp_f0=20_500.0,
    chirp_f1=23_500.0,
    # Same chirp duration (42.7 ms) as the 48 kHz profiles.
    chirp_len=4096,
    mfsk_a_start=448,
    mfsk_b_start=472,
    ofdm_first_bin=444,
    ofdm_last_bin=495,
    experimental=True,
)

PROFILES: Final[dict[str, Profile]] = {p.key: p for p in (PROFILE_NU, PROFILE_AU, PROFILE_US)}


def get_profile(key: str) -> Profile:
    """Return the profile for ``key`` ("nu", "au", "us")."""
    try:
        return PROFILES[key.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown profile {key!r}; expected one of {sorted(PROFILES)}") from exc


def profiles_for_rate(sample_rate: int) -> list[Profile]:
    """Profiles that can be received simultaneously from a capture stream at ``sample_rate``."""
    return [p for p in PROFILES.values() if p.sample_rate == sample_rate]


# --------------------------------------------------------------------------------------
# Waveform constants
# --------------------------------------------------------------------------------------

#: Number of MFSK tones per subband (one nibble).
MFSK_TONES: Final = 16
#: Taper length (samples at 48 kHz; scaled with the profile) for symbol cross-fades.
SYMBOL_TAPER_48K: Final = 64
#: Burst fade-in / fade-out (raised cosine) duration.
RAMP_S: Final = 0.005
#: Tukey alpha for the chirps.
CHIRP_TUKEY_ALPHA: Final = 0.1
#: Sync word (4 MFSK symbols / bytes) that follows the chirp preamble.
SYNC_WORD: Final = bytes([0x5A, 0xC3, 0x96, 0x3C])
#: Minimum number of correct sync symbols to accept a detection.
SYNC_MIN_CORRECT: Final = 3
#: PHY header: 6 plaintext bytes + RS(nsym=8).
PHY_HEADER_LEN: Final = 6
PHY_HEADER_NSYM: Final = 8
#: Number of known OFDM training symbols in a PROBE body.
PROBE_TRAINING_SYMBOLS: Final = 4
#: Silence gap in a PROBE body used for the noise measurement.
PROBE_SILENCE_S: Final = 0.25
#: Every Nth enabled OFDM carrier is a pilot.
PILOT_SPACING: Final = 8
#: Clip level for OFDM PAPR reduction, relative to RMS (dB).
OFDM_CLIP_DB: Final = 8.0
#: OFDM sections (probe training and data alike) are scaled to RMS = peak - this backoff,
#: so per-carrier power is identical for probe and data symbols.
OFDM_BACKOFF_DB: Final = 10.0
#: Iterations of clip-and-filter PAPR reduction.
OFDM_CLIP_ITERATIONS: Final = 2
#: Peak level hard limit (dBFS) regardless of the user slider.
MAX_PEAK_DBFS: Final = -3.0
MIN_PEAK_DBFS: Final = -20.0
DEFAULT_TX_LEVEL_DBFS: Final = -6.0
#: Fraction of the guard skipped before the MFSK FFT window (timing tolerance vs multipath;
#: MFSK sync/header are demodulated before sample-rate correction).
FFT_OFFSET_FRACTION: Final = 0.5
#: OFDM bodies are demodulated after resampling and fine re-timing, so their FFT window can
#: use 7/8 of the guard to absorb more multipath (measured ~+0.8 dB carrier SNR at RT60 0.3 s).
OFDM_FFT_OFFSET_FRACTION: Final = 0.875
#: Allowed sample-rate mismatch between machines (beyond this a postamble is rejected).
MAX_PPM: Final = 500.0
#: Tolerance on the up/down chirp spacing (samples at 48 kHz, scaled with profile).
CHIRP_PAIR_TOLERANCE_48K: Final = 64

# --------------------------------------------------------------------------------------
# Link modes
# --------------------------------------------------------------------------------------

PHY_MFSK: Final = "mfsk"
PHY_OFDM1: Final = "ofdm1"  # DBPSK
PHY_OFDM2: Final = "ofdm2"  # DQPSK

NSYM_C0: Final = 64
NSYM_C1: Final = 32
NSYM_C2: Final = 16


@dataclass(frozen=True)
class ModeSpec:
    """One row of the link mode table (§3.7)."""

    mode_id: int
    phy: str
    nsym: int
    label: str

    @property
    def is_ofdm(self) -> bool:
        """True for the OFDM modes."""
        return self.phy != PHY_MFSK

    @property
    def bits_per_carrier(self) -> int:
        """Bits per data carrier per OFDM symbol (0 for MFSK)."""
        return {PHY_MFSK: 0, PHY_OFDM1: 1, PHY_OFDM2: 2}[self.phy]


MODES: Final[dict[int, ModeSpec]] = {
    0: ModeSpec(0, PHY_MFSK, NSYM_C0, "R/C0 (MFSK, robust)"),
    1: ModeSpec(1, PHY_MFSK, NSYM_C1, "R/C1 (MFSK)"),
    2: ModeSpec(2, PHY_OFDM1, NSYM_C0, "F1/C0 (OFDM DBPSK)"),
    3: ModeSpec(3, PHY_OFDM1, NSYM_C1, "F1/C1 (OFDM DBPSK)"),
    4: ModeSpec(4, PHY_OFDM2, NSYM_C0, "F2/C0 (OFDM DQPSK)"),
    5: ModeSpec(5, PHY_OFDM2, NSYM_C1, "F2/C1 (OFDM DQPSK)"),
}
CONTROL_MODE: Final = 0
PROBE_MODE_ID: Final = 0xFE
ALL_MODES_MASK: Final = 0b111111
MAX_MODE: Final = 5


def get_mode(mode_id: int) -> ModeSpec:
    """Return the :class:`ModeSpec` for ``mode_id`` (the probe pseudo-mode maps to mode 0 coding)."""
    if mode_id == PROBE_MODE_ID:
        return MODES[CONTROL_MODE]
    try:
        return MODES[mode_id]
    except KeyError as exc:
        raise ValueError(f"unknown mode id {mode_id}") from exc


# --------------------------------------------------------------------------------------
# SNR thresholds for mode selection.
#
# Units: the receiver's *estimator* outputs — MFSK per-tone bin SNR for modes 0/1
# and mean per-carrier SNR over the enabled carriers for OFDM modes 2–5 (probe
# measurement or differential-EVM estimate, which agree within ~0.5 dB).
# Derived from docs/SNR_CURVES.md (tools/snr_curves.py: 50 frames per point,
# 1 dB steps, AWGN + 150 ppm): the lowest selection-metric SNR at which the
# frame error rate is <= 2 % at that and every stronger point, rounded up to
# 0.5 dB.  On the 1 dB grid C0 and C1 OFDM curves coincide, so the C1 entries
# are nudged 0.5 dB above C0.  Mode selection adds MODE_SNR_MARGIN_DB on top.
# --------------------------------------------------------------------------------------

REQUIRED_SNR_DB: Final[dict[int, float]] = {
    0: 10.5,
    1: 11.5,
    2: 7.5,
    3: 8.0,
    4: 10.5,
    5: 12.5,
}
MODE_SNR_MARGIN_DB: Final = 3.0
#: An OFDM carrier is enabled only if its SNR is at least this far above the
#: mode's required mean SNR minus this slack.
CARRIER_SLACK_DB: Final = 3.0
#: Minimum number of enabled carriers for an OFDM mode to be considered.
MIN_OFDM_CARRIERS: Final = 16
#: Differential detection doubles noise; EVM-derived SNR is corrected by this.
DIFFERENTIAL_SNR_CORRECTION_DB: Final = 3.0

# --------------------------------------------------------------------------------------
# Protocol constants
# --------------------------------------------------------------------------------------

PROTO_VERSION: Final = 1
LINK_MAGIC: Final = 0xA5
UNIT_SIZE: Final = 32
MAX_FRAME_AIRTIME_S: Final = 6.0
MAX_OFDM_BODY_AIRTIME_S: Final = 4.0
WINDOW_AIRTIME_CAP_S: Final = 20.0
MAX_MANIFEST_BYTES: Final = 256 * 1024
MAX_MANIFEST_FILES: Final = 1000
MAX_SENDER_NAME_BYTES: Final = 32
SALT_LEN: Final = 16
ACK_BITMAP_BITS: Final = 64
MAX_ACK_RANGES: Final = 4
MAX_ENDACK_RANGES: Final = 8
MAX_RESUME_RANGES: Final = 48
COMPRESSION_MIN_SAVING: Final = 0.05
SCRYPT_N: Final = 2**15
SCRYPT_R: Final = 8
SCRYPT_P: Final = 1
AUTH_FAILURE_LIMIT: Final = 3
PARTIAL_DIR_NAME: Final = ".airdrop-partial"
DEFAULT_MAX_ACCEPT_BYTES: Final = 50 * 1024 * 1024


# --------------------------------------------------------------------------------------
# User-tunable settings
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Runtime settings.  Persisted by the GUI via QSettings; the CLI uses defaults/flags."""

    detect_ratio: float = 8.0
    erasure_ratio: float = 1.6
    window_mfsk: int = 6
    window_ofdm: int = 8
    tx_level_dbfs: float = DEFAULT_TX_LEVEL_DBFS
    offer_listen_s: float = 6.0
    offer_max_sends: int = 5
    ack_extra_s: float = 2.5
    max_polls: int = 6
    reply_delay_s: float = 0.3
    rx_window_extra_s: float = 3.0
    rx_inactivity_s: float = 90.0
    probe_attempts: int = 3
    end_max_attempts: int = 6
    half_duplex_tail_s: float = 0.25
    max_accept_bytes: int = DEFAULT_MAX_ACCEPT_BYTES
    auto_accept_bytes: int = 0
    compression: bool = True
    exclusive_mode: bool = False
    broadcast_passes: int = 2
    broadcast_mode: int = 1
    output_latency_s: float = 0.15
    log_level: str = "INFO"
    supported_modes: int = ALL_MODES_MASK
    linger_s: float = 30.0

    def replace(self, **changes: object) -> Settings:
        """Return a copy with ``changes`` applied (validated)."""
        new = dataclasses.replace(self, **changes)  # type: ignore[arg-type]
        new.validate()
        return new

    def validate(self) -> None:
        """Raise ``ValueError`` if any setting is out of range."""
        if not 2.0 <= self.detect_ratio <= 100.0:
            raise ValueError("detect_ratio must be within [2, 100]")
        if not 1.0 <= self.erasure_ratio <= 10.0:
            raise ValueError("erasure_ratio must be within [1, 10]")
        if not 1 <= self.window_mfsk <= 32 or not 1 <= self.window_ofdm <= 32:
            raise ValueError("window sizes must be within [1, 32]")
        if not MIN_PEAK_DBFS <= self.tx_level_dbfs <= MAX_PEAK_DBFS:
            raise ValueError(f"tx level must be within [{MIN_PEAK_DBFS}, {MAX_PEAK_DBFS}] dBFS")
        if self.max_accept_bytes < 0 or self.auto_accept_bytes < 0:
            raise ValueError("size limits must be non-negative")
        if self.broadcast_mode not in MODES:
            raise ValueError("broadcast_mode must be a valid data mode")
        if self.broadcast_passes < 1:
            raise ValueError("broadcast_passes must be >= 1")
        for name in (
            "offer_listen_s",
            "ack_extra_s",
            "reply_delay_s",
            "rx_window_extra_s",
            "rx_inactivity_s",
            "half_duplex_tail_s",
            "output_latency_s",
            "linger_s",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


DEFAULT_SETTINGS: Final = Settings()


@dataclass
class TransferOptions:
    """Per-transfer choices made by the sending user."""

    profile_key: str = "nu"
    broadcast: bool = False
    encrypt: bool = False
    passphrase: str = ""
    compression: bool = True
    sender_name: str = "Ultrasonic AirDrop"
    fixed_mode: int | None = None
    extra: dict[str, object] = field(default_factory=dict)
