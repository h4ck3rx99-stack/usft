"""Link-quality aggregation on the receiver: SNR tracking, FER window, per-carrier estimates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ultrasonic_airdrop.phy.numerology import FloatArray

EMA_ALPHA = 0.5
N_CARRIERS = 52


@dataclass
class QualityTracker:
    """Tracks what the receiver measured, for ACK feedback and display.

    * ``carrier_snr_db``: latest per-carrier SNR (OFDM-equivalent), from the probe
      or OFDM frames (EMA), NaN if never measured.
    * ``mfsk_snr_db``: EMA of MFSK SNR; ``_mfsk_ref`` is the MFSK SNR measured at
      the time the carrier estimate was last refreshed so that a broad level
      change seen on MFSK frames can shift the carrier estimate.
    """

    carrier_snr_db: FloatArray = field(default_factory=lambda: np.full(N_CARRIERS, np.nan))
    mfsk_snr_db: float = float("nan")
    _mfsk_ref: float = float("nan")
    window_frames: int = 0
    window_errors: int = 0
    window_snr: list[float] = field(default_factory=list)
    total_frames: int = 0
    total_errors: int = 0
    last_mode: int = 0

    def _ema(self, old: float, new: float) -> float:
        if np.isnan(old):
            return new
        return (1 - EMA_ALPHA) * old + EMA_ALPHA * new

    def record_probe(self, carrier_snr: FloatArray, mfsk_snr: float) -> None:
        """Replace the carrier estimate with a fresh probe measurement."""
        self.carrier_snr_db = np.asarray(carrier_snr, dtype=np.float64).copy()
        if not np.isnan(mfsk_snr):
            self.mfsk_snr_db = self._ema(self.mfsk_snr_db, mfsk_snr)
            self._mfsk_ref = mfsk_snr

    def record_frame(
        self,
        ok: bool,
        mode_id: int | None,
        snr_db: float,
        mfsk_snr_db: float,
        carrier_snr: FloatArray | None,
    ) -> None:
        """Account one received (or failed) frame."""
        self.window_frames += 1
        self.total_frames += 1
        if mode_id is not None:
            self.last_mode = mode_id
        if not ok:
            self.window_errors += 1
            self.total_errors += 1
            return
        if not np.isnan(snr_db):
            self.window_snr.append(snr_db)
        if carrier_snr is not None:
            valid = ~np.isnan(carrier_snr)
            old = self.carrier_snr_db
            upd = np.where(np.isnan(old), carrier_snr, (1 - EMA_ALPHA) * old + EMA_ALPHA * carrier_snr)
            self.carrier_snr_db = np.where(valid, upd, old)
            if not np.isnan(mfsk_snr_db):
                self._mfsk_ref = mfsk_snr_db
        if not np.isnan(mfsk_snr_db):
            self.mfsk_snr_db = self._ema(self.mfsk_snr_db, mfsk_snr_db)

    def effective_carrier_snr(self) -> FloatArray:
        """Carrier estimate shifted by the MFSK level change since it was measured."""
        shift = 0.0
        if not np.isnan(self._mfsk_ref) and not np.isnan(self.mfsk_snr_db):
            shift = min(0.0, self.mfsk_snr_db - self._mfsk_ref) + 0.5 * max(0.0, self.mfsk_snr_db - self._mfsk_ref)
        return self.carrier_snr_db + shift

    @property
    def window_fer(self) -> float:
        """Frame error rate in the current ACK window."""
        return self.window_errors / self.window_frames if self.window_frames else 0.0

    def window_mean_snr(self) -> float:
        """Mean SNR (dB) of frames in the current window (NaN if none)."""
        return float(np.mean(self.window_snr)) if self.window_snr else float("nan")

    def reset_window(self) -> None:
        """Start a new ACK window."""
        self.window_frames = 0
        self.window_errors = 0
        self.window_snr = []
