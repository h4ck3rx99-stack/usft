"""Mode selection (receiver side) and rate control (sender side).

Receiver: :func:`recommend` picks the highest mode whose required SNR (from
``config.REQUIRED_SNR_DB``, derived from simulation) is met with
``MODE_SNR_MARGIN_DB`` of headroom, and a carrier mask of carriers good enough
for that mode.

Sender: :class:`RateController` applies the rules of §4.3 step 5:
step down one mode at once if the window frame-error rate exceeds 25 % or two
consecutive ACKs are missed; step up at most one mode after three consecutive
clean windows, and only if the receiver suggests a higher mode; never exceed
the receiver's suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultrasonic_airdrop.config import (
    CARRIER_SLACK_DB,
    MAX_MODE,
    MIN_OFDM_CARRIERS,
    MODE_SNR_MARGIN_DB,
    MODES,
    REQUIRED_SNR_DB,
)
from ultrasonic_airdrop.phy.numerology import FloatArray
from ultrasonic_airdrop.phy.ofdm import CarrierMask

FER_STEP_DOWN = 0.25
CLEAN_WINDOWS_TO_STEP_UP = 3
MISSED_ACKS_TO_STEP_DOWN = 2


def supported(mode_id: int, caps: int) -> bool:
    """Is ``mode_id`` in the capability bitmap?"""
    return bool(caps >> mode_id & 1)


def recommend(
    carrier_snr_db: FloatArray, mfsk_snr_db: float, caps: int = 0x3F, margin_db: float = MODE_SNR_MARGIN_DB
) -> tuple[int, CarrierMask]:
    """Return ``(mode_id, mask)`` for the measured SNRs."""
    snr = np.asarray(carrier_snr_db, dtype=np.float64)
    if snr.size == 52 and not np.all(np.isnan(snr)):
        for mode_id in sorted((m for m in MODES if MODES[m].is_ofdm), reverse=True):
            if not supported(mode_id, caps):
                continue
            need = REQUIRED_SNR_DB[mode_id] + margin_db
            mask_arr = np.nan_to_num(snr, nan=-99.0) >= need - CARRIER_SLACK_DB
            if mask_arr.sum() < MIN_OFDM_CARRIERS:
                continue
            # Median in dB: robust to a few carriers whose (4-symbol) estimate is
            # optimistic, which a linear mean over a hand-picked subset would reward.
            typical_db = float(np.median(snr[mask_arr]))
            if typical_db >= need:
                return mode_id, CarrierMask.from_array(mask_arr)
    full = CarrierMask.all()
    if not np.isnan(mfsk_snr_db) and supported(1, caps) and mfsk_snr_db >= REQUIRED_SNR_DB[1] + margin_db:
        return 1, full
    return 0, full


@dataclass
class RateController:
    """Sender-side mode state."""

    mode: int
    caps: int = 0x3F
    clean_windows: int = 0
    missed_acks: int = 0
    last_suggestion: int = MAX_MODE

    def _clamp(self, mode: int) -> int:
        mode = max(0, min(mode, MAX_MODE, self.last_suggestion))
        while mode > 0 and not supported(mode, self.caps):
            mode -= 1
        return mode

    def step_down(self) -> int:
        """Drop one mode."""
        self.mode = self._clamp(self.mode - 1)
        self.clean_windows = 0
        return self.mode

    def on_ack(self, frames_in_window: int, lost: int, suggested: int) -> int:
        """Update after an ACK; returns the mode to use next."""
        self.missed_acks = 0
        self.last_suggestion = suggested
        fer = lost / frames_in_window if frames_in_window else 0.0
        if fer > FER_STEP_DOWN:
            return self.step_down()
        if lost == 0 and frames_in_window > 0:
            self.clean_windows += 1
        elif lost:
            self.clean_windows = 0
        if self.clean_windows >= CLEAN_WINDOWS_TO_STEP_UP and suggested > self.mode:
            self.mode = self._clamp(self.mode + 1)
            self.clean_windows = 0
        self.mode = self._clamp(self.mode)
        return self.mode

    def on_ack_missed(self) -> int:
        """Account a missed ACK; returns the mode to use next."""
        self.missed_acks += 1
        if self.missed_acks >= MISSED_ACKS_TO_STEP_DOWN:
            self.missed_acks = 0
            return self.step_down()
        return self.mode
