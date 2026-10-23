"""PHY header: 6 bytes (mode, mask id, body length, CRC-16) protected by RS(nsym=8)."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from ultrasonic_airdrop.config import MODES, PHY_HEADER_LEN, PHY_HEADER_NSYM, PROBE_MODE_ID
from ultrasonic_airdrop.errors import FecError, FrameError
from ultrasonic_airdrop.fec.crc import crc16_ccitt_false
from ultrasonic_airdrop.fec.rs import rs_decode, rs_encode
from ultrasonic_airdrop.phy.numerology import BoolArray, FloatArray

_FMT = "<BBH"
MAX_BODY_LEN = 16384


@dataclass(frozen=True)
class PhyHeader:
    """Decoded PHY header."""

    mode_id: int
    mask_id: int
    body_len: int

    def __post_init__(self) -> None:
        if self.mode_id not in MODES and self.mode_id != PROBE_MODE_ID:
            raise FrameError(f"unknown mode id {self.mode_id}")
        if not 0 <= self.mask_id <= 255:
            raise FrameError("mask id out of range")
        if not 0 < self.body_len <= MAX_BODY_LEN:
            raise FrameError(f"invalid body length {self.body_len}")

    def pack(self) -> bytes:
        """6 plaintext bytes."""
        head = struct.pack(_FMT, self.mode_id, self.mask_id, self.body_len)
        return head + struct.pack("<H", crc16_ccitt_false(head))

    def encode(self) -> bytes:
        """14 on-air bytes (plaintext + RS parity)."""
        return rs_encode(self.pack(), PHY_HEADER_NSYM)

    @classmethod
    def unpack(cls, raw: bytes) -> PhyHeader:
        """Parse 6 plaintext bytes, verifying the CRC."""
        if len(raw) != PHY_HEADER_LEN:
            raise FrameError("PHY header must be 6 bytes")
        (crc,) = struct.unpack_from("<H", raw, 4)
        if crc16_ccitt_false(raw[:4]) != crc:
            raise FrameError("PHY header CRC mismatch")
        mode_id, mask_id, body_len = struct.unpack_from(_FMT, raw, 0)
        return cls(mode_id, mask_id, body_len)

    @classmethod
    def decode(cls, coded: bytes, erasures: BoolArray | None = None, confidence: FloatArray | None = None) -> PhyHeader:
        """RS-decode 14 received bytes and parse.  Raises :class:`FrameError`."""
        if len(coded) != PHY_HEADER_LEN + PHY_HEADER_NSYM:
            raise FrameError("coded PHY header must be 14 bytes")
        erase: list[int] = []
        if erasures is not None:
            flagged = np.flatnonzero(erasures)
            if confidence is not None:
                flagged = flagged[np.argsort(confidence[flagged])]
            erase = sorted(int(i) for i in flagged[: PHY_HEADER_NSYM - 2])
        for attempt in (erase, []):
            try:
                data, _ = rs_decode(coded, PHY_HEADER_NSYM, attempt)
                return cls.unpack(data)
            except (FecError, FrameError):
                continue
        raise FrameError("PHY header undecodable")
