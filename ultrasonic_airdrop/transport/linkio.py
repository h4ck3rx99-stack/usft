"""Glue between sessions and the radio: frame building (seq, encryption), sending, parsing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ultrasonic_airdrop.config import CONTROL_MODE, Profile, Settings
from ultrasonic_airdrop.crypto.aead import TAG_LEN, FrameCipher
from ultrasonic_airdrop.errors import CryptoError, FrameError
from ultrasonic_airdrop.link.codec import FrameCodec, frame_airtime
from ultrasonic_airdrop.link.frames import (
    ENCRYPTED_TYPES,
    FrameType,
    LinkFrame,
    Payload,
    link_frame_len,
    parse_payload,
)
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy.numerology import FloatArray
from ultrasonic_airdrop.phy.ofdm import CarrierMask

log = logging.getLogger(__name__)
INTER_FRAME_GAP_S = 0.05


class Radio(Protocol):
    """What a session needs from the station it runs on."""

    def transmit(self, samples: FloatArray) -> float:
        """Queue a burst for playback; returns the clock time at which it ends."""
        ...

    def tx_busy(self) -> bool:
        """True while our own audio is still playing."""
        ...

    def rx_busy(self) -> bool:
        """True while a frame is being received (detected but not yet decoded)."""
        ...

    def now(self) -> float:
        """Current clock time."""
        ...


@dataclass
class OutFrame:
    """A packed link frame with the PHY mode it will be sent in."""

    raw: bytes
    ftype: FrameType
    seq: int
    mode_id: int = CONTROL_MODE
    mask: CarrierMask | None = None
    poll: bool = False


class LinkIO:
    """Per-session frame factory and parser for one direction."""

    def __init__(self, radio: Radio, profile: Profile, settings: Settings, direction: int, session_id: int) -> None:
        self.radio = radio
        self.profile = profile
        self.settings = settings
        self.direction = direction
        self.session_id = session_id
        self.codec = FrameCodec(profile, settings.tx_level_dbfs)
        self.cipher: FrameCipher | None = None
        self.seq = 0
        self.frames_sent = 0
        self.airtime_sent = 0.0

    def set_cipher(self, cipher: FrameCipher | None) -> None:
        """Enable/disable payload encryption."""
        self.cipher = cipher

    def tag_overhead(self, ftype: FrameType) -> int:
        """Extra bytes added by encryption for ``ftype``."""
        return TAG_LEN if self.cipher is not None and ftype in ENCRYPTED_TYPES else 0

    def build(
        self,
        payload: Payload,
        poll: bool = False,
        mode_id: int = CONTROL_MODE,
        mask: CarrierMask | None = None,
    ) -> OutFrame:
        """Pack ``payload`` into a link frame with the next seq (encrypting if configured)."""
        ftype = payload.TYPE
        plain = payload.pack()
        seq = self.seq
        self.seq = (self.seq + 1) & 0xFFFFFFFF
        enc = self.cipher is not None and ftype in ENCRYPTED_TYPES
        frame = LinkFrame(ftype, self.session_id, seq, plain, self.direction, enc, poll)
        if enc:
            assert self.cipher is not None
            frame.payload = bytes(len(plain) + TAG_LEN)  # length placeholder for the AAD header
            header = frame.header_bytes()
            frame.payload = self.cipher.seal(header, plain, self.direction, seq)
        return OutFrame(frame.pack(), ftype, seq, mode_id, mask, poll)

    def airtime(self, out: OutFrame) -> float:
        """Airtime of one frame."""
        n_en = out.mask.count if out.mask is not None else None
        return frame_airtime(self.profile, out.mode_id, len(out.raw), n_en)

    def control_airtime(self, payload_len: int) -> float:
        """Airtime of a mode-0 control frame with ``payload_len`` payload bytes."""
        return frame_airtime(self.profile, CONTROL_MODE, link_frame_len(payload_len))

    def render(self, frames: list[OutFrame]) -> FloatArray:
        """Encode frames into one burst sequence separated by short gaps."""
        gap = np.zeros(int(INTER_FRAME_GAP_S * self.profile.sample_rate))
        parts: list[FloatArray] = []
        for i, f in enumerate(frames):
            if i:
                parts.append(gap)
            parts.append(self.codec.encode(f.raw, f.mode_id, f.mask))
        return np.concatenate(parts) if parts else np.zeros(0)

    def send(self, frames: list[OutFrame]) -> float:
        """Transmit frames back-to-back; returns the end time."""
        samples = self.render(frames)
        end = self.radio.transmit(samples)
        self.frames_sent += len(frames)
        self.airtime_sent += len(samples) / self.profile.sample_rate
        for f in frames:
            log.debug(kv("tx frame", type=f.ftype.name, seq=f.seq, mode=f.mode_id, poll=f.poll, bytes=len(f.raw)))
        return end

    def open(self, frame: LinkFrame) -> Payload:
        """Decrypt (if needed) and parse a received frame's payload.

        Raises :class:`CryptoError` on authentication failure and
        :class:`FrameError` on malformed payloads or encryption downgrades.
        """
        plain = frame.payload
        if frame.enc:
            if self.cipher is None:
                raise CryptoError("Encrypted frame but no passphrase configured")
            plain = self.cipher.open(frame.header_bytes(), frame.payload, frame.direction, frame.seq)
        elif self.cipher is not None and frame.ftype in ENCRYPTED_TYPES:
            raise FrameError(f"unencrypted {frame.ftype.name} in an encrypted session")
        return parse_payload(frame.ftype, plain)
