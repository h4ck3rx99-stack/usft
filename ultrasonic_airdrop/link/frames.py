"""Link frames (§4.1) and the payload of every frame type (§4.2).

This module is the single definition of the wire format, shared by transmitter
and receiver.  All integers are little-endian.  Every ``unpack`` raises only
:class:`~ultrasonic_airdrop.errors.FrameError` for malformed input.

Link frame::

    0  u8   magic 0xA5
    1  u8   version (1)
    2  u8   type
    3  u8   flags: bit0 dir, bit1 enc, bit2 poll, bits3-7 reserved (0)
    4  u32  session_id
    8  u32  seq
    12 u16  payload_len
    14 n    payload
    14+n u32 CRC-32 over bytes 0 .. 14+n-1
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar

from ultrasonic_airdrop.config import (
    ACK_BITMAP_BITS,
    LINK_MAGIC,
    MAX_ACK_RANGES,
    MAX_ENDACK_RANGES,
    MAX_RESUME_RANGES,
    MAX_SENDER_NAME_BYTES,
    PROTO_VERSION,
    SALT_LEN,
)
from ultrasonic_airdrop.errors import FrameError
from ultrasonic_airdrop.fec.crc import crc32

HEADER_LEN = 14
CRC_LEN = 4
OVERHEAD = HEADER_LEN + CRC_LEN
MAX_PAYLOAD = 4000

FLAG_DIR = 0x01
FLAG_ENC = 0x02
FLAG_POLL = 0x04
RESERVED_FLAGS = 0xF8

DIR_S2R = 0
DIR_R2S = 1


class FrameType(IntEnum):
    """Link frame types."""

    OFFER = 0x01
    ACCEPT = 0x02
    PROBE = 0x03
    PROBE_REPORT = 0x04
    MANIFEST = 0x05
    DATA = 0x06
    ACK = 0x07
    END = 0x08
    END_ACK = 0x09
    CANCEL = 0x0A
    POLL = 0x0B
    RESUME_INFO = 0x0C


#: Frame types whose payload is encrypted when the session uses encryption.
ENCRYPTED_TYPES = frozenset(
    {FrameType.MANIFEST, FrameType.DATA, FrameType.ACK, FrameType.END_ACK, FrameType.RESUME_INFO}
)
#: Expected direction of each frame type (None = either).
FRAME_DIR: dict[FrameType, int | None] = {
    FrameType.OFFER: DIR_S2R,
    FrameType.ACCEPT: DIR_R2S,
    FrameType.PROBE: DIR_S2R,
    FrameType.PROBE_REPORT: DIR_R2S,
    FrameType.MANIFEST: DIR_S2R,
    FrameType.DATA: DIR_S2R,
    FrameType.ACK: DIR_R2S,
    FrameType.END: DIR_S2R,
    FrameType.END_ACK: DIR_R2S,
    FrameType.CANCEL: None,
    FrameType.POLL: DIR_S2R,
    FrameType.RESUME_INFO: DIR_R2S,
}


class UnknownFrameTypeError(FrameError):
    """A structurally valid frame with a type this implementation does not know."""


# --------------------------------------------------------------------------------------
# Reader helper
# --------------------------------------------------------------------------------------


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise FrameError("payload truncated")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str) -> tuple[int, ...]:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.take(size))

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def u16(self) -> int:
        return self.unpack("<H")[0]

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def u64(self) -> int:
        return self.unpack("<Q")[0]

    def i16(self) -> int:
        return self.unpack("<h")[0]

    def rest(self) -> bytes:
        out = self.data[self.pos :]
        self.pos = len(self.data)
        return out

    def done(self) -> None:
        if self.pos != len(self.data):
            raise FrameError("trailing bytes in payload")


def _utf8(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FrameError("invalid UTF-8") from exc


# --------------------------------------------------------------------------------------
# Link frame
# --------------------------------------------------------------------------------------


@dataclass
class LinkFrame:
    """A link frame with raw (possibly encrypted) payload bytes."""

    ftype: FrameType
    session_id: int
    seq: int
    payload: bytes = b""
    direction: int = DIR_S2R
    enc: bool = False
    poll: bool = False

    @property
    def flags(self) -> int:
        """Flag byte."""
        return (FLAG_DIR if self.direction else 0) | (FLAG_ENC if self.enc else 0) | (FLAG_POLL if self.poll else 0)

    def header_bytes(self) -> bytes:
        """The 14-byte header (also the AEAD associated data)."""
        if not 0 <= self.session_id <= 0xFFFFFFFF or not 0 <= self.seq <= 0xFFFFFFFF:
            raise FrameError("session_id/seq out of range")
        if len(self.payload) > MAX_PAYLOAD:
            raise FrameError("payload too long")
        return struct.pack(
            "<BBBBIIH",
            LINK_MAGIC,
            PROTO_VERSION,
            int(self.ftype),
            self.flags,
            self.session_id,
            self.seq,
            len(self.payload),
        )

    def pack(self) -> bytes:
        """Serialize including the trailing CRC-32."""
        body = self.header_bytes() + self.payload
        return body + struct.pack("<I", crc32(body))

    @classmethod
    def unpack(cls, raw: bytes, allow_unknown_type: bool = False) -> LinkFrame:
        """Parse and validate.  Raises :class:`FrameError` (or :class:`UnknownFrameTypeError`)."""
        if len(raw) < OVERHEAD:
            raise FrameError("frame shorter than header + CRC")
        magic, version, ftype, flags, session_id, seq, plen = struct.unpack_from("<BBBBIIH", raw, 0)
        if magic != LINK_MAGIC:
            raise FrameError("bad magic")
        if len(raw) != OVERHEAD + plen:
            raise FrameError("length mismatch")
        (crc,) = struct.unpack_from("<I", raw, HEADER_LEN + plen)
        if crc32(raw[: HEADER_LEN + plen]) != crc:
            raise FrameError("CRC-32 mismatch")
        if version != PROTO_VERSION:
            raise FrameError(f"unsupported protocol version {version}")
        if flags & RESERVED_FLAGS:
            raise FrameError("reserved flag bits set")
        try:
            ft = FrameType(ftype)
        except ValueError as exc:
            raise UnknownFrameTypeError(f"unknown frame type 0x{ftype:02x}") from exc
        frame = cls(
            ftype=ft,
            session_id=session_id,
            seq=seq,
            payload=bytes(raw[HEADER_LEN : HEADER_LEN + plen]),
            direction=DIR_R2S if flags & FLAG_DIR else DIR_S2R,
            enc=bool(flags & FLAG_ENC),
            poll=bool(flags & FLAG_POLL),
        )
        expected = FRAME_DIR[ft]
        if expected is not None and frame.direction != expected:
            raise FrameError(f"{ft.name} with wrong direction flag")
        if frame.enc and ft not in ENCRYPTED_TYPES:
            raise FrameError(f"{ft.name} must not be encrypted")
        return frame

    @staticmethod
    def peek_version(raw: bytes) -> int | None:
        """Return the version byte of a CRC-valid frame, else ``None`` (for version mismatch replies)."""
        if len(raw) < OVERHEAD or raw[0] != LINK_MAGIC:
            return None
        plen = struct.unpack_from("<H", raw, 12)[0]
        if len(raw) != OVERHEAD + plen:
            return None
        if crc32(raw[: HEADER_LEN + plen]) != struct.unpack_from("<I", raw, HEADER_LEN + plen)[0]:
            return None
        return raw[1]


def link_frame_len(payload_len: int) -> int:
    """Total link frame length for a payload length."""
    return OVERHEAD + payload_len


# --------------------------------------------------------------------------------------
# Payloads
# --------------------------------------------------------------------------------------

CAP_ENC = 1 << 8
CAP_COMPRESSION = 1 << 9
MODE_BITS = 0x3F

TRANSFER_INTERACTIVE = 0
TRANSFER_BROADCAST = 1


@dataclass(frozen=True)
class Offer:
    """OFFER payload."""

    capabilities: int
    transfer_mode: int
    file_count: int
    total_bytes: int
    manifest_len: int
    manifest_sha256: bytes
    encrypted: bool
    salt: bytes
    sender_name: str
    version: int = PROTO_VERSION

    TYPE: ClassVar[FrameType] = FrameType.OFFER

    def pack(self) -> bytes:
        """Serialize."""
        name = self.sender_name.encode("utf-8")[:MAX_SENDER_NAME_BYTES]
        # never cut a multi-byte character in half
        name = name.decode("utf-8", "ignore").encode("utf-8")
        if len(self.manifest_sha256) != 32 or len(self.salt) != SALT_LEN:
            raise FrameError("bad hash/salt length")
        return (
            struct.pack(
                "<BHBHQI",
                self.version,
                self.capabilities,
                self.transfer_mode,
                self.file_count,
                self.total_bytes,
                self.manifest_len,
            )
            + self.manifest_sha256
            + struct.pack("<B", 1 if self.encrypted else 0)
            + self.salt
            + struct.pack("<B", len(name))
            + name
        )

    @classmethod
    def unpack(cls, data: bytes) -> Offer:
        """Parse."""
        r = _Reader(data)
        version, caps, mode, count, total, mlen = r.unpack("<BHBHQI")
        sha = r.take(32)
        enc = r.u8()
        salt = r.take(SALT_LEN)
        nlen = r.u8()
        if nlen > MAX_SENDER_NAME_BYTES:
            raise FrameError("sender name too long")
        name = _utf8(r.take(nlen))
        r.done()
        if mode not in (TRANSFER_INTERACTIVE, TRANSFER_BROADCAST):
            raise FrameError("unknown transfer mode")
        if enc not in (0, 1):
            raise FrameError("bad encryption flag")
        return cls(caps, mode, count, total, mlen, sha, bool(enc), salt, name, version)


ACCEPT_OK = 0
ACCEPT_REJECTED = 1
ACCEPT_TOO_LARGE = 2
ACCEPT_BUSY = 3
ACCEPT_UNSUPPORTED = 4


@dataclass(frozen=True)
class Accept:
    """ACCEPT payload."""

    code: int
    capabilities: int
    resume_available: bool
    max_bytes: int

    TYPE: ClassVar[FrameType] = FrameType.ACCEPT

    def pack(self) -> bytes:
        """Serialize."""
        return struct.pack("<BHBQ", self.code, self.capabilities, int(self.resume_available), self.max_bytes)

    @classmethod
    def unpack(cls, data: bytes) -> Accept:
        """Parse."""
        r = _Reader(data)
        code, caps, resume, max_bytes = r.unpack("<BHBQ")
        r.done()
        if code > ACCEPT_UNSUPPORTED or resume > 1:
            raise FrameError("bad ACCEPT fields")
        return cls(code, caps, bool(resume), max_bytes)


@dataclass(frozen=True)
class Probe:
    """PROBE payload (empty; the sounding sequence follows in the PHY body)."""

    TYPE: ClassVar[FrameType] = FrameType.PROBE

    def pack(self) -> bytes:
        """Serialize."""
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> Probe:
        """Parse."""
        _Reader(data).done()
        return cls()


def snr_to_u8(snr_db: float) -> int:
    """Encode a per-carrier SNR as u8 in 0.25 dB steps with a -10 dB offset (-10 .. 53.75 dB)."""
    if snr_db != snr_db:  # NaN
        return 0
    return int(min(255, max(0, round((snr_db + 10.0) * 4.0))))


def u8_to_snr(value: int) -> float:
    """Inverse of :func:`snr_to_u8`."""
    return value / 4.0 - 10.0


def db_to_i16(db: float) -> int:
    """Encode dB as i16 in 0.1 dB steps."""
    if db != db:
        return -32768
    return int(min(32767, max(-32767, round(db * 10.0))))


def i16_to_db(value: int) -> float:
    """Inverse of :func:`db_to_i16` (``-32768`` = unknown → NaN)."""
    return float("nan") if value == -32768 else value / 10.0


@dataclass(frozen=True)
class ProbeReport:
    """PROBE_REPORT payload."""

    carrier_snr: tuple[int, ...]  # 52 u8 (see snr_to_u8)
    mfsk_snr: int  # i16, 0.1 dB
    recommended_mode: int
    mask: bytes  # 7 bytes
    ppm: int  # i16

    TYPE: ClassVar[FrameType] = FrameType.PROBE_REPORT

    def pack(self) -> bytes:
        """Serialize."""
        if len(self.carrier_snr) != 52 or len(self.mask) != 7:
            raise FrameError("bad PROBE_REPORT sizes")
        return (
            bytes(self.carrier_snr)
            + struct.pack("<hB", self.mfsk_snr, self.recommended_mode)
            + self.mask
            + struct.pack("<h", self.ppm)
        )

    @classmethod
    def unpack(cls, data: bytes) -> ProbeReport:
        """Parse."""
        r = _Reader(data)
        snr = tuple(r.take(52))
        mfsk_snr, mode = r.unpack("<hB")
        mask = r.take(7)
        (ppm,) = r.unpack("<h")
        r.done()
        if mode > 5:
            raise FrameError("bad recommended mode")
        if int.from_bytes(mask, "little") >> 52:
            raise FrameError("carrier mask bits beyond 52")
        return cls(snr, mfsk_snr, mode, mask, ppm)


@dataclass(frozen=True)
class ManifestPart:
    """MANIFEST payload."""

    part_index: int
    part_count: int
    data: bytes

    TYPE: ClassVar[FrameType] = FrameType.MANIFEST

    def pack(self) -> bytes:
        """Serialize."""
        return struct.pack("<HH", self.part_index, self.part_count) + self.data

    @classmethod
    def unpack(cls, data: bytes) -> ManifestPart:
        """Parse."""
        r = _Reader(data)
        idx, count = r.unpack("<HH")
        if count == 0 or idx >= count:
            raise FrameError("bad manifest part index")
        return cls(idx, count, r.rest())


DATA_HEADER_LEN = 7


@dataclass(frozen=True)
class Data:
    """DATA payload: ``unit_count`` units of file ``file_index`` starting at ``start_unit``."""

    file_index: int
    start_unit: int
    unit_count: int
    data: bytes

    TYPE: ClassVar[FrameType] = FrameType.DATA

    def pack(self) -> bytes:
        """Serialize."""
        return struct.pack("<HIB", self.file_index, self.start_unit, self.unit_count) + self.data

    @classmethod
    def unpack(cls, data: bytes) -> Data:
        """Parse (unit lengths are validated against the manifest by the receiver)."""
        r = _Reader(data)
        fi, start, count = r.unpack("<HIB")
        payload = r.rest()
        if count == 0 or not payload:
            raise FrameError("empty DATA frame")
        return cls(fi, start, count, payload)


@dataclass(frozen=True)
class UnitRange:
    """A run of units of one file."""

    file_index: int
    start: int
    count: int


def _pack_ranges(ranges: tuple[UnitRange, ...]) -> bytes:
    out = struct.pack("<B", len(ranges))
    for rg in ranges:
        out += struct.pack("<HIH", rg.file_index, rg.start, min(rg.count, 0xFFFF))
    return out


def _unpack_ranges(r: _Reader, limit: int) -> tuple[UnitRange, ...]:
    n = r.u8()
    if n > limit:
        raise FrameError("too many ranges")
    out = []
    for _ in range(n):
        fi, start, count = r.unpack("<HIH")
        if count == 0:
            raise FrameError("empty range")
        out.append(UnitRange(fi, start, count))
    return tuple(out)


@dataclass(frozen=True)
class Ack:
    """ACK payload (selective acknowledgement + link feedback)."""

    base_seq: int
    bitmap: int  # bit i => seq base_seq + i received
    mean_snr: int  # i16 0.1 dB
    frame_errors: int
    suggested_mode: int
    mask: bytes
    missing: tuple[UnitRange, ...] = field(default_factory=tuple)

    TYPE: ClassVar[FrameType] = FrameType.ACK

    def pack(self) -> bytes:
        """Serialize."""
        if len(self.missing) > MAX_ACK_RANGES:
            raise FrameError("too many missing ranges")
        return (
            struct.pack(
                "<IQhBB",
                self.base_seq,
                self.bitmap & ((1 << ACK_BITMAP_BITS) - 1),
                self.mean_snr,
                min(self.frame_errors, 255),
                self.suggested_mode,
            )
            + self.mask
            + _pack_ranges(self.missing)
        )

    @classmethod
    def unpack(cls, data: bytes) -> Ack:
        """Parse."""
        r = _Reader(data)
        base, bitmap, snr, fer, mode = r.unpack("<IQhBB")
        mask = r.take(7)
        missing = _unpack_ranges(r, MAX_ACK_RANGES)
        r.done()
        if mode > 5:
            raise FrameError("bad suggested mode")
        if int.from_bytes(mask, "little") >> 52:
            raise FrameError("carrier mask bits beyond 52")
        return cls(base, bitmap, snr, fer, mode, mask, missing)

    def acked_seqs(self) -> list[int]:
        """Sequence numbers marked received."""
        return [self.base_seq + i for i in range(ACK_BITMAP_BITS) if (self.bitmap >> i) & 1]


@dataclass(frozen=True)
class End:
    """END payload (empty)."""

    TYPE: ClassVar[FrameType] = FrameType.END

    def pack(self) -> bytes:
        """Serialize."""
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> End:
        """Parse."""
        _Reader(data).done()
        return cls()


FILE_OK = 0
FILE_HASH_MISMATCH = 1
FILE_INCOMPLETE = 2
FILE_WRITE_ERROR = 3
FILE_SKIPPED = 4
FILE_STATUS_NAMES = {
    FILE_OK: "OK",
    FILE_HASH_MISMATCH: "HASH_MISMATCH",
    FILE_INCOMPLETE: "INCOMPLETE",
    FILE_WRITE_ERROR: "WRITE_ERROR",
    FILE_SKIPPED: "SKIPPED",
}


@dataclass(frozen=True)
class EndAck:
    """END_ACK payload: statuses for files ``first .. first+len(statuses)-1`` of ``total_files``."""

    total_files: int
    first: int
    statuses: tuple[int, ...]
    missing: tuple[UnitRange, ...] = field(default_factory=tuple)

    TYPE: ClassVar[FrameType] = FrameType.END_ACK

    def pack(self) -> bytes:
        """Serialize."""
        if len(self.missing) > MAX_ENDACK_RANGES:
            raise FrameError("too many missing ranges")
        return (
            struct.pack("<HHH", self.total_files, self.first, len(self.statuses))
            + bytes(self.statuses)
            + _pack_ranges(self.missing)
        )

    @classmethod
    def unpack(cls, data: bytes) -> EndAck:
        """Parse."""
        r = _Reader(data)
        total, first, n = r.unpack("<HHH")
        statuses = tuple(r.take(n))
        missing = _unpack_ranges(r, MAX_ENDACK_RANGES)
        r.done()
        if first + n > total:
            raise FrameError("END_ACK statuses beyond file count")
        if any(s not in FILE_STATUS_NAMES for s in statuses):
            raise FrameError("unknown file status")
        return cls(total, first, statuses, missing)


CANCEL_USER = 0
CANCEL_VERSION = 1
CANCEL_AUTH = 2
CANCEL_TIMEOUT = 3
CANCEL_ERROR = 4
CANCEL_REJECTED = 5
MAX_CANCEL_TEXT = 64


@dataclass(frozen=True)
class Cancel:
    """CANCEL payload."""

    reason: int
    text: str = ""

    TYPE: ClassVar[FrameType] = FrameType.CANCEL

    def pack(self) -> bytes:
        """Serialize."""
        raw = self.text.encode("utf-8")[:MAX_CANCEL_TEXT].decode("utf-8", "ignore").encode("utf-8")
        return struct.pack("<BB", self.reason, len(raw)) + raw

    @classmethod
    def unpack(cls, data: bytes) -> Cancel:
        """Parse."""
        r = _Reader(data)
        reason, n = r.unpack("<BB")
        if n > MAX_CANCEL_TEXT:
            raise FrameError("cancel text too long")
        text = _utf8(r.take(n))
        r.done()
        return cls(reason, text)


@dataclass(frozen=True)
class Poll:
    """POLL payload (empty)."""

    TYPE: ClassVar[FrameType] = FrameType.POLL

    def pack(self) -> bytes:
        """Serialize."""
        return b""

    @classmethod
    def unpack(cls, data: bytes) -> Poll:
        """Parse."""
        _Reader(data).done()
        return cls()


@dataclass(frozen=True)
class ResumeInfo:
    """RESUME_INFO payload: unit ranges of ``file_index`` the receiver already holds."""

    file_index: int
    ranges: tuple[tuple[int, int], ...]  # (start_unit u32, count u32)

    TYPE: ClassVar[FrameType] = FrameType.RESUME_INFO

    def pack(self) -> bytes:
        """Serialize."""
        if len(self.ranges) > MAX_RESUME_RANGES:
            raise FrameError("too many resume ranges")
        out = struct.pack("<HB", self.file_index, len(self.ranges))
        for start, count in self.ranges:
            out += struct.pack("<II", start, count)
        return out

    @classmethod
    def unpack(cls, data: bytes) -> ResumeInfo:
        """Parse."""
        r = _Reader(data)
        fi, n = r.unpack("<HB")
        if n > MAX_RESUME_RANGES:
            raise FrameError("too many resume ranges")
        ranges = []
        for _ in range(n):
            start, count = r.unpack("<II")
            if count == 0:
                raise FrameError("empty resume range")
            ranges.append((start, count))
        r.done()
        return cls(fi, tuple(ranges))


Payload = Offer | Accept | Probe | ProbeReport | ManifestPart | Data | Ack | End | EndAck | Cancel | Poll | ResumeInfo

PAYLOAD_TYPES: dict[FrameType, type[Payload]] = {
    FrameType.OFFER: Offer,
    FrameType.ACCEPT: Accept,
    FrameType.PROBE: Probe,
    FrameType.PROBE_REPORT: ProbeReport,
    FrameType.MANIFEST: ManifestPart,
    FrameType.DATA: Data,
    FrameType.ACK: Ack,
    FrameType.END: End,
    FrameType.END_ACK: EndAck,
    FrameType.CANCEL: Cancel,
    FrameType.POLL: Poll,
    FrameType.RESUME_INFO: ResumeInfo,
}


def parse_payload(ftype: FrameType, plaintext: bytes) -> Payload:
    """Parse a (decrypted) payload for ``ftype``.  Raises :class:`FrameError`."""
    cls = PAYLOAD_TYPES[ftype]
    try:
        return cls.unpack(plaintext)
    except FrameError:
        raise
    except (struct.error, ValueError, IndexError) as exc:  # defensive: never leak other errors
        raise FrameError(f"malformed {ftype.name} payload: {exc}") from exc
