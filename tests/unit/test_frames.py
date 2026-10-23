import contextlib
import struct

import numpy as np
import pytest

from ultrasonic_airdrop.errors import FrameError
from ultrasonic_airdrop.fec.crc import crc32
from ultrasonic_airdrop.link import frames as fr
from ultrasonic_airdrop.link.phy_header import PhyHeader

PAYLOADS = [
    fr.Offer(0x33F, fr.TRANSFER_INTERACTIVE, 3, 123456, 210, bytes(range(32)), True, bytes(16), "Lab PC ünï"),
    fr.Offer(0x3F, fr.TRANSFER_BROADCAST, 1, 0, 50, bytes(32), False, bytes(16), ""),
    fr.Accept(fr.ACCEPT_OK, 0x33F, True, 50 * 1024 * 1024),
    fr.Probe(),
    fr.ProbeReport(tuple(range(52)), -15, 3, bytes([0xFF] * 6 + [0x0F]), -123),
    fr.ManifestPart(1, 3, b"\x01\x02\x03"),
    fr.Data(2, 100000, 4, bytes(range(128))),
    fr.Ack(10, (1 << 63) | 5, 123, 2, 4, bytes(7), (fr.UnitRange(0, 5, 3), fr.UnitRange(1, 0, 65535))),
    fr.End(),
    fr.EndAck(5, 0, (0, 1, 2, 3, 4), (fr.UnitRange(3, 7, 9),)),
    fr.Cancel(fr.CANCEL_AUTH, "passphrase mismatch ✓"),
    fr.Poll(),
    fr.ResumeInfo(7, ((0, 10), (20, 5))),
]


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: type(p).__name__)
def test_payload_roundtrip(payload):
    raw = payload.pack()
    assert type(payload).unpack(raw) == payload
    direction = fr.FRAME_DIR[payload.TYPE] or 0
    frame = fr.LinkFrame(payload.TYPE, 0xDEADBEEF, 42, raw, direction, False, True)
    packed = frame.pack()
    back = fr.LinkFrame.unpack(packed)
    assert back == frame
    assert fr.parse_payload(back.ftype, back.payload) == payload


def test_header_layout():
    frame = fr.LinkFrame(fr.FrameType.DATA, 0x01020304, 0x0A0B0C0D, b"xyz", fr.DIR_S2R, False, True)
    raw = frame.pack()
    assert raw[0] == 0xA5 and raw[1] == 1 and raw[2] == 0x06 and raw[3] == 0x04
    assert raw[4:8] == bytes([4, 3, 2, 1])
    assert raw[8:12] == bytes([0x0D, 0x0C, 0x0B, 0x0A])
    assert raw[12:14] == b"\x03\x00"
    assert struct.unpack("<I", raw[-4:])[0] == crc32(raw[:-4])


def _good() -> bytes:
    return fr.LinkFrame(fr.FrameType.POLL, 1, 2, b"", fr.DIR_S2R, False, True).pack()


def _recrc(raw: bytearray) -> bytes:
    raw[-4:] = struct.pack("<I", crc32(bytes(raw[:-4])))
    return bytes(raw)


def test_rejections():
    good = bytearray(_good())
    bad_magic = bytearray(good)
    bad_magic[0] = 0x5A
    with pytest.raises(FrameError):
        fr.LinkFrame.unpack(_recrc(bad_magic))
    bad_version = bytearray(good)
    bad_version[1] = 2
    with pytest.raises(FrameError, match="version"):
        fr.LinkFrame.unpack(_recrc(bad_version))
    assert fr.LinkFrame.peek_version(_recrc(bad_version)) == 2
    reserved = bytearray(good)
    reserved[3] |= 0x80
    with pytest.raises(FrameError, match="reserved"):
        fr.LinkFrame.unpack(_recrc(reserved))
    bad_crc = bytearray(good)
    bad_crc[-1] ^= 1
    with pytest.raises(FrameError, match="CRC"):
        fr.LinkFrame.unpack(bytes(bad_crc))
    with pytest.raises(FrameError):
        fr.LinkFrame.unpack(bytes(good[:-1]))
    with pytest.raises(FrameError):
        fr.LinkFrame.unpack(bytes(good) + b"\x00")
    unknown = bytearray(good)
    unknown[2] = 0x7F
    with pytest.raises(fr.UnknownFrameTypeError):
        fr.LinkFrame.unpack(_recrc(unknown))
    wrong_dir = bytearray(good)
    wrong_dir[3] |= fr.FLAG_DIR  # POLL must be sender->receiver
    with pytest.raises(FrameError, match="direction"):
        fr.LinkFrame.unpack(_recrc(wrong_dir))


def test_payload_rejections():
    with pytest.raises(FrameError):
        fr.Offer.unpack(PAYLOADS[0].pack()[:-1])
    with pytest.raises(FrameError):
        fr.Accept.unpack(bytes([9]) + bytes(11))
    with pytest.raises(FrameError):
        fr.ManifestPart.unpack(struct.pack("<HH", 3, 3))
    with pytest.raises(FrameError):
        fr.Data.unpack(struct.pack("<HIB", 0, 0, 0) + b"x")
    with pytest.raises(FrameError):
        fr.EndAck.unpack(struct.pack("<HHH", 1, 0, 2) + bytes(2) + b"\x00")
    with pytest.raises(FrameError):
        fr.Ack.unpack(PAYLOADS[7].pack() + b"x")
    with pytest.raises(FrameError):
        fr.Cancel.unpack(bytes([0, 2]) + b"\xff\xfe")
    with pytest.raises(FrameError):
        fr.ResumeInfo.unpack(struct.pack("<HB", 0, 49) + bytes(49 * 8))


def test_fuzz_only_frame_errors(rng):
    for _ in range(3000):
        n = int(rng.integers(0, 80))
        raw = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        if rng.random() < 0.5 and n >= 18:
            # make header/CRC plausible so payload parsers get exercised
            b = bytearray(raw)
            b[0], b[1] = 0xA5, 1
            b[2] = int(rng.integers(1, 13))
            b[3] = int(rng.integers(0, 8))
            struct.pack_into("<H", b, 12, n - 18)
            raw = _recrc(b)
        try:
            frame = fr.LinkFrame.unpack(raw)
            fr.parse_payload(frame.ftype, frame.payload)
        except FrameError:
            pass


def test_payload_fuzz(rng):
    for ftype, cls in fr.PAYLOAD_TYPES.items():
        for _ in range(300):
            raw = rng.integers(0, 256, int(rng.integers(0, 120)), dtype=np.uint8).tobytes()
            with contextlib.suppress(FrameError):
                fr.parse_payload(ftype, raw)
        assert ftype == cls.TYPE


def test_phy_header_roundtrip_and_errors(rng):
    h = PhyHeader(4, 17, 1234)
    assert PhyHeader.unpack(h.pack()) == h
    coded = bytearray(h.encode())
    assert len(coded) == 14
    coded[0] ^= 0xFF
    coded[5] ^= 0x01
    assert PhyHeader.decode(bytes(coded)) == h
    # erasures extend capacity
    coded = bytearray(h.encode())
    for i in (1, 3, 6, 9, 11, 12):
        coded[i] = 0
    flags = np.zeros(14, dtype=bool)
    flags[[1, 3, 6, 9, 11, 12]] = True
    assert PhyHeader.decode(bytes(coded), flags) == h
    with pytest.raises(FrameError):
        PhyHeader(7, 0, 10)
    with pytest.raises(FrameError):
        PhyHeader(0, 0, 0)
    raw = bytearray(h.pack())
    raw[4] ^= 1
    with pytest.raises(FrameError):
        PhyHeader.unpack(bytes(raw))
    for _ in range(200):
        with contextlib.suppress(FrameError):
            PhyHeader.decode(rng.integers(0, 256, 14, dtype=np.uint8).tobytes())


def test_snr_encoding():
    assert fr.u8_to_snr(fr.snr_to_u8(12.25)) == 12.25
    assert fr.snr_to_u8(-50) == 0 and fr.snr_to_u8(100) == 255
    assert fr.snr_to_u8(float("nan")) == 0
    assert np.isnan(fr.i16_to_db(fr.db_to_i16(float("nan"))))
    assert fr.i16_to_db(fr.db_to_i16(-3.14)) == pytest.approx(-3.1)
