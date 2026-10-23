"""CRC-16/CCITT-FALSE (PHY header) and CRC-32 (link frames, zlib polynomial)."""

from __future__ import annotations

import zlib


def _make_crc16_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
        table.append(crc)
    return table


_CRC16_TABLE = _make_crc16_table()


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, xorout 0.

    ``crc16_ccitt_false(b"123456789") == 0x29B1``.
    """
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


def crc32(data: bytes) -> int:
    """CRC-32 as used by zlib/PNG/Ethernet. ``crc32(b"123456789") == 0xCBF43926``."""
    return zlib.crc32(data) & 0xFFFFFFFF
