import numpy as np
import pytest

from ultrasonic_airdrop.errors import FecError
from ultrasonic_airdrop.fec.crc import crc16_ccitt_false, crc32
from ultrasonic_airdrop.fec.interleave import deinterleave, deinterleave_array, interleave
from ultrasonic_airdrop.fec.rs import coded_length, decode_body, encode_body, rs_decode, rs_encode, split_plan


def test_crc_known_vectors():
    assert crc16_ccitt_false(b"123456789") == 0x29B1
    assert crc16_ccitt_false(b"") == 0xFFFF
    assert crc32(b"123456789") == 0xCBF43926
    assert crc32(b"") == 0


@pytest.mark.parametrize("nsym", [8, 16, 32, 64])
def test_rs_errors_at_capacity(nsym, rng):
    data = rng.integers(0, 256, 255 - nsym, dtype=np.uint8).tobytes()
    cw = bytearray(rs_encode(data, nsym))
    pos = rng.choice(len(cw), nsym // 2, replace=False)
    for p in pos:
        cw[p] ^= 0xA5
    out, corrected = rs_decode(bytes(cw), nsym)
    assert out == data
    assert corrected == nsym // 2


@pytest.mark.parametrize("nsym", [16, 64])
def test_rs_erasures_at_capacity(nsym, rng):
    data = rng.integers(0, 256, 100, dtype=np.uint8).tobytes()
    cw = bytearray(rs_encode(data, nsym))
    pos = sorted(int(p) for p in rng.choice(len(cw), nsym, replace=False))
    for p in pos:
        cw[p] = 0
    out, _ = rs_decode(bytes(cw), nsym, pos)
    assert out == data


def test_rs_mixed_errors_and_erasures(rng):
    nsym = 32
    data = rng.integers(0, 256, 150, dtype=np.uint8).tobytes()
    cw = bytearray(rs_encode(data, nsym))
    pos = [int(p) for p in rng.choice(len(cw), 20, replace=False)]
    erase, errs = pos[:12], pos[12:18]  # 2*6 + 12 = 24 <= 32
    for p in erase + errs:
        cw[p] ^= 0x3C
    out, _ = rs_decode(bytes(cw), nsym, erase)
    assert out == data


def test_rs_beyond_capacity_never_silently_returns_original_claim(rng):
    nsym = 16
    data = rng.integers(0, 256, 100, dtype=np.uint8).tobytes()
    failures = 0
    for trial in range(20):
        cw = bytearray(rs_encode(data, nsym))
        for p in rng.choice(len(cw), nsym, replace=False):  # 16 errors > capacity 8
            cw[p] ^= int(rng.integers(1, 256))
        try:
            out, _ = rs_decode(bytes(cw), nsym)
        except FecError:
            failures += 1
            continue
        # A mis-correction is possible; it must then differ from the original (CRC catches it upstream).
        assert out != data or trial < 0
    assert failures >= 15


def test_rs_too_many_erasures_raises(rng):
    data = bytes(50)
    cw = rs_encode(data, 8)
    with pytest.raises(FecError):
        rs_decode(cw, 8, list(range(9)))


def test_interleave_inverse(rng):
    k, n = 5, 37
    cws = [rng.integers(0, 256, n, dtype=np.uint8).tobytes() for _ in range(k)]
    inter = interleave(cws)
    assert inter[1] == cws[1][0] and inter[k] == cws[0][1]
    assert deinterleave(inter, k) == cws
    flags = np.arange(k * n)
    per = deinterleave_array(flags, k)
    assert all(per[i][j] == j * k + i for i in range(k) for j in range(n))


def test_interleave_rejects_bad_sizes():
    with pytest.raises(ValueError):
        interleave([b"ab", b"abc"])
    with pytest.raises(ValueError):
        deinterleave(b"abcde", 2)


@pytest.mark.parametrize("length", [1, 50, 191, 192, 500, 1500])
@pytest.mark.parametrize("nsym", [16, 32, 64])
def test_body_roundtrip_and_burst(length, nsym, rng):
    body = rng.integers(0, 256, length, dtype=np.uint8).tobytes()
    coded = encode_body(body, nsym)
    plan = split_plan(length, nsym)
    assert len(coded) == coded_length(length, nsym) == plan.coded_len
    assert plan.codeword_len <= 255
    corrupted = bytearray(coded)
    burst = min(len(coded), plan.k * (nsym // 2))  # a burst spread over k codewords
    start = int(rng.integers(0, len(coded) - burst + 1))
    for i in range(start, start + burst):
        corrupted[i] ^= 0xFF
    out, stats = decode_body(bytes(corrupted), length, nsym)
    assert out == body
    assert stats.codewords == plan.k


def test_body_with_erasure_flags(rng):
    body = rng.integers(0, 256, 300, dtype=np.uint8).tobytes()
    nsym = 32
    coded = bytearray(encode_body(body, nsym))
    plan = split_plan(300, nsym)
    er = np.zeros(len(coded), dtype=bool)
    # erase (and destroy) nsym-4 bytes per codeword: more than errors-only capacity
    for i in range(plan.k):
        for j in range(nsym - 4):
            pos = j * plan.k + i
            coded[pos] ^= 0x55
            er[pos] = True
    out, stats = decode_body(bytes(coded), 300, nsym, er)
    assert out == body
    assert stats.erasures_used > 0
