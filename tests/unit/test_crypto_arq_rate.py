import numpy as np
import pytest

from ultrasonic_airdrop.config import REQUIRED_SNR_DB, Settings
from ultrasonic_airdrop.crypto.aead import FrameCipher, derive_key, make_nonce, new_salt
from ultrasonic_airdrop.errors import CryptoError
from ultrasonic_airdrop.files.manifest import Manifest, ManifestEntry
from ultrasonic_airdrop.link import frames as fr
from ultrasonic_airdrop.phy.quality import QualityTracker
from ultrasonic_airdrop.transport.arq import Outstanding, PartTracker, SentFrame, UnitTracker
from ultrasonic_airdrop.transport.rate_adapt import RateController, recommend


def test_nonce_layout():
    n = make_nonce(0x11223344, 1, 0xAABBCCDD)
    assert n == bytes([0x44, 0x33, 0x22, 0x11, 1, 0, 0, 0, 0xDD, 0xCC, 0xBB, 0xAA])
    with pytest.raises(ValueError):
        make_nonce(1, 2, 3)


def test_aead_roundtrip_and_tamper():
    salt = new_salt()
    c1 = FrameCipher.from_passphrase("correct horse", salt, 77)
    frame = fr.LinkFrame(fr.FrameType.DATA, 77, 5, bytes(16 + 10), fr.DIR_S2R, True)
    header = frame.header_bytes()
    ct = c1.seal(header, b"0123456789", 0, 5)
    assert len(ct) == 26
    c2 = FrameCipher.from_passphrase("correct horse", salt, 77)
    assert c2.open(header, ct, 0, 5) == b"0123456789"
    bad = bytearray(ct)
    bad[0] ^= 1
    with pytest.raises(CryptoError):
        c2.open(header, bytes(bad), 0, 5)
    with pytest.raises(CryptoError):
        c2.open(header[:5] + bytes([header[5] ^ 1]) + header[6:], ct, 0, 5)  # AAD tampered
    with pytest.raises(CryptoError):
        c2.open(header, ct, 1, 5)  # other direction nonce
    with pytest.raises(CryptoError):
        FrameCipher.from_passphrase("wrong", salt, 77).open(header, ct, 0, 5)


def test_kdf_deterministic_and_salted():
    s = bytes(16)
    assert derive_key("pw", s) == derive_key("pw", s)
    assert derive_key("pw", s) != derive_key("pw", bytes([1] * 16))
    assert derive_key("e\u0301", s) == derive_key("é", s)  # NFC normalised
    with pytest.raises(ValueError):
        derive_key("", s)


def _manifest():
    return Manifest(
        (
            ManifestEntry(0, "a", 320, "1" * 64, 320, False, 0),  # 10 units
            ManifestEntry(1, "b", 50, "2" * 64, 50, False, 0),  # 2 units, last short
            ManifestEntry(2, "c", 320, "1" * 64, 320, False, 0),  # alias of a
        )
    )


def test_unit_tracker_selection_and_retransmit_first():
    t = UnitTracker(_manifest())
    assert t.alias == [False, False, True]
    assert t.total_bytes == 370
    r1 = t.next_run(4)
    assert r1 == (0, 0, 4)
    t.mark_inflight(*r1)
    r2 = t.next_run(4)
    assert r2 == (0, 4, 4)
    t.mark_inflight(*r2)
    t.mark_acked(*r2)
    t.release(*r1)  # r1 lost
    assert t.next_run(4) == (0, 0, 4)  # retransmission before new data
    t.mark_inflight(0, 0, 4)
    assert t.retransmitted_units == 4
    t.mark_acked(0, 0, 4)
    assert t.next_run(4) == (0, 8, 2)
    t.mark_inflight(0, 8, 2)
    t.mark_acked(0, 8, 2)
    assert t.next_run(4) == (1, 0, 2)
    t.mark_acked(1, 0, 2)
    assert t.all_acked() and t.acked_bytes() == 370
    t.mark_missing(1, 1, 1)
    assert not t.all_acked() and t.next_run(4) == (1, 1, 1)


def test_outstanding_resolution():
    o = Outstanding()
    for s in range(5, 10):
        o.add(SentFrame(s, "data", 0, s, 1))
    ack = fr.Ack(0, (1 << 5) | (1 << 7), 0, 0, 1, bytes(7))
    acked, lost = o.resolve(set(ack.acked_seqs()), final=False)
    assert [f.seq for f in acked] == [5, 7] and lost == []
    acked, lost = o.resolve({9}, final=True)
    assert [f.seq for f in acked] == [9] and sorted(f.seq for f in lost) == [6, 8]
    assert len(o) == 0


def test_part_tracker():
    p = PartTracker(3)
    assert p.next_part() == 0
    p.inflight.add(0)
    assert p.next_part() == 1
    p.acked.update({0, 1, 2})
    assert p.all_acked()


def test_rate_controller_rules():
    rc = RateController(mode=3)
    assert rc.on_ack(8, 3, 5) == 2  # FER 37.5% > 25%: step down at once
    for _ in range(2):
        assert rc.on_ack(8, 0, 5) == 2
    assert rc.on_ack(8, 0, 5) == 3  # third clean window: step up one
    assert rc.on_ack(8, 0, 2) == 2  # never exceed suggestion
    assert rc.on_ack_missed() == 2
    assert rc.on_ack_missed() == 1  # two missed ACKs: step down
    rc2 = RateController(mode=2, caps=0b000011)
    assert rc2.step_down() == 1
    rc3 = RateController(mode=1)
    for _ in range(3):
        rc3.on_ack(3, 0, 1)
    assert rc3.mode == 1  # suggestion not higher: no step up


def test_recommend():
    hi = np.full(52, 40.0)
    mode, mask = recommend(hi, 50.0)
    assert mode == 5 and mask.count == 52
    lo = np.full(52, REQUIRED_SNR_DB[2] + 3.5)
    assert recommend(lo, 30.0)[0] in (2, 3)
    mixed = np.full(52, 40.0)
    mixed[10:30] = -5.0
    mode, mask = recommend(mixed, 40.0)
    assert mode == 5 and mask.count == 32 and not mask.bits[15]
    assert recommend(np.full(52, np.nan), REQUIRED_SNR_DB[1] + 3.1)[0] == 1
    assert recommend(np.full(52, np.nan), 0.0)[0] == 0
    assert recommend(hi, 50.0, caps=0b000011)[0] == 1


def test_quality_tracker():
    q = QualityTracker()
    q.record_probe(np.full(52, 20.0), 30.0)
    q.record_frame(True, 1, 25.0, 25.0, None)
    eff = q.effective_carrier_snr()
    assert eff[0] < 20.0  # MFSK SNR dropped by ~2.5 dB (EMA): carrier estimate follows
    q.record_frame(False, 1, float("nan"), float("nan"), None)
    assert q.window_fer == 0.5
    q.reset_window()
    assert q.window_frames == 0


def test_settings_validation():
    s = Settings()
    s.validate()
    assert s.replace(detect_ratio=10).detect_ratio == 10
    with pytest.raises(ValueError):
        s.replace(tx_level_dbfs=0.0)
    with pytest.raises(ValueError):
        s.replace(window_mfsk=0)
