from __future__ import annotations

import hashlib

import numpy as np

from ultrasonic_airdrop.channel.simulator import ChannelConfig, apply_channel
from ultrasonic_airdrop.config import Settings, TransferOptions
from ultrasonic_airdrop.link import frames as fr
from ultrasonic_airdrop.link.receiver_pipeline import RxFrame, RxInfo
from ultrasonic_airdrop.transport.offline import decode_recording, read_wav, render_broadcast, write_wav
from ultrasonic_airdrop.transport.session_receiver import ReceiverSession


def test_wav_encode_decode_roundtrip(tmp_path, make_file, rng):
    a = make_file("a.bin", rng.integers(0, 256, 400, dtype=np.uint8).tobytes())
    b = make_file("b é.txt", b"text " * 50)
    x, fs = render_broadcast([a, b], TransferOptions(compression=True), Settings(broadcast_passes=1))
    wav = tmp_path / "sig.wav"
    write_wav(wav, x, fs)
    y, fs2 = read_wav(wav)
    assert fs2 == fs == 48000
    noisy = apply_channel(y, ChannelConfig(snr_db=12, ppm=120, seed=3), fs)
    res = decode_recording(noisy, fs, tmp_path / "out")
    assert res.success, res.message
    for src in (a, b):
        assert (
            hashlib.sha256((tmp_path / "out" / src.name).read_bytes()).digest()
            == hashlib.sha256(src.read_bytes()).digest()
        )


class FakeRadio:
    def __init__(self) -> None:
        self.sent: list[np.ndarray] = []
        self.t = 0.0

    def transmit(self, samples):
        self.sent.append(samples)
        return self.t + len(samples) / 48000

    def tx_busy(self):
        return False

    def rx_busy(self):
        return False

    def now(self):
        return self.t


def test_version_mismatch_gets_cancel(tmp_path):
    radio = FakeRadio()
    rx = ReceiverSession(radio, Settings(), tmp_path, lambda e: None)
    offer = fr.Offer(0x3F, 0, 1, 10, 10, bytes(32), False, bytes(16), "x", version=2).pack()
    raw = bytearray(fr.LinkFrame(fr.FrameType.OFFER, 5, 0, offer).pack())
    raw[1] = 2  # protocol version 2 on the link header
    import struct

    from ultrasonic_airdrop.fec.crc import crc32

    raw[-4:] = struct.pack("<I", crc32(bytes(raw[:-4])))
    rx.on_frame(RxFrame(bytes(raw), RxInfo("nu", 0, 100)), 0.0)
    assert len(radio.sent) == 1  # CANCEL(VERSION) transmitted
    assert rx.sess is None


def test_self_echo_and_foreign_sessions_ignored(tmp_path):
    radio = FakeRadio()
    events = []
    rx = ReceiverSession(radio, Settings(), tmp_path, events.append)
    ack = fr.LinkFrame(fr.FrameType.ACK, 1, 0, fr.Ack(0, 1, 0, 0, 1, bytes(7)).pack(), fr.DIR_R2S).pack()
    rx.on_frame(RxFrame(ack, RxInfo("nu", 0, 100)), 0.0)
    data = fr.LinkFrame(fr.FrameType.DATA, 99, 3, fr.Data(0, 0, 1, b"x").pack()).pack()
    rx.on_frame(RxFrame(data, RxInfo("nu", 0, 100)), 0.0)
    assert rx.sess is None and not radio.sent
