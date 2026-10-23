"""SoundDeviceBackend against a fake ``sounddevice`` module (no hardware needed)."""

from __future__ import annotations

import types
from typing import ClassVar

import numpy as np
import pytest

import ultrasonic_airdrop.audio.devices as devices
import ultrasonic_airdrop.audio.sd_backend as sdb
from ultrasonic_airdrop.errors import AudioDeviceError


class FakeStream:
    instances: ClassVar[list[FakeStream]] = []

    def __init__(self, samplerate, blocksize, device, channels, dtype, latency, callback, extra_settings):
        if extra_settings is not None and getattr(extra_settings, "exclusive", False) and FakeSD.reject_exclusive:
            raise RuntimeError("exclusive mode not supported")
        self.samplerate = FakeSD.opened_rate or samplerate
        self.callback = callback
        self.latency = 0.05
        self.extra = extra_settings
        self.started = self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class FakeSD(types.SimpleNamespace):
    reject_exclusive = False
    opened_rate: int | None = None
    rate_error: str | None = None


def make_fake() -> FakeSD:
    sd = FakeSD()
    sd.InputStream = FakeStream
    sd.OutputStream = FakeStream
    sd.default = types.SimpleNamespace(device=[0, 0])
    sd.WasapiSettings = lambda exclusive: types.SimpleNamespace(exclusive=exclusive)
    sd.query_hostapis = lambda: [{"name": "Windows WASAPI"}]
    sd.query_devices = lambda: [
        {
            "name": "Mic+Spk",
            "hostapi": 0,
            "max_input_channels": 2,
            "max_output_channels": 2,
            "default_samplerate": 48000,
        }
    ]

    def check(**kw):
        if FakeSD.rate_error:
            raise RuntimeError(FakeSD.rate_error)

    sd.check_input_settings = check
    sd.check_output_settings = check
    return sd


@pytest.fixture
def fake(monkeypatch):
    sd = make_fake()
    FakeStream.instances = []
    FakeSD.reject_exclusive = False
    FakeSD.opened_rate = None
    FakeSD.rate_error = None
    monkeypatch.setattr(devices, "load_sounddevice", lambda: sd)
    monkeypatch.setattr(sdb, "load_sounddevice", lambda: sd)
    return sd


def test_capture_playback_and_close(fake):
    b = sdb.SoundDeviceBackend(48000, 0, 0)
    b.start()
    inp, out = FakeStream.instances
    # stereo capture is down-mixed
    inp.callback(np.ones((1024, 2), dtype=np.float32) * 0.5, 1024, None, None)
    data, start = b.read_input()
    assert start == 0 and len(data) == 1024 and np.allclose(data, 0.5)
    info = b.play(np.arange(3000, dtype=np.float64) / 3000)
    assert info.mute_start <= 1024 <= info.mute_end
    assert info.end_time > info.start_time
    assert b.playing()
    buf = np.zeros((2048, 1), dtype=np.float32)
    out.callback(buf, 2048, None, None)
    assert buf[1, 0] == pytest.approx(1 / 3000)
    out.callback(buf, 2048, None, None)
    assert not b.playing()
    assert np.all(buf[952:, 0] == 0)  # silence after the burst
    b.close()
    assert all(s.closed for s in FakeStream.instances)
    b.close()  # idempotent


def test_exclusive_falls_back_to_shared(fake):
    FakeSD.reject_exclusive = True
    b = sdb.SoundDeviceBackend(48000, 0, 0, exclusive=True)
    b.start()
    assert not b.exclusive_active
    assert all(s.extra is None for s in FakeStream.instances if not s.closed)
    b.close()


def test_rate_mismatch_and_unsupported_rate(fake):
    FakeSD.opened_rate = 44100
    with pytest.raises(AudioDeviceError, match="44100"):
        sdb.SoundDeviceBackend(48000, 0, 0).start()
    assert all(s.closed for s in FakeStream.instances)
    FakeSD.opened_rate = None
    FakeSD.rate_error = "Invalid sample rate"
    with pytest.raises(AudioDeviceError, match="sample rate"):
        sdb.SoundDeviceBackend(96000, 0, 0).start()


def test_error_translation():
    assert "denied" in devices.describe_error(RuntimeError("Access denied"), "input").user_message
    assert "busy" in devices.describe_error(RuntimeError("Device unavailable"), "output").user_message
    assert "No usable" in devices.describe_error(RuntimeError("Error querying device -1"), "input").user_message
