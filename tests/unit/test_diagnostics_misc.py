import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from ultrasonic_airdrop.audio.backend import RingBuffer
from ultrasonic_airdrop.audio.diagnostics import analyze_hardware_check, log_sweep, make_test_tone
from ultrasonic_airdrop.config import PROFILE_NU, Settings
from ultrasonic_airdrop.transport.estimate import (
    estimate_transfer_s,
    max_units_per_frame,
    window_frames,
)


def _simulate(fs, f1, rng, lowpass=None, noise=1e-4, gain=0.5):
    sweeps = [log_sweep(fs, 2000, f1, 3.0, 10 ** (lv / 20)) for lv in (-12, -6)]
    recs = []
    for sw in sweeps:
        y = sw * gain
        if lowpass is not None:
            sos = butter(16, lowpass, fs=fs, output="sos")
            y = sosfilt(sos, y)
        rec = np.concatenate([np.zeros(3000), y, np.zeros(fs)])
        recs.append(rec + rng.standard_normal(len(rec)) * noise)
    silence = rng.standard_normal(fs) * noise
    return sweeps, recs, silence


def test_hardware_check_flat_response_supported(rng):
    fs = 48000
    sweeps, recs, sil = _simulate(fs, 23000, rng)
    res = analyze_hardware_check(fs, sweeps, recs, sil, 23000, 48000.0)
    assert res.verdict("nu") == "supported"
    assert res.verdict("au") == "supported"
    assert res.verdict("us") == "n/a"
    assert res.brick_wall_hz is None
    assert not res.clipping


def test_hardware_check_detects_brick_wall(rng):
    fs = 48000
    sweeps, recs, sil = _simulate(fs, 23000, rng, lowpass=16000)
    res = analyze_hardware_check(fs, sweeps, recs, sil, 23000)
    assert res.verdict("nu") == "not supported"
    assert res.brick_wall_hz is not None and 14500 <= res.brick_wall_hz <= 17000
    assert res.level_independent is True
    assert any("enhancement" in m.lower() for m in res.messages)


def test_hardware_check_clipping_flag(rng):
    fs = 48000
    sweeps, recs, sil = _simulate(fs, 23000, rng, gain=4.0)
    recs = [np.clip(r, -1, 1) for r in recs]
    assert analyze_hardware_check(fs, sweeps, recs, sil, 23000).clipping


def test_make_test_tone_level():
    x = make_test_tone(48000, 1000, -12, 0.5)
    assert np.max(np.abs(x)) == pytest.approx(10 ** (-12 / 20), rel=1e-3)


def test_ring_buffer():
    rb = RingBuffer(10)
    rb.write(np.arange(4))
    data, start = rb.read_since(0)
    assert start == 0 and list(data) == [0, 1, 2, 3]
    rb.write(np.arange(4, 16))
    data, start = rb.read_since(4)
    assert start == 6 and list(data) == list(range(6, 16))
    assert rb.overruns == 1
    data, start = rb.read_since(16)
    assert len(data) == 0 and start == 16


def test_estimates_are_sane():
    s = Settings()
    assert max_units_per_frame(PROFILE_NU, 1, None, False) == 4
    assert max_units_per_frame(PROFILE_NU, 0, None, False) == 3
    assert max_units_per_frame(PROFILE_NU, 5, None, False) > 20
    assert window_frames(s, PROFILE_NU, 1, None, False) == 3  # 20 s window cap
    t_small = estimate_transfer_s(PROFILE_NU, [1000], 1, s)
    t_big = estimate_transfer_s(PROFILE_NU, [100_000], 1, s)
    assert 30 < t_small < 300
    assert t_big > 60 * 60  # 100 KB at robust MFSK takes over an hour
    assert estimate_transfer_s(PROFILE_NU, [100_000], 5, s) < t_big / 4
    assert estimate_transfer_s(PROFILE_NU, [1000], 1, s, broadcast=True) > 0


def test_lowpass_inside_band_is_not_supported(rng):
    fs = 48000
    sweeps, recs, sil = _simulate(fs, 23000, rng, lowpass=18000)
    res = analyze_hardware_check(fs, sweeps, recs, sil, 23000)
    assert res.verdict("nu") in ("marginal", "not supported")
    assert res.verdict("au") == "supported"
    assert res.brick_wall_hz is not None and 17000 < res.brick_wall_hz < 18500
