"""Decode success at stated operating points and graceful failure below them (simulation)."""

import numpy as np
import pytest

from tests.sim.helpers import run_frames
from ultrasonic_airdrop.channel.simulator import ChannelConfig, synthetic_rir
from ultrasonic_airdrop.config import PROBE_MODE_ID, PROFILE_AU, PROFILE_NU
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.rate_adapt import recommend


def test_mode0_operating_point_fast():
    """Mode 0 at 6 dB in-band SNR + 150 ppm + RT60 0.3 s (DRR 6 dB): every frame decodes."""
    cfg = ChannelConfig(snr_db=6, ppm=150, rt60_s=0.3, drr_db=6, seed=11)
    out = run_frames(PROFILE_NU, 0, 20, cfg, seed=11)
    assert out.wrong == 0
    assert out.correct == 20


@pytest.mark.slow
def test_mode0_operating_point_99_percent():
    cfg = ChannelConfig(snr_db=6, ppm=150, rt60_s=0.3, drr_db=6, seed=12)
    out = run_frames(PROFILE_NU, 0, 100, cfg, seed=12)
    assert out.wrong == 0
    assert out.correct >= 99


@pytest.mark.parametrize("mode", [0, 1, 2, 4])
def test_graceful_failure_far_below_threshold(mode):
    """At -15 dB in-band SNR nothing decodes, and nothing wrong is ever accepted."""
    out = run_frames(PROFILE_NU, mode, 6, ChannelConfig(snr_db=-15, ppm=150, seed=3), seed=3)
    assert out.wrong == 0
    assert out.correct == 0


def test_graceful_failure_near_threshold_never_wrong():
    out = run_frames(PROFILE_NU, 1, 15, ChannelConfig(snr_db=-6.5, ppm=-200, seed=4), seed=4)
    assert out.wrong == 0


@pytest.mark.parametrize("ppm", [-250, -150, -50, 50, 150, 250])
@pytest.mark.parametrize("mode", [1, 3, 5])
def test_sample_rate_mismatch(mode, ppm):
    out = run_frames(PROFILE_NU, mode, 3, ChannelConfig(snr_db=22, ppm=ppm, seed=ppm & 0xFF), seed=5)
    assert out.wrong == 0 and out.correct == 3
    for f in out.frames:
        assert abs(f.info.ppm - ppm) < 20


@pytest.mark.parametrize("cfo", [-20.0, 20.0])
@pytest.mark.parametrize("mode", [0, 3, 5])
def test_carrier_frequency_offset(mode, cfo):
    out = run_frames(PROFILE_NU, mode, 3, ChannelConfig(snr_db=22, cfo_hz=cfo, ppm=80, seed=6), seed=6)
    assert out.wrong == 0 and out.correct == 3
    for f in out.frames:
        assert abs(f.info.cfo_hz - cfo) < 1.5


def test_timing_offsets_fractional_delays():
    for i, delay in enumerate((0.0, 0.0123457, 0.0371, 0.1)):
        out = run_frames(PROFILE_NU, 3, 2, ChannelConfig(snr_db=22, delay_s=delay, seed=i), seed=i)
        assert out.correct == 2


@pytest.mark.parametrize("burst_s", [0.05, 0.15, 0.3])
def test_noise_burst_inside_frame(burst_s):
    """A 0 dB-SIR noise burst of up to 300 ms inside a mode-0 frame is corrected by interleaved RS."""
    cfg = ChannelConfig(snr_db=10, ppm=100, burst_sir_db=0.0, seed=7)
    out = run_frames(PROFILE_NU, 0, 3, cfg, seed=7, burst_at=0.5, burst_s=burst_s)
    assert out.wrong == 0 and out.correct == 3


def test_burst_inside_ofdm_frame():
    cfg = ChannelConfig(snr_db=22, ppm=100, burst_sir_db=0.0, seed=8)
    out = run_frames(PROFILE_NU, 2, 3, cfg, seed=8, burst_at=0.5, burst_s=0.1)
    assert out.wrong == 0 and out.correct == 3


def test_amplitude_scaling_and_slow_fading():
    cfg = ChannelConfig(snr_db=8, gain_db=-30, fading_hz=0.5, fading_depth_db=6, ppm=50, seed=9)
    out = run_frames(PROFILE_NU, 1, 4, cfg, seed=9)
    assert out.wrong == 0 and out.correct == 4


@pytest.mark.parametrize("rt60", [0.2, 0.5])
def test_reverberation_mfsk(rt60):
    cfg = ChannelConfig(snr_db=6, ppm=150, rt60_s=rt60, drr_db=6, seed=10)
    out = run_frames(PROFILE_NU, 1, 4, cfg, seed=10)
    assert out.wrong == 0 and out.correct == 4


def test_reverberation_ofdm_short_distance():
    """OFDM DBPSK with RT60 0.3 s at a high direct-to-reverberant ratio (short distance)."""
    cfg = ChannelConfig(snr_db=25, ppm=150, rt60_s=0.3, drr_db=12, seed=13)
    out = run_frames(PROFILE_NU, 2, 4, cfg, seed=13)
    assert out.wrong == 0 and out.correct == 4


def test_strong_reverb_lowers_probe_estimate_so_ofdm_is_not_chosen():
    """With DRR 3 dB the probe measures the ISI and the receiver stays on MFSK."""
    cfg = ChannelConfig(snr_db=30, ppm=100, rt60_s=0.5, drr_db=3, seed=14)
    out = run_frames(PROFILE_NU, PROBE_MODE_ID, 2, cfg, seed=14)
    snr = np.mean([f.info.probe_snr_db for f in out.frames if f.ok], axis=0)
    mode, _ = recommend(snr, 30.0)
    assert mode <= 2


def test_band_limited_device_and_carrier_masking():
    """Low-pass at 18.5 kHz + notch at 17.9 kHz: MFSK still decodes; the probe masks dead carriers
    and OFDM with that mask decodes."""
    cfg = ChannelConfig(snr_db=24, ppm=60, lowpass_hz=18_500, notch_hz=17_900, seed=15)
    out = run_frames(PROFILE_NU, 1, 3, cfg, seed=15)
    assert out.wrong == 0 and out.correct == 3
    probe = run_frames(PROFILE_NU, PROBE_MODE_ID, 2, cfg, seed=15)
    snr = np.mean([f.info.probe_snr_db for f in probe.frames if f.ok], axis=0)
    hz = (374 + np.arange(52)) * 46.875
    assert snr[hz > 19_500].max() < snr[hz < 18_000].max() - 10
    mode, mask = recommend(snr, 40.0)
    assert mode >= 2 and mask.count < 52
    assert not any(mask.bits[i] for i in np.flatnonzero(hz > 19_600))
    ofdm_out = run_frames(PROFILE_NU, mode, 3, cfg, seed=16, mask=mask)
    assert ofdm_out.wrong == 0 and ofdm_out.correct == 3


def test_speaker_soft_clipping():
    cfg = ChannelConfig(snr_db=20, soft_clip_drive=2.5, ppm=40, seed=17)
    assert run_frames(PROFILE_NU, 1, 3, cfg, seed=17).correct == 3
    assert run_frames(PROFILE_NU, 2, 3, cfg, seed=18).correct == 3


def test_frame_loss_drops_whole_frames():
    cfg = ChannelConfig(snr_db=20, loss=0.5, seed=19)
    out = run_frames(PROFILE_NU, 1, 10, cfg, seed=19)
    assert out.wrong == 0 and 1 <= out.correct <= 9


def test_audible_profile():
    out = run_frames(PROFILE_AU, 0, 2, ChannelConfig(snr_db=0, ppm=150, rt60_s=0.3, seed=20), seed=20)
    assert out.correct == 2


def test_rir_shape():
    rng = np.random.default_rng(0)
    h = synthetic_rir(48000, 0.3, 6.0, rng)
    direct = h[0] ** 2
    rest = np.sum(h[1:] ** 2)
    assert 10 * np.log10(direct / rest) == pytest.approx(6.0, abs=0.01)
    assert len(h) == int(1.2 * 0.3 * 48000) + 1


def test_full_mask_is_default():
    assert CarrierMask.all().mask_id == 0
