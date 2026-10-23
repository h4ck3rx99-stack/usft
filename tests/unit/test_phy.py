import numpy as np
import pytest

from ultrasonic_airdrop.config import PROBE_MODE_ID, PROFILE_AU, PROFILE_NU, PROFILE_US
from ultrasonic_airdrop.link.codec import FrameCodec, body_layout, frame_airtime, max_body_len
from ultrasonic_airdrop.link.receiver_pipeline import ReceiverPipeline
from ultrasonic_airdrop.phy import mfsk, ofdm
from ultrasonic_airdrop.phy.chirp import chirp
from ultrasonic_airdrop.phy.detector import ChirpDetector
from ultrasonic_airdrop.phy.numerology import GRAY, INV_GRAY, numerology
from ultrasonic_airdrop.phy.resample import fractional_delay, frequency_shift, resample_by_ratio
from ultrasonic_airdrop.phy.synth import apply_ramps, db_to_amp


@pytest.mark.parametrize("profile", [PROFILE_NU, PROFILE_AU, PROFILE_US], ids=lambda p: p.key)
def test_numerology(profile):
    num = numerology(profile)
    assert profile.bin_hz == pytest.approx(46.875)
    assert num.sym_len / num.fs == pytest.approx(0.0266667, rel=1e-4)
    assert len(num.ofdm_bins) == 52
    assert set(num.mfsk_bins_a) <= set(num.ofdm_bins) and set(num.mfsk_bins_b) <= set(num.ofdm_bins)
    assert num.fft_offset >= num.taper and num.fft_offset + num.n <= num.sym_len


def test_gray_mapping_adjacent_differ_by_one_bit():
    for i in range(15):
        assert bin(int(GRAY[i]) ^ int(GRAY[i + 1])).count("1") == 1
    assert all(GRAY[INV_GRAY[v]] == v for v in range(16))


def test_nu_band_values():
    num = numerology(PROFILE_NU)
    assert num.mfsk_bins_a[0] * PROFILE_NU.bin_hz == pytest.approx(17625.0)
    assert num.mfsk_bins_b[-1] * PROFILE_NU.bin_hz == pytest.approx(19453.125)


@pytest.mark.parametrize("profile", [PROFILE_NU, PROFILE_AU, PROFILE_US], ids=lambda p: p.key)
def test_mfsk_clean_bit_exact(profile, rng):
    num = numerology(profile)
    data = bytes(range(256))
    x = mfsk.modulate(data, num, 0.5, abs_start=0)
    assert np.max(np.abs(x)) <= 0.5 + 1e-9
    starts = np.arange(len(data)) * num.sym_len
    res = mfsk.demodulate(x, starts, num, 1.6)
    assert res.data == data
    assert not res.erasures.any()
    assert res.snr_db > 40


def test_mfsk_erasure_flags_on_ambiguous_symbol():
    num = numerology(PROFILE_NU)
    x = mfsk.modulate(b"\x00\x11", num, 0.5, 0)
    # add a second equally strong tone to symbol 0 subband A
    t = np.arange(num.sym_len)
    k = num.mfsk_bins_a[9]
    x[: num.sym_len] += 0.25 * np.cos(2 * np.pi * k * t / num.n)
    res = mfsk.demodulate(x, np.array([0, num.sym_len]), num, 1.6)
    assert res.erasures[0] and not res.erasures[1]


@pytest.mark.parametrize("bits", [1, 2])
@pytest.mark.parametrize("masked", [False, True])
def test_ofdm_clean_bit_exact(bits, masked, rng):
    num = numerology(PROFILE_NU)
    arr = np.ones(52, dtype=bool)
    if masked:
        arr[[0, 5, 6, 7, 30, 51]] = False
    mask = ofdm.CarrierMask.from_array(arr)
    lay = ofdm.layout(num, mask)
    coded = rng.integers(0, 256, 700, dtype=np.uint8).tobytes()
    x = ofdm.modulate_body(coded, bits, lay, num)
    x = np.concatenate([np.zeros(300), x, np.zeros(300)])
    res = ofdm.demodulate_body(x, 300, len(coded), bits, lay, num)
    assert res.data == coded
    assert res.snr_db > 20
    assert not res.erasures.any()


def test_carrier_mask_bytes_and_id():
    m = ofdm.CarrierMask.all()
    assert m.mask_id == 0 and m.count == 52
    arr = np.ones(52, dtype=bool)
    arr[3] = False
    m2 = ofdm.CarrierMask.from_array(arr)
    assert ofdm.CarrierMask.from_bytes(m2.to_bytes()) == m2
    assert 1 <= m2.mask_id <= 255
    with pytest.raises(ValueError):
        ofdm.CarrierMask.from_bytes(bytes([0xFF] * 7))


def test_ofdm_value_mapping_inverse(rng):
    data = rng.integers(0, 256, 123, dtype=np.uint8).tobytes()
    for bits in (1, 2):
        vals = ofdm.bytes_to_values(data, 45, bits)
        assert ofdm.values_to_bytes(vals, len(data), bits) == data


def test_probe_analysis_detects_dead_carriers(rng):
    num = numerology(PROFILE_NU)
    train = ofdm.training_samples(num)
    sil = num.probe_silence_len
    x = np.concatenate([np.zeros(500), train, np.zeros(sil), np.zeros(500)])
    x = x / np.max(np.abs(x)) * 0.3
    x += rng.standard_normal(len(x)) * 1e-3
    # notch out carriers 20..25 with an FFT mask
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / num.fs)
    lo, hi = num.ofdm_bins[20] * PROFILE_NU.bin_hz - 20, num.ofdm_bins[25] * PROFILE_NU.bin_hz + 20
    spec[(f > lo) & (f < hi)] *= 1e-3
    x = np.fft.irfft(spec, len(x))
    snr = ofdm.analyze_probe(x, 500, 500 + len(train) - num.taper, sil, num)
    assert np.median(snr[:15]) > 20
    assert np.all(snr[21:25] < np.median(snr[:15]) - 15)


def test_chirps_and_ramps():
    c = chirp(PROFILE_NU)
    assert np.max(np.abs(c)) <= 1.0 and len(c) == 2048
    spec = np.abs(np.fft.rfft(c))
    f = np.fft.rfftfreq(len(c), 1 / 48000)
    assert f[np.argmax(spec)] > 16000
    y = apply_ramps(np.ones(1000), 240)
    assert y[0] < 0.01 and y[-1] < 0.01 and y[500] == 1.0


@pytest.mark.parametrize("profile", [PROFILE_NU, PROFILE_AU], ids=lambda p: p.key)
def test_detector_timing_accuracy(profile, rng):
    det = ChirpDetector(profile)
    num = numerology(profile)
    for _ in range(5):
        lead = float(rng.uniform(1000, 5000))
        sig = np.concatenate([chirp(profile, True), chirp(profile, False)]) * 0.3
        x = fractional_delay(sig, lead - int(lead))
        x = np.concatenate([np.zeros(int(lead)), x, np.zeros(2000)])
        x += rng.standard_normal(len(x)) * 0.01
        eu, ed = det.envelopes(det.bandpass(x))
        p = det.refine(eu, det.first_arrival(eu, int(np.argmax(eu))))
        q = det.refine(ed, det.first_arrival(ed, int(np.argmax(ed))))
        t0 = 0.5 * (p + q - num.chirp_len)
        assert abs(t0 - lead) < 1.0


@pytest.mark.parametrize("ppm", [-250, -150, -50, 50, 150, 250])
def test_sample_rate_ratio_estimation(ppm, rng):
    codec = FrameCodec(PROFILE_NU, -6)
    body = rng.integers(0, 256, 150, dtype=np.uint8).tobytes()
    sig = resample_by_ratio(codec.encode(body, 1), 1 / (1 + ppm * 1e-6))
    x = np.concatenate([np.zeros(20000), sig, np.zeros(48000)])
    x += rng.standard_normal(len(x)) * 0.002
    pl = ReceiverPipeline(PROFILE_NU)
    pl.feed(x)
    frames = pl.process()
    assert len(frames) == 1 and frames[0].body == body
    assert frames[0].info.postamble_found
    assert abs(frames[0].info.ppm - ppm) < 20


def test_frequency_shift_and_resample_sanity():
    fs = 48000
    t = np.arange(4800) / fs
    x = np.sin(2 * np.pi * 1000 * t)
    y = frequency_shift(x, 100.0, fs)
    f = np.fft.rfftfreq(len(y), 1 / fs)
    assert f[np.argmax(np.abs(np.fft.rfft(y)))] == pytest.approx(1100, abs=11)
    assert len(resample_by_ratio(x, 1.0001)) == round(len(x) / 1.0001)


@pytest.mark.parametrize("profile", [PROFILE_NU, PROFILE_AU, PROFILE_US], ids=lambda p: p.key)
@pytest.mark.parametrize("mode", [0, 1, 2, 3, 4, 5, PROBE_MODE_ID])
def test_codec_pipeline_roundtrip(profile, mode, rng):
    codec = FrameCodec(profile, -6)
    body = rng.integers(0, 256, 120, dtype=np.uint8).tobytes()
    sig = codec.encode(body, mode)
    assert np.max(np.abs(sig)) <= db_to_amp(-6) + 1e-9
    assert len(sig) / profile.sample_rate == pytest.approx(frame_airtime(profile, mode, len(body)), abs=1e-6)
    x = np.concatenate([np.zeros(int(0.6 * profile.sample_rate) + 17), sig, np.zeros(profile.sample_rate)])
    x += rng.standard_normal(len(x)) * 1e-3
    pl = ReceiverPipeline(profile)
    out = []
    step = profile.sample_rate // 10
    for i in range(0, len(x), step):
        pl.feed(x[i : i + step])
        out += pl.process()
    assert len(out) == 1
    assert out[0].body == body
    if mode == PROBE_MODE_ID:
        assert out[0].info.probe_snr_db is not None and np.median(out[0].info.probe_snr_db) > 15


def test_airtime_limits():
    for mode in range(6):
        n = max_body_len(PROFILE_NU, mode, 6.0)
        assert frame_airtime(PROFILE_NU, mode, n) <= 6.0 < frame_airtime(PROFILE_NU, mode, n + 64)
    lay = body_layout(PROFILE_NU, 2, 100)
    assert lay.ofdm_symbols >= 2


def test_half_duplex_mute_suppresses_detection(rng):
    codec = FrameCodec(PROFILE_NU, -6)
    sig = codec.encode(b"x" * 40, 0)
    x = np.concatenate([np.zeros(10000), sig, np.zeros(48000)])
    pl = ReceiverPipeline(PROFILE_NU)
    pl.add_mute(9000, 10000 + len(sig) + 100)
    pl.feed(x)
    assert pl.process() == []
