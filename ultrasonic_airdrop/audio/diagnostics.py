"""Hardware self-check (single machine), test tone and two-machine link test.

The self-check plays a logarithmic sweep (2 kHz → min(23 kHz, Nyquist − 1 kHz))
through the selected output while recording the selected input, at two levels,
plus one second of silence for the noise floor.  :func:`analyze_hardware_check`
(pure DSP, unit-tested with synthetic recordings) deconvolves the speaker→mic
magnitude response, reports each profile band relative to 2–4 kHz, the noise
floor, clipping, and detects the "brick wall" signature (a steep, level
independent drop at a fixed frequency) typical of audio enhancements/APOs,
driver low-pass filters or shared-mode sample-rate conversion.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt, welch

from ultrasonic_airdrop.audio.backend import AudioBackend
from ultrasonic_airdrop.config import PROBE_MODE_ID, PROFILES, Profile, Settings
from ultrasonic_airdrop.errors import AudioDeviceError
from ultrasonic_airdrop.link import frames as fr
from ultrasonic_airdrop.link.receiver_pipeline import RxFrame
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy.numerology import FloatArray
from ultrasonic_airdrop.phy.synth import apply_ramps, db_to_amp
from ultrasonic_airdrop.transport.events import EventSink, LinkTestEvent
from ultrasonic_airdrop.transport.linkio import LinkIO, Radio
from ultrasonic_airdrop.transport.rate_adapt import recommend

log = logging.getLogger(__name__)

SWEEP_S = 3.0
SWEEP_F0 = 2000.0
NOISE_S = 1.0
REF_BAND = (2000.0, 4000.0)
LEVELS_DB = (-12.0, -6.0)

TROUBLESHOOTING = [
    "Disable audio enhancements: Settings → System → Sound → (device) → Advanced → Audio enhancements: Off "
    "(or Sound Control Panel → device Properties → Enhancements/Advanced → disable all enhancements).",
    "Try WASAPI exclusive mode (Diagnostics page) — it bypasses the shared-mode processing chain on many drivers.",
    "Set both devices to 48000 Hz in the Sound Control Panel (Advanced → Default Format).",
    "Turn off vendor suites (Dell MaxxAudio, Realtek/Waves Audio, 'Voice Clarity', noise suppression, beamforming).",
    "Close applications that hold the microphone or speaker (conferencing tools, browsers).",
    "Avoid Bluetooth headsets/speakers: their codecs remove content above ~16 kHz.",
    "Use an external USB microphone/speaker, or switch to the Audible profile.",
]


def log_sweep(fs: int, f0: float, f1: float, duration: float, amplitude: float) -> FloatArray:
    """Exponential (log) sine sweep with 10 ms fades."""
    t = np.arange(int(duration * fs)) / fs
    k = np.log(f1 / f0)
    x = amplitude * np.sin(2 * np.pi * f0 * duration / k * (np.exp(t * k / duration) - 1.0))
    return apply_ramps(x, int(0.01 * fs))


@dataclass
class BandVerdict:
    """Result for one profile band."""

    profile_key: str
    band_hz: tuple[float, float]
    relative_db: float
    snr_db: float
    verdict: str  # supported | marginal | not supported | n/a


@dataclass
class HardwareCheckResult:
    """Outcome of :func:`analyze_hardware_check`."""

    sample_rate: int
    freqs_hz: FloatArray
    response_db: FloatArray
    noise_db: FloatArray
    bands: list[BandVerdict]
    clipping: bool
    peak_dbfs: float
    noise_floor_dbfs: float
    brick_wall_hz: float | None
    level_independent: bool | None
    actual_rate: float | None = None
    messages: list[str] = field(default_factory=list)

    def verdict(self, profile_key: str) -> str:
        """Verdict for a profile."""
        for b in self.bands:
            if b.profile_key == profile_key:
                return b.verdict
        return "n/a"


def _smooth_db(freqs: FloatArray, power: FloatArray, centers: FloatArray, frac: float = 1 / 12) -> FloatArray:
    out = np.full(len(centers), np.nan)
    for i, c in enumerate(centers):
        lo, hi = c * 2 ** (-frac / 2), c * 2 ** (frac / 2)
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            out[i] = 10 * np.log10(np.mean(power[sel]) + 1e-20)
    return out


def _response(rec: FloatArray, sweep: FloatArray, fs: int, centers: FloatArray) -> FloatArray:
    n = 1 << int(np.ceil(np.log2(len(rec) + len(sweep))))
    x = np.fft.rfft(sweep, n)
    y = np.fft.rfft(rec, n)
    mag2 = np.abs(x) ** 2
    h2 = np.abs(y) ** 2 / (mag2 + 1e-6 * mag2.max())
    freqs = np.fft.rfftfreq(n, 1 / fs)
    return _smooth_db(freqs, h2, centers)


def _align(rec: FloatArray, sweep: FloatArray) -> int:
    c = correlate(rec, sweep, mode="valid", method="fft")
    return int(np.argmax(np.abs(c))) if len(c) else 0


def _band_snr(
    rec: FloatArray,
    offset: int,
    noise: FloatArray,
    fs: int,
    f0: float,
    f1: float,
    band: tuple[float, float],
    duration: float,
) -> float:
    k = np.log(f1 / f0)
    lo, hi = max(band[0], f0), min(band[1], f1)
    if hi <= lo:
        return float("nan")
    t_lo = duration * np.log(lo / f0) / k
    t_hi = duration * np.log(hi / f0) / k
    a, b = offset + int(t_lo * fs), offset + int(t_hi * fs)
    seg = rec[max(0, a) : max(0, b)]
    if len(seg) < 256 or len(noise) < 256:
        return float("nan")
    sos = butter(6, [lo, min(hi, 0.99 * fs / 2)], btype="bandpass", fs=fs, output="sos")
    ps = float(np.mean(sosfiltfilt(sos, seg) ** 2))
    pn = float(np.mean(sosfiltfilt(sos, noise) ** 2)) + 1e-20
    return float(10 * np.log10(max(ps - pn, 1e-20) / pn))


def _brick_wall(centers: FloatArray, resp: FloatArray, noise_rel: FloatArray) -> float | None:
    """Frequency of a steep drop (>= 25 dB within 1.5 kHz) above 10 kHz, or ``None``.

    Natural speaker/microphone roll-off is a few dB per kHz; a drop this steep is
    the signature of a digital low-pass (APO/enhancement, driver filter, or
    shared-mode sample-rate conversion).  ``noise_rel`` is kept for diagnostics.
    """
    best_drop = 0.0
    best_i = -1
    for i, c in enumerate(centers):
        if c < 10_000 or np.isnan(resp[i]):
            continue
        j = np.flatnonzero((centers > c) & (centers <= c + 1500))
        if j.size == 0 or np.isnan(resp[j[-1]]):
            continue
        drop = float(resp[i] - np.nanmin(resp[j]))
        if drop > best_drop:
            best_drop, best_i = drop, i
    if best_drop < 25.0 or best_i < 0:
        return None
    # Report the -3 dB edge: the highest frequency below the steep region that is
    # still within 3 dB of the pass-band level.
    c_best = centers[best_i]
    band = (centers >= 4000) & (centers <= c_best - 1000)
    passband = float(np.nanmedian(resp[band])) if band.any() else float(resp[best_i])
    edge = np.flatnonzero((centers <= c_best + 1500) & (resp >= passband - 3.0))
    return float(centers[edge[-1]]) if edge.size else float(c_best)


def analyze_hardware_check(
    fs: int,
    sweeps: list[FloatArray],
    recordings: list[FloatArray],
    noise: FloatArray,
    f1: float,
    actual_rate: float | None = None,
) -> HardwareCheckResult:
    """Analyse sweep recordings (one per level) and a silence recording."""
    centers = np.geomspace(1000.0, f1, 160)
    responses = []
    offsets = []
    for sw, rec in zip(sweeps, recordings, strict=True):
        off = _align(rec, sw)
        offsets.append(off)
        responses.append(_response(rec[off : off + len(sw) + fs // 4], sw, fs, centers))
    ref_sel = (centers >= REF_BAND[0]) & (centers <= REF_BAND[1])
    resp_abs = responses[-1]
    ref = np.nanmean(resp_abs[ref_sel])
    resp = resp_abs - ref
    # Noise floor in the same "response" units: noise PSD over the sweep's PSD.
    sw = sweeps[-1]
    fn, pn = welch(noise, fs, nperseg=4096)
    _fsw, psw = welch(sw, fs, nperseg=4096)
    noise_rel = _smooth_db(fn, pn / (psw + 1e-20), centers) - ref
    noise_dbfs = float(10 * np.log10(np.mean(noise**2) + 1e-20))
    peak = max(float(np.max(np.abs(r))) for r in recordings) if recordings else 0.0
    clipping = peak >= 0.99
    bands = []
    for p in PROFILES.values():
        lo, hi = p.ofdm_first_bin * p.bin_hz, p.ofdm_last_bin * p.bin_hz
        if hi >= fs / 2 - 500 or (p.sample_rate != fs and p.sample_rate > fs):
            bands.append(BandVerdict(p.key, (lo, hi), float("nan"), float("nan"), "n/a"))
            continue
        sel = (centers >= lo) & (centers <= hi)
        rel = float(np.nanmean(resp[sel])) if sel.any() else float("nan")
        snr = _band_snr(recordings[-1], offsets[-1], noise, fs, SWEEP_F0, f1, (lo, hi), len(sw) / fs)
        if np.isnan(rel) or np.isnan(snr):
            verdict = "not supported"
        elif snr >= 25 and rel >= -25:
            verdict = "supported"
        elif snr >= 12 and rel >= -40:
            verdict = "marginal"
        else:
            verdict = "not supported"
        bands.append(BandVerdict(p.key, (lo, hi), rel, snr, verdict))
    wall = _brick_wall(centers, resp, noise_rel)
    level_indep: bool | None = None
    if wall is not None and len(responses) >= 2:
        r0 = responses[0] - np.nanmean(responses[0][ref_sel])
        w0 = _brick_wall(centers, r0, noise_rel + (LEVELS_DB[1] - LEVELS_DB[0]))
        level_indep = w0 is not None and abs(w0 - wall) <= 400
    messages = []
    if clipping:
        messages.append("The recording clipped: lower the transmit level or the microphone gain.")
    if actual_rate is not None and abs(actual_rate - fs) > 0.5:
        messages.append(f"The device opened at {actual_rate:.0f} Hz instead of {fs} Hz.")
    if wall is not None:
        messages.append(
            f"Steep 'brick wall' drop at ≈{wall / 1000:.1f} kHz"
            + (" that does not move with level" if level_indep else "")
            + ". Likely causes: audio enhancements/APO processing, a driver low-pass filter, or sample-rate "
            "conversion in shared mode."
        )
    nu = next(b for b in bands if b.profile_key == "nu")
    if nu.verdict != "supported":
        messages.append(f"Near-ultrasonic band verdict: {nu.verdict}. Try the fixes below or the Audible profile.")
        messages.extend(TROUBLESHOOTING)
    log.info(
        kv(
            "hardware check analysed",
            nu=nu.verdict,
            rel=nu.relative_db,
            snr=nu.snr_db,
            wall=wall,
            clipping=clipping,
        )
    )
    return HardwareCheckResult(
        fs,
        centers,
        resp,
        noise_rel,
        bands,
        clipping,
        20 * np.log10(peak + 1e-12),
        noise_dbfs,
        wall,
        level_indep,
        actual_rate,
        messages,
    )


def run_hardware_check(
    backend: AudioBackend,
    level_dbfs: float = -6.0,
    progress: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> HardwareCheckResult:
    """Play/record the sweeps on a real backend and analyse.  The backend is closed afterwards."""
    fs = backend.sample_rate
    f1 = min(23_000.0, fs / 2 - 1000.0)
    levels = (level_dbfs - 6.0, level_dbfs)
    sweeps = [log_sweep(fs, SWEEP_F0, f1, SWEEP_S, db_to_amp(lv)) for lv in levels]

    def capture(seconds: float) -> FloatArray:
        chunks = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if should_stop is not None and should_stop():
                raise AudioDeviceError("Hardware check cancelled.")
            time.sleep(0.05)
            data, _ = backend.read_input()
            chunks.append(data)
        data, _ = backend.read_input()
        chunks.append(data)
        return np.concatenate(chunks) if chunks else np.zeros(0)

    try:
        backend.start()
        time.sleep(0.3)
        backend.read_input()
        if progress:
            progress("Measuring background noise (1 s of silence)…")
        noise = capture(NOISE_S)
        recs = []
        for lv, sw in zip(levels, sweeps, strict=True):
            if progress:
                progress(f"Playing a {SWEEP_S:.0f} s sweep 2–{f1 / 1000:.0f} kHz at {lv:.0f} dBFS…")
            backend.read_input()
            backend.play(sw)
            recs.append(capture(SWEEP_S + 1.0))
        actual = getattr(backend, "actual_rate", None)
    finally:
        backend.close()
    if len(noise) < fs // 2 or any(len(r) < len(sweeps[0]) for r in recs):
        raise AudioDeviceError("Too little audio was captured.", "Check the microphone selection and permissions.")
    return analyze_hardware_check(fs, sweeps, recs, noise, f1, actual)


def make_test_tone(fs: int, freq_hz: float, level_dbfs: float, seconds: float) -> FloatArray:
    """A ramped sine test tone."""
    t = np.arange(int(seconds * fs)) / fs
    return apply_ramps(db_to_amp(level_dbfs) * np.sin(2 * np.pi * freq_hz * t), int(0.02 * fs))


# --------------------------------------------------------------------------------------
# Link test
# --------------------------------------------------------------------------------------


class LinkTestSender:
    """Transmits PROBE frames back-to-back for ``duration_s``."""

    role = "linktest-sender"

    def __init__(
        self, radio: Radio, profile: Profile, settings: Settings, events: EventSink, duration_s: float = 20.0
    ) -> None:
        self.radio = radio
        self.events = events
        self.duration_s = duration_s
        self.io = LinkIO(radio, profile, settings, fr.DIR_S2R, random.getrandbits(32))
        self.start = radio.now()
        self.sent = 0
        self._done = False

    @property
    def done(self) -> bool:
        """Finished."""
        return self._done

    def stop(self) -> None:
        """Stop early."""
        self._done = True

    def on_frame(self, rx: RxFrame, now: float) -> None:
        """Ignored."""

    def tick(self, now: float) -> None:
        """Queue the next probe once the previous one has played."""
        if self._done or self.radio.tx_busy():
            return
        if now - self.start >= self.duration_s:
            self._done = True
            return
        self.io.send([self.io.build(fr.Probe(), mode_id=PROBE_MODE_ID)])
        self.sent += 1
        self.events(LinkTestEvent(self.role, probes_sent=self.sent, elapsed_s=now - self.start))


class LinkTestReceiver:
    """Measures incoming PROBE frames and reports live statistics."""

    role = "linktest-receiver"

    def __init__(
        self,
        events: EventSink,
        pipeline_stats: Callable[[], tuple[int, int]],
        duration_s: float | None = None,
        now: float = 0.0,
    ) -> None:
        self.events = events
        self.pipeline_stats = pipeline_stats
        self.duration_s = duration_s
        self.start = now
        self.frames_ok = 0
        self.frame_errors = 0
        self.snr_hist: list[FloatArray] = []
        self.mfsk_hist: list[float] = []
        self.ppm_hist: list[float] = []
        self._done = False
        self.last: LinkTestEvent | None = None

    @property
    def done(self) -> bool:
        """Finished."""
        return self._done

    def stop(self) -> None:
        """Stop."""
        self._done = True

    def on_frame(self, rx: RxFrame, now: float) -> None:
        """Account a frame."""
        if rx.info.header is None or rx.info.header.mode_id != PROBE_MODE_ID:
            return
        if rx.body is None:
            self.frame_errors += 1
        else:
            self.frames_ok += 1
            if rx.info.probe_snr_db is not None:
                self.snr_hist.append(rx.info.probe_snr_db)
            self.mfsk_hist.append(rx.info.mfsk_snr_db)
            if rx.info.postamble_found:
                self.ppm_hist.append(rx.info.ppm)
        self._emit(now)

    def summary(self, now: float) -> LinkTestEvent:
        """Current statistics."""
        detections, _ = self.pipeline_stats()
        snr = np.mean(np.stack(self.snr_hist[-5:]), axis=0) if self.snr_hist else None
        mfsk = float(np.mean(self.mfsk_hist[-5:])) if self.mfsk_hist else float("nan")
        mode, mask = (-1, None)
        if snr is not None or self.mfsk_hist:
            mode, mask = recommend(snr if snr is not None else np.full(52, np.nan), mfsk)
        return LinkTestEvent(
            self.role,
            detections=detections,
            frames_ok=self.frames_ok,
            frame_errors=self.frame_errors,
            carrier_snr_db=tuple(float(v) for v in snr) if snr is not None else None,
            mfsk_snr_db=mfsk,
            ppm=float(np.median(self.ppm_hist)) if self.ppm_hist else float("nan"),
            recommended_mode=mode,
            carriers_enabled=mask.count if mask is not None else 0,
            elapsed_s=now - self.start,
        )

    def _emit(self, now: float) -> None:
        self.last = self.summary(now)
        self.events(self.last)

    def tick(self, now: float) -> None:
        """Stop after ``duration_s`` (if set)."""
        if self.duration_s is not None and now - self.start >= self.duration_s:
            self._emit(now)
            self._done = True
