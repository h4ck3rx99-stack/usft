"""Station: binds an audio backend, a receive pipeline and a session; real-time runner."""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol

import numpy as np

from ultrasonic_airdrop.audio.backend import AudioBackend
from ultrasonic_airdrop.config import Profile, Settings, profiles_for_rate
from ultrasonic_airdrop.errors import AirDropError
from ultrasonic_airdrop.link.receiver_pipeline import MultiPipeline, ReceiverPipeline, RxFrame
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy.numerology import FloatArray
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.events import ErrorEvent, EventSink, SpectrumEvent, null_sink

log = logging.getLogger(__name__)
SPECTRUM_INTERVAL_S = 0.1
SPECTRUM_N = 4096


class Session(Protocol):
    """Anything a station can drive."""

    @property
    def done(self) -> bool:
        """Finished."""
        ...

    def on_frame(self, rx: RxFrame, now: float) -> None:
        """A frame was decoded."""
        ...

    def tick(self, now: float) -> None:
        """Time passes."""
        ...


def build_pipeline(
    sample_rate: int, settings: Settings, masks: dict[int, CarrierMask], profiles: list[Profile] | None = None
) -> MultiPipeline:
    """Receive pipelines for every profile receivable at ``sample_rate``."""
    profs = profiles if profiles is not None else profiles_for_rate(sample_rate)
    if not profs:
        raise AirDropError(f"No modem profile works at {sample_rate} Hz.", "Use 48 kHz (or 96 kHz for ultrasonic).")
    return MultiPipeline([ReceiverPipeline(p, settings.detect_ratio, settings.erasure_ratio, masks.get) for p in profs])


class Station:
    """Implements the session-facing :class:`~ultrasonic_airdrop.transport.linkio.Radio` interface."""

    def __init__(
        self,
        backend: AudioBackend,
        pipeline: MultiPipeline,
        settings: Settings,
        events: EventSink = null_sink,
        spectrum_profile: Profile | None = None,
    ) -> None:
        self.backend = backend
        self.pipeline = pipeline
        self.settings = settings
        self.events = events
        self.session: Session | None = None
        self.spectrum_profile = spectrum_profile
        self._tx_end = 0.0
        self._last_spectrum = 0.0
        self._started = False
        self.frames_seen = 0

    # Radio -----------------------------------------------------------------
    def transmit(self, samples: FloatArray) -> float:
        """Queue audio; gate our own capture for its duration plus the half-duplex tail."""
        info = self.backend.play(samples)
        tail = int(self.settings.half_duplex_tail_s * self.backend.sample_rate)
        self.pipeline.add_mute(info.mute_start, info.mute_end + tail)
        self._tx_end = max(self._tx_end, info.end_time)
        return info.end_time

    def tx_busy(self) -> bool:
        """Our audio is still playing."""
        return self.backend.now() < self._tx_end or self.backend.playing()

    def rx_busy(self) -> bool:
        """A frame is currently being received."""
        return self.pipeline.busy()

    def now(self) -> float:
        """Clock time."""
        return self.backend.now()

    # Loop --------------------------------------------------------------------
    def start(self) -> None:
        """Start the backend once."""
        if not self._started:
            self.backend.start()
            self._started = True

    def service(self) -> None:
        """Pull captured audio, decode frames, drive the session."""
        samples, start = self.backend.read_input()
        if len(samples):
            self.pipeline.feed(samples, start)
        frames = self.pipeline.process()
        now = self.backend.now()
        sess = self.session
        for f in frames:
            self.frames_seen += 1
            if sess is not None:
                sess.on_frame(f, now)
        if sess is not None:
            sess.tick(now)
        if self.spectrum_profile is not None and now - self._last_spectrum >= SPECTRUM_INTERVAL_S:
            self._last_spectrum = now
            self._emit_spectrum()

    def _emit_spectrum(self) -> None:
        prof = self.spectrum_profile
        assert prof is not None
        x = self.pipeline.recent(SPECTRUM_N)
        if len(x) < SPECTRUM_N:
            return
        win = np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x * win)) ** 2 / (np.sum(win**2) * len(x)) * 2
        freqs = np.fft.rfftfreq(len(x), 1.0 / prof.sample_rate)
        lo, hi = prof.band_low_hz - 1500, prof.band_high_hz + 1500
        sel = (freqs >= lo) & (freqs <= hi)
        pdb = 10 * np.log10(spec[sel] + 1e-14)
        inband = (freqs >= prof.band_low_hz) & (freqs <= prof.band_high_hz)
        outband = sel & ~inband
        band_snr = float(10 * np.log10((spec[inband].mean() + 1e-14) / (spec[outband].mean() + 1e-14)))
        level = float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))
        self.events(SpectrumEvent(tuple(freqs[sel].tolist()), tuple(pdb.tolist()), level, band_snr))

    def close(self) -> None:
        """Close the backend."""
        self.backend.close()


class RealtimeRunner:
    """Runs a station's service loop on a background thread until the session is done or stopped."""

    def __init__(self, station: Station, poll_s: float = 0.02, stop_when_done: bool = True) -> None:
        self.station = station
        self.poll_s = poll_s
        self.stop_when_done = stop_when_done
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        """Start the background thread."""
        self._thread = threading.Thread(target=self.run, name="airdrop-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request the loop to end."""
        self._stop.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the thread."""
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        """Thread alive."""
        return self._thread is not None and self._thread.is_alive()

    def run(self) -> None:
        """Blocking loop (also usable directly from the CLI)."""
        st = self.station
        try:
            st.start()
            while not self._stop.is_set():
                t = time.monotonic()
                st.service()
                sess = st.session
                if self.stop_when_done and sess is not None and sess.done and not st.tx_busy():
                    break
                self._stop.wait(max(0.0, self.poll_s - (time.monotonic() - t)))
        except AirDropError as exc:
            self.error = exc
            log.error(kv("engine stopped", error=exc.user_message, hint=exc.hint))
            st.events(ErrorEvent(exc.user_message, exc.hint, fatal=True))
        except Exception as exc:  # a crashed worker must still release the device
            self.error = exc
            log.exception("engine crashed")
            st.events(ErrorEvent(f"Internal error: {exc}", "See the log page for details.", fatal=True))
        finally:
            try:
                deadline = time.monotonic() + 5.0
                while st.tx_busy() and time.monotonic() < deadline and self.error is None:
                    time.sleep(0.05)
            finally:
                st.close()
