"""Real audio I/O through sounddevice/PortAudio (WASAPI preferred on Windows).

* One persistent mono float32 InputStream at the profile rate writes into a 30 s
  :class:`~ultrasonic_airdrop.audio.backend.RingBuffer` (multi-channel devices
  are down-mixed).
* A separate OutputStream plays queued bursts; when the queue is empty it plays
  silence.  The output stream is opened only when the station starts (i.e.
  after an explicit user action) — the app never plays audio at launch.
* Optional WASAPI exclusive mode (bypasses the shared-mode effects chain on many
  drivers) with automatic fallback to shared mode and a logged warning.
* Streams are always closed in ``close()`` (idempotent; called from ``finally``
  blocks, ``aboutToQuit`` and ``atexit``).
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
import weakref
from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from ultrasonic_airdrop.audio.backend import PlayInfo, RingBuffer
from ultrasonic_airdrop.audio.devices import check_rate, describe_error, list_devices, load_sounddevice
from ultrasonic_airdrop.errors import AudioDeviceError
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy.numerology import FloatArray

log = logging.getLogger(__name__)
_OPEN_BACKENDS: weakref.WeakSet[SoundDeviceBackend] = weakref.WeakSet()


def _close_all() -> None:
    for b in list(_OPEN_BACKENDS):
        b.close()


atexit.register(_close_all)


class SoundDeviceBackend:
    """Duplex audio via two PortAudio streams."""

    def __init__(
        self,
        sample_rate: int,
        input_device: int | None = None,
        output_device: int | None = None,
        exclusive: bool = False,
        blocksize: int = 1024,
        latency: str | float = "high",
        enable_output: bool = True,
        enable_input: bool = True,
        ring_seconds: float = 30.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.input_device = input_device
        self.output_device = output_device
        self.exclusive = exclusive
        self.blocksize = blocksize
        self.latency = latency
        self.enable_output = enable_output
        self.enable_input = enable_input
        self.ring = RingBuffer(int(ring_seconds * sample_rate))
        self._read_idx = 0
        self._in: Any = None
        self._out: Any = None
        self._in_channels = 1
        self._out_channels = 1
        self._queue: deque[npt.NDArray[np.float32]] = deque()
        self._cur: npt.NDArray[np.float32] | None = None
        self._cur_pos = 0
        self._queued = 0
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self.status_flags = 0
        self.exclusive_active = False
        self.actual_rate: float | None = None

    # ------------------------------------------------------------------ helpers
    def _extra(self, sd: Any, device: int | None) -> Any:
        if not self.exclusive:
            return None
        try:
            devs = list_devices()
            idx = device if device is not None else -1
            if idx >= 0 and devs[idx].hostapi != "Windows WASAPI":
                log.warning(kv("exclusive mode needs a WASAPI device; using shared mode", device=devs[idx].label))
                return None
            return sd.WasapiSettings(exclusive=True)
        except Exception as exc:
            log.warning(kv("WASAPI exclusive settings unavailable; using shared mode", error=exc))
            return None

    def _channels_for(self, device: int | None, kind: str) -> int:
        try:
            devs = list_devices()
            sd = load_sounddevice()
            idx = device if device is not None else sd.default.device[0 if kind == "input" else 1]
            if idx is None or idx < 0:
                return 1
            maxc = devs[idx].max_input_channels if kind == "input" else devs[idx].max_output_channels
            return 1 if maxc >= 1 else 0
        except Exception:
            return 1

    def _in_cb(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.status_flags += 1
        x = indata[:, 0] if indata.shape[1] == 1 else indata.mean(axis=1)
        self.ring.write(x)

    def _out_cb(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            self.status_flags += 1
        out = np.zeros(frames, dtype=np.float32)
        pos = 0
        with self._lock:
            while pos < frames:
                if self._cur is None:
                    if not self._queue:
                        break
                    self._cur = self._queue.popleft()
                    self._cur_pos = 0
                n = min(frames - pos, len(self._cur) - self._cur_pos)
                out[pos : pos + n] = self._cur[self._cur_pos : self._cur_pos + n]
                pos += n
                self._cur_pos += n
                self._queued -= n
                if self._cur_pos >= len(self._cur):
                    self._cur = None
        outdata[:] = np.repeat(out[:, None], outdata.shape[1], axis=1)

    def _open(self, kind: str, device: int | None) -> Any:
        sd = load_sounddevice()
        extra = self._extra(sd, device)
        err = check_rate(device, self.sample_rate, kind, extra)
        if err is not None and extra is not None:
            log.warning(kv("exclusive mode rejected; falling back to shared mode", kind=kind, error=err))
            extra = None
            err = check_rate(device, self.sample_rate, kind, None)
        if err is not None:
            raise describe_error(RuntimeError(err), kind)
        cls = sd.InputStream if kind == "input" else sd.OutputStream
        cb = self._in_cb if kind == "input" else self._out_cb
        try:
            stream = cls(
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                device=device,
                channels=1,
                dtype="float32",
                latency=self.latency,
                callback=cb,
                extra_settings=extra,
            )
            stream.start()
        except Exception as exc:
            if extra is not None:
                log.warning(kv("exclusive open failed; retrying shared mode", kind=kind, error=exc))
                self.exclusive = False
                return self._open(kind, device)
            raise describe_error(exc, kind) from exc
        if extra is not None:
            self.exclusive_active = True
        self.actual_rate = float(stream.samplerate)
        if abs(self.actual_rate - self.sample_rate) > 0.5:
            stream.close()
            raise AudioDeviceError(
                f"The {kind} device opened at {self.actual_rate:.0f} Hz instead of {self.sample_rate} Hz.",
                "Set the device's default format to the required rate in the Sound Control Panel.",
            )
        log.info(
            kv(
                "audio stream open",
                kind=kind,
                device=device,
                rate=self.actual_rate,
                latency=stream.latency,
                exclusive=extra is not None,
            )
        )
        return stream

    # ------------------------------------------------------------------ AudioBackend
    def start(self) -> None:
        """Open the streams (input first so capture indices start at 0)."""
        _OPEN_BACKENDS.add(self)
        try:
            if self.enable_input:
                self._in = self._open("input", self.input_device)
            if self.enable_output:
                self._out = self._open("output", self.output_device)
        except Exception:
            self.close()
            raise
        self._t0 = time.monotonic()

    def read_input(self) -> tuple[FloatArray, int]:
        """New captured samples since the previous call."""
        data, start = self.ring.read_since(self._read_idx)
        self._read_idx = start + len(data)
        return data, start

    def _latency(self, stream: Any) -> float:
        try:
            lat = stream.latency
            return float(lat if not isinstance(lat, tuple) else lat[0])
        except Exception:
            return 0.1

    def play(self, samples: FloatArray) -> PlayInfo:
        """Queue a burst; returns its estimated timing and the input indices to gate."""
        if self._out is None:
            raise AudioDeviceError("Output stream is not open.", "Start the engine before transmitting.")
        x = np.asarray(samples, dtype=np.float32)
        with self._lock:
            ahead = self._queued
            self._queue.append(x)
            self._queued += len(x)
        fs = self.sample_rate
        out_lat = self._latency(self._out)
        in_lat = self._latency(self._in) if self._in is not None else 0.0
        now = time.monotonic()
        start_t = now + ahead / fs + out_lat
        end_t = start_t + len(x) / fs
        in_now = self.ring.total
        margin = int(0.05 * fs)
        mute_start = in_now + int(ahead) - margin
        mute_end = in_now + int(ahead + len(x) + (out_lat + in_lat) * fs) + margin
        return PlayInfo(start_t, end_t, mute_start, mute_end)

    def playing(self) -> bool:
        """Queued audio remains."""
        with self._lock:
            return self._queued > 0 or self._cur is not None

    def now(self) -> float:
        """Monotonic clock."""
        return time.monotonic()

    def stop_output(self) -> None:
        """Drop any queued audio immediately (e.g. stop a test tone)."""
        with self._lock:
            self._queue.clear()
            self._cur = None
            self._queued = 0

    def close(self) -> None:
        """Stop and close both streams (idempotent)."""
        for name in ("_out", "_in"):
            stream = getattr(self, name)
            setattr(self, name, None)
            if stream is None:
                continue
            try:
                stream.stop()
            except Exception as exc:
                log.debug(kv("stream stop failed", error=exc))
            try:
                stream.close()
            except Exception as exc:
                log.debug(kv("stream close failed", error=exc))
        _OPEN_BACKENDS.discard(self)
