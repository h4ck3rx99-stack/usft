"""Offline operation: render a broadcast session to WAV, and decode a recording/WAV file."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from ultrasonic_airdrop.config import Settings, TransferOptions, get_profile, profiles_for_rate
from ultrasonic_airdrop.errors import AirDropError
from ultrasonic_airdrop.files.chunker import cleanup_prepared, prepare_files
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy.numerology import FloatArray
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.engine import build_pipeline
from ultrasonic_airdrop.transport.events import EventSink, null_sink
from ultrasonic_airdrop.transport.session_receiver import ReceiverSession, ReceiverState
from ultrasonic_airdrop.transport.session_sender import SenderSession

log = logging.getLogger(__name__)
BURST_GAP_S = 0.2


class RenderRadio:
    """A radio that appends every burst to an in-memory timeline (no real time passes)."""

    def __init__(self, fs: int) -> None:
        self.fs = fs
        self.parts: list[FloatArray] = [np.zeros(int(0.5 * fs))]
        self.samples = len(self.parts[0])

    def transmit(self, samples: FloatArray) -> float:
        """Append a burst plus a short gap."""
        gap = np.zeros(int(BURST_GAP_S * self.fs))
        self.parts.extend([samples, gap])
        self.samples += len(samples) + len(gap)
        return self.samples / self.fs

    def tx_busy(self) -> bool:
        """Never busy."""
        return False

    def rx_busy(self) -> bool:
        """Never receiving."""
        return False

    def now(self) -> float:
        """Timeline position."""
        return self.samples / self.fs

    def audio(self) -> FloatArray:
        """The rendered signal."""
        return np.concatenate([*self.parts, np.zeros(self.fs)])


def render_broadcast(
    paths: list[Path],
    options: TransferOptions,
    settings: Settings | None = None,
    events: EventSink = null_sink,
) -> tuple[FloatArray, int]:
    """Render a full broadcast session (OFFER×3, MANIFEST×2, data passes, END) to samples."""
    settings = settings or Settings()
    options.broadcast = True
    profile = get_profile(options.profile_key)
    radio = RenderRadio(profile.sample_rate)
    with tempfile.TemporaryDirectory(prefix="airdrop-wav-") as tmp:
        manifest, prepared = prepare_files(paths, options.compression, Path(tmp))
        try:
            sess = SenderSession(radio, profile, settings, options, manifest, prepared, events)
            while not sess.done:
                sess.tick(radio.now())
        finally:
            cleanup_prepared(prepared)
    return radio.audio(), profile.sample_rate


def write_wav(path: Path, samples: FloatArray, fs: int) -> None:
    """Write 32-bit float WAV."""
    wavfile.write(str(path), fs, np.asarray(samples, dtype=np.float32))


def read_wav(path: Path) -> tuple[FloatArray, int]:
    """Read a WAV as mono float64 in [-1, 1], resampled to 48 kHz if its rate has no profile."""
    fs, data = wavfile.read(str(path))
    x = np.asarray(data)
    if x.dtype.kind == "i":
        x = x.astype(np.float64) / float(np.iinfo(data.dtype).max)
    elif x.dtype.kind == "u":
        x = (x.astype(np.float64) - 128.0) / 128.0
    else:
        x = x.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if not profiles_for_rate(fs):
        from math import gcd

        g = gcd(fs, 48_000)
        x = resample_poly(x, 48_000 // g, fs // g)
        log.warning(kv("WAV resampled", from_rate=fs, to_rate=48_000))
        fs = 48_000
    return x, fs


class FileRadio:
    """Clock driven by the position in a recording; never transmits."""

    def __init__(self, fs: int) -> None:
        self.fs = fs
        self.samples = 0

    def transmit(self, samples: FloatArray) -> float:
        """Ignore transmissions (decoding a recording is receive-only)."""
        return self.now()

    def tx_busy(self) -> bool:
        """Never busy."""
        return False

    def rx_busy(self) -> bool:
        """Never busy."""
        return False

    def now(self) -> float:
        """Position in seconds."""
        return self.samples / self.fs


@dataclass
class DecodeResult:
    """Outcome of :func:`decode_recording`."""

    success: bool
    message: str
    files: list[tuple[str, str, str]]
    frames: int


def decode_recording(
    x: FloatArray,
    fs: int,
    dest: Path,
    settings: Settings | None = None,
    passphrase: str = "",
    events: EventSink = null_sink,
) -> DecodeResult:
    """Decode a broadcast transfer from recorded samples into ``dest``."""
    settings = (settings or Settings()).replace(rx_inactivity_s=1e9, linger_s=0.0)
    masks: dict[int, CarrierMask] = {}
    pipe = build_pipeline(fs, settings, masks)
    radio = FileRadio(fs)
    sess = ReceiverSession(
        radio,
        settings,
        dest,
        events,
        masks=masks,
        passphrase=passphrase,
        accept_all=True,
        on_profile_lock=pipe.restrict,
    )
    chunk = fs // 10
    frames = 0
    padded = np.concatenate([x, np.zeros(2 * fs)])
    for i in range(0, len(padded), chunk):
        block = padded[i : i + chunk]
        pipe.feed(block)
        radio.samples += len(block)
        for f in pipe.process():
            frames += 1
            sess.on_frame(f, radio.now())
        sess.tick(radio.now())
    s = sess.sess
    if s is None:
        raise AirDropError("No transfer was found in the recording.", "Check the profile and recording level.")
    sess.finish_recording()
    sess.stop()
    return DecodeResult(sess.state == ReceiverState.COMPLETE, sess.message, list(sess.results), frames)
