"""Simulated audio: two endpoints joined by a channel simulator, driven by a sample clock.

Each endpoint implements :class:`~ultrasonic_airdrop.audio.backend.AudioBackend`.
When an endpoint plays a burst, the burst is passed through the channel for
that direction and scheduled into the peer's input timeline; a loud copy is
also scheduled into the endpoint's *own* input (speaker-to-own-mic coupling),
which the station must gate out (half-duplex).  :meth:`SimLink.run` advances
the clock in fixed chunks and services every station, so whole transfers run
deterministically and much faster than real time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ultrasonic_airdrop.audio.backend import PlayInfo
from ultrasonic_airdrop.channel.simulator import Channel, ChannelConfig
from ultrasonic_airdrop.phy.numerology import FloatArray
from ultrasonic_airdrop.transport.clock import SampleClock

if TYPE_CHECKING:
    from ultrasonic_airdrop.transport.engine import Station

log = logging.getLogger(__name__)
SELF_ECHO_GAIN = 0.8


class SimEndpoint:
    """One simulated machine (speaker + microphone)."""

    def __init__(self, link: SimLink, name: str, incoming: Channel) -> None:
        self.link = link
        self.name = name
        self.sample_rate = link.fs
        self.incoming = incoming
        self.peer: SimEndpoint | None = None
        self._scheduled: list[tuple[int, FloatArray]] = []
        self._read_pos = 0
        self._out_busy_until = 0
        self.started = False
        self.closed = False
        self.muted_output = False

    def start(self) -> None:
        """Start streams (no-op apart from bookkeeping)."""
        self.started = True
        self._read_pos = self.link.clock.samples

    def now(self) -> float:
        """Simulated clock time."""
        return self.link.clock.now()

    def schedule(self, start: int, samples: FloatArray) -> None:
        """Add ``samples`` to this endpoint's input starting at absolute index ``start``."""
        if len(samples):
            self._scheduled.append((start, samples))

    def read_input(self) -> tuple[FloatArray, int]:
        """Captured audio since the last read: noise plus every overlapping burst."""
        end = self.link.clock.samples
        start = self._read_pos
        n = end - start
        if n <= 0:
            return np.zeros(0), start
        out = self.incoming.noise(n)
        keep = []
        for s, x in self._scheduled:
            e = s + len(x)
            lo, hi = max(s, start), min(e, end)
            if hi > lo:
                out[lo - start : hi - start] += x[lo - s : hi - s]
            if e > end:
                keep.append((s, x))
        self._scheduled = keep
        self._read_pos = end
        return out, start

    def play(self, samples: FloatArray) -> PlayInfo:
        """Queue a burst; it starts when previously queued audio has finished."""
        start = max(self.link.clock.samples, self._out_busy_until)
        end = start + len(samples)
        self._out_busy_until = end
        if not self.muted_output:
            self.link.deliver(self, samples, start)
        self.schedule(start, samples * SELF_ECHO_GAIN)
        fs = self.sample_rate
        return PlayInfo(start / fs, end / fs, start, end)

    def playing(self) -> bool:
        """True while queued audio is still playing."""
        return self.link.clock.samples < self._out_busy_until

    def close(self) -> None:
        """Close (idempotent)."""
        self.closed = True


class SimLink:
    """Two endpoints ``a`` and ``b`` and the channels between them."""

    def __init__(
        self,
        fs: int = 48_000,
        a_to_b: ChannelConfig | None = None,
        b_to_a: ChannelConfig | None = None,
        seed: int = 0,
    ) -> None:
        self.fs = fs
        self.clock = SampleClock(fs)
        rng = np.random.default_rng(seed)
        cab = a_to_b or ChannelConfig()
        cba = b_to_a or ChannelConfig()
        self.ch_ab = Channel(cab, fs, np.random.default_rng(rng.integers(1 << 32)))
        self.ch_ba = Channel(cba, fs, np.random.default_rng(rng.integers(1 << 32)))
        self.a = SimEndpoint(self, "A", incoming=self.ch_ba)
        self.b = SimEndpoint(self, "B", incoming=self.ch_ab)
        self.a.peer, self.b.peer = self.b, self.a
        self.ack_drop: Callable[[int], bool] | None = None

    def deliver(self, src: SimEndpoint, samples: FloatArray, start: int) -> None:
        """Pass a burst through the channel towards the peer."""
        ch = self.ch_ab if src is self.a else self.ch_ba
        y, delay = ch.process_burst(samples)
        assert src.peer is not None
        src.peer.schedule(start + delay, y)

    def run(
        self,
        stations: list[Station],
        until: Callable[[], bool],
        max_seconds: float,
        chunk_s: float = 0.1,
    ) -> float:
        """Advance time until ``until()`` is true or ``max_seconds`` elapse; returns elapsed seconds."""
        chunk = int(chunk_s * self.fs)
        t0 = self.clock.now()
        for st in stations:
            st.start()
        while self.clock.now() - t0 < max_seconds:
            self.clock.advance(chunk)
            for st in stations:
                st.service()
            if until():
                break
        return self.clock.now() - t0
