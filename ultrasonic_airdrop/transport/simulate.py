"""In-process two-endpoint transfers through the channel simulator (CLI ``simulate``, tests, tools)."""

from __future__ import annotations

import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ultrasonic_airdrop.audio.sim_backend import SimLink
from ultrasonic_airdrop.channel.simulator import ChannelConfig
from ultrasonic_airdrop.config import Settings, TransferOptions, get_profile
from ultrasonic_airdrop.files.chunker import cleanup_prepared, prepare_files
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.engine import Station, build_pipeline
from ultrasonic_airdrop.transport.events import DoneEvent, Event, EventSink
from ultrasonic_airdrop.transport.session_receiver import ReceiverSession
from ultrasonic_airdrop.transport.session_sender import SenderSession


@dataclass
class SimResult:
    """Outcome of a simulated transfer."""

    sender_success: bool
    receiver_success: bool
    sender_message: str
    receiver_message: str
    sim_seconds: float
    sender_stats: dict[str, float]
    receiver_stats: dict[str, float]
    sender_events: list[Event] = field(default_factory=list)
    receiver_events: list[Event] = field(default_factory=list)
    sender: SenderSession | None = None
    receiver: ReceiverSession | None = None
    link: SimLink | None = None


def run_transfer(
    paths: list[Path],
    dest: Path,
    settings: Settings | None = None,
    options: TransferOptions | None = None,
    a_to_b: ChannelConfig | None = None,
    b_to_a: ChannelConfig | None = None,
    seed: int = 1,
    max_seconds: float = 3600.0,
    receiver_passphrase: str = "",
    accept_all: bool = True,
    hooks: Callable[[SimLink, SenderSession, ReceiverSession], None] | None = None,
    stop_when: Callable[[SenderSession, ReceiverSession], bool] | None = None,
    sender_session_id: int | None = None,
) -> SimResult:
    """Run a complete sender→receiver transfer in simulation."""
    settings = settings or Settings()
    options = options or TransferOptions()
    profile = get_profile(options.profile_key)
    link = SimLink(profile.sample_rate, a_to_b, b_to_a, seed=seed)
    s_events: list[Event] = []
    r_events: list[Event] = []
    s_sink: EventSink = s_events.append
    r_sink: EventSink = r_events.append
    masks_a: dict[int, CarrierMask] = {}
    masks_b: dict[int, CarrierMask] = {}
    st_a = Station(link.a, build_pipeline(profile.sample_rate, settings, masks_a, [profile]), settings, s_sink)
    st_b = Station(link.b, build_pipeline(profile.sample_rate, settings, masks_b), settings, r_sink)
    with tempfile.TemporaryDirectory(prefix="airdrop-sim-") as tmp:
        manifest, prepared = prepare_files(paths, options.compression, Path(tmp))
        try:
            rng = random.Random(seed)
            sender = SenderSession(
                st_a,
                profile,
                settings,
                options,
                manifest,
                prepared,
                s_sink,
                session_id=sender_session_id,
                rng=rng,
            )
            receiver = ReceiverSession(
                st_b,
                settings,
                dest,
                r_sink,
                masks=masks_b,
                passphrase=receiver_passphrase,
                accept_all=accept_all,
                on_profile_lock=st_b.pipeline.restrict,
            )
            st_a.session = sender
            st_b.session = receiver
            if hooks is not None:
                hooks(link, sender, receiver)

            def finished() -> bool:
                if stop_when is not None and stop_when(sender, receiver):
                    return True
                return sender.done and (receiver.done or receiver.state.value in ("failed", "cancelled", "interrupted"))

            elapsed = link.run([st_a, st_b], finished, max_seconds)
            # Let the receiver observe the end of the session (e.g. broadcast tail).
            if not receiver.done and sender.done:
                link.run([st_a, st_b], lambda: receiver.done or receiver.state.value != "receiving", 5.0)
        finally:
            cleanup_prepared(prepared)
    s_done = next((e for e in reversed(s_events) if isinstance(e, DoneEvent)), None)
    r_done = next((e for e in reversed(r_events) if isinstance(e, DoneEvent)), None)
    return SimResult(
        sender_success=sender.success,
        receiver_success=receiver.success,
        sender_message=s_done.message if s_done else sender.message,
        receiver_message=r_done.message if r_done else receiver.message,
        sim_seconds=elapsed,
        sender_stats=sender.stats(),
        receiver_stats=r_done.stats if r_done else {},
        sender_events=s_events,
        receiver_events=r_events,
        sender=sender,
        receiver=receiver,
        link=link,
    )
