"""Stream frames through the channel simulator and the real receive pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ultrasonic_airdrop.channel.simulator import Channel, ChannelConfig
from ultrasonic_airdrop.config import PROBE_MODE_ID, Profile
from ultrasonic_airdrop.errors import FrameError
from ultrasonic_airdrop.link.codec import FrameCodec
from ultrasonic_airdrop.link.frames import Data, FrameType, LinkFrame
from ultrasonic_airdrop.link.receiver_pipeline import ReceiverPipeline, RxFrame
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.estimate import max_units_per_frame


@dataclass
class Outcome:
    sent: int
    correct: int
    wrong: int  # decoded, CRC-valid, but different from what was sent (must never happen)
    frames: list[RxFrame]


def run_frames(
    profile: Profile,
    mode: int,
    n: int,
    cfg: ChannelConfig,
    seed: int = 0,
    units: int | None = None,
    mask: CarrierMask | None = None,
    burst_at: float | None = None,
    burst_s: float = 0.3,
) -> Outcome:
    rng = np.random.default_rng(seed)
    fs = profile.sample_rate
    codec = FrameCodec(profile, cfg.tx_level_dbfs)
    ch = Channel(cfg, fs, rng)
    pipe = ReceiverPipeline(profile, mask_lookup={mask.mask_id: mask}.get if mask is not None else None)
    n_en = mask.count if mask is not None else None
    u = units if units is not None else max_units_per_frame(profile, mode, n_en, False)
    bodies = []
    parts = [ch.noise(int(rng.uniform(0.4, 0.9) * fs))]
    for i in range(n):
        if mode == PROBE_MODE_ID:
            body = LinkFrame(FrameType.PROBE, 9, i).pack()
        else:
            payload = Data(0, i, u, rng.integers(0, 256, u * 32, dtype=np.uint8).tobytes()).pack()
            body = LinkFrame(FrameType.DATA, 9, i, payload).pack()
        bodies.append(body)
        y, d = ch.process_burst(codec.encode(body, mode, mask))
        seg = ch.noise(d + len(y) + int(rng.uniform(0.3, 0.6) * fs))
        seg[d : d + len(y)] += y
        if burst_at is not None:
            b0 = d + int(burst_at * len(y))
            bl = int(burst_s * fs)
            seg[b0 : b0 + bl] += rng.standard_normal(min(bl, len(seg) - b0)) * ch.burst_sigma
        parts.append(seg)
    parts.append(ch.noise(fs))
    x = np.concatenate(parts)
    out: list[RxFrame] = []
    step = fs // 10
    for i in range(0, len(x), step):
        pipe.feed(x[i : i + step])
        out.extend(pipe.process())
    sent = set(bodies)
    correct = wrong = 0
    for f in out:
        if f.body is None:
            continue
        try:
            LinkFrame.unpack(f.body)
        except FrameError:
            continue  # rejected by CRC: a frame error, never accepted
        if f.body in sent:
            correct += 1
        else:
            wrong += 1
    return Outcome(n, correct, wrong, out)
