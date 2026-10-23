"""Frame-error-rate vs SNR curves for every link mode, from the channel simulator.

For each mode and in-band SNR point, ``--frames`` typical DATA-sized frames are
sent through an AWGN channel with a sample-clock offset (default 150 ppm) and
decoded by the real receive pipeline.  For each point we record:

* FER (frame lost or failed RS/CRC),
* the receiver's own SNR estimate for that mode family (MFSK per-tone SNR for
  modes 0/1; mean per-carrier SNR from the differential EVM for OFDM modes),
* the per-carrier SNR measured by a PROBE frame at the same channel SNR (the
  quantity the receiver uses for mode selection).

``REQUIRED_SNR_DB`` in ``config.py`` is read from this output: the lowest
estimator SNR at which FER <= the target (default 2 %, i.e. at most 1 lost
frame in 50) at that point and every stronger point.

Usage::

    python tools/snr_curves.py --frames 50 --out docs/SNR_CURVES.md --csv docs/snr_curves.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultrasonic_airdrop.channel.simulator import Channel, ChannelConfig
from ultrasonic_airdrop.config import MODES, PROBE_MODE_ID, get_profile
from ultrasonic_airdrop.link.codec import FrameCodec
from ultrasonic_airdrop.link.frames import Data, FrameType, LinkFrame
from ultrasonic_airdrop.link.receiver_pipeline import ReceiverPipeline
from ultrasonic_airdrop.transport.estimate import max_units_per_frame


@dataclass
class Point:
    """One (mode, SNR) measurement."""

    mode: int
    snr_db: float
    frames: int
    errors: int
    est_snr_db: float
    probe_snr_db: float

    @property
    def fer(self) -> float:
        """Frame error rate."""
        return self.errors / self.frames if self.frames else 1.0


def run_point(args: tuple[str, int, float, int, float, int, float]) -> Point:
    """Measure one point (separate process)."""
    profile_key, mode, snr, n_frames, ppm, seed, rt60 = args
    prof = get_profile(profile_key)
    rng = np.random.default_rng(seed)
    codec = FrameCodec(prof, -6.0)
    fs = prof.sample_rate
    cfg = ChannelConfig(snr_db=snr, ppm=ppm, rt60_s=rt60, drr_db=6.0, seed=seed)
    ch = Channel(cfg, fs, rng)
    pipe = ReceiverPipeline(prof)
    errors = 0
    ests: list[float] = []
    bodies: list[bytes] = []
    stream: list[np.ndarray] = []
    gap = ch.noise(int(0.6 * fs))
    stream.append(gap)
    units = max_units_per_frame(prof, mode, None, False) if mode != PROBE_MODE_ID else 0
    for i in range(n_frames):
        if mode == PROBE_MODE_ID:
            body = LinkFrame(FrameType.PROBE, 1, i).pack()
        else:
            payload = Data(0, i, units, rng.integers(0, 256, units * 32, dtype=np.uint8).tobytes()).pack()
            body = LinkFrame(FrameType.DATA, 1, i, payload).pack()
        bodies.append(body)
        y, d = ch.process_burst(codec.encode(body, mode))
        seg = ch.noise(len(y) + d + int(0.4 * fs))
        seg[d : d + len(y)] += y
        stream.append(seg)
    x = np.concatenate([*stream, ch.noise(fs)])
    got: dict[bytes, float] = {}
    probe_vals: list[float] = []
    for i in range(0, len(x), 4800):
        pipe.feed(x[i : i + 4800])
        for f in pipe.process():
            if f.body is not None:
                info = f.info
                if mode == PROBE_MODE_ID:
                    if info.probe_snr_db is not None:
                        probe_vals.append(float(10 * np.log10(np.mean(10 ** (info.probe_snr_db / 10)))))
                    got[f.body] = info.mfsk_snr_db
                else:
                    got[f.body] = info.snr_db
    for b in bodies:
        if b in got:
            ests.append(got[b])
        else:
            errors += 1
    est = float(np.mean(ests)) if ests else float("nan")
    probe = float(np.mean(probe_vals)) if probe_vals else float("nan")
    return Point(mode, snr, n_frames, errors, est, probe)


def threshold(points: list[Point], target: float, key: str = "est_snr_db") -> float:
    """Lowest estimator SNR from which FER <= target holds for all stronger points."""
    pts = sorted(points, key=lambda p: p.snr_db)
    best = math.nan
    for i in range(len(pts)):
        if all(p.fer <= target for p in pts[i:]):
            best = getattr(pts[i], key)
            break
    return best


def main() -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="nu")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--ppm", type=float, default=150.0)
    ap.add_argument("--rt60", type=float, default=0.0)
    ap.add_argument("--target", type=float, default=0.02)
    ap.add_argument("--snr-min", type=float, default=-16.0)
    ap.add_argument("--snr-max", type=float, default=24.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--modes", default="0,1,2,3,4,5")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    modes = [int(m) for m in args.modes.split(",")]
    snrs = list(np.arange(args.snr_min, args.snr_max + 1e-9, args.step))
    jobs = []
    seed = 1000
    for m in [*modes, PROBE_MODE_ID]:
        for s in snrs:
            n = max(10, args.frames // 3) if m == PROBE_MODE_ID else args.frames
            jobs.append((args.profile, m, float(s), n, args.ppm, seed, args.rt60))
            seed += 1
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        points = list(ex.map(run_point, jobs))
    probe_by_snr = {p.snr_db: p.probe_snr_db for p in points if p.mode == PROBE_MODE_ID}
    mfsk_by_snr = {p.snr_db: p.est_snr_db for p in points if p.mode == PROBE_MODE_ID}
    lines = [
        "# FER vs SNR (simulation)",
        "",
        f"Generated by `tools/snr_curves.py` — profile `{args.profile}`, {args.frames} DATA frames per point, "
        f"AWGN + {args.ppm:g} ppm clock offset, RT60 {args.rt60:g} s, FER target {args.target:.0%}. "
        f"Runtime {time.time() - t0:.0f} s.",
        "",
        "*In-band SNR*: received signal power (two-tone MFSK reference at the transmit level) over noise power in a "
        "3 kHz band. *Estimator SNR*: the receiver's own per-bin estimate for that frame "
        "(MFSK per-tone SNR, or mean OFDM per-carrier SNR from differential EVM). *Probe carrier SNR*: mean "
        "per-carrier SNR measured by a PROBE frame at the same channel SNR (used for mode selection). "
        "**These are simulation results, not hardware measurements.**",
        "",
    ]
    rows = []
    thresholds: dict[int, tuple[float, float, float]] = {}
    for m in modes:
        pts = sorted((p for p in points if p.mode == m), key=lambda p: p.snr_db)
        lines += [
            f"## Mode {m} — {MODES[m].label}",
            "",
            "| in-band SNR (dB) | FER | estimator SNR (dB) | probe carrier SNR (dB) | probe MFSK SNR (dB) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for p in pts:
            lines.append(
                f"| {p.snr_db:.0f} | {p.fer:.3f} ({p.errors}/{p.frames}) | {p.est_snr_db:.1f} | "
                f"{probe_by_snr.get(p.snr_db, float('nan')):.1f} | {mfsk_by_snr.get(p.snr_db, float('nan')):.1f} |"
            )
            rows.append(
                [
                    m,
                    p.snr_db,
                    p.frames,
                    p.errors,
                    p.fer,
                    p.est_snr_db,
                    probe_by_snr.get(p.snr_db),
                    mfsk_by_snr.get(p.snr_db),
                ]
            )
        in_band = threshold(pts, args.target, "snr_db")
        est = threshold(pts, args.target)
        if MODES[m].is_ofdm:
            sel = probe_by_snr.get(in_band, float("nan"))
        else:
            sel = mfsk_by_snr.get(in_band, float("nan"))
        thresholds[m] = (in_band, est, sel)
        lines += [
            "",
            f"Threshold (FER ≤ {args.target:.0%}): in-band {in_band:.0f} dB, estimator {est:.1f} dB, "
            f"selection metric {sel:.1f} dB.",
            "",
        ]
    lines += [
        "## Summary",
        "",
        "| mode | in-band SNR (dB) | estimator SNR (dB) | selection-metric SNR (dB) |",
        "|---:|---:|---:|---:|",
    ]
    for m, (a, b, c) in thresholds.items():
        lines.append(f"| {m} | {a:.0f} | {b:.1f} | {c:.1f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "mode",
                    "inband_snr_db",
                    "frames",
                    "errors",
                    "fer",
                    "estimator_snr_db",
                    "probe_carrier_snr_db",
                    "probe_mfsk_snr_db",
                ]
            )
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
