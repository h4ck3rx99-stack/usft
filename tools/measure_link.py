"""Hardware link measurement harness → CSV.

Run ``receive`` on one computer and ``send`` on the other.  The sender transmits
``--frames`` DATA frames per mode (deterministic pseudo-random payloads, mode-0
sequence first, then each requested mode), pausing between frames.  The
receiver records one CSV row per detected frame and a per-mode summary
(frames decoded / sent, FER, mean SNR, clock offset, goodput if that mode were
used).  Copy the summary into docs/HARDWARE_TESTING.md.

    python tools/measure_link.py receive --csv results.csv --label "desk 1 m, quiet"
    python tools/measure_link.py send --modes 0,1,2,3,4,5 --frames 20

``--sim`` runs both roles in-process through the channel simulator (for testing
the harness itself; results are NOT hardware measurements).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib

from ultrasonic_airdrop.config import MODES, PROFILES, Settings, get_profile
from ultrasonic_airdrop.link import frames as fr
from ultrasonic_airdrop.link.codec import frame_airtime
from ultrasonic_airdrop.link.receiver_pipeline import RxFrame
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.engine import Station, build_pipeline
from ultrasonic_airdrop.transport.estimate import max_units_per_frame
from ultrasonic_airdrop.transport.linkio import LinkIO, Radio

SESSION = 0x4D454153  # "MEAS"
GAP_S = 0.6


def payload_for(mode: int, i: int, units: int) -> bytes:
    """Deterministic payload so the receiver can verify content and count losses."""
    rng = np.random.default_rng(mode * 100_000 + i)
    return fr.Data(mode, i, units, rng.integers(0, 256, units * 32, dtype=np.uint8).tobytes()).pack()


class MeasureSender:
    """Plays frames for each mode."""

    def __init__(self, radio: Radio, profile_key: str, settings: Settings, modes: list[int], frames: int) -> None:
        self.radio = radio
        self.profile = get_profile(profile_key)
        self.io = LinkIO(radio, self.profile, settings, fr.DIR_S2R, SESSION)
        self.plan = [(m, i) for m in modes for i in range(frames)]
        self.pos = 0
        self.next_at = 0.0
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def on_frame(self, rx: RxFrame, now: float) -> None:
        """Unused."""

    def tick(self, now: float) -> None:
        """Send the next frame after the previous one plus a gap."""
        if self._done or self.radio.tx_busy() or now < self.next_at:
            return
        if self.pos >= len(self.plan):
            self._done = True
            return
        mode, i = self.plan[self.pos]
        self.pos += 1
        units = max_units_per_frame(self.profile, mode, None, False)
        out = self.io.build(fr.Data.unpack(payload_for(mode, i, units)), mode_id=mode, mask=CarrierMask.all())
        end = self.io.send([out])
        self.next_at = end + GAP_S
        print(f"sent mode {mode} frame {i + 1}", flush=True)


@dataclass
class ModeStats:
    """Per-mode accumulation."""

    ok: set[int] = field(default_factory=set)
    errors: int = 0
    snr: list[float] = field(default_factory=list)
    ppm: list[float] = field(default_factory=list)


class MeasureReceiver:
    """Records frames to CSV."""

    def __init__(self, writer: csv.writer, label: str, profile_key: str, idle_s: float) -> None:
        self.writer = writer
        self.label = label
        self.profile = get_profile(profile_key)
        self.stats: dict[int, ModeStats] = {}
        self.last = time.monotonic()
        self.idle_s = idle_s
        self.started = False
        self._done = False
        self.now = 0.0

    @property
    def done(self) -> bool:
        return self._done

    def on_frame(self, rx: RxFrame, now: float) -> None:
        """Record one detected frame."""
        info = rx.info
        mode = info.header.mode_id if info.header else -1
        st = self.stats.setdefault(mode, ModeStats())
        ok = False
        idx = -1
        if rx.body is not None:
            try:
                frame = fr.LinkFrame.unpack(rx.body)
                data = fr.Data.unpack(frame.payload)
                units = max_units_per_frame(self.profile, mode, None, False)
                ok = frame.session_id == SESSION and frame.payload == payload_for(
                    data.file_index, data.start_unit, units
                )
                idx = data.start_unit
            except fr.FrameError:
                ok = False
        if ok:
            st.ok.add(idx)
            st.snr.append(info.snr_db)
            if info.postamble_found:
                st.ppm.append(info.ppm)
        else:
            st.errors += 1
        self.writer.writerow(
            [
                self.label,
                f"{now:.2f}",
                mode,
                idx,
                int(ok),
                f"{info.snr_db:.2f}",
                f"{info.mfsk_snr_db:.2f}",
                f"{info.ppm:.1f}",
                f"{info.cfo_hz:.2f}",
                f"{info.detect_ratio:.1f}",
                info.rs_corrected,
                info.erasures,
            ]
        )
        self.started = True
        self.now = now

    def tick(self, now: float) -> None:
        """Stop after ``idle_s`` without frames once something was received."""
        if self.started and now - self.now > self.idle_s:
            self._done = True

    def summary(self, frames_per_mode: int) -> str:
        """Per-mode summary table (markdown)."""
        lines = [
            "| mode | decoded / sent | FER | mean SNR (dB) | clock offset (ppm) | payload rate if used (B/s) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for mode in sorted(m for m in self.stats if m in MODES):
            st = self.stats[mode]
            n = len(st.ok)
            fer = 1 - n / frames_per_mode if frames_per_mode else math.nan
            units = max_units_per_frame(self.profile, mode, None, False)
            air = frame_airtime(self.profile, mode, fr.link_frame_len(fr.DATA_HEADER_LEN + units * 32))
            rate = units * 32 / air * (1 - fer)
            snr = float(np.mean(st.snr)) if st.snr else math.nan
            ppm = float(np.median(st.ppm)) if st.ppm else math.nan
            lines.append(f"| {mode} | {n} / {frames_per_mode} | {fer:.2f} | {snr:.1f} | {ppm:+.0f} | {rate:.0f} |")
        return "\n".join(lines)


HEADER = [
    "label",
    "time_s",
    "mode",
    "frame",
    "ok",
    "snr_db",
    "mfsk_snr_db",
    "ppm",
    "cfo_hz",
    "detect_ratio",
    "rs_corrected",
    "erasures",
]


def main() -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=["send", "receive"])
    ap.add_argument("--profile", default="nu", choices=sorted(PROFILES))
    ap.add_argument("--modes", default="0,1,2,3,4,5")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--csv", type=Path, default=Path("link_measurements.csv"))
    ap.add_argument("--label", default="")
    ap.add_argument("--level", type=float, default=-6.0)
    ap.add_argument("--in", dest="input", type=int, default=None)
    ap.add_argument("--out", dest="output", type=int, default=None)
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--idle", type=float, default=20.0, help="receiver stops after this many idle seconds")
    ap.add_argument("--sim", action="store_true", help="run both roles through the simulator (not hardware!)")
    ap.add_argument("--snr", type=float, default=15.0, help="--sim only: in-band SNR")
    args = ap.parse_args()
    modes = [int(m) for m in args.modes.split(",")]
    settings = Settings(tx_level_dbfs=args.level, exclusive_mode=args.exclusive)
    profile = get_profile(args.profile)
    new = not args.csv.exists()
    with args.csv.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(HEADER)
        if args.sim:
            from ultrasonic_airdrop.audio.sim_backend import SimLink
            from ultrasonic_airdrop.channel.simulator import ChannelConfig

            link = SimLink(profile.sample_rate, ChannelConfig(snr_db=args.snr, ppm=120, rt60_s=0.2, seed=1))
            st_a = Station(link.a, build_pipeline(profile.sample_rate, settings, {}, [profile]), settings)
            st_b = Station(link.b, build_pipeline(profile.sample_rate, settings, {}, [profile]), settings)
            snd = MeasureSender(st_a, args.profile, settings, modes, args.frames)
            rcv = MeasureReceiver(writer, args.label or f"SIMULATION snr={args.snr}", args.profile, args.idle)
            st_a.session, st_b.session = snd, rcv
            link.run([st_a, st_b], lambda: snd.done and rcv.done, 3600.0)
            print("\nSIMULATION (not a hardware measurement):\n" + rcv.summary(args.frames))
            return 0
        from ultrasonic_airdrop.audio.sd_backend import SoundDeviceBackend
        from ultrasonic_airdrop.transport.engine import RealtimeRunner

        backend = SoundDeviceBackend(
            profile.sample_rate, args.input, args.output, args.exclusive, enable_output=args.role == "send"
        )
        station = Station(backend, build_pipeline(profile.sample_rate, settings, {}, [profile]), settings)
        if args.role == "send":
            station.session = MeasureSender(station, args.profile, settings, modes, args.frames)
            RealtimeRunner(station).run()
            return 0
        rcv = MeasureReceiver(writer, args.label, args.profile, args.idle)
        station.session = rcv
        print("Listening… start the sender on the other machine.", flush=True)
        with contextlib.suppress(KeyboardInterrupt):
            RealtimeRunner(station).run()
        print(rcv.summary(args.frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
