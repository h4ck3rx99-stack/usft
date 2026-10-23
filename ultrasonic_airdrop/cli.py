"""Command-line interface: a thin layer over the same core used by the GUI.

airdrop devices
airdrop check [--in N --out N]
airdrop linktest send|receive
airdrop send FILE... [--profile nu|au|us] [--broadcast] [--encrypt]
airdrop receive --out DIR [--accept-all]
airdrop simulate FILE... --snr 12 --reverb 0.3 --ppm 150 --loss 0.05
airdrop wav-encode FILE... --out sig.wav
airdrop wav-decode sig.wav --out DIR
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import logging
import math
import signal
import sys
import tempfile
import threading
from pathlib import Path

from ultrasonic_airdrop import __version__
from ultrasonic_airdrop.config import (
    DEFAULT_TX_LEVEL_DBFS,
    MODES,
    PROFILES,
    Profile,
    Settings,
    TransferOptions,
    get_profile,
)
from ultrasonic_airdrop.errors import AirDropError
from ultrasonic_airdrop.logging_setup import kv, setup_logging
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.engine import Session, Station
from ultrasonic_airdrop.transport.events import (
    DoneEvent,
    ErrorEvent,
    Event,
    FileEvent,
    LinkTestEvent,
    OfferEvent,
    ProgressEvent,
    StateEvent,
)

log = logging.getLogger("ultrasonic_airdrop.cli")


def fmt_duration(seconds: float | None) -> str:
    """Human-readable duration."""
    if seconds is None or not math.isfinite(seconds):
        return "unknown"
    seconds = round(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"


def fmt_bytes(n: float) -> str:
    """Human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class Printer:
    """Prints engine events to the terminal."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self._last_state = ""

    def __call__(self, ev: Event) -> None:
        if isinstance(ev, StateEvent):
            if ev.label != self._last_state:
                self._last_state = ev.label
                print(f"[{ev.role}] {ev.label}", flush=True)
        elif isinstance(ev, ProgressEvent) and not self.quiet:
            pct = 100.0 * ev.done_bytes / ev.total_bytes if ev.total_bytes else 100.0
            print(
                f"  {pct:5.1f}%  {fmt_bytes(ev.done_bytes)}/{fmt_bytes(ev.total_bytes)}  "
                f"goodput {ev.goodput_bps:.1f} B/s  ETA {fmt_duration(ev.eta_s)}  mode {ev.mode_id}  "
                f"FER {ev.frame_error_rate:.0%}  SNR {ev.snr_db:.1f} dB",
                flush=True,
            )
        elif isinstance(ev, OfferEvent):
            print(
                f"Offer from '{ev.sender_name}': {ev.file_count} file(s), {fmt_bytes(ev.total_bytes)}, "
                f"≈{fmt_duration(ev.estimated_s)}, profile {ev.profile_key}, "
                f"{'encrypted, ' if ev.encrypted else ''}{'broadcast' if ev.broadcast else 'interactive'}",
                flush=True,
            )
        elif isinstance(ev, FileEvent) and ev.status != "receiving":
            print(f"  file {ev.name}: {ev.status}{' → ' + str(ev.path) if ev.path else ''}", flush=True)
        elif isinstance(ev, ErrorEvent):
            print(f"ERROR: {ev.message}" + (f"\n  → {ev.hint}" if ev.hint else ""), file=sys.stderr, flush=True)
        elif isinstance(ev, DoneEvent):
            print(f"[{ev.role}] {'SUCCESS' if ev.success else 'NOT SUCCESSFUL'}: {ev.message}", flush=True)
            for name, status, path in ev.files:
                print(f"    {name}: {status} {path}", flush=True)
        elif isinstance(ev, LinkTestEvent) and not self.quiet:
            if ev.role == "linktest-sender":
                print(f"  probes sent: {ev.probes_sent} ({ev.elapsed_s:.0f} s)", flush=True)
            else:
                med = "n/a"
                if ev.carrier_snr_db:
                    vals = sorted(ev.carrier_snr_db)
                    med = f"{vals[len(vals) // 2]:.1f} dB"
                print(
                    f"  probes ok {ev.frames_ok}, errors {ev.frame_errors}, detections {ev.detections}, "
                    f"MFSK SNR {ev.mfsk_snr_db:.1f} dB, median carrier SNR {med}, {ev.ppm:+.0f} ppm, "
                    f"recommended mode {ev.recommended_mode} ({ev.carriers_enabled} carriers)",
                    flush=True,
                )


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings()
    changes: dict[str, object] = {}
    if getattr(args, "level", None) is not None:
        changes["tx_level_dbfs"] = args.level
    if getattr(args, "exclusive", False):
        changes["exclusive_mode"] = True
    if getattr(args, "detect_ratio", None) is not None:
        changes["detect_ratio"] = args.detect_ratio
    if getattr(args, "max_size_mb", None) is not None:
        changes["max_accept_bytes"] = int(args.max_size_mb * 1024 * 1024)
    if getattr(args, "passes", None) is not None:
        changes["broadcast_passes"] = args.passes
    return s.replace(**changes) if changes else s


def _passphrase(args: argparse.Namespace, confirm: bool) -> str:
    if getattr(args, "passphrase", None):
        return str(args.passphrase)
    first = getpass.getpass("Passphrase: ")
    if confirm and getpass.getpass("Repeat passphrase: ") != first:
        raise AirDropError("Passphrases do not match.")
    if not first:
        raise AirDropError("Empty passphrase.")
    return first


def _station(
    profile_key: str, settings: Settings, args: argparse.Namespace, events: Printer, restrict: bool
) -> tuple[Station, dict[int, CarrierMask], Profile]:
    from ultrasonic_airdrop.audio.sd_backend import SoundDeviceBackend
    from ultrasonic_airdrop.transport.engine import build_pipeline

    profile = get_profile(profile_key)
    backend = SoundDeviceBackend(profile.sample_rate, args.input, args.output, settings.exclusive_mode)
    masks: dict[int, CarrierMask] = {}
    pipe = build_pipeline(profile.sample_rate, settings, masks, [profile] if restrict else None)
    return Station(backend, pipe, settings, events), masks, profile


def _run(station: Station, session: Session) -> int:
    from ultrasonic_airdrop.transport.engine import RealtimeRunner

    station.session = session
    runner = RealtimeRunner(station)
    stop = threading.Event()

    def on_sigint(_sig: int, _frame: object) -> None:
        if stop.is_set():
            runner.stop()
            return
        stop.set()
        print("\nCancelling… (Ctrl+C again to abort immediately)", file=sys.stderr)
        cancel = getattr(session, "cancel", None)
        if callable(cancel):
            cancel()
        else:
            runner.stop()

    old = signal.signal(signal.SIGINT, on_sigint)
    try:
        runner.start()
        while runner.running:
            runner.join(0.2)
    finally:
        signal.signal(signal.SIGINT, old)
        runner.stop()
        runner.join(5.0)
    if runner.error is not None:
        return 2
    return 0 if getattr(session, "success", True) else 1


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def cmd_devices(args: argparse.Namespace) -> int:
    """List audio devices."""
    from ultrasonic_airdrop.audio.devices import check_rate, list_devices

    devs = list_devices()
    if not devs:
        print("No audio devices found.")
        return 1
    for d in devs:
        caps = []
        for rate in (48_000, 96_000):
            ok_in = d.max_input_channels > 0 and check_rate(d.index, rate, "input") is None
            ok_out = d.max_output_channels > 0 and check_rate(d.index, rate, "output") is None
            if ok_in or ok_out:
                caps.append(
                    f"{rate // 1000}k:{'in' if ok_in else ''}{'/' if ok_in and ok_out else ''}{'out' if ok_out else ''}"
                )
        print(
            f"{d.index:3d}  {d.label:60s} in={d.max_input_channels} out={d.max_output_channels} "
            f"default={d.default_samplerate:.0f} Hz  {' '.join(caps)}"
        )
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Hardware self-check."""
    from ultrasonic_airdrop.audio.diagnostics import run_hardware_check
    from ultrasonic_airdrop.audio.sd_backend import SoundDeviceBackend

    settings = _settings(args)
    print(
        f"Hardware check: this will play ~1 s of silence and two {3} s sweeps (2 kHz → ultrasonic) through "
        f"the selected speaker at up to {args.level:.0f} dBFS. Children and pets may hear high tones."
    )
    backend = SoundDeviceBackend(args.rate, args.input, args.output, settings.exclusive_mode)
    res = run_hardware_check(backend, args.level, progress=lambda m: print("  " + m, flush=True))
    print(f"\nSample rate: {res.sample_rate} Hz (device reported {res.actual_rate})")
    print(
        f"Peak recorded level: {res.peak_dbfs:.1f} dBFS{'  (CLIPPING)' if res.clipping else ''}; "
        f"noise floor {res.noise_floor_dbfs:.1f} dBFS"
    )
    for b in res.bands:
        print(
            f"  {PROFILES[b.profile_key].name:48s} response {b.relative_db:+6.1f} dB vs 2–4 kHz, "
            f"band SNR {b.snr_db:5.1f} dB  →  {b.verdict}"
        )
    if res.brick_wall_hz:
        print(f"  Brick-wall drop detected at ≈{res.brick_wall_hz / 1000:.1f} kHz")
    for m in res.messages:
        print("  • " + m)
    return 0 if res.verdict("nu") == "supported" else 1


def cmd_linktest(args: argparse.Namespace) -> int:
    """Two-machine link test."""
    from ultrasonic_airdrop.audio.diagnostics import LinkTestReceiver, LinkTestSender

    settings = _settings(args)
    printer = Printer()
    if args.role == "send":
        station, _masks, profile = _station(args.profile, settings, args, printer, True)
        print(f"Link test: transmitting PROBE frames for {args.duration:.0f} s on the {profile.name} profile.")
        sess = LinkTestSender(station, profile, settings, printer, args.duration)
        return _run(station, sess)
    station, _masks, profile = _station(args.profile, settings, args, printer, False)
    print("Link test: listening for PROBE frames (Ctrl+C to stop)…")

    def stats() -> tuple[int, int]:
        return sum(p.stats.detections for p in station.pipeline.pipelines), 0

    rx = LinkTestReceiver(printer, stats, args.duration + 10.0, now=0.0)
    station.session = rx
    import time as _t

    rx.start = _t.monotonic()
    return _run(station, rx)


def cmd_send(args: argparse.Namespace) -> int:
    """Send files over the air."""
    from ultrasonic_airdrop.files.chunker import cleanup_prepared, prepare_files
    from ultrasonic_airdrop.transport.estimate import estimate_transfer_s
    from ultrasonic_airdrop.transport.session_sender import SenderSession

    settings = _settings(args)
    paths = [Path(p) for p in args.files]
    options = TransferOptions(
        profile_key=args.profile,
        broadcast=args.broadcast,
        encrypt=args.encrypt,
        compression=not args.no_compress,
        sender_name=args.name,
        fixed_mode=args.mode,
    )
    if args.encrypt:
        options.passphrase = _passphrase(args, confirm=True)
    profile = get_profile(args.profile)
    with tempfile.TemporaryDirectory(prefix="airdrop-") as tmp:
        print("Preparing files (hashing/compressing)…", flush=True)
        manifest, prepared = prepare_files(paths, options.compression, Path(tmp))
        try:
            mode = args.mode if args.mode is not None else (settings.broadcast_mode if args.broadcast else 1)
            est = estimate_transfer_s(
                profile,
                [f.tx_size for f in manifest.files],
                mode,
                settings,
                args.encrypt,
                args.broadcast,
                None,
                len(manifest.to_blob()),
            )
            print(
                f"{len(manifest.files)} file(s), {fmt_bytes(manifest.total_size)} "
                f"({fmt_bytes(manifest.total_tx)} on air). Estimated airtime ≈ {fmt_duration(est)} "
                f"at mode {mode} ({MODES[mode].label}); actual time depends on link quality and retransmissions."
            )
            if not args.yes:
                reply = input("Start transmitting? [y/N] ").strip().lower()
                if reply not in ("y", "yes"):
                    print("Aborted.")
                    return 1
            printer = Printer()
            station, _masks, _p = _station(args.profile, settings, args, printer, True)
            sess = SenderSession(station, profile, settings, options, manifest, prepared, printer)
            return _run(station, sess)
        finally:
            cleanup_prepared(prepared)


def cmd_receive(args: argparse.Namespace) -> int:
    """Receive files."""
    from ultrasonic_airdrop.transport.session_receiver import ReceiverSession

    settings = _settings(args)
    printer = Printer()
    rate = get_profile(args.profile).sample_rate if args.profile != "auto" else 48_000
    station, masks, _profile = _station("nu" if rate == 48_000 else "us", settings, args, printer, False)
    dest = Path(args.out).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    phrase = args.passphrase or ""
    sess = ReceiverSession(
        station,
        settings,
        dest,
        printer,
        masks=masks,
        passphrase=phrase,
        accept_all=args.accept_all,
        keep_listening=args.keep_listening,
        on_profile_lock=station.pipeline.restrict,
    )
    if not args.accept_all:

        def ask(ev: Event) -> None:
            printer(ev)
            if isinstance(ev, OfferEvent):

                def prompt() -> None:
                    ans = input("Accept? [y/N] ").strip().lower()
                    pw = ""
                    if ans in ("y", "yes") and ev.encrypted and not phrase:
                        pw = getpass.getpass("Passphrase: ")
                    sess.decide(ans in ("y", "yes"), pw)

                threading.Thread(target=prompt, daemon=True).start()

        sess.events = ask
        station.events = ask
    print(f"Listening… files will be saved to {dest} (Ctrl+C to stop).")
    return _run(station, sess)


def cmd_simulate(args: argparse.Namespace) -> int:
    """Full in-process transfer through the channel simulator."""
    from ultrasonic_airdrop.channel.simulator import ChannelConfig
    from ultrasonic_airdrop.transport.simulate import run_transfer

    settings = _settings(args)
    cfg = dict(
        snr_db=args.snr,
        ppm=args.ppm,
        rt60_s=args.reverb,
        loss=args.loss,
        cfo_hz=args.cfo,
        burst_rate_hz=args.bursts,
        lowpass_hz=args.lowpass,
    )
    options = TransferOptions(
        profile_key=args.profile,
        broadcast=args.broadcast,
        encrypt=args.encrypt,
        passphrase=args.passphrase or ("simulation" if args.encrypt else ""),
        compression=not args.no_compress,
        fixed_mode=args.mode,
    )
    paths = [Path(p) for p in args.files]
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="airdrop-sim-out-"))
    print(
        f"Simulating transfer of {len(paths)} file(s) (SNR {args.snr} dB, {args.ppm:+g} ppm, RT60 {args.reverb} s, "
        f"loss {args.loss:.0%}) — output in {out}"
    )
    res = run_transfer(
        paths,
        out,
        settings,
        options,
        a_to_b=ChannelConfig(**cfg, seed=args.seed),
        b_to_a=ChannelConfig(**cfg, seed=args.seed + 1),
        seed=args.seed,
        max_seconds=args.max_seconds,
        receiver_passphrase=options.passphrase,
    )
    st = res.sender_stats
    print(f"Sender:   {'SUCCESS' if res.sender_success else 'FAILED'} — {res.sender_message}")
    print(f"Receiver: {'SUCCESS' if res.receiver_success else 'FAILED'} — {res.receiver_message}")
    print(
        f"Simulated time {fmt_duration(res.sim_seconds)}; goodput {st['goodput_bps']:.1f} B/s (on-air bytes "
        f"acknowledged / elapsed); frames sent {st['frames_sent']:.0f}; retransmitted units "
        f"{st['retransmitted_units']:.0f}; data-frame loss rate {st['frame_loss_rate']:.1%}; final mode "
        f"{st['mode']:.0f}.  (SIMULATION — not a hardware measurement.)"
    )
    ok = True
    for p in paths:
        cands = sorted(out.glob(p.name)) + sorted(out.glob(f"{p.stem} (*){p.suffix}"))
        src_hash = hashlib.sha256(p.read_bytes()).hexdigest()
        match = any(hashlib.sha256(c.read_bytes()).hexdigest() == src_hash for c in cands)
        ok &= match
        print(f"  {p.name}: {'SHA-256 verified' if match else 'MISSING or MISMATCH'}")
    return 0 if ok and res.receiver_success else 1


def cmd_wav_encode(args: argparse.Namespace) -> int:
    """Render a broadcast transfer to WAV."""
    from ultrasonic_airdrop.transport.offline import render_broadcast, write_wav

    settings = _settings(args)
    options = TransferOptions(
        profile_key=args.profile,
        broadcast=True,
        encrypt=args.encrypt,
        compression=not args.no_compress,
        fixed_mode=args.mode,
    )
    if args.encrypt:
        options.passphrase = _passphrase(args, confirm=True)
    x, fs = render_broadcast([Path(p) for p in args.files], options, settings)
    write_wav(Path(args.out), x, fs)
    print(f"Wrote {args.out}: {len(x) / fs:.1f} s at {fs} Hz")
    return 0


def cmd_wav_decode(args: argparse.Namespace) -> int:
    """Decode a WAV recording."""
    from ultrasonic_airdrop.transport.offline import decode_recording, read_wav

    x, fs = read_wav(Path(args.wav))
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    res = decode_recording(x, fs, dest, _settings(args), args.passphrase or "", Printer(quiet=True))
    print(f"{'SUCCESS' if res.success else 'INCOMPLETE'}: {res.message} ({res.frames} frames detected)")
    for name, status, path in res.files:
        print(f"  {name}: {status} {path}")
    return 0 if res.success else 1


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    p = argparse.ArgumentParser(prog="airdrop", description="Ultrasonic AirDrop — file transfer over sound.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    def audio_opts(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--in", dest="input", type=int, default=None, help="input device index (see 'devices')")
        sp.add_argument("--out", dest="output", type=int, default=None, help="output device index")
        sp.add_argument("--exclusive", action="store_true", help="try WASAPI exclusive mode")
        sp.add_argument("--level", type=float, default=DEFAULT_TX_LEVEL_DBFS, help="transmit peak level dBFS")

    sp = sub.add_parser("devices", help="list audio devices")
    sp.set_defaults(func=cmd_devices)

    sp = sub.add_parser("check", help="single-machine hardware self-check (plays sweeps)")
    audio_opts(sp)
    sp.add_argument("--rate", type=int, default=48_000, choices=[48_000, 96_000])
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("linktest", help="two-machine link test")
    sp.add_argument("role", choices=["send", "receive"])
    sp.add_argument("--profile", default="nu", choices=sorted(PROFILES))
    sp.add_argument("--duration", type=float, default=20.0)
    audio_opts(sp)
    sp.set_defaults(func=cmd_linktest)

    sp = sub.add_parser("send", help="send files")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--profile", default="nu", choices=sorted(PROFILES))
    sp.add_argument("--broadcast", action="store_true", help="one-way mode (no ACKs)")
    sp.add_argument("--passes", type=int, default=None, help="broadcast passes")
    sp.add_argument("--encrypt", action="store_true")
    sp.add_argument("--passphrase", default=None, help="(insecure on shared machines; prompts if omitted)")
    sp.add_argument("--no-compress", action="store_true")
    sp.add_argument("--mode", type=int, choices=sorted(MODES), default=None, help="fixed link mode")
    sp.add_argument("--name", default="Ultrasonic AirDrop", help="sender display name")
    sp.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")
    audio_opts(sp)
    sp.set_defaults(func=cmd_send)

    sp = sub.add_parser("receive", help="receive files")
    sp.add_argument("--out", dest="out", required=True, help="destination folder")
    sp.add_argument("--accept-all", action="store_true")
    sp.add_argument("--keep-listening", action="store_true")
    sp.add_argument("--passphrase", default=None)
    sp.add_argument("--profile", default="auto", choices=["auto", *sorted(PROFILES)])
    sp.add_argument("--max-size-mb", type=float, default=None)
    sp.add_argument("--in", dest="input", type=int, default=None)
    sp.add_argument("--output-device", dest="output", type=int, default=None)
    sp.add_argument("--exclusive", action="store_true")
    sp.add_argument("--level", type=float, default=DEFAULT_TX_LEVEL_DBFS)
    sp.set_defaults(func=cmd_receive)

    sp = sub.add_parser("simulate", help="in-process transfer through the channel simulator")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--snr", type=float, default=15.0, help="in-band SNR dB")
    sp.add_argument("--reverb", type=float, default=0.0, help="RT60 seconds")
    sp.add_argument("--ppm", type=float, default=0.0)
    sp.add_argument("--loss", type=float, default=0.0, help="frame loss probability")
    sp.add_argument("--cfo", type=float, default=0.0, help="carrier frequency offset Hz")
    sp.add_argument("--bursts", type=float, default=0.0, help="noise bursts per second")
    sp.add_argument("--lowpass", type=float, default=None, help="device low-pass cutoff Hz")
    sp.add_argument("--profile", default="nu", choices=sorted(PROFILES))
    sp.add_argument("--broadcast", action="store_true")
    sp.add_argument("--passes", type=int, default=None)
    sp.add_argument("--encrypt", action="store_true")
    sp.add_argument("--passphrase", default=None)
    sp.add_argument("--no-compress", action="store_true")
    sp.add_argument("--mode", type=int, choices=sorted(MODES), default=None)
    sp.add_argument("--seed", type=int, default=1)
    sp.add_argument("--out", default=None, help="receiver destination (default: temp dir)")
    sp.add_argument("--max-seconds", type=float, default=6 * 3600.0, help="simulated time limit")
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("wav-encode", help="render broadcast frames to a WAV file")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--out", required=True)
    sp.add_argument("--profile", default="nu", choices=sorted(PROFILES))
    sp.add_argument("--mode", type=int, choices=sorted(MODES), default=None)
    sp.add_argument("--passes", type=int, default=None)
    sp.add_argument("--encrypt", action="store_true")
    sp.add_argument("--passphrase", default=None)
    sp.add_argument("--no-compress", action="store_true")
    sp.set_defaults(func=cmd_wav_encode)

    sp = sub.add_parser("wav-decode", help="decode a WAV recording of a broadcast")
    sp.add_argument("wav")
    sp.add_argument("--out", required=True)
    sp.add_argument("--passphrase", default=None)
    sp.set_defaults(func=cmd_wav_decode)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``airdrop``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    try:
        return int(args.func(args))
    except AirDropError as exc:
        print(f"ERROR: {exc.user_message}" + (f"\n  → {exc.hint}" if exc.hint else ""), file=sys.stderr)
        log.debug(kv("command failed", error=exc))
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
