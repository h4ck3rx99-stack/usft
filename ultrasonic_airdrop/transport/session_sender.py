"""Sender state machines: interactive half-duplex ARQ (§4.3) and one-way broadcast (§4.5).

The session is driven by :meth:`tick` (time passing) and :meth:`on_frame`
(a decoded frame from the far side).  It never blocks; all waiting is done by
comparing the clock against deadlines, so the same code runs in real time and
in the sample-clock simulator.
"""

from __future__ import annotations

import enum
import logging
import math
import random
from collections import deque
from collections.abc import Iterator

import numpy as np

from ultrasonic_airdrop.config import (
    ALL_MODES_MASK,
    MAX_MODE,
    PROBE_MODE_ID,
    UNIT_SIZE,
    Profile,
    Settings,
    TransferOptions,
    get_mode,
)
from ultrasonic_airdrop.crypto.aead import FrameCipher, new_salt
from ultrasonic_airdrop.errors import CryptoError, FrameError
from ultrasonic_airdrop.files.chunker import PreparedFile, UnitReader
from ultrasonic_airdrop.files.manifest import Manifest, split_blob
from ultrasonic_airdrop.link import frames as fr
from ultrasonic_airdrop.link.receiver_pipeline import RxFrame
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport import estimate
from ultrasonic_airdrop.transport.arq import Outstanding, PartTracker, SentFrame, UnitTracker
from ultrasonic_airdrop.transport.events import (
    DoneEvent,
    ErrorEvent,
    EventSink,
    LinkEvent,
    ProgressEvent,
    StateEvent,
)
from ultrasonic_airdrop.transport.linkio import LinkIO, OutFrame, Radio
from ultrasonic_airdrop.transport.rate_adapt import RateController

log = logging.getLogger(__name__)
GOODPUT_WINDOW_S = 60.0
AUTH_FAIL_LIMIT = 3


class SenderState(enum.Enum):
    """Sender states."""

    OFFERING = "offering"
    PROBING = "probing"
    MANIFEST = "manifest"
    DATA = "data"
    ENDING = "ending"
    BROADCAST = "broadcast"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


STATE_LABELS = {
    SenderState.OFFERING: "Offering… waiting for receiver",
    SenderState.PROBING: "Probing the channel…",
    SenderState.MANIFEST: "Sending manifest…",
    SenderState.DATA: "Sending data…",
    SenderState.ENDING: "Verifying…",
    SenderState.BROADCAST: "Broadcasting…",
    SenderState.DONE: "Done",
    SenderState.FAILED: "Failed",
    SenderState.CANCELLED: "Cancelled",
}
TERMINAL = {SenderState.DONE, SenderState.FAILED, SenderState.CANCELLED}


def capabilities(settings: Settings) -> int:
    """Our capability bitmap."""
    return (settings.supported_modes & ALL_MODES_MASK) | fr.CAP_ENC | fr.CAP_COMPRESSION


class SenderSession:
    """Interactive or broadcast sender."""

    role = "sender"

    def __init__(
        self,
        radio: Radio,
        profile: Profile,
        settings: Settings,
        options: TransferOptions,
        manifest: Manifest,
        prepared: list[PreparedFile],
        events: EventSink,
        session_id: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.radio = radio
        self.profile = profile
        self.settings = settings
        self.options = options
        self.manifest = manifest
        self.events = events
        self.rng = rng or random.Random()
        self.session_id = session_id if session_id is not None else self.rng.getrandbits(32)
        self.io = LinkIO(radio, profile, settings, fr.DIR_S2R, self.session_id)
        self.reader = UnitReader(prepared)
        self.blob = manifest.to_blob()
        self.salt = new_salt() if options.encrypt else bytes(16)
        if options.encrypt:
            if not options.passphrase:
                raise ValueError("encryption requires a passphrase")
            self.io.set_cipher(FrameCipher.from_passphrase(options.passphrase, self.salt, self.session_id))
        self.units = UnitTracker(manifest)
        self.outstanding = Outstanding()
        self.state = SenderState.BROADCAST if options.broadcast else SenderState.OFFERING
        self.peer_caps = capabilities(settings)
        initial = options.fixed_mode if options.fixed_mode is not None else 1
        self.rate = RateController(mode=initial, caps=self.peer_caps)
        if options.fixed_mode is not None:
            self.rate.last_suggestion = options.fixed_mode
        self.mask = CarrierMask.all()
        self.deadline: float | None = None
        self.expect: str = ""
        self.offer_sends = 0
        self.probe_sends = 0
        self.polls = 0
        self.end_sends = 0
        self.auth_failures = 0
        self.window_size = 0
        self.frame_errors_reported = 0
        self.last_snr = float("nan")
        self.end_statuses: dict[int, int] = {}
        self.retransmissions = 0
        self.acks = 0
        self.start_time = radio.now()
        self.progress_hist: deque[tuple[float, int]] = deque()
        self.message = ""
        self._parts: list[bytes] = []
        self._part_tracker: PartTracker | None = None
        self._broadcast_iter: Iterator[list[OutFrame]] | None = None
        self._cancel_requested = False
        self._cancel_reason: str | None = None
        self._windows_total = 0
        self.hold_until = 0.0
        self.polls_total = 0
        self.data_start: float | None = None
        self.end_time: float | None = None
        self.mode_history: list[int] = []
        self._frames_total = 0
        self._frames_lost = 0
        self._emit_state()
        log.info(
            kv(
                "sender session start",
                session=f"{self.session_id:08x}",
                files=len(manifest.files),
                tx_bytes=manifest.total_tx,
                broadcast=options.broadcast,
                encrypted=options.encrypt,
                profile=profile.key,
            )
        )

    # ------------------------------------------------------------------ helpers
    @property
    def done(self) -> bool:
        """Terminal state reached."""
        return self.state in TERMINAL

    @property
    def success(self) -> bool:
        """Finished successfully (interactive: all files verified by the receiver)."""
        return self.state == SenderState.DONE

    def _emit_state(self, detail: str = "") -> None:
        label = STATE_LABELS[self.state]
        if self.state == SenderState.DATA and self.expect == "ack":
            label = "Waiting for ACK…"
        self.events(StateEvent(self.role, self.state.value, label + (f" ({detail})" if detail else "")))

    def _set_state(self, state: SenderState, detail: str = "") -> None:
        if state == SenderState.DATA and self.data_start is None:
            self.data_start = self.radio.now()
        if state != self.state:
            log.info(kv("sender state", old=self.state.value, new=state.value, detail=detail))
        self.state = state
        self._emit_state(detail)

    def _finish(self, state: SenderState, message: str) -> None:
        self.end_time = self.radio.now()
        self.message = message
        self.deadline = None
        self._set_state(state, message)
        self.reader.close()
        files = []
        for e in self.manifest.files:
            st = self.end_statuses.get(e.index)
            files.append((e.name, fr.FILE_STATUS_NAMES.get(st, "UNKNOWN") if st is not None else "-", ""))
        self.events(
            DoneEvent(
                self.role,
                state == SenderState.DONE,
                message,
                tuple(files),
                stats=self.stats(),
            )
        )

    def stats(self) -> dict[str, float]:
        """Summary statistics (frozen at the moment the session ended)."""
        now = self.end_time if self.end_time is not None else self.radio.now()
        elapsed = max(1e-9, now - self.start_time)
        return {
            "elapsed_s": elapsed,
            "acked_bytes": float(self.units.acked_bytes()),
            "goodput_bps": self.units.acked_bytes() / elapsed,
            "frames_sent": float(self.io.frames_sent),
            "airtime_s": self.io.airtime_sent,
            "retransmitted_units": float(self.units.retransmitted_units),
            "frame_loss_rate": self._frames_lost / self._frames_total if self._frames_total else 0.0,
            "mode": float(self.rate.mode),
            "polls": float(self.polls_total),
            "data_phase_goodput_bps": (
                self.units.acked_bytes() / max(1e-9, now - self.data_start) if self.data_start is not None else 0.0
            ),
        }

    def _progress(self) -> None:
        now = self.radio.now()
        acked = self.units.acked_bytes()
        self.progress_hist.append((now, acked))
        while self.progress_hist and now - self.progress_hist[0][0] > GOODPUT_WINDOW_S:
            self.progress_hist.popleft()
        t0, b0 = self.progress_hist[0]
        span = now - t0
        if span < 5.0:
            span = max(now - self.start_time, 1e-6)
            b0 = 0
        goodput = (acked - b0) / span if span > 0 else 0.0
        remaining = self.units.total_bytes - acked
        eta = remaining / goodput if goodput > 0 else None
        fer = self._frames_lost / self._frames_total if self._frames_total else 0.0
        self.events(
            ProgressEvent(
                self.role,
                acked,
                self.units.total_bytes,
                goodput,
                eta,
                self.rate.mode,
                self.units.retransmitted_units,
                fer,
                self.last_snr,
                now - self.start_time,
            )
        )

    def _wait_for(self, expect: str, tx_end: float, reply_airtime: float) -> None:
        self.expect = expect
        self.deadline = tx_end + estimate.reply_wait_s(self.profile, self.settings, reply_airtime)

    def _ack_air(self) -> float:
        return estimate.ack_airtime(self.profile, self.io.cipher is not None)

    # ------------------------------------------------------------------ control API
    def request_cancel(self, reason: str = "Cancelled by sender") -> None:
        """Thread-safe cancel request (applied on the engine thread at the next tick)."""
        self._cancel_reason = reason

    def cancel(self, reason: str = "Cancelled by sender") -> None:
        """User cancel: send CANCEL twice and stop."""
        if self.done:
            return
        self._cancel_requested = True
        payload = fr.Cancel(fr.CANCEL_USER, reason)
        self.io.send([self.io.build(payload), self.io.build(payload)])
        self._finish(SenderState.CANCELLED, reason)

    # ------------------------------------------------------------------ main loop
    def tick(self, now: float) -> None:
        """Advance the state machine."""
        if self._cancel_reason is not None and not self.done:
            reason, self._cancel_reason = self._cancel_reason, None
            self.cancel(reason)
            return
        if self.done or self.radio.tx_busy():
            return
        if now < self.hold_until:
            return  # turnaround: let the peer's half-duplex gate and reverberation clear
        if self.state == SenderState.BROADCAST:
            self._broadcast_step()
            return
        if self.deadline is not None:
            if now < self.deadline:
                return
            if self.radio.rx_busy():
                self.deadline = now + 0.5
                return
            self._on_timeout()
            return
        self._act()

    def _act(self) -> None:
        if self.state == SenderState.OFFERING:
            self._send_offer()
        elif self.state == SenderState.PROBING:
            self._send_probe()
        elif self.state == SenderState.MANIFEST:
            self._send_manifest_window()
        elif self.state == SenderState.DATA:
            self._send_data_window()
        elif self.state == SenderState.ENDING:
            self._send_end()

    def _on_timeout(self) -> None:
        exp = self.expect
        self.deadline = None
        if exp == "accept":
            if self.offer_sends >= self.settings.offer_max_sends:
                self._finish(SenderState.FAILED, "No receiver answered the offer")
            else:
                self._send_offer()
        elif exp == "report":
            if self.probe_sends >= self.settings.probe_attempts:
                log.warning(kv("probe failed; falling back to robust MFSK", attempts=self.probe_sends))
                self.rate = RateController(mode=1, caps=self.peer_caps, last_suggestion=1)
                self.mask = CarrierMask.all()
                self._set_state(SenderState.MANIFEST, "probe failed, using mode 1")
                self._act()
            else:
                self._send_probe()
        elif exp in ("ack", "end_ack"):
            if exp == "ack" and self.state == SenderState.DATA:
                self.rate.on_ack_missed()
            if exp == "end_ack":
                if self.end_sends >= self.settings.end_max_attempts:
                    self._finish(SenderState.FAILED, "Link lost while finishing (no END_ACK)")
                else:
                    self._send_end()
                return
            self.polls += 1
            self.polls_total += 1
            if self.polls > self.settings.max_polls:
                self._finish(SenderState.FAILED, "Link lost: no ACK after repeated polls (resume is possible)")
                return
            out = self.io.build(fr.Poll(), poll=True)
            end = self.io.send([out])
            log.info(kv("ACK timeout, polling", poll=self.polls))
            self._wait_for("ack", end, self._ack_air())

    # ------------------------------------------------------------------ phases
    def _send_offer(self) -> None:
        self.offer_sends += 1
        offer = fr.Offer(
            capabilities=capabilities(self.settings),
            transfer_mode=fr.TRANSFER_BROADCAST if self.options.broadcast else fr.TRANSFER_INTERACTIVE,
            file_count=len(self.manifest.files),
            total_bytes=self.manifest.total_tx,
            manifest_len=len(self.blob),
            manifest_sha256=Manifest.blob_sha256(self.blob),
            encrypted=self.options.encrypt,
            salt=self.salt,
            sender_name=self.options.sender_name,
        )
        end = self.io.send([self.io.build(offer)])
        accept_air = self.io.control_airtime(12)
        self.expect = "accept"
        self.deadline = end + max(
            self.settings.offer_listen_s, estimate.reply_wait_s(self.profile, self.settings, accept_air)
        )
        self._emit_state(f"attempt {self.offer_sends}/{self.settings.offer_max_sends}")

    def _send_probe(self) -> None:
        self.probe_sends += 1
        out = self.io.build(fr.Probe(), mode_id=PROBE_MODE_ID)
        end = self.io.send([out])
        self._wait_for("report", end, self.io.control_airtime(64))

    def _manifest_parts(self) -> list[bytes]:
        if not self._parts:
            size = estimate.max_control_payload(self.profile, self.io.cipher is not None) - 4
            self._parts = split_blob(self.blob, size)
            self._part_tracker = PartTracker(len(self._parts))
        return self._parts

    def _send_manifest_window(self) -> None:
        parts = self._manifest_parts()
        tracker = self._part_tracker
        assert tracker is not None
        if tracker.all_acked():
            self._set_state(SenderState.DATA)
            self._act()
            return
        chosen: list[int] = []
        air = 0.0
        part_air = self.io.control_airtime(4 + len(parts[0]) + self.io.tag_overhead(fr.FrameType.MANIFEST))
        while len(chosen) < self.settings.window_mfsk:
            idx = tracker.next_part()
            if idx is None or (chosen and air + part_air > 20.0):
                break
            air += part_air
            tracker.inflight.add(idx)
            chosen.append(idx)
        frames: list[OutFrame] = []
        for k, idx in enumerate(chosen):
            payload = fr.ManifestPart(idx, len(parts), parts[idx])
            out = self.io.build(payload, poll=k == len(chosen) - 1)
            self.outstanding.add(SentFrame(out.seq, "manifest", part=idx))
            frames.append(out)
        if not frames:
            frames = [self.io.build(fr.Poll(), poll=True)]
        end = self.io.send(frames)
        self.polls = 0
        self._wait_for("ack", end, self._ack_air())

    def _data_mode(self) -> tuple[int, CarrierMask | None, int | None]:
        mode = self.rate.mode
        if get_mode(mode).is_ofdm:
            return mode, self.mask, self.mask.count
        return mode, None, None

    def _send_data_window(self) -> None:
        if self.units.all_acked():
            self._set_state(SenderState.ENDING)
            self._act()
            return
        mode, mask, n_en = self._data_mode()
        enc = self.io.cipher is not None
        units_pf = estimate.max_units_per_frame(self.profile, mode, n_en, enc)
        window = estimate.window_frames(self.settings, self.profile, mode, n_en, enc)
        runs: list[tuple[int, int, int]] = []
        for _ in range(window):
            run = self.units.next_run(units_pf)
            if run is None:
                break
            self.units.mark_inflight(*run)
            runs.append(run)
        frames: list[OutFrame] = []
        for k, (fi, start, count) in enumerate(runs):
            data = self.reader.read(fi, start, count)
            out = self.io.build(fr.Data(fi, start, count, data), poll=k == len(runs) - 1, mode_id=mode, mask=mask)
            self.outstanding.add(SentFrame(out.seq, "data", fi, start, count, nbytes=len(data)))
            frames.append(out)
        if not frames:
            frames = [self.io.build(fr.Poll(), poll=True)]
        self.window_size = len(runs)
        self._windows_total += 1
        end = self.io.send(frames)
        self.polls = 0
        self._wait_for("ack", end, self._ack_air())
        self._emit_state(f"mode {mode}, {self.window_size} frames")

    def _send_end(self) -> None:
        self.end_sends += 1
        end = self.io.send([self.io.build(fr.End(), poll=True)])
        n_frames = max(1, math.ceil(len(self.manifest.files) / 100))
        self._wait_for("end_ack", end, n_frames * self.io.control_airtime(120))

    # ------------------------------------------------------------------ broadcast
    def _broadcast_frames(self) -> Iterator[list[OutFrame]]:
        offer = fr.Offer(
            capabilities=capabilities(self.settings),
            transfer_mode=fr.TRANSFER_BROADCAST,
            file_count=len(self.manifest.files),
            total_bytes=self.manifest.total_tx,
            manifest_len=len(self.blob),
            manifest_sha256=Manifest.blob_sha256(self.blob),
            encrypted=self.options.encrypt,
            salt=self.salt,
            sender_name=self.options.sender_name,
        )
        for _ in range(3):
            yield [self.io.build(offer)]
        parts = self._manifest_parts()
        for _ in range(2):
            for i, p in enumerate(parts):
                yield [self.io.build(fr.ManifestPart(i, len(parts), p))]
        mode = self.options.fixed_mode if self.options.fixed_mode is not None else self.settings.broadcast_mode
        n_en = None
        mask = CarrierMask.all() if get_mode(mode).is_ofdm else None
        units_pf = estimate.max_units_per_frame(self.profile, mode, n_en, self.io.cipher is not None)
        for pass_no in range(self.settings.broadcast_passes):
            log.info(kv("broadcast pass", n=pass_no + 1, of=self.settings.broadcast_passes))
            for f in self.manifest.files:
                if self.units.alias[f.index]:
                    continue
                for start in range(0, f.n_units, units_pf):
                    count = min(units_pf, f.n_units - start)
                    data = self.reader.read(f.index, start, count)
                    self.units.mark_inflight(f.index, start, count)
                    yield [self.io.build(fr.Data(f.index, start, count, data), mode_id=mode, mask=mask)]
                    self.units.mark_acked(f.index, start, count)
            yield [self.io.build(fr.End())]
        yield [self.io.build(fr.End())]

    def _broadcast_step(self) -> None:
        if self._broadcast_iter is None:
            self._broadcast_iter = self._broadcast_frames()
        batch = next(self._broadcast_iter, None)
        if batch is None:
            self._finish(SenderState.DONE, "Broadcast finished (the receiver verifies each file independently)")
            return
        self.io.send(batch)
        self._progress()

    # ------------------------------------------------------------------ receive path
    def on_frame(self, rx: RxFrame, now: float) -> None:
        """Handle a frame decoded by our receive pipeline."""
        if rx.info.header is not None:
            self.events(
                LinkEvent(
                    rx.info.profile_key,
                    rx.ok,
                    rx.info.snr_db,
                    rx.info.ppm,
                    rx.info.detect_ratio,
                    rx.info.header.mode_id,
                )
            )
        if self.done or rx.body is None:
            return
        try:
            frame = fr.LinkFrame.unpack(rx.body)
        except FrameError as exc:
            log.info(kv("discarding frame", error=exc.user_message))
            return
        if frame.direction != fr.DIR_R2S or frame.session_id != self.session_id:
            return  # self-echo or someone else's session
        try:
            payload = self.io.open(frame)
        except CryptoError:
            self.auth_failures += 1
            log.warning(kv("authentication failure on receiver frame", count=self.auth_failures))
            if self.auth_failures >= AUTH_FAIL_LIMIT:
                self.events(ErrorEvent("Passphrase mismatch or corrupted data", "Check the passphrase on both sides."))
                self.io.send([self.io.build(fr.Cancel(fr.CANCEL_AUTH, "authentication failed"))] * 2)
                self._finish(SenderState.FAILED, "Passphrase mismatch or corrupted data")
            return
        except FrameError as exc:
            log.info(kv("bad payload", type=frame.ftype.name, error=exc.user_message))
            return
        self.auth_failures = 0
        self.hold_until = now + self.settings.reply_delay_s
        if self.deadline is not None and self.expect in ("ack", "end_ack"):
            # The receiver is talking: allow its whole reply sequence to arrive.
            self.deadline = max(self.deadline, now + self._ack_air() + self.settings.ack_extra_s)
        if isinstance(payload, fr.Cancel):
            reason = {
                fr.CANCEL_VERSION: "protocol version mismatch",
                fr.CANCEL_AUTH: "passphrase mismatch",
            }.get(payload.reason, payload.text or "cancelled by receiver")
            self._finish(SenderState.CANCELLED, f"Receiver cancelled: {reason}")
        elif isinstance(payload, fr.Accept):
            self._on_accept(payload)
        elif isinstance(payload, fr.ProbeReport):
            self._on_report(payload)
        elif isinstance(payload, fr.ResumeInfo):
            for start, count in payload.ranges:
                self.units.mark_have(payload.file_index, start, count)
            log.info(kv("resume info", file=payload.file_index, ranges=len(payload.ranges)))
        elif isinstance(payload, fr.Ack):
            self._on_ack(payload)
        elif isinstance(payload, fr.EndAck):
            self._on_end_ack(payload)

    def _on_accept(self, acc: fr.Accept) -> None:
        if self.state != SenderState.OFFERING:
            return
        self.deadline = None
        if acc.code != fr.ACCEPT_OK:
            reason = {
                fr.ACCEPT_REJECTED: "the receiver declined",
                fr.ACCEPT_TOO_LARGE: "the transfer exceeds the receiver's size limit",
                fr.ACCEPT_BUSY: "the receiver is busy",
                fr.ACCEPT_UNSUPPORTED: "the receiver does not support this transfer",
            }.get(acc.code, "rejected")
            self._finish(SenderState.FAILED, f"Offer rejected: {reason}")
            return
        self.peer_caps = capabilities(self.settings) & acc.capabilities
        if self.options.encrypt and not acc.capabilities & fr.CAP_ENC:
            self._finish(SenderState.FAILED, "Receiver does not support encryption")
            return
        self.rate.caps = self.peer_caps
        log.info(kv("offer accepted", caps=hex(acc.capabilities), resume=acc.resume_available))
        if self.options.fixed_mode is not None:
            self._set_state(SenderState.MANIFEST, f"fixed mode {self.options.fixed_mode}")
        else:
            self._set_state(SenderState.PROBING)

    def _on_report(self, rep: fr.ProbeReport) -> None:
        if self.state != SenderState.PROBING:
            return
        self.deadline = None
        mode = min(rep.recommended_mode, MAX_MODE)
        self.rate = RateController(mode=mode, caps=self.peer_caps, last_suggestion=mode)
        self.rate.mode = self.rate._clamp(mode)
        mask = CarrierMask.from_bytes(rep.mask)
        self.mask = mask if mask.count >= 8 else CarrierMask.all()
        self.last_snr = fr.i16_to_db(rep.mfsk_snr)
        snrs = [fr.u8_to_snr(v) for v in rep.carrier_snr]
        log.info(
            kv(
                "probe report",
                mode=self.rate.mode,
                carriers=self.mask.count,
                mfsk_snr=self.last_snr,
                median_carrier_snr=float(np.median(snrs)),
                ppm=rep.ppm,
            )
        )
        self.mode_history.append(self.rate.mode)
        self._set_state(SenderState.MANIFEST, f"mode {self.rate.mode}")

    def _on_ack(self, ack: fr.Ack) -> None:
        if self.state not in (SenderState.MANIFEST, SenderState.DATA):
            return
        self.acks += 1
        final = self.expect == "ack" and self.deadline is not None
        acked, lost = self.outstanding.resolve(set(ack.acked_seqs()), final)
        tracker = self._part_tracker
        for f in acked:
            if f.kind == "data":
                self.units.mark_acked(f.file_index, f.start, f.count)
            elif tracker is not None:
                tracker.inflight.discard(f.part)
                tracker.acked.add(f.part)
        for f in lost:
            if f.kind == "data":
                self.units.release(f.file_index, f.start, f.count)
            elif tracker is not None:
                tracker.inflight.discard(f.part)
        for rg in ack.missing:
            self.units.mark_missing(rg.file_index, rg.start, rg.count)
        self.last_snr = fr.i16_to_db(ack.mean_snr)
        n_data = sum(1 for f in acked + lost if f.kind == "data")
        n_lost = sum(1 for f in lost if f.kind == "data")
        self._frames_total += n_data
        self._frames_lost += n_lost
        if self.state == SenderState.DATA and self.options.fixed_mode is None:
            old = self.rate.mode
            self.rate.on_ack(n_data, n_lost, ack.suggested_mode)
            mask = CarrierMask.from_bytes(ack.mask)
            if mask.count >= 8:
                self.mask = mask
            if self.rate.mode != old:
                self.mode_history.append(self.rate.mode)
                log.info(
                    kv(
                        "mode change",
                        old=old,
                        new=self.rate.mode,
                        lost=n_lost,
                        of=n_data,
                        suggested=ack.suggested_mode,
                    )
                )
        if final:
            self.deadline = None
            self.polls = 0
            self.expect = ""
        log.debug(kv("ack", base=ack.base_seq, acked=len(acked), lost=len(lost), snr=self.last_snr))
        self._progress()

    def _on_end_ack(self, ea: fr.EndAck) -> None:
        if self.state != SenderState.ENDING:
            return
        for i, st in enumerate(ea.statuses):
            self.end_statuses[ea.first + i] = st
        for rg in ea.missing:
            self.units.mark_missing(rg.file_index, rg.start, rg.count)
        if len(self.end_statuses) < ea.total_files:
            return  # more END_ACK frames to come
        self.deadline = None
        statuses = [self.end_statuses.get(i, fr.FILE_INCOMPLETE) for i in range(len(self.manifest.files))]
        if all(s == fr.FILE_OK for s in statuses):
            self._progress()
            self._finish(SenderState.DONE, "All files delivered and verified by the receiver")
            return
        fatal = [i for i, s in enumerate(statuses) if s in (fr.FILE_HASH_MISMATCH, fr.FILE_WRITE_ERROR)]
        if fatal:
            names = ", ".join(self.manifest.files[i].name for i in fatal[:5])
            self._finish(SenderState.FAILED, f"Receiver could not verify/store: {names}")
            return
        # Incomplete: go back to data for the missing ranges.
        if not ea.missing:
            for i, s in enumerate(statuses):
                if s == fr.FILE_INCOMPLETE:
                    f = self.manifest.files[i]
                    self.units.mark_missing(i, 0, f.n_units)
        self.end_statuses.clear()
        self.end_sends = 0
        log.info(kv("receiver reports missing data; resuming data phase"))
        self._set_state(SenderState.DATA, "retransmitting missing data")


def units_for_bytes(n: int) -> int:
    """Units needed for ``n`` bytes."""
    return math.ceil(n / UNIT_SIZE)
