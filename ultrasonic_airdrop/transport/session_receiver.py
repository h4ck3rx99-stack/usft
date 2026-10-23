"""Receiver state machine (interactive replies and broadcast collection).

The receiver is reactive: it answers OFFER (after the user's decision), PROBE,
poll-flagged frames / POLL (with ACK) and END (with END_ACK).  It never
transmits except in reply, plus CANCEL on user request or authentication
failure.  Retransmissions are driven entirely by the sender's timeouts.
"""

from __future__ import annotations

import enum
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ultrasonic_airdrop.config import (
    AUTH_FAILURE_LIMIT,
    MAX_RESUME_RANGES,
    PROFILES,
    PROTO_VERSION,
    Profile,
    Settings,
)
from ultrasonic_airdrop.crypto.aead import FrameCipher
from ultrasonic_airdrop.errors import CryptoError, FrameError, ManifestError
from ultrasonic_airdrop.files import resume
from ultrasonic_airdrop.files.manifest import Manifest
from ultrasonic_airdrop.files.reassembly import Reassembly
from ultrasonic_airdrop.link import frames as fr
from ultrasonic_airdrop.link.codec import frame_airtime
from ultrasonic_airdrop.link.receiver_pipeline import RxFrame
from ultrasonic_airdrop.logging_setup import kv
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.phy.quality import QualityTracker
from ultrasonic_airdrop.transport import estimate
from ultrasonic_airdrop.transport.events import (
    DoneEvent,
    ErrorEvent,
    EventSink,
    FileEvent,
    LinkEvent,
    ManifestEvent,
    OfferEvent,
    ProgressEvent,
    StateEvent,
)
from ultrasonic_airdrop.transport.linkio import LinkIO, OutFrame, Radio
from ultrasonic_airdrop.transport.rate_adapt import recommend
from ultrasonic_airdrop.transport.session_sender import capabilities

log = logging.getLogger(__name__)
PROGRESS_INTERVAL_S = 1.0
ENDACK_STATUSES_PER_FRAME = 100


class ReceiverState(enum.Enum):
    """Receiver states."""

    LISTENING = "listening"
    OFFERED = "offered"
    RECEIVING = "receiving"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


LABELS = {
    ReceiverState.LISTENING: "Listening…",
    ReceiverState.OFFERED: "Incoming offer — waiting for your decision",
    ReceiverState.RECEIVING: "Receiving…",
    ReceiverState.COMPLETE: "Complete",
    ReceiverState.FAILED: "Failed",
    ReceiverState.CANCELLED: "Cancelled",
    ReceiverState.INTERRUPTED: "Interrupted (partial data kept)",
}
TERMINAL = {ReceiverState.COMPLETE, ReceiverState.FAILED, ReceiverState.CANCELLED, ReceiverState.INTERRUPTED}


@dataclass
class _Session:
    session_id: int
    offer: fr.Offer
    profile: Profile
    io: LinkIO
    started: float
    last_activity: float
    accepted: bool = False
    decided: bool = False
    parts: dict[int, bytes] = field(default_factory=dict)
    part_count: int = 0
    manifest: Manifest | None = None
    reassembly: Reassembly | None = None
    received_seqs: set[int] = field(default_factory=set)
    max_seq: int = -1
    quality: QualityTracker = field(default_factory=QualityTracker)
    auth_failures: int = 0
    resume_pending: bool = False
    ack_deadline: float | None = None
    finished_at: float | None = None
    end_seen: bool = False
    data_frames: int = 0
    duplicate_frames: int = 0
    last_progress: float = 0.0

    @property
    def broadcast(self) -> bool:
        return self.offer.transfer_mode == fr.TRANSFER_BROADCAST


class ReceiverSession:
    """Listens for offers and receives transfers into ``dest``."""

    role = "receiver"

    def __init__(
        self,
        radio: Radio,
        settings: Settings,
        dest: Path,
        events: EventSink,
        masks: dict[int, CarrierMask] | None = None,
        passphrase: str = "",
        accept_all: bool = False,
        keep_listening: bool = False,
        on_profile_lock: object = None,
    ) -> None:
        self.radio = radio
        self.settings = settings
        self.dest = Path(dest)
        self.events = events
        self.masks: dict[int, CarrierMask] = masks if masks is not None else {}
        self.passphrase = passphrase
        self.accept_all = accept_all
        self.keep_listening = keep_listening
        self.on_profile_lock = on_profile_lock
        self.state = ReceiverState.LISTENING
        self.sess: _Session | None = None
        self.replies: list[OutFrame] = []
        self.reply_at = 0.0
        self.discard_on_cancel = False
        self._lock = threading.Lock()
        self._decision: tuple[bool, str] | None = None
        self._cancel_req: bool | None = None
        self.results: list[tuple[str, str, str]] = []
        self.message = ""
        self._stopped = False
        self.events(StateEvent(self.role, self.state.value, LABELS[self.state]))

    # ------------------------------------------------------------------ public API
    @property
    def done(self) -> bool:
        """True when a one-shot receiver has finished its session (incl. linger)."""
        return self._stopped

    @property
    def idle(self) -> bool:
        """No transfer in progress (listening, or the last session has ended)."""
        return self.state == ReceiverState.LISTENING or self.state in TERMINAL

    @property
    def success(self) -> bool:
        """Last session completed with every file verified."""
        return self.state == ReceiverState.COMPLETE

    def decide(self, accept: bool, passphrase: str = "") -> None:
        """User decision on the pending offer (thread-safe)."""
        with self._lock:
            self._decision = (accept, passphrase)

    def cancel(self, discard: bool = False) -> None:
        """User cancel of the active session (thread-safe)."""
        with self._lock:
            self._cancel_req = discard

    def finish_recording(self) -> None:
        """End of a recording/broadcast: summarise what was verified (used by WAV decoding)."""
        s = self.sess
        if s is None or self.state in TERMINAL:
            return
        if s.reassembly is not None:
            self._broadcast_summary(final=True)
        else:
            self._terminal(ReceiverState.INTERRUPTED, "The manifest could not be recovered from the recording")

    def stop(self) -> None:
        """Stop listening: interrupt an active session (partials kept)."""
        s = self.sess
        if s is not None and self.state not in TERMINAL and self.state != ReceiverState.LISTENING:
            self._terminal(ReceiverState.INTERRUPTED, "Stopped listening; partial data kept for resume")
        if s is not None and s.reassembly is not None:
            s.reassembly.close()
        self._stopped = True

    # ------------------------------------------------------------------ internals
    def _set_state(self, state: ReceiverState, detail: str = "") -> None:
        if state != self.state:
            log.info(kv("receiver state", old=self.state.value, new=state.value, detail=detail))
        self.state = state
        self.events(StateEvent(self.role, state.value, LABELS[state] + (f" ({detail})" if detail else "")))

    def _queue(self, frames: list[OutFrame], now: float) -> None:
        # replace queued frames of the same type (no duplicate ACKs etc.)
        types = {f.ftype for f in frames}
        self.replies = [f for f in self.replies if f.ftype not in types] + frames
        self.reply_at = max(self.reply_at, now + self.settings.reply_delay_s) if self.replies else 0.0

    def _terminal(self, state: ReceiverState, message: str) -> None:
        s = self.sess
        self.message = message
        if s is not None:
            s.finished_at = self.radio.now()
            s.ack_deadline = None
            if s.reassembly is not None:
                discard = state == ReceiverState.CANCELLED and self.discard_on_cancel
                s.reassembly.close(discard=discard)
        self._set_state(state, message)
        stats: dict[str, float] = {}
        if s is not None:
            elapsed = max(1e-9, self.radio.now() - s.started)
            got = s.reassembly.received_tx() if s.reassembly else 0
            stats = {
                "elapsed_s": elapsed,
                "received_bytes": float(got),
                "frames": float(s.quality.total_frames),
                "frame_errors": float(s.quality.total_errors),
                "duplicate_frames": float(s.duplicate_frames),
            }
        self.events(DoneEvent(self.role, state == ReceiverState.COMPLETE, message, tuple(self.results), stats))

    # ------------------------------------------------------------------ tick
    def tick(self, now: float) -> None:
        """Advance timers, apply user decisions and send queued replies."""
        with self._lock:
            decision, self._decision = self._decision, None
            cancel_req, self._cancel_req = self._cancel_req, None
        s = self.sess
        if decision is not None and s is not None and self.state == ReceiverState.OFFERED:
            self._apply_decision(decision[0], decision[1], now)
        if cancel_req is not None and s is not None and self.state not in TERMINAL:
            self.discard_on_cancel = cancel_req
            if not s.broadcast and self.state != ReceiverState.LISTENING:
                c = fr.Cancel(fr.CANCEL_USER, "cancelled by receiver")
                s.io.send([s.io.build(c), s.io.build(c)])
                self.replies = []
            self._terminal(ReceiverState.CANCELLED, "Cancelled by receiver")
        s = self.sess
        if s is None:
            return
        # Replies.
        if self.replies and now >= self.reply_at and not self.radio.tx_busy() and not self.radio.rx_busy():
            frames, self.replies = self.replies, []
            s.io.send(frames)
        # Window timeout (poll frame lost): ACK anyway.
        if (
            s.ack_deadline is not None
            and now >= s.ack_deadline
            and not self.radio.rx_busy()
            and self.state == ReceiverState.RECEIVING
        ):
            s.ack_deadline = None
            log.info(kv("window timeout, sending ACK without poll"))
            self._queue([self._build_ack(s)], now)
        # Inactivity / linger.
        if self.state in (ReceiverState.OFFERED, ReceiverState.RECEIVING):
            limit = self.settings.rx_inactivity_s
            if now - s.last_activity > limit:
                if s.reassembly is not None and s.broadcast:
                    self._broadcast_summary(final=True)
                    return
                self._terminal(
                    ReceiverState.INTERRUPTED,
                    f"No signal for {limit:.0f} s; partial data kept for resume",
                )
        elif self.state in TERMINAL and s.finished_at is not None:
            if now - s.finished_at >= self.settings.linger_s and not self.replies and not self.radio.tx_busy():
                if self.keep_listening:
                    self.sess = None
                    self._set_state(ReceiverState.LISTENING)
                else:
                    self._stopped = True

    def _apply_decision(self, accept: bool, passphrase: str, now: float) -> None:
        s = self.sess
        assert s is not None
        s.decided = True
        if not accept:
            if not s.broadcast:
                self._queue([s.io.build(fr.Accept(fr.ACCEPT_REJECTED, capabilities(self.settings), False, 0))], now)
            self._terminal(ReceiverState.CANCELLED, "Offer declined")
            return
        if s.offer.encrypted:
            phrase = passphrase or self.passphrase
            if not phrase:
                self.events(ErrorEvent("This transfer is encrypted.", "Enter the passphrase before accepting."))
                s.decided = False
                return
            s.io.set_cipher(FrameCipher.from_passphrase(phrase, s.offer.salt, s.session_id))
        s.accepted = True
        s.last_activity = now
        self._set_state(ReceiverState.RECEIVING, "waiting for manifest")
        if not s.broadcast:
            self._queue([self._build_accept(s)], now)

    def _build_accept(self, s: _Session) -> OutFrame:
        resume_flag = resume.manifest_seen(self.dest / ".airdrop-partial", s.offer.manifest_sha256.hex())
        return s.io.build(
            fr.Accept(fr.ACCEPT_OK, capabilities(self.settings), resume_flag, self.settings.max_accept_bytes)
        )

    # ------------------------------------------------------------------ frames
    def on_frame(self, rx: RxFrame, now: float) -> None:
        """Handle one frame from the pipeline."""
        info = rx.info
        mode_id = info.header.mode_id if info.header is not None else None
        carrier = tuple(float(v) for v in info.carrier_snr_db) if info.carrier_snr_db is not None else None
        if info.header is not None:
            self.events(
                LinkEvent(
                    info.profile_key,
                    rx.ok,
                    info.snr_db,
                    info.ppm,
                    info.detect_ratio,
                    mode_id if mode_id is not None else -1,
                    carrier,
                )
            )
        s = self.sess
        if rx.body is None:
            if s is not None and s.accepted and info.profile_key == s.profile.key:
                s.quality.record_frame(False, mode_id, float("nan"), float("nan"), None)
                s.last_activity = now
            return
        try:
            frame = fr.LinkFrame.unpack(rx.body)
        except fr.UnknownFrameTypeError as exc:
            log.info(kv("ignoring unknown frame type", error=exc.user_message))
            return
        except FrameError as exc:
            self._maybe_version_reply(rx, now)
            log.info(kv("discarding invalid frame", error=exc.user_message))
            if s is not None and s.accepted:
                s.quality.record_frame(False, mode_id, float("nan"), float("nan"), None)
            return
        if frame.direction != fr.DIR_S2R:
            return
        if frame.ftype == fr.FrameType.OFFER and (
            s is None
            or (
                frame.session_id != s.session_id
                and (self.state in TERMINAL or self.state in (ReceiverState.LISTENING, ReceiverState.OFFERED))
            )
        ):
            self._new_session(frame, rx, now)
            return
        if s is None or frame.session_id != s.session_id or info.profile_key != s.profile.key:
            return
        s.last_activity = now
        try:
            payload = s.io.open(frame)
        except CryptoError:
            s.auth_failures += 1
            log.warning(kv("authentication failure", count=s.auth_failures, type=frame.ftype.name))
            if s.auth_failures >= AUTH_FAILURE_LIMIT and self.state not in TERMINAL:
                self.events(
                    ErrorEvent(
                        "Passphrase mismatch or corrupted data",
                        "Make sure both sides typed exactly the same passphrase.",
                    )
                )
                if not s.broadcast:
                    c = fr.Cancel(fr.CANCEL_AUTH, "passphrase mismatch")
                    self._queue([s.io.build(c), s.io.build(c)], now)
                self._terminal(ReceiverState.FAILED, "Passphrase mismatch or corrupted data")
            return
        except FrameError as exc:
            log.info(kv("bad payload", type=frame.ftype.name, error=exc.user_message))
            s.quality.record_frame(False, mode_id, float("nan"), float("nan"), None)
            return
        s.auth_failures = 0
        s.received_seqs.add(frame.seq)
        s.max_seq = max(s.max_seq, frame.seq)
        if len(s.received_seqs) > 4096:
            floor = s.max_seq - 1024
            s.received_seqs = {q for q in s.received_seqs if q >= floor}
        if s.accepted:
            s.quality.record_frame(True, mode_id, info.snr_db, info.mfsk_snr_db, info.carrier_snr_db)
        self._dispatch(s, frame, payload, rx, now)

    def _maybe_version_reply(self, rx: RxFrame, now: float) -> None:
        body = rx.body or b""
        version = fr.LinkFrame.peek_version(body)
        if version is None or version == PROTO_VERSION or len(body) < 8 or body[2] != fr.FrameType.OFFER:
            return
        session_id = int.from_bytes(body[4:8], "little")
        profile = PROFILES[rx.info.profile_key]
        io = LinkIO(self.radio, profile, self.settings, fr.DIR_R2S, session_id)
        log.warning(kv("protocol version mismatch", peer_version=version))
        self.events(
            ErrorEvent(
                f"Sender uses protocol version {version}; this app speaks {PROTO_VERSION}.",
                "Update both machines to the same version.",
            )
        )
        io.send([io.build(fr.Cancel(fr.CANCEL_VERSION, f"version {PROTO_VERSION} required"))])

    def _new_session(self, frame: fr.LinkFrame, rx: RxFrame, now: float) -> None:
        try:
            offer = fr.Offer.unpack(frame.payload)
        except FrameError as exc:
            log.info(kv("bad OFFER", error=exc.user_message))
            return
        old = self.sess
        if old is not None and old.reassembly is not None and self.state not in TERMINAL:
            old.reassembly.close()
        profile = PROFILES[rx.info.profile_key]
        io = LinkIO(self.radio, profile, self.settings, fr.DIR_R2S, frame.session_id)
        s = _Session(frame.session_id, offer, profile, io, started=now, last_activity=now)
        s.received_seqs.add(frame.seq)
        s.max_seq = frame.seq
        self.sess = s
        self.replies = []
        self.results = []
        if callable(self.on_profile_lock):
            self.on_profile_lock(profile.key)
        log.info(
            kv(
                "offer received",
                session=f"{frame.session_id:08x}",
                sender=offer.sender_name,
                files=offer.file_count,
                bytes=offer.total_bytes,
                encrypted=offer.encrypted,
                broadcast=offer.transfer_mode == fr.TRANSFER_BROADCAST,
                profile=profile.key,
                snr=rx.info.snr_db,
            )
        )
        if offer.version != PROTO_VERSION:
            self._queue([io.build(fr.Cancel(fr.CANCEL_VERSION, f"version {PROTO_VERSION} required"))], now)
            self._terminal(ReceiverState.FAILED, f"Sender uses unsupported protocol version {offer.version}")
            return
        est = estimate.estimate_transfer_s(
            profile,
            [offer.total_bytes],
            1,
            self.settings,
            offer.encrypted,
            s.broadcast,
            None,
            offer.manifest_len,
        )
        self.events(
            OfferEvent(
                frame.session_id,
                offer.sender_name,
                offer.file_count,
                offer.total_bytes,
                est,
                offer.encrypted,
                s.broadcast,
                profile.key,
                rx.info.snr_db,
            )
        )
        if offer.total_bytes > self.settings.max_accept_bytes:
            if not s.broadcast:
                self._queue(
                    [
                        s.io.build(
                            fr.Accept(
                                fr.ACCEPT_TOO_LARGE,
                                capabilities(self.settings),
                                False,
                                self.settings.max_accept_bytes,
                            )
                        )
                    ],
                    now,
                )
            self._terminal(ReceiverState.FAILED, "Offer exceeds the maximum accepted size")
            return
        self._set_state(ReceiverState.OFFERED, f"from {offer.sender_name or 'unknown'}")
        auto = self.accept_all or (
            self.settings.auto_accept_bytes > 0 and offer.total_bytes <= self.settings.auto_accept_bytes
        )
        if auto and (not offer.encrypted or self.passphrase):
            self._apply_decision(True, self.passphrase, now)

    def _dispatch(self, s: _Session, frame: fr.LinkFrame, payload: fr.Payload, rx: RxFrame, now: float) -> None:
        if isinstance(payload, fr.Cancel):
            if payload.reason == fr.CANCEL_AUTH:
                self.events(
                    ErrorEvent(
                        "Passphrase mismatch or corrupted data",
                        "Make sure both sides typed exactly the same passphrase.",
                    )
                )
                self._terminal(ReceiverState.FAILED, "Passphrase mismatch or corrupted data (reported by sender)")
            elif payload.reason == fr.CANCEL_VERSION:
                self._terminal(ReceiverState.FAILED, "Protocol version mismatch (reported by sender)")
            else:
                self._terminal(ReceiverState.CANCELLED, f"Sender cancelled: {payload.text or 'no reason given'}")
            return
        if isinstance(payload, fr.Offer):
            if s.accepted and not s.broadcast and self.state == ReceiverState.RECEIVING and s.manifest is None:
                self._queue([self._build_accept(s)], now)  # our ACCEPT was lost
            return
        if not s.accepted:
            return
        if self.state in TERMINAL and (s.broadcast or not isinstance(payload, fr.End | fr.Poll)):
            return  # finished: only re-answer END/POLL retransmissions of an interactive sender
        if isinstance(payload, fr.Probe):
            self._on_probe(s, rx, now)
        elif isinstance(payload, fr.ManifestPart):
            self._on_manifest_part(s, payload, now)
        elif isinstance(payload, fr.Data):
            self._on_data(s, payload, now)
        elif isinstance(payload, fr.End):
            self._on_end(s, now)
            return
        if s.broadcast:
            return
        if frame.poll and isinstance(payload, fr.ManifestPart | fr.Data | fr.Poll):
            s.ack_deadline = None
            frames: list[OutFrame] = []
            if s.resume_pending and s.reassembly is not None:
                frames.extend(self._build_resume(s))
                s.resume_pending = False
            frames.append(self._build_ack(s))
            self._queue(frames, now)
        elif isinstance(payload, fr.ManifestPart | fr.Data):
            mode_id = rx.info.header.mode_id if rx.info.header else 0
            n_en = None
            if rx.info.header is not None and rx.info.header.mask_id:
                m = self.masks.get(rx.info.header.mask_id)
                n_en = m.count if m else None
            air = frame_airtime(s.profile, mode_id, len(rx.body or b""), n_en)
            s.ack_deadline = now + air + self.settings.rx_window_extra_s

    def _on_probe(self, s: _Session, rx: RxFrame, now: float) -> None:
        info = rx.info
        carrier = info.probe_snr_db if info.probe_snr_db is not None else np.full(52, np.nan)
        s.quality.record_probe(carrier, info.mfsk_snr_db)
        mode, mask = recommend(
            s.quality.effective_carrier_snr(),
            s.quality.mfsk_snr_db,
            capabilities(self.settings) & s.offer.capabilities,
        )
        self.masks[mask.mask_id] = mask
        report = fr.ProbeReport(
            carrier_snr=tuple(fr.snr_to_u8(float(v)) for v in carrier),
            mfsk_snr=fr.db_to_i16(info.mfsk_snr_db),
            recommended_mode=mode,
            mask=mask.to_bytes(),
            ppm=int(max(-32767, min(32767, round(info.ppm)))),
        )
        log.info(
            kv(
                "probe measured",
                mode=mode,
                carriers=mask.count,
                mfsk_snr=info.mfsk_snr_db,
                median_carrier_snr=float(np.nanmedian(carrier)) if np.isfinite(carrier).any() else float("nan"),
                ppm=info.ppm,
            )
        )
        self._queue([s.io.build(report)], now)

    def _on_manifest_part(self, s: _Session, part: fr.ManifestPart, now: float) -> None:
        if s.manifest is not None:
            return
        if s.part_count and part.part_count != s.part_count:
            log.warning(kv("inconsistent manifest part count"))
            return
        s.part_count = part.part_count
        s.parts[part.part_index] = part.data
        if len(s.parts) < s.part_count:
            return
        blob = b"".join(s.parts[i] for i in range(s.part_count))
        try:
            if len(blob) != s.offer.manifest_len or Manifest.blob_sha256(blob) != s.offer.manifest_sha256:
                raise ManifestError("manifest hash does not match the offer")
            manifest = Manifest.from_blob(blob, self.settings.max_accept_bytes)
            if len(manifest.files) != s.offer.file_count or manifest.total_tx != s.offer.total_bytes:
                raise ManifestError("manifest does not match the offer")
            reasm = Reassembly(self.dest, manifest, s.session_id, s.offer.manifest_sha256.hex())
        except (ManifestError, OSError) as exc:
            msg = exc.user_message if isinstance(exc, ManifestError) else f"cannot write to destination: {exc}"
            log.error(kv("manifest rejected", error=msg))
            if not s.broadcast:
                c = fr.Cancel(fr.CANCEL_ERROR, msg[:60])
                self._queue([s.io.build(c), s.io.build(c)], now)
            self.events(ErrorEvent(msg, "The transfer was cancelled."))
            self._terminal(ReceiverState.FAILED, msg)
            return
        s.manifest = manifest
        s.reassembly = reasm
        s.resume_pending = any(a.resumed_units > 0 for a in reasm.unique())
        self.events(ManifestEvent(tuple((f.index, f.name, f.size) for f in manifest.files)))
        for f in manifest.files:
            asm = reasm.assembler(f.index)
            self.events(FileEvent(f.index, f.name, "receiving", asm.received_bytes(), f.tx_size))
        self._set_state(ReceiverState.RECEIVING, f"{len(manifest.files)} file(s)")
        if s.broadcast:
            for f in manifest.files:
                if reasm.assembler(f.index).complete:
                    self._finalize_one(s, f.index)

    def _on_data(self, s: _Session, d: fr.Data, now: float) -> None:
        reasm = s.reassembly
        if reasm is None or s.manifest is None:
            return
        s.data_frames += 1
        try:
            asm = reasm.assembler(d.file_index)
            new = asm.write_units(d.start_unit, d.unit_count, d.data)
        except (IndexError, ValueError) as exc:
            log.warning(kv("invalid DATA frame", file=d.file_index, start=d.start_unit, error=exc))
            return
        except OSError as exc:
            log.error(kv("write error", error=exc))
            self.events(ErrorEvent(f"Cannot write received data: {exc}", "Check free disk space and permissions."))
            return
        if new == 0:
            s.duplicate_frames += 1
        if s.broadcast and asm.complete and asm.status is None:
            self._finalize_one(s, d.file_index)
        if now - s.last_progress >= PROGRESS_INTERVAL_S:
            s.last_progress = now
            self._emit_progress(s, now)

    def _emit_progress(self, s: _Session, now: float) -> None:
        reasm = s.reassembly
        if reasm is None:
            return
        got = reasm.received_tx()
        elapsed = max(1e-9, now - s.started)
        q = s.quality
        fer = q.total_errors / q.total_frames if q.total_frames else 0.0
        goodput = got / elapsed
        remaining = reasm.total_tx - got
        self.events(
            ProgressEvent(
                self.role,
                got,
                reasm.total_tx,
                goodput,
                remaining / goodput if goodput > 0 else None,
                q.last_mode,
                s.duplicate_frames,
                fer,
                q.mfsk_snr_db,
                elapsed,
            )
        )
        assert s.manifest is not None
        for f in s.manifest.files:
            asm = reasm.assembler(f.index)
            if asm.status is None:
                self.events(FileEvent(f.index, f.name, "receiving", asm.received_bytes(), f.tx_size))

    def _finalize_one(self, s: _Session, index: int) -> None:
        reasm = s.reassembly
        assert reasm is not None and s.manifest is not None
        st = reasm.finalize_file(index)
        asm = reasm.assembler(index)
        for f in s.manifest.files:
            if reasm.assembler(f.index) is asm:
                k = [i for i, a in enumerate(reasm.assemblers) if a is asm].index(f.index)
                path = asm.final_paths[k] if st == fr.FILE_OK and k < len(asm.final_paths) else None
                label = {fr.FILE_OK: "verified", fr.FILE_HASH_MISMATCH: "failed"}.get(st, "incomplete")
                self.events(
                    FileEvent(
                        f.index,
                        f.name,
                        label,
                        asm.received_bytes(),
                        f.tx_size,
                        path,
                        fr.FILE_STATUS_NAMES.get(st, ""),
                    )
                )

    def _collect_results(self, s: _Session) -> list[int]:
        reasm = s.reassembly
        assert reasm is not None
        results = reasm.finalize_all()
        self.results = [(r.name, fr.FILE_STATUS_NAMES[r.status], str(r.path) if r.path else "") for r in results]
        for r in results:
            f = s.manifest.files[r.index] if s.manifest else None
            label = {
                fr.FILE_OK: "verified",
                fr.FILE_HASH_MISMATCH: "failed",
                fr.FILE_WRITE_ERROR: "failed",
            }.get(r.status, "incomplete")
            if f is not None:
                self.events(
                    FileEvent(
                        r.index,
                        r.name,
                        label,
                        reasm.assembler(r.index).received_bytes(),
                        f.tx_size,
                        r.path,
                        fr.FILE_STATUS_NAMES[r.status],
                    )
                )
        return [r.status for r in results]

    def _on_end(self, s: _Session, now: float) -> None:
        if s.reassembly is None:
            if not s.broadcast:
                log.info(kv("END before manifest; ignoring"))
            return
        if s.broadcast:
            if self.state in TERMINAL:
                return
            s.end_seen = True
            if s.reassembly.complete():
                self._broadcast_summary(final=True)
            return
        statuses = self._collect_results(s)
        missing: list[fr.UnitRange] = []
        for asm in s.reassembly.unique():
            if asm.status is None or asm.status == fr.FILE_INCOMPLETE or not asm.complete:
                for start, count in asm.missing_ranges():
                    while count > 0 and len(missing) < 8:
                        c = min(count, 0xFFFF)
                        missing.append(fr.UnitRange(asm.entry.index, start, c))
                        start += c
                        count -= c
        frames: list[OutFrame] = []
        n = len(statuses)
        for first in range(0, max(n, 1), ENDACK_STATUSES_PER_FRAME):
            chunk = tuple(statuses[first : first + ENDACK_STATUSES_PER_FRAME])
            last = first + ENDACK_STATUSES_PER_FRAME >= n
            frames.append(s.io.build(fr.EndAck(n, first, chunk, tuple(missing) if last else ())))
        self._queue(frames, now)
        if all(st == fr.FILE_OK for st in statuses):
            if self.state != ReceiverState.COMPLETE:
                self._terminal(ReceiverState.COMPLETE, f"{n} file(s) received and verified")
        elif any(st in (fr.FILE_HASH_MISMATCH, fr.FILE_WRITE_ERROR) for st in statuses):
            if self.state not in TERMINAL:
                self._terminal(ReceiverState.FAILED, "One or more files failed verification")
        else:
            log.info(kv("END with missing data", ranges=len(missing)))

    def _broadcast_summary(self, final: bool) -> None:
        s = self.sess
        assert s is not None and s.reassembly is not None
        if self.state in TERMINAL:
            return
        statuses = self._collect_results(s)
        ok = sum(1 for st in statuses if st == fr.FILE_OK)
        if ok == len(statuses):
            self._terminal(ReceiverState.COMPLETE, f"{ok} file(s) received and verified")
        elif final:
            self._terminal(
                ReceiverState.INTERRUPTED,
                f"Broadcast ended: {ok} of {len(statuses)} file(s) verified; partial data kept for a later pass",
            )

    def _build_ack(self, s: _Session) -> OutFrame:
        q = s.quality
        base = max(0, s.max_seq - 63)
        bitmap = 0
        for seq in s.received_seqs:
            if base <= seq < base + 64:
                bitmap |= 1 << (seq - base)
        caps = capabilities(self.settings) & s.offer.capabilities
        mode, mask = recommend(q.effective_carrier_snr(), q.mfsk_snr_db, caps)
        if q.window_frames and q.window_fer > 0.25:
            mode = min(mode, max(0, q.last_mode - 1))
        self.masks[mask.mask_id] = mask
        missing: list[fr.UnitRange] = []
        if s.reassembly is not None:
            for asm in s.reassembly.unique():
                if asm.n_units == 0 or not asm.bitmap.any():
                    continue
                last = int(np.flatnonzero(asm.bitmap)[-1])
                for start, count in asm.missing_ranges():
                    if start < last and len(missing) < 4:
                        missing.append(fr.UnitRange(asm.entry.index, start, min(count, last - start, 0xFFFF)))
        snr = q.window_mean_snr()
        if math.isnan(snr):
            snr = q.mfsk_snr_db
        ack = fr.Ack(base, bitmap, fr.db_to_i16(snr), q.window_errors, mode, mask.to_bytes(), tuple(missing))
        log.debug(kv("ack", base=base, bitmap=hex(bitmap), mode=mode, carriers=mask.count, fer=q.window_errors))
        q.reset_window()
        return s.io.build(ack)

    def _build_resume(self, s: _Session) -> list[OutFrame]:
        assert s.reassembly is not None
        per = min(MAX_RESUME_RANGES, (estimate.max_control_payload(s.profile, s.io.cipher is not None) - 3) // 8)
        frames = []
        for asm in s.reassembly.unique():
            ranges = asm.received_ranges()
            for i in range(0, len(ranges), per):
                frames.append(s.io.build(fr.ResumeInfo(asm.entry.index, tuple(ranges[i : i + per]))))
        log.info(kv("sending resume info", frames=len(frames)))
        return frames
