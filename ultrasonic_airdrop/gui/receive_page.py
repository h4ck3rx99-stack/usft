"""Receive page: listen, inspect offers, accept/reject, watch per-file progress."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ultrasonic_airdrop.config import PROFILE_NU, PROFILES
from ultrasonic_airdrop.gui.context import AppContext
from ultrasonic_airdrop.gui.send_page import fmt_duration
from ultrasonic_airdrop.gui.widgets.file_list import FileList, human_size
from ultrasonic_airdrop.gui.widgets.signal_meter import SignalMeter
from ultrasonic_airdrop.gui.widgets.spectrum import SpectrumWidget
from ultrasonic_airdrop.gui.widgets.waterfall import WaterfallWidget
from ultrasonic_airdrop.gui.workers import EngineThread, EventBridge
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.engine import Station, build_pipeline
from ultrasonic_airdrop.transport.events import (
    DoneEvent,
    ErrorEvent,
    FileEvent,
    LinkEvent,
    ManifestEvent,
    OfferEvent,
    ProgressEvent,
    SpectrumEvent,
    StateEvent,
)
from ultrasonic_airdrop.transport.session_receiver import ReceiverSession

log = logging.getLogger(__name__)


class ReceivePage(QWidget):
    """Listening controls, live signal view, offer card, file progress."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.bridge = EventBridge()
        self.bridge.event.connect(self._on_event)
        self.engine: EngineThread | None = None
        self.session: ReceiverSession | None = None
        self.offer: OfferEvent | None = None

        root = QVBoxLayout(self)
        title = QLabel("Receive files")
        title.setObjectName("PageTitle")
        root.addWidget(title)

        top = QHBoxLayout()
        top.addWidget(QLabel("Save to:"))
        self.dest = QLineEdit(str(ctx.destination))
        self.dest.editingFinished.connect(lambda: ctx.set_destination(Path(self.dest.text())))
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._choose_dest)
        open_btn = QPushButton("Open folder")
        open_btn.clicked.connect(self._open_folder)
        top.addWidget(self.dest, 1)
        top.addWidget(browse)
        top.addWidget(open_btn)
        root.addLayout(top)

        ctl = QHBoxLayout()
        self.listen_btn = QPushButton("Start listening")
        self.listen_btn.setCheckable(True)
        self.listen_btn.toggled.connect(self._toggle_listen)
        self.status = QLabel("Not listening")
        self.status.setObjectName("StateLabel")
        ctl.addWidget(self.listen_btn)
        ctl.addWidget(self.status, 1)
        root.addLayout(ctl)

        split = QSplitter(Qt.Orientation.Vertical)
        sig = QGroupBox("Signal")
        sl = QVBoxLayout(sig)
        self.spectrum = SpectrumWidget()
        self.spectrum.set_band(PROFILE_NU.band_low_hz, PROFILE_NU.band_high_hz)
        self.waterfall = WaterfallWidget()
        self.meter = SignalMeter()
        self.detected = QLabel("Detected transmitter: none")
        sl.addWidget(self.spectrum, 2)
        sl.addWidget(self.waterfall, 2)
        sl.addWidget(self.meter)
        sl.addWidget(self.detected)
        split.addWidget(sig)

        lower = QWidget()
        ll = QVBoxLayout(lower)
        ll.setContentsMargins(0, 0, 0, 0)
        self.offer_box = QGroupBox("Incoming offer")
        of = QFormLayout(self.offer_box)
        self.offer_text = QLabel()
        self.offer_text.setWordWrap(True)
        self.offer_pass = QLineEdit()
        self.offer_pass.setEchoMode(QLineEdit.EchoMode.Password)
        self.offer_pass.setPlaceholderText("Passphrase for this encrypted transfer")
        btns = QHBoxLayout()
        self.accept_btn = QPushButton("Accept")
        self.reject_btn = QPushButton("Reject")
        self.accept_btn.clicked.connect(lambda: self._decide(True))
        self.reject_btn.clicked.connect(lambda: self._decide(False))
        btns.addWidget(self.accept_btn)
        btns.addWidget(self.reject_btn)
        btns.addStretch(1)
        of.addRow(self.offer_text)
        of.addRow("Passphrase:", self.offer_pass)
        of.addRow(btns)
        self.offer_box.hide()
        ll.addWidget(self.offer_box)
        self.files = FileList()
        ll.addWidget(self.files, 1)
        self.progress = QProgressBar()
        self.progress.setFormat("%p% received")
        ll.addWidget(self.progress)
        self.stats = QLabel("—")
        self.stats.setWordWrap(True)
        ll.addWidget(self.stats)
        bottom = QHBoxLayout()
        self.discard = QCheckBox("Discard partial data when cancelling")
        self.cancel_btn = QPushButton("Cancel transfer")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        bottom.addWidget(self.discard)
        bottom.addStretch(1)
        bottom.addWidget(self.cancel_btn)
        ll.addLayout(bottom)
        split.addWidget(lower)
        split.setSizes([360, 340])
        root.addWidget(split, 1)

    # ---------------------------------------------------------------- controls
    def _choose_dest(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Destination folder", self.dest.text())
        if d:
            self.dest.setText(d)
            self.ctx.set_destination(Path(d))

    def _open_folder(self) -> None:
        dest = Path(self.dest.text())
        dest.mkdir(parents=True, exist_ok=True)
        # Only the destination directory is ever opened; received files are never opened or executed.
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(dest)))

    def _toggle_listen(self, on: bool) -> None:
        if on:
            self._start()
        else:
            self._stop()

    def _start(self) -> None:
        dest = Path(self.dest.text()).expanduser()
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.ctx.banner("error", f"Cannot create the destination folder: {exc}", "Choose another folder.")
            self.listen_btn.setChecked(False)
            return
        self.ctx.set_destination(dest)
        settings = self.ctx.settings
        rate = 48_000
        try:
            backend = self.ctx.make_backend(rate)
            masks: dict[int, CarrierMask] = {}
            pipeline = build_pipeline(rate, settings, masks)
            station = Station(backend, pipeline, settings, self.bridge.sink, spectrum_profile=PROFILE_NU)
            self.session = ReceiverSession(
                station, settings, dest, self.bridge.sink, masks=masks, keep_listening=True, on_profile_lock=None
            )
        except Exception as exc:
            self.ctx.banner("error", f"Could not start listening: {exc}", "Check the Diagnostics page.")
            self.listen_btn.setChecked(False)
            return
        station.session = self.session
        self.engine = EngineThread(station, stop_when_done=False, parent=self)
        self.engine.failed.connect(self._failed)
        self.engine.finished.connect(self._finished)
        self.engine.start()
        self.listen_btn.setText("Stop listening")
        self.status.setText("Listening… (nothing is played until you accept an offer)")
        self.waterfall.clear()

    def _failed(self, msg: str, hint: str) -> None:
        self.ctx.banner("error", msg, hint)

    def _stop(self) -> None:
        if self.engine is not None:
            self.engine.stop()
            self.engine.wait(5000)

    def _finished(self) -> None:
        if self.session is not None:
            self.session.stop()  # engine thread has ended: safe to touch the session here
        self.engine = None
        self.session = None
        self.listen_btn.blockSignals(True)
        self.listen_btn.setChecked(False)
        self.listen_btn.blockSignals(False)
        self.listen_btn.setText("Start listening")
        self.status.setText("Not listening")
        self.cancel_btn.setEnabled(False)
        self.offer_box.hide()

    def _decide(self, accept: bool) -> None:
        if self.session is not None:
            self.session.decide(accept, self.offer_pass.text())
        self.offer_box.hide()

    def _cancel(self) -> None:
        if self.session is not None:
            self.session.cancel(discard=self.discard.isChecked())

    def shutdown(self) -> None:
        """Stop listening (application exit)."""
        self._stop()

    # ---------------------------------------------------------------- events
    def _on_event(self, ev: object) -> None:
        if isinstance(ev, SpectrumEvent):
            self.spectrum.update_spectrum(ev)
            self.waterfall.add(ev)
            self.meter.level.set_value(ev.level_dbfs)
            self.meter.snr.set_value(ev.band_snr_db)
        elif isinstance(ev, LinkEvent):
            self.meter.detect.set_value(ev.detect_ratio)
            prof = PROFILES.get(ev.profile_key)
            self.detected.setText(
                f"Last frame: {'decoded' if ev.ok else 'detected but not decoded'} · profile "
                f"{prof.name if prof else ev.profile_key} · SNR {ev.snr_db:.1f} dB · {ev.ppm:+.0f} ppm"
            )
        elif isinstance(ev, StateEvent):
            self.status.setText(ev.label)
            active = ev.state in ("offered", "receiving")
            self.cancel_btn.setEnabled(active)
        elif isinstance(ev, OfferEvent):
            self.offer = ev
            prof = PROFILES.get(ev.profile_key)
            self.detected.setText(
                f"Detected transmitter: {ev.sender_name or 'unnamed'} "
                f"({prof.name if prof else ev.profile_key}, SNR {ev.snr_db:.1f} dB)"
            )
            self.offer_text.setText(
                f"<b>{ev.sender_name or 'Unnamed sender'}</b> wants to send <b>{ev.file_count} file(s)</b>, "
                f"{human_size(ev.total_bytes)} on air, estimated ≈ {fmt_duration(ev.estimated_s)}"
                f"{' (broadcast: no replies are sent)' if ev.broadcast else ''}."
                f"{'<br><b>Encrypted</b> — enter the shared passphrase.' if ev.encrypted else ''}"
            )
            self.offer_pass.setVisible(ev.encrypted)
            self.offer_box.show()
            self.files.clear()
            self.progress.setValue(0)
        elif isinstance(ev, ManifestEvent):
            self.files.set_rows(list(ev.files))
        elif isinstance(ev, FileEvent):
            pct = f" {100 * ev.received_bytes // ev.total_bytes}%" if ev.total_bytes else ""
            label = {
                "verified": "✔ verified",
                "failed": "✖ failed (hash mismatch)",
                "incomplete": "incomplete",
                "receiving": f"receiving{pct}",
            }.get(ev.status, ev.status)
            self.files.set_status(ev.index, label)
        elif isinstance(ev, ProgressEvent):
            if ev.total_bytes:
                self.progress.setValue(int(100 * ev.done_bytes / ev.total_bytes))
            self.stats.setText(
                f"{human_size(ev.done_bytes)} of {human_size(ev.total_bytes)} · "
                f"{ev.goodput_bps:.1f} B/s · ETA {fmt_duration(ev.eta_s)} · "
                f"frame errors {ev.frame_error_rate:.0%} · duplicates {ev.retransmissions}"
            )
        elif isinstance(ev, ErrorEvent):
            self.ctx.banner("error" if ev.fatal else "warning", ev.message, ev.hint)
        elif isinstance(ev, DoneEvent):
            self.offer_box.hide()
            if ev.success:
                self.progress.setValue(100)
            self.ctx.banner(
                "success" if ev.success else "warning",
                ev.message,
                "" if ev.success else "Partial data is kept; the same files can resume later.",
            )
