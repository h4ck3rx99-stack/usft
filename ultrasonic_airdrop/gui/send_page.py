"""Send page: choose files and options, confirm the airtime estimate, transmit."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ultrasonic_airdrop.config import MODES, PROFILES, TransferOptions, get_profile
from ultrasonic_airdrop.files.chunker import PreparedFile, cleanup_prepared, prepare_files
from ultrasonic_airdrop.files.manifest import Manifest
from ultrasonic_airdrop.gui.context import AppContext
from ultrasonic_airdrop.gui.widgets.file_list import FileList, human_size
from ultrasonic_airdrop.gui.workers import EngineThread, EventBridge, TaskThread
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.engine import Station, build_pipeline
from ultrasonic_airdrop.transport.estimate import estimate_transfer_s
from ultrasonic_airdrop.transport.events import DoneEvent, ErrorEvent, ProgressEvent, StateEvent
from ultrasonic_airdrop.transport.session_sender import SenderSession

log = logging.getLogger(__name__)


def fmt_duration(s: float | None) -> str:
    """Human duration."""
    if s is None or s != s or s == float("inf"):
        return "—"
    s = round(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h} h {m:02d} min" if h else (f"{m} min {sec:02d} s" if m else f"{sec} s")


class SendPage(QWidget):
    """File selection, options and live transfer status."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.bridge = EventBridge()
        self.bridge.event.connect(self._on_event)
        self.engine: EngineThread | None = None
        self.session: SenderSession | None = None
        self.prep: TaskThread | None = None
        self._prepared: list[PreparedFile] = []
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

        root = QVBoxLayout(self)
        title = QLabel("Send files")
        title.setObjectName("PageTitle")
        root.addWidget(title)

        files_box = QGroupBox("Files (drag and drop here)")
        fl = QVBoxLayout(files_box)
        self.files = FileList(accept_drops=True)
        self.files.files_dropped.connect(self._add_paths)
        fl.addWidget(self.files)
        row = QHBoxLayout()
        add = QPushButton("Add files…")
        add.clicked.connect(self._browse)
        rem = QPushButton("Remove selected")
        rem.clicked.connect(self._remove)
        self.total = QLabel("No files selected")
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        row.addWidget(self.total)
        fl.addLayout(row)
        root.addWidget(files_box, 2)

        opts = QGroupBox("Options")
        form = QFormLayout(opts)
        mode_row = QHBoxLayout()
        self.interactive = QRadioButton("Interactive (with acknowledgements)")
        self.broadcast = QRadioButton("Broadcast (one-way)")
        self.interactive.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.interactive)
        grp.addButton(self.broadcast)
        mode_row.addWidget(self.interactive)
        mode_row.addWidget(self.broadcast)
        mode_row.addStretch(1)
        form.addRow("Transfer mode:", mode_row)
        self.profile = QComboBox()
        for p in PROFILES.values():
            self.profile.addItem(p.name, p.key)
        form.addRow("Frequency profile:", self.profile)
        enc_row = QHBoxLayout()
        self.encrypt = QCheckBox("Encrypt")
        self.passphrase = QLineEdit()
        self.passphrase.setEchoMode(QLineEdit.EchoMode.Password)
        self.passphrase.setPlaceholderText("Shared passphrase (agree on it out of band)")
        self.passphrase.setEnabled(False)
        self.encrypt.toggled.connect(self.passphrase.setEnabled)
        enc_row.addWidget(self.encrypt)
        enc_row.addWidget(self.passphrase, 1)
        form.addRow("Encryption:", enc_row)
        self.compress = QCheckBox("Compress files when it saves ≥ 5 %")
        self.compress.setChecked(True)
        form.addRow("Compression:", self.compress)
        lvl = QHBoxLayout()
        self.level = QSlider(Qt.Orientation.Horizontal)
        self.level.setRange(-20, -3)
        self.level.setValue(round(ctx.settings.tx_level_dbfs))
        self.level_label = QLabel()
        self.level.valueChanged.connect(self._level_changed)
        lvl.addWidget(self.level, 1)
        lvl.addWidget(self.level_label)
        form.addRow("Transmit level:", lvl)
        self.name = QLineEdit(str(ctx.qs.value("send/name", "Ultrasonic AirDrop")))
        self.name.setMaxLength(32)
        form.addRow("Your display name:", self.name)
        root.addWidget(opts)

        status = QGroupBox("Transfer")
        sl = QFormLayout(status)
        self.state = QLabel("Idle")
        self.state.setObjectName("StateLabel")
        self.progress = QProgressBar()
        self.progress.setFormat("%p% acknowledged")
        self.stats = QLabel("—")
        self.stats.setWordWrap(True)
        sl.addRow("State:", self.state)
        sl.addRow("Progress:", self.progress)
        sl.addRow("Link:", self.stats)
        root.addWidget(status)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.send_btn = QPushButton("Send…")
        self.send_btn.setDefault(True)
        self.send_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btns.addWidget(self.send_btn)
        btns.addWidget(self.cancel_btn)
        root.addLayout(btns)
        self._level_changed(self.level.value())
        ctx.devices_changed.connect(self._update_profiles)
        self._update_profiles()

    # ---------------------------------------------------------------- helpers
    def _update_profiles(self) -> None:
        idx = self.profile.findData("us")
        model = self.profile.model()
        item = model.item(idx) if isinstance(model, QStandardItemModel) else None
        if item is not None:
            item.setEnabled(self.ctx.us_hardware_ok)
            item.setToolTip("" if self.ctx.us_hardware_ok else "Enable by passing the 96 kHz hardware check")

    def _level_changed(self, v: int) -> None:
        self.level_label.setText(f"{v} dBFS")

    def _browse(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, "Choose files to send")
        self._add_paths([Path(n) for n in names])

    def _add_paths(self, paths: list[Path]) -> None:
        rejected = self.files.add_paths(paths)
        if rejected:
            self.ctx.banner(
                "warning",
                f"Skipped {len(rejected)} item(s) that are not regular files.",
                "Folders are not supported; add the files inside them.",
            )
        self._update_total()

    def _remove(self) -> None:
        self.files.remove_selected()
        self._update_total()

    def _update_total(self) -> None:
        n = len(self.files.paths())
        self.total.setText(f"{n} file(s), {human_size(self.files.total_size())}" if n else "No files selected")

    def _busy(self, busy: bool) -> None:
        self.send_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        for w in (self.interactive, self.broadcast, self.profile, self.encrypt, self.compress, self.level, self.name):
            w.setEnabled(not busy)
        self.passphrase.setEnabled(not busy and self.encrypt.isChecked())

    # ---------------------------------------------------------------- start
    def _start(self) -> None:
        paths = self.files.paths()
        if not paths:
            self.ctx.banner("warning", "No files selected.", "Add files with “Add files…” or drag them in.")
            return
        if self.encrypt.isChecked() and len(self.passphrase.text()) < 1:
            self.ctx.banner("warning", "Encryption is on but the passphrase is empty.", "Type a passphrase.")
            return
        self.ctx.qs.setValue("send/name", self.name.text())
        self._busy(True)
        self.state.setText("Preparing files (hashing/compressing)…")
        compress = self.compress.isChecked()
        self._tmp = tempfile.TemporaryDirectory(prefix="airdrop-send-")
        tmpdir = Path(self._tmp.name)

        def work(_t: TaskThread):
            return prepare_files(paths, compress, tmpdir)

        self.prep = TaskThread(work, self)
        self.prep.done.connect(self._prepared_ok)
        self.prep.error.connect(self._prep_failed)
        self.prep.start()

    def _prep_failed(self, msg: str, hint: str) -> None:
        self.ctx.banner("error", msg, hint)
        self.state.setText("Idle")
        self._busy(False)

    def _prepared_ok(self, result: object) -> None:
        manifest: Manifest
        prepared: list[PreparedFile]
        manifest, prepared = result  # type: ignore[misc]
        self._prepared = prepared
        settings = self.ctx.settings.replace(tx_level_dbfs=float(self.level.value()))
        profile = get_profile(self.profile.currentData())
        broadcast = self.broadcast.isChecked()
        mode = settings.broadcast_mode if broadcast else 1
        est = estimate_transfer_s(
            profile,
            [f.tx_size for f in manifest.files],
            mode,
            settings,
            self.encrypt.isChecked(),
            broadcast,
            None,
            len(manifest.to_blob()),
        )
        est_fast = (
            None
            if broadcast
            else estimate_transfer_s(
                profile,
                [f.tx_size for f in manifest.files],
                5,
                settings,
                self.encrypt.isChecked(),
                False,
                None,
                len(manifest.to_blob()),
            )
        )
        text = (
            f"<b>{len(manifest.files)} file(s), {human_size(manifest.total_size)}</b> "
            f"({human_size(manifest.total_tx)} on air after compression).<br><br>"
            f"Estimated airtime: <b>≈ {fmt_duration(est)}</b> at the robust mode ({MODES[mode].label})"
            + (f"; ≈ {fmt_duration(est_fast)} if the link supports the fastest OFDM mode" if est_fast else "")
            + ".<br>Actual time depends on link quality and retransmissions.<br><br>"
            f"This computer will now play <b>{profile.name}</b> sound at up to <b>{settings.tx_level_dbfs:.0f} dBFS</b>"
            " through the selected speaker. Children and pets may hear it."
        )
        box = QMessageBox(
            QMessageBox.Icon.Question,
            "Start transmitting?",
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            self,
        )
        box.setTextFormat(Qt.TextFormat.RichText)
        if box.exec() != QMessageBox.StandardButton.Yes:
            self._cleanup()
            self.state.setText("Idle")
            self._busy(False)
            return
        options = TransferOptions(
            profile_key=profile.key,
            broadcast=broadcast,
            encrypt=self.encrypt.isChecked(),
            passphrase=self.passphrase.text(),
            compression=self.compress.isChecked(),
            sender_name=self.name.text() or "Ultrasonic AirDrop",
        )
        try:
            backend = self.ctx.make_backend(profile.sample_rate)
            masks: dict[int, CarrierMask] = {}
            station = Station(
                backend, build_pipeline(profile.sample_rate, settings, masks, [profile]), settings, self.bridge.sink
            )
            self.session = SenderSession(station, profile, settings, options, manifest, prepared, self.bridge.sink)
        except Exception as exc:  # e.g. sounddevice/PortAudio missing
            self.ctx.banner("error", f"Could not start: {exc}", "Check the audio devices on the Diagnostics page.")
            self._cleanup()
            self._busy(False)
            return
        station.session = self.session
        self.progress.setValue(0)
        self.engine = EngineThread(station, parent=self)
        self.engine.failed.connect(lambda m, h: self.ctx.banner("error", m, h))
        self.engine.finished.connect(self._engine_finished)
        self.engine.start()

    def _cancel(self) -> None:
        if self.session is not None:
            self.session.request_cancel()
        elif self.prep is not None and self.prep.isRunning():
            self.prep.cancelled = True

    def _engine_finished(self) -> None:
        self._cleanup()
        self._busy(False)
        self.engine = None
        self.session = None

    def _cleanup(self) -> None:
        if self._prepared:
            cleanup_prepared(self._prepared)
            self._prepared = []
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def shutdown(self) -> None:
        """Stop the engine (application exit)."""
        if self.session is not None:
            self.session.request_cancel()
        if self.engine is not None:
            self.engine.stop()
            self.engine.wait(3000)

    # ---------------------------------------------------------------- events
    def _on_event(self, ev: object) -> None:
        if isinstance(ev, StateEvent):
            self.state.setText(ev.label)
        elif isinstance(ev, ProgressEvent):
            if ev.total_bytes:
                self.progress.setValue(int(100 * ev.done_bytes / ev.total_bytes))
            self.stats.setText(
                f"{human_size(ev.done_bytes)} of {human_size(ev.total_bytes)} acknowledged · "
                f"goodput {ev.goodput_bps:.1f} B/s · ETA {fmt_duration(ev.eta_s)} · mode {ev.mode_id} "
                f"({MODES[ev.mode_id].label}) · retransmitted units {ev.retransmissions} · "
                f"frame error rate {ev.frame_error_rate:.0%} · receiver SNR "
                + ("—" if ev.snr_db != ev.snr_db else f"{ev.snr_db:.1f} dB")
            )
        elif isinstance(ev, ErrorEvent):
            self.ctx.banner("error" if ev.fatal else "warning", ev.message, ev.hint)
        elif isinstance(ev, DoneEvent):
            if ev.success:
                self.progress.setValue(100)
            self.ctx.banner(
                "success" if ev.success else "error",
                ev.message,
                "" if ev.success else "Run the Link test on both machines, or try Broadcast mode.",
            )
