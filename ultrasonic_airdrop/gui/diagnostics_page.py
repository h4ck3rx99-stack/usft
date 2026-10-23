"""Diagnostics: device selection, hardware self-check, link test, test tone, troubleshooting."""

from __future__ import annotations

import logging
import math

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ultrasonic_airdrop.audio.diagnostics import (
    TROUBLESHOOTING,
    HardwareCheckResult,
    LinkTestReceiver,
    LinkTestSender,
    make_test_tone,
    run_hardware_check,
)
from ultrasonic_airdrop.audio.sd_backend import SoundDeviceBackend
from ultrasonic_airdrop.config import MODES, PROFILES, REQUIRED_SNR_DB, get_profile
from ultrasonic_airdrop.gui.context import AppContext
from ultrasonic_airdrop.gui.widgets.plots import CarrierBars, ResponsePlot
from ultrasonic_airdrop.gui.workers import EngineThread, EventBridge, TaskThread
from ultrasonic_airdrop.phy.ofdm import CarrierMask
from ultrasonic_airdrop.transport.engine import Station, build_pipeline
from ultrasonic_airdrop.transport.events import ErrorEvent, LinkTestEvent

log = logging.getLogger(__name__)


class DiagnosticsPage(QWidget):
    """All hardware-facing tools.  Every action that plays sound says so first."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.check_task: TaskThread | None = None
        self.link_engine: EngineThread | None = None
        self.link_session: LinkTestSender | LinkTestReceiver | None = None
        self.tone_backend: SoundDeviceBackend | None = None
        self.bridge = EventBridge()
        self.bridge.event.connect(self._on_event)

        outer = QVBoxLayout(self)
        title = QLabel("Diagnostics")
        title.setObjectName("PageTitle")
        outer.addWidget(title)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        root = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)

        # Devices --------------------------------------------------------------
        dev = QGroupBox("Audio devices")
        df = QFormLayout(dev)
        self.in_combo = QComboBox()
        self.out_combo = QComboBox()
        self.exclusive = QCheckBox("Use WASAPI exclusive mode (bypasses Windows audio effects; falls back to shared)")
        self.exclusive.setChecked(ctx.settings.exclusive_mode)
        self.exclusive.toggled.connect(lambda v: ctx.update_settings(exclusive_mode=v))
        refresh = QPushButton("Refresh device list")
        refresh.clicked.connect(self.refresh_devices)
        df.addRow("Microphone (input):", self.in_combo)
        df.addRow("Speaker (output):", self.out_combo)
        df.addRow(self.exclusive)
        df.addRow(refresh)
        self.in_combo.currentIndexChanged.connect(lambda _i: ctx.set_device("input", self.in_combo.currentData()))
        self.out_combo.currentIndexChanged.connect(lambda _i: ctx.set_device("output", self.out_combo.currentData()))
        root.addWidget(dev)

        # Hardware check ---------------------------------------------------------
        hw = QGroupBox("Hardware self-check (one machine)")
        hl = QVBoxLayout(hw)
        desc = QLabel(
            "Plays 1 s of silence and two 3-second sweeps from 2 kHz to the top of the band through the selected "
            "speaker while recording the microphone, then estimates the speaker→mic response in each profile band."
        )
        desc.setWordWrap(True)
        hl.addWidget(desc)
        row = QHBoxLayout()
        self.rate = QComboBox()
        self.rate.addItem("48 kHz (near-ultrasonic + audible)", 48_000)
        self.rate.addItem("96 kHz (ultrasonic experimental)", 96_000)
        self.check_level = QSpinBox()
        self.check_level.setRange(-20, -3)
        self.check_level.setValue(-6)
        self.check_level.setSuffix(" dBFS")
        self.check_btn = QPushButton("Run hardware check…")
        self.check_btn.clicked.connect(self._run_check)
        row.addWidget(QLabel("Rate:"))
        row.addWidget(self.rate)
        row.addWidget(QLabel("Level:"))
        row.addWidget(self.check_level)
        row.addStretch(1)
        row.addWidget(self.check_btn)
        hl.addLayout(row)
        self.plot = ResponsePlot()
        hl.addWidget(self.plot)
        self.verdict = QLabel("No check run yet.")
        self.verdict.setMinimumWidth(100)
        self.verdict.setTextFormat(Qt.TextFormat.RichText)
        self.verdict.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.verdict.setWordWrap(True)
        hl.addWidget(self.verdict)
        root.addWidget(hw)

        # Link test --------------------------------------------------------------
        lt = QGroupBox("Link test (two machines)")
        ll = QVBoxLayout(lt)
        lt_desc = QLabel(
            "Start “Receiver” on one machine, then “Sender” on the other. The sender plays PROBE "
            "frames for the chosen duration."
        )
        lt_desc.setWordWrap(True)
        ll.addWidget(lt_desc)
        lrow = QHBoxLayout()
        self.role = QComboBox()
        self.role.addItem("Receiver (listen)", "receive")
        self.role.addItem("Sender (plays probes)", "send")
        self.lt_profile = QComboBox()
        for p in PROFILES.values():
            self.lt_profile.addItem(p.name, p.key)
        self.lt_duration = QSpinBox()
        self.lt_duration.setRange(5, 300)
        self.lt_duration.setValue(20)
        self.lt_duration.setSuffix(" s")
        self.lt_btn = QPushButton("Start link test")
        self.lt_btn.setCheckable(True)
        self.lt_btn.toggled.connect(self._toggle_link_test)
        for w in (
            QLabel("Role:"),
            self.role,
            QLabel("Profile:"),
            self.lt_profile,
            QLabel("Duration:"),
            self.lt_duration,
        ):
            lrow.addWidget(w)
        lrow.addStretch(1)
        lrow.addWidget(self.lt_btn)
        ll.addLayout(lrow)
        self.bars = CarrierBars()
        ll.addWidget(self.bars)
        self.lt_stats = QLabel("—")
        self.lt_stats.setWordWrap(True)
        ll.addWidget(self.lt_stats)
        root.addWidget(lt)

        # Test tone --------------------------------------------------------------
        tt = QGroupBox("Test tone")
        tl = QHBoxLayout(tt)
        self.tone_freq = QDoubleSpinBox()
        self.tone_freq.setRange(100, 23_000)
        self.tone_freq.setValue(18_500)
        self.tone_freq.setSuffix(" Hz")
        self.tone_level = QSpinBox()
        self.tone_level.setRange(-40, -3)
        self.tone_level.setValue(-12)
        self.tone_level.setSuffix(" dBFS")
        self.tone_secs = QSpinBox()
        self.tone_secs.setRange(1, 30)
        self.tone_secs.setValue(3)
        self.tone_secs.setSuffix(" s")
        self.tone_btn = QPushButton("Play tone")
        self.tone_btn.clicked.connect(self._play_tone)
        self.tone_stop = QPushButton("Stop")
        self.tone_stop.setEnabled(False)
        self.tone_stop.clicked.connect(self._stop_tone)
        tone_widgets: tuple[QWidget, ...] = (
            QLabel("Frequency:"),
            self.tone_freq,
            QLabel("Level:"),
            self.tone_level,
            QLabel("Duration:"),
            self.tone_secs,
        )
        for tw in tone_widgets:
            tl.addWidget(tw)
        tl.addStretch(1)
        tl.addWidget(self.tone_btn)
        tl.addWidget(self.tone_stop)
        root.addWidget(tt)
        self._tone_timer = QTimer(self)
        self._tone_timer.setSingleShot(True)
        self._tone_timer.timeout.connect(self._stop_tone)

        # Troubleshooting ----------------------------------------------------------
        tb = QGroupBox("Troubleshooting checklist")
        tbl = QVBoxLayout(tb)
        txt = QLabel(
            "<ol>" + "".join(f"<li>{t}</li>" for t in TROUBLESHOOTING) + "<li>Allow microphone access: "
            "Settings → Privacy &amp; security → Microphone → “Let desktop apps access your microphone”."
            "</li></ol>"
        )
        txt.setWordWrap(True)
        tbl.addWidget(txt)
        root.addWidget(tb)
        root.addStretch(1)
        self._devices_loaded = False

    # ---------------------------------------------------------------- devices
    def showEvent(self, e: object) -> None:
        if not self._devices_loaded:
            self.refresh_devices()
        super().showEvent(e)  # type: ignore[arg-type]

    def refresh_devices(self) -> None:
        """(Re)populate device pickers; querying devices never opens a stream or plays sound."""
        from ultrasonic_airdrop.audio.devices import list_devices

        self._devices_loaded = True
        for combo, kind in ((self.in_combo, "input"), (self.out_combo, "output")):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("System default", None)
            try:
                devs = list_devices()
            except Exception as exc:
                self.ctx.banner("error", f"Cannot list audio devices: {exc}", "Install PortAudio / check drivers.")
                devs = []
            for d in devs:
                if (d.max_input_channels if kind == "input" else d.max_output_channels) > 0:
                    combo.addItem(d.label, d.index)
            sel = self.ctx.device(kind)
            i = combo.findData(sel) if sel is not None else 0
            combo.setCurrentIndex(max(0, i))
            combo.blockSignals(False)

    # ---------------------------------------------------------------- hardware check
    def _run_check(self) -> None:
        rate = int(self.rate.currentData())
        level = float(self.check_level.value())
        top = min(23.0, rate / 2000 - 1)
        if (
            QMessageBox.question(
                self,
                "Play test sweeps?",
                f"This will play two 3-second sweeps from 2 kHz to {top:.0f} kHz at up to {level:.0f} dBFS through the "
                "selected speaker (high tones may be audible, especially to children and pets). Continue?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.check_btn.setEnabled(False)
        self.verdict.setText("Running…")
        backend = self.ctx.make_backend(rate)

        def work(task: TaskThread) -> HardwareCheckResult:
            return run_hardware_check(backend, level, progress=task.progress.emit, should_stop=lambda: task.cancelled)

        self.check_task = TaskThread(work, self)
        self.check_task.progress.connect(self.verdict.setText)
        self.check_task.done.connect(self._check_done)
        self.check_task.error.connect(self._check_failed)
        self.check_task.finished.connect(lambda: self.check_btn.setEnabled(True))
        self.check_task.start()

    def _check_failed(self, msg: str, hint: str) -> None:
        self.verdict.setText(f"Hardware check failed: {msg}")
        self.ctx.banner("error", msg, hint)

    def _check_done(self, res: object) -> None:
        assert isinstance(res, HardwareCheckResult)
        bands = [
            (PROFILES[b.profile_key].key.upper(), b.band_hz[0], b.band_hz[1]) for b in res.bands if b.verdict != "n/a"
        ]
        # Points within 3 dB of the noise floor are not measurable: hide them in the plot.
        shown = [
            r if (n != n or r - n >= 3.0) else float("nan") for r, n in zip(res.response_db, res.noise_db, strict=True)
        ]
        self.plot.set_data(list(res.freqs_hz), shown, list(res.noise_db), bands)
        lines = []
        for b in res.bands:
            if b.verdict == "n/a":
                continue
            lines.append(
                f"<b>{PROFILES[b.profile_key].name}</b>: {b.verdict.upper()} "
                f"(response {b.relative_db:+.1f} dB vs 2–4 kHz, band SNR {b.snr_db:.1f} dB)"
            )
        lines.append(
            f"Recorded peak {res.peak_dbfs:.1f} dBFS, noise floor {res.noise_floor_dbfs:.1f} dBFS"
            + (" — <b>CLIPPING</b>" if res.clipping else "")
        )
        lines += [f"• {m}" for m in res.messages]
        self.verdict.setText("<br>".join(lines))
        self.verdict.adjustSize()
        self.verdict.updateGeometry()
        if res.sample_rate == 96_000:
            self.ctx.set_us_ok(res.verdict("us") in ("supported", "marginal"))
        nu = res.verdict("nu")
        self.ctx.banner(
            "success" if nu == "supported" else "warning",
            f"Near-ultrasonic band: {nu}.",
            "" if nu == "supported" else "See the suggestions under the plot.",
        )

    # ---------------------------------------------------------------- link test
    def _toggle_link_test(self, on: bool) -> None:
        if not on:
            if self.link_session is not None:
                self.link_session.stop()
            if self.link_engine is not None:
                self.link_engine.stop()
            return
        profile = get_profile(self.lt_profile.currentData())
        settings = self.ctx.settings
        role = self.role.currentData()
        duration = float(self.lt_duration.value())
        if (
            role == "send"
            and QMessageBox.question(
                self,
                "Play probe frames?",
                f"This will play {profile.name} probe frames for {duration:.0f} s at "
                f"{settings.tx_level_dbfs:.0f} dBFS. Continue?",
            )
            != QMessageBox.StandardButton.Yes
        ):
            self.lt_btn.setChecked(False)
            return
        try:
            backend = self.ctx.make_backend(profile.sample_rate, enable_output=role == "send")
            masks: dict[int, CarrierMask] = {}
            station = Station(
                backend, build_pipeline(profile.sample_rate, settings, masks, [profile]), settings, self.bridge.sink
            )
        except Exception as exc:
            self.ctx.banner("error", f"Could not open audio: {exc}", "")
            self.lt_btn.setChecked(False)
            return
        if role == "send":
            self.link_session = LinkTestSender(station, profile, settings, self.bridge.sink, duration)
        else:

            def stats() -> tuple[int, int]:
                return sum(p.stats.detections for p in station.pipeline.pipelines), 0

            self.link_session = LinkTestReceiver(self.bridge.sink, stats, duration_s=None, now=station.now())
        station.session = self.link_session
        self.link_engine = EngineThread(station, stop_when_done=True, parent=self)
        self.link_engine.failed.connect(lambda m, h: self.ctx.banner("error", m, h))
        self.link_engine.finished.connect(self._link_finished)
        self.link_engine.start()
        self.lt_btn.setText("Stop link test")
        self.lt_stats.setText("Running…")

    def _link_finished(self) -> None:
        self.link_engine = None
        self.link_session = None
        self.lt_btn.blockSignals(True)
        self.lt_btn.setChecked(False)
        self.lt_btn.blockSignals(False)
        self.lt_btn.setText("Start link test")

    def _on_event(self, ev: object) -> None:
        if isinstance(ev, LinkTestEvent):
            if ev.role == "linktest-sender":
                self.lt_stats.setText(f"Probes sent: {ev.probes_sent} ({ev.elapsed_s:.0f} s)")
                return
            if ev.carrier_snr_db:
                self.bars.set_values(list(ev.carrier_snr_db), REQUIRED_SNR_DB[2])
            total = ev.frames_ok + ev.frame_errors
            fer = ev.frame_errors / total if total else 0.0
            mode = MODES[ev.recommended_mode].label if ev.recommended_mode in MODES else "—"
            ppm = "—" if math.isnan(ev.ppm) else f"{ev.ppm:+.0f} ppm"
            mfsk = "—" if math.isnan(ev.mfsk_snr_db) else f"{ev.mfsk_snr_db:.1f} dB"
            self.lt_stats.setText(
                f"Detections {ev.detections} · probes decoded {ev.frames_ok} · frame error rate {fer:.0%} · "
                f"MFSK SNR {mfsk} · clock offset {ppm} · <b>recommended mode: {mode}</b> "
                f"({ev.carriers_enabled} OFDM carriers usable)"
            )
        elif isinstance(ev, ErrorEvent):
            self.ctx.banner("error", ev.message, ev.hint)

    # ---------------------------------------------------------------- test tone
    def _play_tone(self) -> None:
        freq = float(self.tone_freq.value())
        level = float(self.tone_level.value())
        secs = float(self.tone_secs.value())
        rate = 96_000 if freq > 22_000 else 48_000
        try:
            self.tone_backend = self.ctx.make_backend(rate, enable_input=False)
            self.tone_backend.start()
            self.tone_backend.play(make_test_tone(rate, freq, level, secs))
        except Exception as exc:
            self.ctx.banner("error", f"Could not play the tone: {exc}", "Check the output device.")
            self._stop_tone()
            return
        self.tone_btn.setEnabled(False)
        self.tone_stop.setEnabled(True)
        self._tone_timer.start(int(secs * 1000) + 500)

    def _stop_tone(self) -> None:
        self._tone_timer.stop()
        if self.tone_backend is not None:
            self.tone_backend.stop_output()
            self.tone_backend.close()
            self.tone_backend = None
        self.tone_btn.setEnabled(True)
        self.tone_stop.setEnabled(False)

    def shutdown(self) -> None:
        """Stop everything (application exit)."""
        self._stop_tone()
        if self.link_engine is not None:
            self.link_engine.stop()
            self.link_engine.wait(3000)
        if self.check_task is not None and self.check_task.isRunning():
            self.check_task.cancelled = True
            self.check_task.wait(8000)
