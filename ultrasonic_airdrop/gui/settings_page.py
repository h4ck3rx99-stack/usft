"""Settings page: tunables with per-field reset, persisted via QSettings."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ultrasonic_airdrop.config import Settings
from ultrasonic_airdrop.gui.context import AppContext

MB = 1024 * 1024

# name, label, kind, min, max, step/decimals, scale, help
FIELDS = [
    ("detect_ratio", "Detection threshold (peak/median)", "float", 3.0, 50.0, 1, 1, "Higher = fewer false detections"),
    ("erasure_ratio", "MFSK erasure ratio", "float", 1.0, 5.0, 2, 1, "Best/second-best energy below this → erasure"),
    ("window_mfsk", "Window size (MFSK frames)", "int", 1, 32, 1, 1, "Capped so a window is ≤ 20 s"),
    ("window_ofdm", "Window size (OFDM frames)", "int", 1, 32, 1, 1, "Capped so a window is ≤ 20 s"),
    ("offer_listen_s", "Offer listen time (s)", "float", 2.0, 60.0, 1, 1, "Wait after each OFFER"),
    ("ack_extra_s", "Extra ACK wait (s)", "float", 0.5, 20.0, 1, 1, "Added to ACK airtime + latency"),
    ("rx_inactivity_s", "Receiver inactivity timeout (s)", "float", 20.0, 600.0, 0, 1, "Then partials are kept"),
    ("output_latency_s", "Assumed output latency (s)", "float", 0.0, 2.0, 2, 1, "Used in timeouts"),
    ("max_accept_bytes", "Maximum accepted transfer (MB)", "float", 0.0, 10_000.0, 1, MB, "Larger offers are rejected"),
    ("auto_accept_bytes", "Auto-accept offers up to (MB, 0 = off)", "float", 0.0, 1000.0, 2, MB, "Off by default"),
    ("broadcast_passes", "Broadcast passes", "int", 1, 10, 1, 1, "Carousel repetitions"),
]


class SettingsPage(QWidget):
    """Editable settings."""

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.editors: dict[str, QSpinBox | QDoubleSpinBox | QComboBox] = {}
        root = QVBoxLayout(self)
        title = QLabel("Settings")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        box = QGroupBox("Modem and protocol")
        form = QFormLayout(box)
        for name, label, kind, lo, hi, prec, _scale, help_text in FIELDS:
            if kind == "int":
                ed: QSpinBox | QDoubleSpinBox = QSpinBox()
                ed.setRange(int(lo), int(hi))
            else:
                ed = QDoubleSpinBox()
                ed.setRange(lo, hi)
                ed.setDecimals(int(prec))
            ed.setToolTip(help_text)
            ed.setAccessibleName(label)
            ed.editingFinished.connect(lambda n=name: self._commit(n))
            self.editors[name] = ed
            form.addRow(label + ":", self._with_reset(name, ed))
        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level.currentTextChanged.connect(lambda v: self.ctx.update_settings(log_level=v))
        self.editors["log_level"] = self.log_level
        form.addRow("Log level:", self._with_reset("log_level", self.log_level))
        root.addWidget(box)
        root.addWidget(QLabel("Changes apply to the next transfer. Values are stored per user."))
        reset_all = QPushButton("Reset all to defaults")
        reset_all.clicked.connect(self._reset_all)
        root.addWidget(reset_all)
        root.addStretch(1)
        self._load()
        ctx.settings_changed.connect(self._load)

    def _with_reset(self, name: str, ed: QWidget) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(ed, 1)
        b = QPushButton("Reset to default")
        b.clicked.connect(lambda: self.ctx.reset_setting(name))
        h.addWidget(b)
        return w

    def _scale(self, name: str) -> float:
        for f in FIELDS:
            if f[0] == name:
                return float(f[6])
        return 1.0

    def _load(self) -> None:
        s = self.ctx.settings
        for name, ed in self.editors.items():
            ed.blockSignals(True)
            if isinstance(ed, QComboBox):
                ed.setCurrentText(str(getattr(s, name)))
            elif isinstance(ed, QSpinBox):
                ed.setValue(int(getattr(s, name)))
            else:
                ed.setValue(float(getattr(s, name)) / self._scale(name))
            ed.blockSignals(False)

    def _commit(self, name: str) -> None:
        ed = self.editors[name]
        if isinstance(ed, QSpinBox):
            value: object = int(ed.value())
        elif isinstance(ed, QDoubleSpinBox):
            scale = self._scale(name)
            value = round(ed.value() * scale) if scale != 1 else float(ed.value())
        else:
            return
        try:
            self.ctx.update_settings(**{name: value})
        except ValueError as exc:
            self.ctx.banner("warning", f"Invalid value: {exc}", "")
            self._load()

    def _reset_all(self) -> None:
        for name in self.editors:
            self.ctx.reset_setting(name)
        assert Settings()  # defaults always valid
