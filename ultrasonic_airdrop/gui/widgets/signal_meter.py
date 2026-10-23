"""Labelled horizontal meters (input level, band SNR, detection confidence)."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget


class Bar(QWidget):
    """A single meter bar with value text drawn inside."""

    def __init__(self, lo: float, hi: float, unit: str, good: float, warn: float) -> None:
        super().__init__()
        self.lo, self.hi, self.unit = lo, hi, unit
        self.good, self.warn = good, warn
        self.value: float | None = None
        self.setMinimumHeight(18)
        self.setMinimumWidth(120)

    def set_value(self, v: float | None) -> None:
        """Update."""
        self.value = v
        self.update()

    def paintEvent(self, _ev: object) -> None:
        p = QPainter(self)
        r = QRectF(self.rect()).adjusted(0, 1, -1, -1)
        p.fillRect(r, self.palette().base())
        p.setPen(self.palette().mid().color())
        p.drawRect(r)
        if self.value is None or self.value != self.value:
            p.setPen(self.palette().text().color())
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "—")
            return
        t = max(0.0, min(1.0, (self.value - self.lo) / (self.hi - self.lo)))
        if self.value >= self.good:
            color, label = QColor("#2e7d32"), "good"
        elif self.value >= self.warn:
            color, label = QColor("#ef6c00"), "fair"
        else:
            color, label = QColor("#c62828"), "poor"
        p.fillRect(QRectF(r.left(), r.top(), r.width() * t, r.height()), color)
        p.setPen(self.palette().text().color())
        p.drawText(
            r.adjusted(4, 0, -4, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            f"{self.value:.1f} {self.unit} ({label})",
        )


class SignalMeter(QWidget):
    """Three meters with captions."""

    def __init__(self) -> None:
        super().__init__()
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        self.level = Bar(-90, 0, "dBFS", -60, -80)
        self.snr = Bar(-10, 40, "dB", 15, 5)
        self.detect = Bar(0, 40, "×", 12, 8)
        for row, (caption, bar) in enumerate(
            (("Input level", self.level), ("Band SNR", self.snr), ("Detection confidence", self.detect))
        ):
            g.addWidget(QLabel(caption), row, 0)
            g.addWidget(bar, row, 1)
        g.setColumnStretch(1, 1)
