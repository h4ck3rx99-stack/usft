"""Live magnitude spectrum of the profile band (QPainter)."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ultrasonic_airdrop.transport.events import SpectrumEvent


class SpectrumWidget(QWidget):
    """Draws the latest :class:`SpectrumEvent` with the signal band shaded."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.freqs: tuple[float, ...] = ()
        self.power: tuple[float, ...] = ()
        self.band: tuple[float, float] = (17_500.0, 20_000.0)
        self.floor_db = -130.0
        self.ceil_db = -30.0
        self.setAccessibleName("Spectrum")

    def set_band(self, lo: float, hi: float) -> None:
        """Signal band to shade."""
        self.band = (lo, hi)
        self.update()

    def update_spectrum(self, ev: SpectrumEvent) -> None:
        """New data."""
        self.freqs, self.power = ev.freqs_hz, ev.power_db
        if self.power:
            top = max(self.power)
            self.ceil_db = max(self.ceil_db * 0.95 + (top + 10) * 0.05, self.floor_db + 40)
        self.update()

    def paintEvent(self, _ev: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        r = QRectF(self.rect()).adjusted(36, 6, -6, -18)
        p.fillRect(self.rect(), pal.base())
        p.setPen(QPen(pal.mid().color(), 1))
        p.drawRect(r)
        if not self.freqs:
            p.setPen(pal.text().color())
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No input yet — press “Start listening”")
            return
        f0, f1 = self.freqs[0], self.freqs[-1]
        span = max(f1 - f0, 1.0)

        def x(f: float) -> float:
            return r.left() + (f - f0) / span * r.width()

        def y(db: float) -> float:
            t = (db - self.floor_db) / (self.ceil_db - self.floor_db)
            return r.bottom() - max(0.0, min(1.0, t)) * r.height()

        band = QColor(46, 125, 50, 40)
        p.fillRect(QRectF(x(self.band[0]), r.top(), x(self.band[1]) - x(self.band[0]), r.height()), band)
        path = QPainterPath()
        for i, (f, db) in enumerate(zip(self.freqs, self.power, strict=False)):
            pt = QPointF(x(f), y(db))
            if i == 0:
                path.moveTo(pt)
            else:
                path.lineTo(pt)
        p.setPen(QPen(pal.highlight().color(), 1.5))
        p.drawPath(path)
        p.setPen(pal.text().color())
        font = p.font()
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1))
        p.setFont(font)
        for k in range(int(f0 // 1000) + 1, int(f1 // 1000) + 1):
            xx = x(k * 1000.0)
            p.drawLine(QPointF(xx, r.bottom()), QPointF(xx, r.bottom() + 3))
            p.drawText(QPointF(xx - 10, r.bottom() + 15), f"{k}k")
        for db in range(int(self.floor_db), int(self.ceil_db) + 1, 20):
            p.drawText(QPointF(2, y(db) + 4), f"{db}")
