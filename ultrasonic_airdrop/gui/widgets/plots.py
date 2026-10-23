"""Response-curve plot (hardware check) and per-carrier SNR bars (link test)."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class ResponsePlot(QWidget):
    """Speaker→mic magnitude response (dB vs log frequency) with profile bands."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.freqs: list[float] = []
        self.resp: list[float] = []
        self.noise: list[float] = []
        self.bands: list[tuple[str, float, float]] = []

    def set_data(
        self, freqs: list[float], resp: list[float], noise: list[float], bands: list[tuple[str, float, float]]
    ) -> None:
        """Set the curves."""
        self.freqs, self.resp, self.noise, self.bands = freqs, resp, noise, bands
        self.update()

    def paintEvent(self, _ev: object) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        r = QRectF(self.rect()).adjusted(40, 8, -8, -20)
        p.setPen(pal.mid().color())
        p.drawRect(r)
        if not self.freqs:
            p.setPen(pal.text().color())
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Run the hardware check to see the response")
            return
        f0, f1 = math.log10(self.freqs[0]), math.log10(self.freqs[-1])
        lo, hi = -80.0, 20.0

        def x(f: float) -> float:
            return r.left() + (math.log10(f) - f0) / (f1 - f0) * r.width()

        def y(db: float) -> float:
            return r.bottom() - (max(lo, min(hi, db)) - lo) / (hi - lo) * r.height()

        for name, a, b in self.bands:
            p.fillRect(QRectF(x(a), r.top(), x(b) - x(a), r.height()), QColor(46, 125, 50, 35))
            p.setPen(pal.text().color())
            p.drawText(QPointF(x(a) + 2, r.top() + 12), name)
        for series, color in ((self.noise, QColor("#9e9e9e")), (self.resp, pal.highlight().color())):
            path = QPainterPath()
            started = False
            for f, v in zip(self.freqs, series, strict=False):
                if v != v:
                    started = False
                    continue
                pt = QPointF(x(f), y(v))
                if not started:
                    path.moveTo(pt)
                    started = True
                else:
                    path.lineTo(pt)
            p.setPen(QPen(color, 1.6))
            p.drawPath(path)
        p.setPen(pal.text().color())
        for f in (2000, 5000, 10000, 15000, 20000):
            if self.freqs[0] <= f <= self.freqs[-1]:
                p.drawText(QPointF(x(f) - 10, r.bottom() + 14), f"{f // 1000}k")
        for db in range(int(lo), int(hi) + 1, 20):
            p.drawText(QPointF(4, y(db) + 4), f"{db}")
        p.drawText(QPointF(r.left() + 6, r.bottom() - 6), "blue: response · grey: noise floor (dB rel. 2–4 kHz)")


class CarrierBars(QWidget):
    """52 per-carrier SNR bars."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(120)
        self.values: list[float] = []
        self.threshold = 10.0

    def set_values(self, values: list[float], threshold: float) -> None:
        """Update bars; carriers below ``threshold`` are drawn in orange/red."""
        self.values, self.threshold = values, threshold
        self.update()

    def paintEvent(self, _ev: object) -> None:
        p = QPainter(self)
        pal = self.palette()
        p.fillRect(self.rect(), pal.base())
        r = QRectF(self.rect()).adjusted(30, 6, -6, -16)
        p.setPen(pal.mid().color())
        p.drawRect(r)
        if not self.values:
            p.setPen(pal.text().color())
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Waiting for probe frames…")
            return
        lo, hi = -10.0, 40.0
        w = r.width() / len(self.values)
        for i, v in enumerate(self.values):
            vv = max(lo, min(hi, v if v == v else lo))
            h = (vv - lo) / (hi - lo) * r.height()
            color = (
                QColor("#2e7d32")
                if vv >= self.threshold
                else (QColor("#ef6c00") if vv >= self.threshold - 6 else QColor("#c62828"))
            )
            p.fillRect(QRectF(r.left() + i * w + 1, r.bottom() - h, max(1.0, w - 2), h), color)
        p.setPen(pal.text().color())
        for db in (0, 10, 20, 30, 40):
            yy = r.bottom() - (db - lo) / (hi - lo) * r.height()
            p.drawText(QPointF(2, yy + 4), str(db))
        p.drawText(QPointF(r.left(), r.bottom() + 13), "carrier 0 (low freq) → 51 (high freq); SNR in dB")
