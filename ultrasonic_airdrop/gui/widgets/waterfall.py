"""Scrolling waterfall (time × frequency) of the profile band (QPainter + QImage)."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from ultrasonic_airdrop.transport.events import SpectrumEvent

ROWS = 200


def _colormap() -> np.ndarray:
    """Perceptually ordered dark-blue → yellow ramp (ARGB32)."""
    t = np.linspace(0, 1, 256)
    r = np.clip(1.5 * t - 0.2, 0, 1)
    g = np.clip(1.2 * t**1.2, 0, 1)
    b = np.clip(0.6 + 0.4 * np.sin(np.pi * t) - 0.6 * t, 0, 1)
    argb = (
        (0xFF << 24)
        | ((r * 255).astype(np.uint32) << 16)
        | ((g * 255).astype(np.uint32) << 8)
        | (b * 255).astype(np.uint32)
    )
    return argb.astype(np.uint32)


class WaterfallWidget(QWidget):
    """Newest spectrum row at the top."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._cmap = _colormap()
        self._data: np.ndarray | None = None
        self.floor_db = -130.0
        self.range_db = 70.0
        self.setAccessibleName("Waterfall")

    def clear(self) -> None:
        """Forget history."""
        self._data = None
        self.update()

    def add(self, ev: SpectrumEvent) -> None:
        """Append one spectrum row."""
        row = np.asarray(ev.power_db, dtype=np.float64)
        if row.size == 0:
            return
        if self._data is None or self._data.shape[1] != row.size:
            self._data = np.full((ROWS, row.size), self.floor_db)
        self._data = np.roll(self._data, 1, axis=0)
        self._data[0] = row
        self.floor_db = 0.98 * self.floor_db + 0.02 * float(np.percentile(row, 20))
        self.update()

    def paintEvent(self, _ev: object) -> None:
        p = QPainter(self)
        if self._data is None:
            p.fillRect(self.rect(), self.palette().base())
            return
        norm = np.clip((self._data - self.floor_db) / self.range_db, 0, 1)
        idx = (norm * 255).astype(np.uint8)
        pix = np.ascontiguousarray(self._cmap[idx])
        h, w = pix.shape
        img = QImage(pix.data, w, h, 4 * w, QImage.Format.Format_ARGB32)
        p.drawImage(QRect(0, 0, self.width(), self.height()), img)
