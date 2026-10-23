"""Non-modal status banner with a text level label (not colour alone) and an actionable hint."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton

COLORS = {
    "info": ("#1d4f91", "#e7f0fd"),
    "success": ("#1b5e20", "#e6f4ea"),
    "warning": ("#7a4b00", "#fff4e0"),
    "error": ("#8e1c1c", "#fde8e8"),
}
LABELS = {"info": "Info", "success": "Success", "warning": "Warning", "error": "Error"}


class StatusBanner(QFrame):
    """Shows one message at a time; hides on close or after a timeout for info/success."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("StatusBanner")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 8, 8)
        self.level_label = QLabel()
        self.level_label.setStyleSheet("font-weight: 600;")
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        close = QToolButton()
        close.setText("✕")
        close.setToolTip("Dismiss")
        close.setAutoRaise(True)
        close.clicked.connect(self.hide)
        lay.addWidget(self.level_label)
        lay.addWidget(self.text, 1)
        lay.addWidget(close)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.hide()

    def show_message(self, level: str, message: str, hint: str = "") -> None:
        """Display a message (``level`` in info/success/warning/error)."""
        fg, bg = COLORS.get(level, COLORS["info"])
        self.setStyleSheet(
            f"#StatusBanner {{ background: {bg}; border: 1px solid {fg}; border-radius: 6px; }}"
            f" QLabel {{ color: {fg}; }}"
        )
        self.level_label.setText(LABELS.get(level, level.title()) + ":")
        self.text.setText(message + (f"\n→ {hint}" if hint else ""))
        self.show()
        self._timer.stop()
        if level in ("info", "success"):
            self._timer.start(8000)
