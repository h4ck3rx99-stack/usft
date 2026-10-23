"""Live tail of the structured log with copy/export."""

from __future__ import annotations

import contextlib
import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ultrasonic_airdrop.logging_setup import LOG_FORMAT, log_file_path


class _Emitter(QObject):
    line = Signal(str)


class QtLogHandler(logging.Handler):
    """Forwards formatted records to a Qt signal (thread-safe via queued connection)."""

    def __init__(self) -> None:
        super().__init__()
        self.emitter = _Emitter()
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(RuntimeError):  # emitter already deleted during shutdown
            self.emitter.line.emit(self.format(record))


class LogPage(QWidget):
    """Shows log lines as they are produced."""

    MAX_LINES = 5000

    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("Log")
        title.setObjectName("PageTitle")
        root.addWidget(title)
        root.addWidget(QLabel(f"Also written to {log_file_path()}"))
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setMaximumBlockCount(self.MAX_LINES)
        self.view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        root.addWidget(self.view, 1)
        row = QHBoxLayout()
        copy = QPushButton("Copy all")
        copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.view.toPlainText()))
        export = QPushButton("Export…")
        export.clicked.connect(self._export)
        clear = QPushButton("Clear view")
        clear.clicked.connect(self.view.clear)
        row.addStretch(1)
        for b in (copy, export, clear):
            row.addWidget(b)
        root.addLayout(row)
        self.handler = QtLogHandler()
        self.handler.emitter.line.connect(self.view.appendPlainText)
        logging.getLogger("ultrasonic_airdrop").addHandler(self.handler)

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export log", "airdrop-log.txt", "Text files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(self.view.toPlainText())

    def detach(self) -> None:
        """Remove the handler (application exit)."""
        logging.getLogger("ultrasonic_airdrop").removeHandler(self.handler)
