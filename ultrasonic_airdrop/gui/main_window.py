"""Main window: sidebar navigation (Send, Receive, Diagnostics, Settings, Log) + status banner."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ultrasonic_airdrop import __version__
from ultrasonic_airdrop.gui.context import AppContext
from ultrasonic_airdrop.gui.diagnostics_page import DiagnosticsPage
from ultrasonic_airdrop.gui.log_page import LogPage
from ultrasonic_airdrop.gui.receive_page import ReceivePage
from ultrasonic_airdrop.gui.send_page import SendPage
from ultrasonic_airdrop.gui.settings_page import SettingsPage
from ultrasonic_airdrop.gui.widgets.status_banner import StatusBanner


class MainWindow(QMainWindow):
    """Top-level window.  Constructing it never opens an audio device or plays sound."""

    def __init__(self, ctx: AppContext | None = None) -> None:
        super().__init__()
        self.ctx = ctx or AppContext()
        self.setWindowTitle(f"Ultrasonic AirDrop {__version__}")
        self.resize(1100, 780)
        self.setMinimumSize(QSize(760, 560))
        central = QWidget()
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("Sidebar")
        self.nav.setFixedWidth(190)
        self.nav.setIconSize(QSize(20, 20))
        right = QWidget()
        v = QVBoxLayout(right)
        v.setContentsMargins(16, 12, 16, 12)
        self.banner = StatusBanner()
        self.ctx.banner = self.banner.show_message
        v.addWidget(self.banner)
        self.stack = QStackedWidget()
        v.addWidget(self.stack, 1)
        h.addWidget(self.nav)
        h.addWidget(right, 1)
        self.setCentralWidget(central)

        self.log_page = LogPage()
        self.send_page = SendPage(self.ctx)
        self.receive_page = ReceivePage(self.ctx)
        self.diag_page = DiagnosticsPage(self.ctx)
        self.settings_page = SettingsPage(self.ctx)
        for label, page in (
            ("⬆  Send", self.send_page),
            ("⬇  Receive", self.receive_page),
            ("🩺  Diagnostics", self.diag_page),
            ("⚙  Settings", self.settings_page),
            ("📜  Log", self.log_page),
        ):
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(180, 40))
            self.nav.addItem(item)
            self.stack.addWidget(page)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.setCurrentRow(0)
        footer = QLabel("Sound only — no network is used.")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.statusBar().addWidget(footer)

    def shutdown(self) -> None:
        """Stop every engine and close audio devices."""
        for page in (self.send_page, self.receive_page, self.diag_page):
            page.shutdown()
        self.log_page.detach()

    def closeEvent(self, e: QCloseEvent) -> None:
        self.shutdown()
        super().closeEvent(e)
