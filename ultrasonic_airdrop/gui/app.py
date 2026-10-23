"""Application entry point: QApplication, theme (follows system light/dark), single instance."""

from __future__ import annotations

import getpass
import logging
import signal
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPalette
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from ultrasonic_airdrop.logging_setup import kv, setup_logging

log = logging.getLogger(__name__)

STYLE = """
#PageTitle { font-size: 20px; font-weight: 600; padding: 4px 0 8px 0; }
#StateLabel { font-weight: 600; }
#Sidebar { border: none; padding-top: 12px; font-size: 14px; }
#Sidebar::item { padding: 6px 12px; border-radius: 6px; margin: 2px 8px; color: palette(text); }
#Sidebar::item:selected { background: palette(highlight); color: palette(highlighted-text); }
#Sidebar::item:hover:!selected { background: palette(alternate-base); }
QGroupBox { font-weight: 600; margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
QGroupBox QLabel, QGroupBox QCheckBox, QGroupBox QRadioButton { font-weight: normal; }
QPushButton { padding: 5px 14px; }
"""


def _is_dark(app: QApplication) -> bool:
    hints = QGuiApplication.styleHints()
    scheme = getattr(hints, "colorScheme", None)
    if scheme is not None:
        try:
            return bool(scheme() == Qt.ColorScheme.Dark)
        except Exception:
            pass
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def apply_theme(app: QApplication) -> None:
    """Fusion style with a palette that follows the system light/dark preference."""
    app.setStyle("Fusion")
    if _is_dark(app):
        pal = QPalette()
        base, window, text = QColor(30, 31, 34), QColor(40, 42, 46), QColor(232, 232, 232)
        pal.setColor(QPalette.ColorRole.Window, window)
        pal.setColor(QPalette.ColorRole.WindowText, text)
        pal.setColor(QPalette.ColorRole.Base, base)
        pal.setColor(QPalette.ColorRole.AlternateBase, window)
        pal.setColor(QPalette.ColorRole.Text, text)
        pal.setColor(QPalette.ColorRole.Button, window)
        pal.setColor(QPalette.ColorRole.ButtonText, text)
        pal.setColor(QPalette.ColorRole.Highlight, QColor(66, 133, 244))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        pal.setColor(QPalette.ColorRole.Mid, QColor(90, 90, 96))
        pal.setColor(QPalette.ColorRole.ToolTipBase, window)
        pal.setColor(QPalette.ColorRole.ToolTipText, text)
        app.setPalette(pal)
    app.setStyleSheet(STYLE)
    f = app.font()
    if f.pointSizeF() < 10:
        f.setPointSizeF(10)
    app.setFont(QFont(f))


def server_name() -> str:
    """Per-user single-instance server name."""
    try:
        user = getpass.getuser()
    except Exception:
        user = "user"
    return f"UltrasonicAirDrop-{user}"


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI."""
    setup_logging("INFO")
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Ultrasonic AirDrop")
    app.setOrganizationName("UltrasonicAirDrop")

    sock = QLocalSocket()
    sock.connectToServer(server_name())
    if sock.waitForConnected(300):
        sock.write(b"raise")
        sock.flush()
        sock.waitForBytesWritten(300)
        print("Ultrasonic AirDrop is already running.", file=sys.stderr)
        return 0
    server = QLocalServer()
    QLocalServer.removeServer(server_name())
    server.listen(server_name())

    apply_theme(app)
    from ultrasonic_airdrop.gui.main_window import MainWindow

    win = MainWindow()

    def raise_window() -> None:
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.close()
        win.showNormal()
        win.raise_()
        win.activateWindow()

    server.newConnection.connect(raise_window)
    app.aboutToQuit.connect(win.shutdown)
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    timer = QTimer()  # lets Python handle Ctrl+C while Qt's loop runs
    timer.start(250)
    timer.timeout.connect(lambda: None)
    win.show()
    log.info(kv("GUI started"))
    rc = app.exec()
    server.close()
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
