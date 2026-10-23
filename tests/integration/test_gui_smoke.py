"""Offscreen GUI smoke test: the main window builds without touching audio."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


def test_main_window_constructs_without_audio(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    import ultrasonic_airdrop.audio.sd_backend as sdb

    started = []
    monkeypatch.setattr(sdb.SoundDeviceBackend, "start", lambda self: started.append(self))
    monkeypatch.setattr(sdb.SoundDeviceBackend, "play", lambda self, x: started.append(x))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from ultrasonic_airdrop.gui.app import apply_theme
    from ultrasonic_airdrop.gui.context import AppContext
    from ultrasonic_airdrop.gui.main_window import MainWindow
    from ultrasonic_airdrop.transport.events import (
        FileEvent,
        LinkTestEvent,
        ManifestEvent,
        OfferEvent,
        ProgressEvent,
        SpectrumEvent,
        StateEvent,
    )

    apply_theme(app)
    qs = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    ctx = AppContext(qs)
    win = MainWindow(ctx)
    win.show()
    for i in range(win.stack.count()):
        win.nav.setCurrentRow(i)
        app.processEvents()
    # feed representative events through each page's handler
    rp = win.receive_page
    rp._on_event(SpectrumEvent(tuple(float(f) for f in range(16000, 21000, 50)), tuple([-100.0] * 100), -40.0, 10.0))
    rp._on_event(OfferEvent(1, "Lab", 2, 1234, 120.0, True, False, "nu", 20.0))
    rp._on_event(ManifestEvent(((0, "a.txt", 10), (1, "b.bin", 20))))
    rp._on_event(FileEvent(0, "a.txt", "verified", 10, 10))
    rp._on_event(ProgressEvent("receiver", 10, 30, 2.0, 10.0, 1, 0, 0.0, 20.0, 5.0))
    rp._on_event(StateEvent("receiver", "receiving", "Receiving…"))
    win.send_page._on_event(ProgressEvent("sender", 10, 30, 2.0, 10.0, 4, 1, 0.1, 15.0, 5.0))
    win.diag_page._on_event(LinkTestEvent("linktest-receiver", 0, 3, 2, 1, tuple([12.0] * 52), 25.0, 30.0, 4, 52))
    ctx.update_settings(detect_ratio=9.0)
    assert ctx.settings.detect_ratio == 9.0
    win.banner.show_message("error", "test", "hint")
    app.processEvents()
    win.grab()  # renders all custom QPainter widgets of the visible page
    win.close()
    app.processEvents()
    assert started == []  # nothing opened an audio stream or played audio
