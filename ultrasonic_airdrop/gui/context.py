"""Shared GUI state: persisted settings (QSettings), device choices, banner access."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal

from ultrasonic_airdrop.config import Settings
from ultrasonic_airdrop.logging_setup import set_level

DEFAULT_DEST = Path.home() / "Downloads" / "Ultrasonic AirDrop"


class AppContext(QObject):
    """Holds the persisted configuration and notifies pages about changes."""

    settings_changed = Signal()
    devices_changed = Signal()

    def __init__(self, qsettings: QSettings | None = None) -> None:
        super().__init__()
        self.qs = qsettings or QSettings("UltrasonicAirDrop", "UltrasonicAirDrop")
        self.settings = self._load_settings()
        self.banner: Callable[[str, str, str], None] = lambda level, msg, hint: None
        self.us_hardware_ok = bool(self.qs.value("diag/us_ok", False, type=bool))

    # ---------------------------------------------------------------- settings
    def _load_settings(self) -> Settings:
        base = Settings()
        changes: dict[str, object] = {}
        for f in dataclasses.fields(Settings):
            key = f"settings/{f.name}"
            if not self.qs.contains(key):
                continue
            default = getattr(base, f.name)
            try:
                if isinstance(default, bool):
                    changes[f.name] = self.qs.value(key, default, type=bool)
                elif isinstance(default, int):
                    changes[f.name] = int(str(self.qs.value(key, default)))
                elif isinstance(default, float):
                    changes[f.name] = float(str(self.qs.value(key, default)))
                elif isinstance(default, str):
                    changes[f.name] = str(self.qs.value(key, default))
            except (TypeError, ValueError):
                continue
        try:
            return base.replace(**changes)
        except ValueError:
            return base

    def update_settings(self, **changes: object) -> None:
        """Validate, apply and persist settings changes."""
        new = self.settings.replace(**changes)
        for name, value in changes.items():
            self.qs.setValue(f"settings/{name}", value)
        self.settings = new
        set_level(new.log_level)
        self.settings_changed.emit()

    def reset_setting(self, name: str) -> None:
        """Restore one setting to its default."""
        self.qs.remove(f"settings/{name}")
        self.settings = self.settings.replace(**{name: getattr(Settings(), name)})
        self.settings_changed.emit()

    # ---------------------------------------------------------------- devices
    def device(self, kind: str) -> int | None:
        """Selected device index (None = system default)."""
        v = self.qs.value(f"devices/{kind}", -1)
        try:
            i = int(str(v))
        except (TypeError, ValueError):
            return None
        return None if i < 0 else i

    def set_device(self, kind: str, index: int | None) -> None:
        """Persist a device choice."""
        self.qs.setValue(f"devices/{kind}", -1 if index is None else index)
        self.devices_changed.emit()

    @property
    def exclusive(self) -> bool:
        """WASAPI exclusive mode preference."""
        return self.settings.exclusive_mode

    def set_us_ok(self, ok: bool) -> None:
        """Remember whether the 96 kHz hardware check passed (enables the experimental profile)."""
        self.us_hardware_ok = ok
        self.qs.setValue("diag/us_ok", ok)
        self.devices_changed.emit()

    def make_backend(self, sample_rate: int, enable_input: bool = True, enable_output: bool = True):
        """Create (but do not start) the real audio backend for the chosen devices."""
        from ultrasonic_airdrop.audio.sd_backend import SoundDeviceBackend

        return SoundDeviceBackend(
            sample_rate,
            self.device("input"),
            self.device("output"),
            self.exclusive,
            enable_input=enable_input,
            enable_output=enable_output,
        )

    @property
    def destination(self) -> Path:
        """Receive destination folder."""
        return Path(str(self.qs.value("receive/dest", str(DEFAULT_DEST))))

    def set_destination(self, path: Path) -> None:
        """Persist the receive destination."""
        self.qs.setValue("receive/dest", str(path))
