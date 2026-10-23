"""Audio device enumeration and capability checks (sounddevice / PortAudio)."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from ultrasonic_airdrop.errors import AudioDeviceError
from ultrasonic_airdrop.logging_setup import kv

log = logging.getLogger(__name__)

MIC_PRIVACY_HINT = (
    'On Windows open Settings → Privacy & security → Microphone and enable "Let desktop apps access your microphone".'
)


def load_sounddevice() -> ModuleType:
    """Import sounddevice, converting a missing PortAudio library into a clear error."""
    try:
        import sounddevice as sd
    except OSError as exc:  # PortAudio shared library missing
        raise AudioDeviceError(
            "The PortAudio audio library is not available.",
            "On Linux install libportaudio2; on Windows/macOS reinstall the 'sounddevice' package.",
        ) from exc
    return sd  # type: ignore[no-any-return]


@dataclass(frozen=True)
class DeviceInfo:
    """One audio device."""

    index: int
    name: str
    hostapi: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float

    @property
    def label(self) -> str:
        """Human-readable label including the host API."""
        return f"{self.name} [{self.hostapi}]"


def list_devices() -> list[DeviceInfo]:
    """All devices known to PortAudio."""
    sd = load_sounddevice()
    try:
        apis = sd.query_hostapis()
        devs = sd.query_devices()
    except Exception as exc:
        raise AudioDeviceError("Could not enumerate audio devices.", str(exc)) from exc
    out = []
    for i, d in enumerate(devs):
        out.append(
            DeviceInfo(
                index=i,
                name=str(d["name"]),
                hostapi=str(apis[d["hostapi"]]["name"]),
                max_input_channels=int(d["max_input_channels"]),
                max_output_channels=int(d["max_output_channels"]),
                default_samplerate=float(d["default_samplerate"]),
            )
        )
    return out


def preferred_hostapi_order() -> list[str]:
    """Host APIs in order of preference (WASAPI first on Windows)."""
    if sys.platform == "win32":
        return ["Windows WASAPI", "Windows WDM-KS", "Windows DirectSound", "MME"]
    if sys.platform == "darwin":
        return ["Core Audio"]
    return ["ALSA", "JACK Audio Connection Kit", "PulseAudio", "OSS"]


def default_device(kind: str) -> int | None:
    """Default input/output device index, preferring WASAPI on Windows."""
    devices = list_devices()
    need_in = kind == "input"
    sd = load_sounddevice()
    try:
        default = sd.default.device[0 if need_in else 1]
    except Exception:
        default = -1
    candidates = [d for d in devices if (d.max_input_channels if need_in else d.max_output_channels) > 0]
    if not candidates:
        return None
    if sys.platform == "win32" and default is not None and default >= 0:
        # map the system default to its WASAPI twin if there is one
        name = devices[default].name
        for d in candidates:
            if d.hostapi == "Windows WASAPI" and (d.name.startswith(name[:20]) or name.startswith(d.name[:20])):
                return d.index
    if default is not None and default >= 0:
        return int(default)
    for api in preferred_hostapi_order():
        for d in candidates:
            if d.hostapi == api:
                return d.index
    return candidates[0].index


def check_rate(device: int | None, rate: int, kind: str, extra_settings: Any = None) -> str | None:
    """Return ``None`` if ``rate`` is supported for ``device`` (``kind`` = input/output), else an error text."""
    sd = load_sounddevice()
    try:
        if kind == "input":
            sd.check_input_settings(
                device=device, samplerate=rate, channels=1, dtype="float32", extra_settings=extra_settings
            )
        else:
            sd.check_output_settings(
                device=device, samplerate=rate, channels=1, dtype="float32", extra_settings=extra_settings
            )
    except Exception as exc:
        log.info(kv("rate check failed", device=device, rate=rate, kind=kind, error=exc))
        return str(exc)
    return None


def describe_error(exc: BaseException, kind: str) -> AudioDeviceError:
    """Translate a PortAudio error into an actionable :class:`AudioDeviceError`."""
    text = str(exc)
    low = text.lower()
    if "invalid sample rate" in low or "samplerate" in low:
        return AudioDeviceError(
            f"The {kind} device does not support the required sample rate.",
            "Set the device to 48000 Hz in the Windows Sound Control Panel (Advanced tab), or choose another device.",
        )
    if "unanticipated host error" in low or "device unavailable" in low or "busy" in low or "-9985" in low:
        return AudioDeviceError(
            f"The {kind} device is busy or unavailable.",
            "Close other applications using it (or disable exclusive mode) and try again.",
        )
    if "permission" in low or "access" in low or "denied" in low:
        return AudioDeviceError(f"Access to the {kind} device was denied.", MIC_PRIVACY_HINT)
    if "invalid device" in low or "no default" in low or "-9996" in low or "querying device -1" in low:
        return AudioDeviceError(
            f"No usable {kind} device was found.",
            "Connect a microphone/speaker and pick it on the Diagnostics page (or with --in/--out).",
        )
    return AudioDeviceError(
        f"Could not open the {kind} device: {text}", "Check the device selection on the Diagnostics page."
    )
