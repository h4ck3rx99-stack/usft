"""Logging configuration: rotating file log in the user data dir plus console output.

Messages use a structured ``event | key=value ...`` layout produced by :func:`kv`.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def user_data_dir() -> Path:
    """Platform-appropriate per-user data directory for logs and state."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "UltrasonicAirDrop"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "UltrasonicAirDrop"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "ultrasonic-airdrop"


def log_file_path() -> Path:
    """Path of the rotating log file."""
    return user_data_dir() / "logs" / "airdrop.log"


def kv(event: str, **fields: object) -> str:
    """Format a structured log message: ``event | k1=v1 k2=v2``."""
    if not fields:
        return event
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            text = f"{value:.4g}"
        else:
            text = str(value)
        if " " in text or "=" in text:
            text = repr(text)
        parts.append(f"{key}={text}")
    return f"{event} | " + " ".join(parts)


def setup_logging(level: str = "INFO", console: bool = True, file: bool = True) -> None:
    """Configure the root ``ultrasonic_airdrop`` logger (idempotent)."""
    global _configured
    logger = logging.getLogger("ultrasonic_airdrop")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _configured:
        return
    formatter = logging.Formatter(LOG_FORMAT)
    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if file:
        try:
            path = log_file_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError as exc:  # read-only home, etc.
            logger.warning(kv("file logging disabled", error=exc))
    _configured = True


def set_level(level: str) -> None:
    """Change the log level at runtime."""
    logging.getLogger("ultrasonic_airdrop").setLevel(getattr(logging, level.upper(), logging.INFO))
