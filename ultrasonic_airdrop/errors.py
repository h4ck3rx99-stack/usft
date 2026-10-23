"""Typed exceptions.  Each carries a user-facing message with an actionable next step."""

from __future__ import annotations


class AirDropError(Exception):
    """Base class for all application errors.

    ``user_message`` is safe to show in the GUI; ``hint`` suggests what to do next.
    """

    def __init__(self, user_message: str, hint: str = "") -> None:
        super().__init__(user_message if not hint else f"{user_message} — {hint}")
        self.user_message = user_message
        self.hint = hint


class FrameError(AirDropError):
    """A link frame or PHY header failed validation (bad magic, CRC, length, ...)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class FecError(AirDropError):
    """Reed-Solomon decoding failed (too many errors)."""

    def __init__(self, message: str = "uncorrectable codeword") -> None:
        super().__init__(message)


class ManifestError(AirDropError):
    """The received manifest is malformed or violates a limit."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Invalid manifest: {message}")


class FileSafetyError(AirDropError):
    """A filename or path from the peer is unsafe."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Unsafe file name: {message}")


class CryptoError(AirDropError):
    """Authentication failure (wrong passphrase or corrupted data)."""

    def __init__(self, message: str = "Passphrase mismatch or corrupted data") -> None:
        super().__init__(message, "Make sure both sides typed exactly the same passphrase.")


class AudioDeviceError(AirDropError):
    """The audio device could not be opened or failed while running."""


class TransferError(AirDropError):
    """A transfer ended unsuccessfully (link lost, rejected, cancelled, ...)."""
