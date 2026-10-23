"""Streaming SHA-256 helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(path: Path, progress: Callable[[int], None] | None = None) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks (``progress`` gets bytes read so far)."""
    h = hashlib.sha256()
    done = 0
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress is not None:
                progress(done)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


EMPTY_SHA256 = sha256_bytes(b"")
