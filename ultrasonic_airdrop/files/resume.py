"""Resume sidecars: ``<partial dir>/<sha256>.json`` recording received unit ranges.

Partials are matched by file SHA-256 + on-air size + compression flag — never by name.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ultrasonic_airdrop.logging_setup import kv

log = logging.getLogger(__name__)
SIDECAR_VERSION = 1


def bitmap_to_ranges(bitmap: npt.NDArray[np.bool_], value: bool = True) -> list[tuple[int, int]]:
    """Runs of ``value`` in a boolean array as ``(start, count)`` tuples."""
    arr = np.asarray(bitmap, dtype=bool)
    if arr.size == 0:
        return []
    target = arr if value else ~arr
    padded = np.concatenate([[False], target, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e - s)) for s, e in zip(starts, ends, strict=True)]


def ranges_to_bitmap(ranges: list[tuple[int, int]], n: int) -> npt.NDArray[np.bool_]:
    """Inverse of :func:`bitmap_to_ranges` (out-of-range parts are ignored)."""
    out = np.zeros(n, dtype=bool)
    for start, count in ranges:
        if start < 0 or count <= 0:
            continue
        out[start : min(n, start + count)] = True
    return out


@dataclass
class Sidecar:
    """Contents of a resume sidecar."""

    sha256: str
    tx_size: int
    compressed: bool
    unit: int
    ranges: list[tuple[int, int]] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """JSON form."""
        return {
            "v": SIDECAR_VERSION,
            "sha256": self.sha256,
            "tx_size": self.tx_size,
            "compressed": self.compressed,
            "unit": self.unit,
            "ranges": [list(r) for r in self.ranges],
            "sessions": self.sessions[-20:],
        }

    @classmethod
    def from_json(cls, obj: Any) -> Sidecar:
        """Parse; raises ``ValueError`` if malformed."""
        if not isinstance(obj, dict) or obj.get("v") != SIDECAR_VERSION:
            raise ValueError("unsupported sidecar")
        ranges = [(int(a), int(b)) for a, b in obj["ranges"]]
        sessions = [s for s in obj.get("sessions", []) if isinstance(s, dict)]
        return cls(
            str(obj["sha256"]),
            int(obj["tx_size"]),
            bool(obj["compressed"]),
            int(obj["unit"]),
            ranges,
            sessions,
        )


def sidecar_path(partial_dir: Path, sha256: str) -> Path:
    """Path of the sidecar for a file hash."""
    return partial_dir / f"{sha256}.json"


def load_sidecar(partial_dir: Path, sha256: str) -> Sidecar | None:
    """Load a sidecar; returns ``None`` if absent or unreadable."""
    path = sidecar_path(partial_dir, sha256)
    try:
        return Sidecar.from_json(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning(kv("ignoring unreadable resume sidecar", path=path, error=exc))
        return None


def save_sidecar(partial_dir: Path, sidecar: Sidecar) -> None:
    """Atomically write a sidecar."""
    partial_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_path(partial_dir, sidecar.sha256)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sidecar.to_json()), encoding="utf-8")
    os.replace(tmp, path)


def delete_sidecar(partial_dir: Path, sha256: str) -> None:
    """Remove a sidecar if present."""
    with contextlib.suppress(FileNotFoundError):
        sidecar_path(partial_dir, sha256).unlink()


def manifest_seen(partial_dir: Path, manifest_sha256_hex: str) -> bool:
    """True if any sidecar records a session for this manifest hash."""
    if not partial_dir.is_dir():
        return False
    for path in partial_dir.glob("*.json"):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for s in obj.get("sessions", []) if isinstance(obj, dict) else []:
            if isinstance(s, dict) and s.get("manifest") == manifest_sha256_hex:
                return True
    return False


def session_record(session_id: int, manifest_sha256_hex: str) -> dict[str, Any]:
    """A session-history entry for a sidecar."""
    return {"session_id": session_id, "manifest": manifest_sha256_hex, "time": int(time.time())}
