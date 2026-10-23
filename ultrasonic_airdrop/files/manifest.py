"""Transfer manifest: build, serialise (zlib JSON) and strictly validate.

Wire form: UTF-8 JSON, zlib-compressed::

    {"v": 1, "unit": 32,
     "files": [{"i": 0, "name": "...", "size": orig_bytes, "sha256": "hex",
                "tx_size": bytes_on_air, "compressed": bool, "mtime": int}, ...]}
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zlib
from dataclasses import dataclass
from typing import Any

from ultrasonic_airdrop.config import MAX_MANIFEST_BYTES, MAX_MANIFEST_FILES, UNIT_SIZE
from ultrasonic_airdrop.errors import FileSafetyError, ManifestError
from ultrasonic_airdrop.files.sanitize import sanitize_filename

MANIFEST_VERSION = 1
MAX_FILE_SIZE = 1 << 40
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ENTRY_KEYS = {"i", "name", "size", "sha256", "tx_size", "compressed", "mtime"}


@dataclass(frozen=True)
class ManifestEntry:
    """One file of a transfer."""

    index: int
    name: str
    size: int
    sha256: str
    tx_size: int
    compressed: bool
    mtime: int

    @property
    def n_units(self) -> int:
        """Number of 32-byte units on air."""
        return math.ceil(self.tx_size / UNIT_SIZE)

    def unit_len(self, unit: int) -> int:
        """Length of unit ``unit`` (the last one may be short)."""
        if not 0 <= unit < self.n_units:
            raise IndexError("unit out of range")
        return min(UNIT_SIZE, self.tx_size - unit * UNIT_SIZE)

    @property
    def content_key(self) -> tuple[str, int, bool]:
        """Identity of the bytes on air (used for resume and duplicate-content aliasing)."""
        return (self.sha256, self.tx_size, self.compressed)

    def to_json(self) -> dict[str, Any]:
        """JSON object form."""
        return {
            "i": self.index,
            "name": self.name,
            "size": self.size,
            "sha256": self.sha256,
            "tx_size": self.tx_size,
            "compressed": self.compressed,
            "mtime": self.mtime,
        }


@dataclass(frozen=True)
class Manifest:
    """Validated list of files."""

    files: tuple[ManifestEntry, ...]
    unit: int = UNIT_SIZE

    @property
    def total_size(self) -> int:
        """Sum of original sizes."""
        return sum(f.size for f in self.files)

    @property
    def total_tx(self) -> int:
        """Sum of on-air sizes."""
        return sum(f.tx_size for f in self.files)

    def canonical_index(self, index: int) -> int:
        """Index of the first file with identical on-air content (itself if unique)."""
        key = self.files[index].content_key
        for f in self.files:
            if f.content_key == key:
                return f.index
        return index

    def to_json_bytes(self) -> bytes:
        """Canonical JSON encoding."""
        obj = {"v": MANIFEST_VERSION, "files": [f.to_json() for f in self.files], "unit": self.unit}
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def to_blob(self) -> bytes:
        """zlib-compressed JSON (the bytes sent in MANIFEST parts)."""
        raw = self.to_json_bytes()
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ManifestError("manifest exceeds 256 KB")
        return zlib.compress(raw, 9)

    @staticmethod
    def blob_sha256(blob: bytes) -> bytes:
        """SHA-256 of the blob (sent in OFFER)."""
        return hashlib.sha256(blob).digest()

    @classmethod
    def from_blob(cls, blob: bytes, max_total_bytes: int | None = None) -> Manifest:
        """Decompress (bounded) and validate."""
        d = zlib.decompressobj()
        try:
            raw = d.decompress(blob, MAX_MANIFEST_BYTES + 1)
        except zlib.error as exc:
            raise ManifestError("not valid zlib data") from exc
        if len(raw) > MAX_MANIFEST_BYTES or d.unconsumed_tail:
            raise ManifestError("manifest exceeds 256 KB")
        if not d.eof:
            raise ManifestError("truncated zlib stream")
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ManifestError("not valid UTF-8 JSON") from exc
        return cls.from_json(obj, max_total_bytes)

    @classmethod
    def from_json(cls, obj: Any, max_total_bytes: int | None = None) -> Manifest:
        """Validate a parsed JSON object strictly."""
        if not isinstance(obj, dict) or set(obj) != {"v", "files", "unit"}:
            raise ManifestError("unexpected top-level structure")
        if type(obj["v"]) is not int or obj["v"] != MANIFEST_VERSION:
            raise ManifestError("unsupported manifest version")
        if type(obj["unit"]) is not int or obj["unit"] != UNIT_SIZE:
            raise ManifestError("unsupported unit size")
        files = obj["files"]
        if not isinstance(files, list) or not 1 <= len(files) <= MAX_MANIFEST_FILES:
            raise ManifestError(f"file list must have 1..{MAX_MANIFEST_FILES} entries")
        entries = []
        for pos, f in enumerate(files):
            entries.append(_validate_entry(f, pos))
        m = cls(tuple(entries))
        if max_total_bytes is not None and m.total_size > max_total_bytes:
            raise ManifestError(f"total size {m.total_size} exceeds the limit of {max_total_bytes} bytes")
        return m


def _int_field(f: dict[str, Any], key: str, lo: int, hi: int) -> int:
    v = f[key]
    if type(v) is not int or not lo <= v <= hi:
        raise ManifestError(f"field {key!r} out of range")
    return v


def _validate_entry(f: Any, pos: int) -> ManifestEntry:
    if not isinstance(f, dict) or set(f) != _ENTRY_KEYS:
        raise ManifestError(f"entry {pos} has unexpected keys")
    index = _int_field(f, "i", 0, MAX_MANIFEST_FILES - 1)
    if index != pos:
        raise ManifestError("file indices must be 0..n-1 in order (duplicate or missing index)")
    name = f["name"]
    if not isinstance(name, str) or not name or len(name) > 1024:
        raise ManifestError(f"entry {pos} has an invalid name")
    try:
        clean = sanitize_filename(name, strict=True)
    except FileSafetyError as exc:
        raise ManifestError(exc.user_message) from exc
    size = _int_field(f, "size", 0, MAX_FILE_SIZE)
    tx_size = _int_field(f, "tx_size", 0, MAX_FILE_SIZE)
    mtime = _int_field(f, "mtime", -(1 << 40), 1 << 40)
    sha = f["sha256"]
    if not isinstance(sha, str) or not _HEX64.match(sha):
        raise ManifestError(f"entry {pos} has an invalid sha256")
    compressed = f["compressed"]
    if type(compressed) is not bool:
        raise ManifestError(f"entry {pos} has an invalid compressed flag")
    if not compressed and tx_size != size:
        raise ManifestError(f"entry {pos}: tx_size must equal size for uncompressed files")
    if compressed and (size == 0 or tx_size == 0):
        raise ManifestError(f"entry {pos}: empty files cannot be compressed")
    if compressed and tx_size > size + size // 10 + 64:
        raise ManifestError(f"entry {pos}: implausible compressed size")
    return ManifestEntry(index, clean, size, sha, tx_size, compressed, mtime)


def split_blob(blob: bytes, part_size: int) -> list[bytes]:
    """Split the manifest blob into MANIFEST parts."""
    if part_size <= 0:
        raise ValueError("part size must be positive")
    return [blob[i : i + part_size] for i in range(0, len(blob), part_size)] or [b""]
