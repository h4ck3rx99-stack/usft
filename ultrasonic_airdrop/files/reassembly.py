"""Receiver-side reassembly: preallocated ``.part`` files, unit bitmaps, verify, atomic rename.

* Units are written at their offsets into ``<dest>/.airdrop-partial/<sha256>.part``.
* Duplicates are ignored (counted); out-of-order units are fine.
* On finalisation the original bytes (decompressed if needed) are SHA-256
  verified; only then is the file moved with :func:`os.replace` to a unique,
  never-overwriting name in the destination.  A mismatch leaves
  ``<sha256>.corrupt`` in the partial directory and reports failure.
* Files whose on-air content is identical share one assembler (aliasing).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np

from ultrasonic_airdrop.config import PARTIAL_DIR_NAME, UNIT_SIZE
from ultrasonic_airdrop.errors import FileSafetyError
from ultrasonic_airdrop.files import resume
from ultrasonic_airdrop.files.manifest import Manifest, ManifestEntry
from ultrasonic_airdrop.files.sanitize import safe_destination
from ultrasonic_airdrop.link.frames import (
    FILE_HASH_MISMATCH,
    FILE_INCOMPLETE,
    FILE_OK,
    FILE_WRITE_ERROR,
)
from ultrasonic_airdrop.logging_setup import kv

log = logging.getLogger(__name__)
CHUNK = 1 << 20


class FileAssembler:
    """Collects the units of one on-air content (one or more aliased manifest entries)."""

    def __init__(self, partial_dir: Path, entry: ManifestEntry, session_id: int, manifest_hex: str) -> None:
        self.partial_dir = partial_dir
        self.entry = entry
        self.n_units = entry.n_units
        self.part_path = partial_dir / f"{entry.sha256}.part"
        self.bitmap = np.zeros(self.n_units, dtype=bool)
        self.duplicates = 0
        self.resumed_units = 0
        self.status: int | None = None
        self.final_paths: list[Path] = []
        self._fh: IO[bytes] | None = None
        self._dirty = 0
        self._session = resume.session_record(session_id, manifest_hex)
        self._sessions: list[dict[str, object]] = []
        partial_dir.mkdir(parents=True, exist_ok=True)
        side = resume.load_sidecar(partial_dir, entry.sha256)
        if (
            side is not None
            and side.tx_size == entry.tx_size
            and side.compressed == entry.compressed
            and side.unit == UNIT_SIZE
            and self.part_path.exists()
            and self.part_path.stat().st_size == entry.tx_size
        ):
            self.bitmap = resume.ranges_to_bitmap(side.ranges, self.n_units)
            self.resumed_units = int(self.bitmap.sum())
            self._sessions = list(side.sessions)
            log.info(kv("resuming partial", sha256=entry.sha256[:12], units=self.resumed_units, total=self.n_units))
        elif self.part_path.exists():
            self.part_path.unlink()

    def _open(self) -> None:
        if self._fh is None:
            mode = "r+b" if self.part_path.exists() else "w+b"
            fh = self.part_path.open(mode)
            fh.truncate(self.entry.tx_size)
            self._fh = fh

    @property
    def complete(self) -> bool:
        """All units received."""
        return bool(self.bitmap.all()) if self.n_units else True

    @property
    def received_units(self) -> int:
        """Units held."""
        return int(self.bitmap.sum())

    def received_bytes(self) -> int:
        """On-air bytes held."""
        if self.n_units == 0:
            return 0
        full = int(self.bitmap[:-1].sum()) * UNIT_SIZE
        return full + (self.entry.unit_len(self.n_units - 1) if self.bitmap[-1] else 0)

    def expected_len(self, start: int, count: int) -> int:
        """Byte length of ``count`` units from ``start`` (raises ``ValueError`` if out of range)."""
        if count <= 0 or start < 0 or start + count > self.n_units:
            raise ValueError("unit range outside file")
        return min(count * UNIT_SIZE, self.entry.tx_size - start * UNIT_SIZE)

    def write_units(self, start: int, count: int, data: bytes) -> int:
        """Write units; returns the number of *new* units stored.  Raises ``ValueError`` if inconsistent."""
        if len(data) != self.expected_len(start, count):
            raise ValueError("DATA length does not match unit range")
        new = ~self.bitmap[start : start + count]
        if not new.any():
            self.duplicates += count
            return 0
        self._open()
        assert self._fh is not None
        self._fh.seek(start * UNIT_SIZE)
        self._fh.write(data)
        n_new = int(new.sum())
        self.duplicates += count - n_new
        self.bitmap[start : start + count] = True
        self._dirty += n_new
        return n_new

    def missing_ranges(self) -> list[tuple[int, int]]:
        """Runs of missing units."""
        return resume.bitmap_to_ranges(self.bitmap, value=False)

    def received_ranges(self) -> list[tuple[int, int]]:
        """Runs of received units."""
        return resume.bitmap_to_ranges(self.bitmap, value=True)

    def save_sidecar(self) -> None:
        """Persist progress for resume."""
        if self.status == FILE_OK or self.n_units == 0:
            return
        if self._fh is not None:
            self._fh.flush()
        side = resume.Sidecar(
            sha256=self.entry.sha256,
            tx_size=self.entry.tx_size,
            compressed=self.entry.compressed,
            unit=UNIT_SIZE,
            ranges=self.received_ranges(),
            sessions=[*self._sessions, self._session],
        )
        try:
            resume.save_sidecar(self.partial_dir, side)
        except OSError as exc:
            log.warning(kv("could not save resume sidecar", error=exc))
        self._dirty = 0

    def close(self) -> None:
        """Close the part file."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _verify_to_temp(self) -> tuple[bool, Path]:
        """Produce the original bytes in a temp file and check the hash."""
        e = self.entry
        h = hashlib.sha256()
        if e.tx_size == 0:
            tmp = self.partial_dir / f"{e.sha256}.empty"
            tmp.write_bytes(b"")
            return h.hexdigest() == e.sha256, tmp
        if not e.compressed:
            with self.part_path.open("rb") as fh:
                while block := fh.read(CHUNK):
                    h.update(block)
            return h.hexdigest() == e.sha256, self.part_path
        tmp = self.partial_dir / f"{e.sha256}.dec"
        d = zlib.decompressobj()
        written = 0
        ok = True
        try:
            with self.part_path.open("rb") as src, tmp.open("wb") as dst:
                while block := src.read(CHUNK):
                    out = d.decompress(block, e.size + 1 - written)
                    while True:
                        written += len(out)
                        if written > e.size:
                            ok = False
                            break
                        h.update(out)
                        dst.write(out)
                        if not d.unconsumed_tail:
                            break
                        out = d.decompress(d.unconsumed_tail, e.size + 1 - written)
                    if not ok:
                        break
                if ok:
                    tail = d.flush()
                    written += len(tail)
                    h.update(tail)
                    dst.write(tail)
        except zlib.error:
            ok = False
        ok = ok and d.eof and written == e.size and h.hexdigest() == e.sha256
        return ok, tmp

    def finalize(self, dest: Path, names: list[str]) -> int:
        """Verify and move into ``dest`` under each of ``names`` (aliases).  Returns a file status."""
        if self.status in (FILE_OK, FILE_HASH_MISMATCH):
            return self.status
        if not self.complete:
            self.save_sidecar()
            return FILE_INCOMPLETE
        self.close()
        try:
            ok, produced = self._verify_to_temp()
        except OSError as exc:
            log.error(kv("verification I/O error", error=exc))
            self.status = FILE_WRITE_ERROR
            return self.status
        if not ok:
            if produced != self.part_path:
                produced.unlink(missing_ok=True)
            corrupt = self.partial_dir / f"{self.entry.sha256}.corrupt"
            if self.part_path.exists():
                os.replace(self.part_path, corrupt)
            resume.delete_sidecar(self.partial_dir, self.entry.sha256)
            self.bitmap[:] = False
            log.error(kv("hash mismatch", file=names[0], expected=self.entry.sha256[:16]))
            self.status = FILE_HASH_MISMATCH
            return self.status
        try:
            first = safe_destination(dest, names[0])
            os.replace(produced, first)
            self.final_paths = [first]
            for alias in names[1:]:
                target = safe_destination(dest, alias)
                shutil.copyfile(first, target)
                self.final_paths.append(target)
            if self.entry.mtime > 0:
                for p in self.final_paths:
                    os.utime(p, (self.entry.mtime, self.entry.mtime))
        except (OSError, FileSafetyError) as exc:
            log.error(kv("could not place received file", error=exc))
            self.status = FILE_WRITE_ERROR
            return self.status
        if self.part_path.exists():
            self.part_path.unlink()
        resume.delete_sidecar(self.partial_dir, self.entry.sha256)
        self.status = FILE_OK
        log.info(kv("file verified", path=self.final_paths[0], sha256=self.entry.sha256[:16]))
        return self.status

    def discard(self) -> None:
        """Delete partial data and sidecar."""
        self.close()
        self.part_path.unlink(missing_ok=True)
        resume.delete_sidecar(self.partial_dir, self.entry.sha256)


@dataclass
class FileResult:
    """Outcome for one manifest entry."""

    index: int
    name: str
    status: int
    path: Path | None


class Reassembly:
    """All assemblers for one manifest."""

    def __init__(self, dest: Path, manifest: Manifest, session_id: int, manifest_hex: str) -> None:
        self.dest = dest
        self.partial_dir = dest / PARTIAL_DIR_NAME
        self.manifest = manifest
        self._by_key: dict[tuple[str, int, bool], FileAssembler] = {}
        self.assemblers: list[FileAssembler] = []
        for e in manifest.files:
            asm = self._by_key.get(e.content_key)
            if asm is None:
                asm = FileAssembler(self.partial_dir, e, session_id, manifest_hex)
                self._by_key[e.content_key] = asm
            self.assemblers.append(asm)

    def assembler(self, index: int) -> FileAssembler:
        """Assembler for manifest entry ``index`` (raises ``IndexError``)."""
        if not 0 <= index < len(self.assemblers):
            raise IndexError("file index out of range")
        return self.assemblers[index]

    def unique(self) -> list[FileAssembler]:
        """Distinct assemblers."""
        return list(self._by_key.values())

    @property
    def total_tx(self) -> int:
        """Total on-air bytes of distinct contents."""
        return sum(a.entry.tx_size for a in self.unique())

    def received_tx(self) -> int:
        """On-air bytes held."""
        return sum(a.received_bytes() for a in self.unique())

    def complete(self) -> bool:
        """Every file complete."""
        return all(a.complete for a in self.unique())

    def finalize_file(self, index: int) -> int:
        """Finalise the content of entry ``index`` and all its aliases."""
        asm = self.assembler(index)
        names = [e.name for e, a in zip(self.manifest.files, self.assemblers, strict=True) if a is asm]
        return asm.finalize(self.dest, names)

    def finalize_all(self) -> list[FileResult]:
        """Finalise every file; returns per-entry results."""
        results = []
        for e, asm in zip(self.manifest.files, self.assemblers, strict=True):
            if asm.status is None or asm.status != FILE_OK:
                self.finalize_file(e.index)
            status = asm.status if asm.status is not None else FILE_INCOMPLETE
            if not asm.complete and status != FILE_HASH_MISMATCH:
                status = FILE_INCOMPLETE
            path = None
            if status == FILE_OK:
                k = [i for i, a in enumerate(self.assemblers) if a is asm].index(e.index)
                path = asm.final_paths[k] if k < len(asm.final_paths) else None
            results.append(FileResult(e.index, e.name, status, path))
        return results

    def save(self) -> None:
        """Persist all sidecars."""
        for a in self.unique():
            a.save_sidecar()

    def close(self, discard: bool = False) -> None:
        """Close files; optionally delete partial data.  Removes the partial dir if it is empty."""
        for a in self.unique():
            if discard and a.status != FILE_OK:
                a.discard()
            else:
                a.save_sidecar()
                a.close()
        with contextlib.suppress(OSError):  # not empty (partials kept for resume) or already gone
            self.partial_dir.rmdir()
