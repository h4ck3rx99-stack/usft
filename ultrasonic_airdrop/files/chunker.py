"""Sender-side file preparation and unit reads.

Each file is streamed once to compute its SHA-256 and (optionally) a zlib
compressed copy in a temp file; the compressed copy is used only if it is at
least 5 % smaller.  Units (32 bytes) are then read on demand by seeking, so
large files are never loaded into memory.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import tempfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from ultrasonic_airdrop.config import COMPRESSION_MIN_SAVING, MAX_MANIFEST_FILES, UNIT_SIZE
from ultrasonic_airdrop.errors import AirDropError
from ultrasonic_airdrop.files.manifest import Manifest, ManifestEntry
from ultrasonic_airdrop.files.sanitize import sanitize_filename
from ultrasonic_airdrop.logging_setup import kv

log = logging.getLogger(__name__)
CHUNK = 1 << 20


def n_units(tx_size: int, unit: int = UNIT_SIZE) -> int:
    """Number of units for ``tx_size`` bytes."""
    return math.ceil(tx_size / unit)


def outgoing_name(path: Path) -> str:
    """Name advertised for a local file (safe on every OS)."""
    return sanitize_filename(path.name.replace(":", "_").replace("\\", "_"), strict=False)


@dataclass
class PreparedFile:
    """A file ready to send: ``source`` holds exactly the on-air bytes."""

    entry: ManifestEntry
    origin: Path
    source: Path
    temporary: bool


def prepare_files(
    paths: list[Path],
    compression: bool = True,
    workdir: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Manifest, list[PreparedFile]]:
    """Hash (and optionally compress) ``paths``; return the manifest and prepared files."""
    if not paths:
        raise AirDropError("No files selected.", "Add at least one file.")
    if len(paths) > MAX_MANIFEST_FILES:
        raise AirDropError(f"Too many files ({len(paths)}).", f"Send at most {MAX_MANIFEST_FILES} files at once.")
    total = 0
    for p in paths:
        if not p.is_file():
            raise AirDropError(f"Not a regular file: {p}", "Folders are not supported; add the files inside.")
        total += p.stat().st_size
    done = 0
    prepared: list[PreparedFile] = []
    tmp_root = Path(workdir or tempfile.gettempdir())
    for i, p in enumerate(paths):
        st = p.stat()
        h = hashlib.sha256()
        tmp_path: Path | None = None
        comp_size = 0
        comp = zlib.compressobj(6) if compression and st.st_size > 0 else None
        out: BinaryIO | None = None
        try:
            if comp is not None:
                fd, name = tempfile.mkstemp(prefix="airdrop-", suffix=".z", dir=tmp_root)
                tmp_path = Path(name)
                out = os.fdopen(fd, "wb")
            with p.open("rb") as fh:
                while True:
                    block = fh.read(CHUNK)
                    if not block:
                        break
                    h.update(block)
                    if comp is not None and out is not None:
                        c = comp.compress(block)
                        comp_size += len(c)
                        out.write(c)
                    done += len(block)
                    if progress:
                        progress(done, total)
            if comp is not None and out is not None:
                c = comp.flush()
                comp_size += len(c)
                out.write(c)
        finally:
            if out is not None:
                out.close()
        use_comp = tmp_path is not None and comp_size <= st.st_size * (1.0 - COMPRESSION_MIN_SAVING)
        if tmp_path is not None and not use_comp:
            tmp_path.unlink(missing_ok=True)
            tmp_path = None
        entry = ManifestEntry(
            index=i,
            name=outgoing_name(p),
            size=st.st_size,
            sha256=h.hexdigest(),
            tx_size=comp_size if use_comp else st.st_size,
            compressed=use_comp,
            mtime=int(st.st_mtime),
        )
        log.info(kv("prepared file", name=entry.name, size=entry.size, tx_size=entry.tx_size, compressed=use_comp))
        prepared.append(
            PreparedFile(entry=entry, origin=p, source=tmp_path if tmp_path else p, temporary=tmp_path is not None)
        )
    return Manifest(tuple(pf.entry for pf in prepared)), prepared


def cleanup_prepared(files: list[PreparedFile]) -> None:
    """Delete temporary compressed copies."""
    for pf in files:
        if pf.temporary:
            pf.source.unlink(missing_ok=True)


class UnitReader:
    """Random-access reads of units from prepared files."""

    def __init__(self, files: list[PreparedFile]) -> None:
        self._files = files
        self._handles: dict[int, BinaryIO] = {}

    def read(self, file_index: int, start_unit: int, count: int) -> bytes:
        """Read ``count`` units starting at ``start_unit`` (the last unit may be short)."""
        pf = self._files[file_index]
        tx = pf.entry.tx_size
        off = start_unit * UNIT_SIZE
        length = min(count * UNIT_SIZE, tx - off)
        if off < 0 or length <= 0:
            raise ValueError("unit range outside file")
        fh = self._handles.get(file_index)
        if fh is None:
            fh = pf.source.open("rb")
            self._handles[file_index] = fh
        fh.seek(off)
        data = fh.read(length)
        if len(data) != length:
            raise AirDropError(f"{pf.origin} changed while sending.", "Do not modify files during a transfer.")
        return data

    def close(self) -> None:
        """Close all open handles."""
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()
