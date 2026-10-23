"""Sender-side ARQ bookkeeping: unit/part tracking, in-flight frames, SACK processing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ultrasonic_airdrop.config import UNIT_SIZE
from ultrasonic_airdrop.files.manifest import Manifest


@dataclass
class SentFrame:
    """A frame awaiting acknowledgement."""

    seq: int
    kind: str  # "data" | "manifest"
    file_index: int = -1
    start: int = 0
    count: int = 0
    part: int = -1
    nbytes: int = 0


class UnitTracker:
    """Per-file acknowledged / in-flight unit bitmaps.

    Units are always chosen lowest-first among those neither acknowledged nor in
    flight, so retransmissions of lost units naturally go ahead of new data.
    Files whose on-air content duplicates an earlier file are pre-acknowledged.
    """

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        self.acked = [np.zeros(f.n_units, dtype=bool) for f in manifest.files]
        self.inflight = [np.zeros(f.n_units, dtype=bool) for f in manifest.files]
        self.alias = [manifest.canonical_index(f.index) != f.index for f in manifest.files]
        for i, is_alias in enumerate(self.alias):
            if is_alias:
                self.acked[i][:] = True
        self.sent_once = [np.zeros(f.n_units, dtype=bool) for f in manifest.files]
        self.retransmitted_units = 0

    def next_run(self, max_units: int) -> tuple[int, int, int] | None:
        """``(file_index, start, count)`` of the next units to send, or ``None``."""
        for fi, (ack, inf) in enumerate(zip(self.acked, self.inflight, strict=True)):
            free = ~(ack | inf)
            if not free.any():
                continue
            start = int(np.argmax(free))
            end = start
            limit = min(len(free), start + max_units)
            while end < limit and free[end]:
                end += 1
            return fi, start, end - start
        return None

    def mark_inflight(self, fi: int, start: int, count: int) -> None:
        """Units handed to a frame."""
        sl = slice(start, start + count)
        self.retransmitted_units += int(self.sent_once[fi][sl].sum())
        self.sent_once[fi][sl] = True
        self.inflight[fi][sl] = True

    def release(self, fi: int, start: int, count: int) -> None:
        """Frame lost: units become eligible again."""
        self.inflight[fi][start : start + count] = False

    def mark_acked(self, fi: int, start: int, count: int) -> None:
        """Frame acknowledged."""
        sl = slice(start, start + count)
        self.acked[fi][sl] = True
        self.inflight[fi][sl] = False

    def mark_missing(self, fi: int, start: int, count: int) -> None:
        """Receiver says these units are missing (authoritative)."""
        if not 0 <= fi < len(self.acked) or self.alias[fi]:
            return
        sl = slice(max(0, start), min(len(self.acked[fi]), start + count))
        self.acked[fi][sl] = False
        self.inflight[fi][sl] = False

    def mark_have(self, fi: int, start: int, count: int) -> None:
        """Receiver already holds these units (resume)."""
        if not 0 <= fi < len(self.acked):
            return
        self.acked[fi][max(0, start) : min(len(self.acked[fi]), start + count)] = True

    def acked_bytes(self) -> int:
        """Acknowledged on-air bytes (aliases excluded)."""
        total = 0
        for f, ack, alias in zip(self.manifest.files, self.acked, self.alias, strict=True):
            if alias or f.n_units == 0:
                continue
            full = int(ack[:-1].sum()) * UNIT_SIZE
            total += full + (f.unit_len(f.n_units - 1) if ack[-1] else 0)
        return total

    @property
    def total_bytes(self) -> int:
        """Total on-air bytes (aliases excluded)."""
        return sum(f.tx_size for f, a in zip(self.manifest.files, self.alias, strict=True) if not a)

    def all_acked(self) -> bool:
        """Everything acknowledged."""
        return all(bool(a.all()) for a in self.acked)

    def unit_bytes(self, fi: int, start: int, count: int) -> int:
        """Bytes in a unit range."""
        f = self.manifest.files[fi]
        return min(count * UNIT_SIZE, f.tx_size - start * UNIT_SIZE)


@dataclass
class PartTracker:
    """Manifest part bookkeeping."""

    count: int
    acked: set[int] = field(default_factory=set)
    inflight: set[int] = field(default_factory=set)

    def next_part(self) -> int | None:
        """Lowest part neither acked nor in flight."""
        for i in range(self.count):
            if i not in self.acked and i not in self.inflight:
                return i
        return None

    def all_acked(self) -> bool:
        """All parts acknowledged."""
        return len(self.acked) == self.count


class Outstanding:
    """Frames sent in the current window, keyed by seq."""

    def __init__(self) -> None:
        self.frames: dict[int, SentFrame] = {}

    def add(self, frame: SentFrame) -> None:
        """Track a sent frame."""
        self.frames[frame.seq] = frame

    def resolve(self, acked_seqs: set[int], final: bool) -> tuple[list[SentFrame], list[SentFrame]]:
        """Split outstanding frames into (acked, lost).

        With ``final`` (the ACK answers our poll) every unacknowledged frame is lost;
        otherwise only positive acknowledgements are applied.
        """
        acked = [f for s, f in self.frames.items() if s in acked_seqs]
        for f in acked:
            del self.frames[f.seq]
        lost: list[SentFrame] = []
        if final:
            lost = list(self.frames.values())
            self.frames.clear()
        return acked, lost

    def __len__(self) -> int:
        return len(self.frames)
