"""Reed-Solomon FEC over GF(2^8) with shortened codes, erasures and interleaving.

Uses ``creedsolo`` (Cython) automatically when importable, else pure-Python ``reedsolo``.
A frame body is split into ``k`` equal-size shortened codewords, each protected by
``nsym`` parity bytes, then byte-interleaved (see :mod:`.interleave`).  Erasure
positions roughly double the correction power: ``2*errors + erasures <= nsym``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from ultrasonic_airdrop.errors import FecError
from ultrasonic_airdrop.fec.interleave import deinterleave, deinterleave_array, interleave

try:  # pragma: no cover - depends on the environment
    from creedsolo import ReedSolomonError, RSCodec  # type: ignore[import-not-found,unused-ignore]

    RS_BACKEND = "creedsolo"
except ImportError:
    from reedsolo import ReedSolomonError, RSCodec

    RS_BACKEND = "reedsolo"

log = logging.getLogger(__name__)

RS_N_MAX = 255


@lru_cache(maxsize=16)
def _codec(nsym: int) -> RSCodec:
    return RSCodec(nsym=nsym, nsize=RS_N_MAX)


@dataclass(frozen=True)
class SplitPlan:
    """How a body of ``body_len`` bytes maps onto ``k`` codewords of ``d`` data bytes."""

    body_len: int
    nsym: int
    k: int
    d: int

    @property
    def codeword_len(self) -> int:
        """Bytes per codeword including parity."""
        return self.d + self.nsym

    @property
    def coded_len(self) -> int:
        """Total bytes on air after FEC."""
        return self.k * self.codeword_len


def split_plan(body_len: int, nsym: int) -> SplitPlan:
    """Compute the equal-size shortened-code split for a body."""
    if body_len <= 0:
        raise ValueError("body must not be empty")
    if not 0 < nsym < RS_N_MAX - 1:
        raise ValueError("invalid nsym")
    max_d = RS_N_MAX - nsym
    k = math.ceil(body_len / max_d)
    d = math.ceil(body_len / k)
    return SplitPlan(body_len, nsym, k, d)


def coded_length(body_len: int, nsym: int) -> int:
    """Number of bytes on air for a body of ``body_len`` bytes with ``nsym`` parity per codeword."""
    return split_plan(body_len, nsym).coded_len


def rs_encode(data: bytes, nsym: int) -> bytes:
    """Encode one (shortened) codeword."""
    if len(data) + nsym > RS_N_MAX:
        raise ValueError("codeword too long")
    return bytes(_codec(nsym).encode(bytearray(data)))


def rs_decode(codeword: bytes, nsym: int, erase_pos: list[int] | None = None) -> tuple[bytes, int]:
    """Decode one codeword; returns ``(data, corrected_symbol_count)``.  Raises :class:`FecError`."""
    erase = sorted(set(erase_pos or []))
    if len(erase) > nsym:
        raise FecError("too many erasures")
    try:
        decoded, _full, errata = _codec(nsym).decode(bytearray(codeword), erase_pos=erase or None)
    except ReedSolomonError as exc:
        raise FecError(str(exc)) from exc
    return bytes(decoded), len(errata)


def encode_body(body: bytes, nsym: int) -> bytes:
    """Split ``body`` into equal shortened codewords, add parity and interleave."""
    plan = split_plan(len(body), nsym)
    padded = body + bytes(plan.k * plan.d - len(body))
    codewords = [rs_encode(padded[i * plan.d : (i + 1) * plan.d], nsym) for i in range(plan.k)]
    return interleave(codewords)


@dataclass
class DecodeStats:
    """Side information from :func:`decode_body`."""

    corrected: int = 0
    erasures_used: int = 0
    codewords: int = 0


def _try_decode(cw: bytes, nsym: int, erase: list[int]) -> tuple[bytes, int] | None:
    try:
        return rs_decode(cw, nsym, erase)
    except FecError:
        return None


def decode_body(
    received: bytes,
    body_len: int,
    nsym: int,
    erasures: npt.NDArray[np.bool_] | None = None,
    confidence: npt.NDArray[np.floating] | None = None,
) -> tuple[bytes, DecodeStats]:
    """De-interleave and RS-decode a received body.

    ``erasures`` flags unreliable bytes (in on-air order); ``confidence`` (higher is
    better) ranks them so that at most ``nsym - 2`` of the least reliable flagged bytes
    are passed as erasures.  If decoding with erasures fails the codeword is retried
    without erasures and with half of them.  Raises :class:`FecError` on failure.
    """
    plan = split_plan(body_len, nsym)
    if len(received) != plan.coded_len:
        raise FecError(f"coded length {len(received)} != expected {plan.coded_len}")
    cws = deinterleave(received, plan.k)
    if erasures is None:
        erasures = np.zeros(plan.coded_len, dtype=bool)
    if confidence is None:
        confidence = np.where(erasures, 0.0, 1.0)
    er_cw = deinterleave_array(np.asarray(erasures, dtype=bool), plan.k)
    conf_cw = deinterleave_array(np.asarray(confidence, dtype=float), plan.k)
    stats = DecodeStats(codewords=plan.k)
    out = []
    for cw, er, conf in zip(cws, er_cw, conf_cw, strict=True):
        flagged = np.flatnonzero(er)
        order = flagged[np.argsort(conf[flagged], kind="stable")]
        cap = max(0, nsym - 2)
        primary = sorted(int(i) for i in order[:cap])
        attempts = [primary, [], sorted(int(i) for i in order[: len(primary) // 2])]
        result = None
        used = 0
        seen: set[tuple[int, ...]] = set()
        for erase in attempts:
            key = tuple(erase)
            if key in seen:
                continue
            seen.add(key)
            result = _try_decode(cw, nsym, erase)
            if result is not None:
                used = len(erase)
                break
        if result is None:
            raise FecError("codeword uncorrectable")
        data, corrected = result
        stats.corrected += corrected
        stats.erasures_used += used
        out.append(data)
    return b"".join(out)[:body_len], stats
