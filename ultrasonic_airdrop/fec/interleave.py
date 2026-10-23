"""Byte block interleaver across RS codewords.

Byte ``j`` of codeword ``i`` (of ``k`` equal-length codewords) is transmitted at
position ``j * k + i`` so that a burst of consecutive bad symbols is spread over
all codewords.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def interleave(codewords: list[bytes]) -> bytes:
    """Interleave equal-length ``codewords`` into one byte string."""
    if not codewords:
        return b""
    n = len(codewords[0])
    if any(len(c) != n for c in codewords):
        raise ValueError("all codewords must have equal length")
    mat = np.frombuffer(b"".join(codewords), dtype=np.uint8).reshape(len(codewords), n)
    return mat.T.reshape(-1).tobytes()


def deinterleave(data: bytes, k: int) -> list[bytes]:
    """Inverse of :func:`interleave` for ``k`` codewords."""
    if k <= 0 or len(data) % k:
        raise ValueError("data length must be a multiple of the codeword count")
    n = len(data) // k
    mat = np.frombuffer(data, dtype=np.uint8).reshape(n, k)
    return [mat[:, i].tobytes() for i in range(k)]


def deinterleave_array(values: npt.NDArray[np.generic], k: int) -> list[npt.NDArray[np.generic]]:
    """De-interleave a per-byte side-information array (erasure flags, confidences)."""
    if k <= 0 or values.shape[0] % k:
        raise ValueError("array length must be a multiple of the codeword count")
    n = values.shape[0] // k
    mat = values.reshape(n, k)
    return [mat[:, i].copy() for i in range(k)]
