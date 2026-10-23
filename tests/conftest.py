"""Shared fixtures.  No test requires audio hardware."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def make_file(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()

    def _make(name: str, data: bytes) -> Path:
        p = src / name
        p.write_bytes(data)
        return p

    return _make
