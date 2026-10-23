import hashlib
import json
import os
import zlib
from pathlib import Path

import numpy as np
import pytest

from ultrasonic_airdrop.config import UNIT_SIZE
from ultrasonic_airdrop.errors import FileSafetyError, ManifestError
from ultrasonic_airdrop.files import resume
from ultrasonic_airdrop.files.chunker import UnitReader, cleanup_prepared, n_units, prepare_files
from ultrasonic_airdrop.files.hashing import EMPTY_SHA256, sha256_file
from ultrasonic_airdrop.files.manifest import Manifest, ManifestEntry, split_blob
from ultrasonic_airdrop.files.reassembly import Reassembly
from ultrasonic_airdrop.files.sanitize import safe_destination, sanitize_filename
from ultrasonic_airdrop.link.frames import FILE_HASH_MISMATCH, FILE_INCOMPLETE, FILE_OK

# ------------------------------------------------------------------ sanitizer


@pytest.mark.parametrize(
    "name",
    [
        "../etc/passwd",
        "..\\windows\\system32",
        "/abs/path",
        "C:\\x.txt",
        "C:x.txt",
        "\\\\server\\share\\f",
        "file.txt:stream",
        "a/b",
        "..",
        ".",
    ],
)
def test_sanitize_rejects_paths(name):
    with pytest.raises(FileSafetyError):
        sanitize_filename(name, strict=True)


def test_sanitize_non_strict_takes_last_component():
    assert sanitize_filename("../../etc/passwd", strict=False) == "passwd"
    assert sanitize_filename("C:\\dir\\evil.exe", strict=False) == "evil.exe"
    assert sanitize_filename("file.txt:ads", strict=False) == "ads"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CON", "_CON"),
        ("con.txt", "_con.txt"),
        ("LPT9.tar.gz", "_LPT9.tar.gz"),
        ("NUL", "_NUL"),
        ("COM1", "_COM1"),
        ("COM10", "COM10"),
        ("trailing. . ", "trailing"),
        ('a<b>c"d|e?f*g', "abcdefg"),
        ("tab\there\x00", "tabhere"),
        ("", "received_file"),
        ("...", "received_file"),
        (".bashrc", ".bashrc"),
        ("résumé.pdf", "résumé.pdf"),
        ("e\u0301.txt", "é.txt"),
        ("📄 notes.txt", "📄 notes.txt"),
        ("evil\u202etxt.exe", "eviltxt.exe"),
        ("שלום.txt", "שלום.txt"),
    ],
)
def test_sanitize_cleans(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_length_limit_preserves_extension():
    name = "😀" * 200 + ".jpeg"
    out = sanitize_filename(name)
    assert out.endswith(".jpeg")
    assert len(out.encode("utf-16-le")) // 2 <= 180
    combining = "a" + "\u0301" * 300 + ".txt"
    out2 = sanitize_filename(combining)
    assert out2.endswith(".txt") and len(out2.encode("utf-16-le")) // 2 <= 180


def test_safe_destination_never_overwrites(tmp_path: Path):
    (tmp_path / "a.txt").write_text("original")
    p1 = safe_destination(tmp_path, "a.txt")
    p2 = safe_destination(tmp_path, "a.txt")
    assert p1.name == "a (1).txt" and p2.name == "a (2).txt"
    assert (tmp_path / "a.txt").read_text() == "original"
    p3 = safe_destination(tmp_path, "../escape.txt")
    assert p3.parent == tmp_path.resolve()


# ------------------------------------------------------------------ manifest


def _entry(i: int, **kw) -> dict:
    d = {
        "i": i,
        "name": f"f{i}.bin",
        "size": 10,
        "sha256": "a" * 64,
        "tx_size": 10,
        "compressed": False,
        "mtime": 0,
    }
    d.update(kw)
    return d


def _blob(files, **top) -> bytes:
    obj = {"v": 1, "files": files, "unit": 32}
    obj.update(top)
    return zlib.compress(json.dumps(obj).encode())


def test_manifest_roundtrip():
    m = Manifest(
        (
            ManifestEntry(0, "a.txt", 100, "b" * 64, 60, True, 5),
            ManifestEntry(1, "e", 0, EMPTY_SHA256, 0, False, 0),
        )
    )
    m2 = Manifest.from_blob(m.to_blob())
    assert m2 == m
    assert m2.files[0].n_units == 2 and m2.files[0].unit_len(1) == 28
    assert m2.files[1].n_units == 0
    parts = split_blob(m.to_blob(), 7)
    assert b"".join(parts) == m.to_blob()


@pytest.mark.parametrize(
    "blob",
    [
        b"not zlib",
        zlib.compress(b"\xff\xfe"),
        zlib.compress(b"[]"),
        _blob([]),
        _blob([_entry(0)], v=2),
        _blob([_entry(0)], unit=64),
        _blob([_entry(0), _entry(0)]),
        _blob([_entry(1)]),
        _blob([_entry(0, name="../x")]),
        _blob([_entry(0, name="C:\\x")]),
        _blob([_entry(0, size=-1, tx_size=-1)]),
        _blob([_entry(0, sha256="xyz")]),
        _blob([_entry(0, compressed=1)]),
        _blob([_entry(0, size=True)]),
        _blob([_entry(0, tx_size=11)]),
        _blob([{**_entry(0), "extra": 1}]),
        _blob([_entry(0, compressed=True, size=0, tx_size=0)]),
        _blob([_entry(i) for i in range(1001)]),
        zlib.compress(b"{" + b" " * (300 * 1024) + b"}"),
    ],
)
def test_manifest_rejects_malicious(blob):
    with pytest.raises(ManifestError):
        Manifest.from_blob(blob)


def test_manifest_size_limit():
    blob = _blob([_entry(0, size=1000, tx_size=1000)])
    with pytest.raises(ManifestError, match="limit"):
        Manifest.from_blob(blob, max_total_bytes=999)
    assert Manifest.from_blob(blob, max_total_bytes=1000).total_size == 1000


def test_manifest_aliases():
    m = Manifest((ManifestEntry(0, "a", 5, "c" * 64, 5, False, 0), ManifestEntry(1, "b", 5, "c" * 64, 5, False, 0)))
    assert m.canonical_index(1) == 0 and m.canonical_index(0) == 0


# ------------------------------------------------------------------ chunker


def test_prepare_and_read_units(tmp_path: Path, rng):
    data = rng.integers(0, 256, 1000, dtype=np.uint8).tobytes()
    comp = b"abc" * 5000
    (tmp_path / "r.bin").write_bytes(data)
    (tmp_path / "c.txt").write_bytes(comp)
    (tmp_path / "e").write_bytes(b"")
    m, prepared = prepare_files([tmp_path / "r.bin", tmp_path / "c.txt", tmp_path / "e"], True, tmp_path)
    try:
        r, c, e = m.files
        assert not r.compressed and r.tx_size == 1000 and r.sha256 == hashlib.sha256(data).hexdigest()
        assert c.compressed and c.tx_size < len(comp) * 0.95 and c.size == len(comp)
        assert e.size == 0 and e.n_units == 0 and e.sha256 == EMPTY_SHA256
        reader = UnitReader(prepared)
        assert n_units(1000) == 32
        assert reader.read(0, 0, 1) == data[:32]
        assert reader.read(0, 31, 1) == data[992:]  # short last unit (8 bytes)
        assert reader.read(0, 30, 5) == data[960:]
        assert zlib.decompress(b"".join(reader.read(1, u, 1) for u in range(c.n_units))) == comp
        with pytest.raises(ValueError):
            reader.read(0, 40, 1)
        reader.close()
    finally:
        cleanup_prepared(prepared)
    assert not any(p.source.exists() for p in prepared if p.temporary)


# ------------------------------------------------------------------ reassembly / resume


def _manifest_for(datas: dict[str, bytes]) -> Manifest:
    return Manifest(
        tuple(
            ManifestEntry(i, n, len(d), hashlib.sha256(d).hexdigest(), len(d), False, 0)
            for i, (n, d) in enumerate(datas.items())
        )
    )


def test_reassembly_out_of_order_duplicates_and_verify(tmp_path: Path, rng):
    data = rng.integers(0, 256, 1000, dtype=np.uint8).tobytes()
    m = _manifest_for({"x.bin": data, "empty": b""})
    ra = Reassembly(tmp_path, m, 1, "ab")
    asm = ra.assembler(0)
    order = rng.permutation(asm.n_units)
    for u in order:
        assert asm.write_units(int(u), 1, data[u * 32 : u * 32 + 32]) == 1
    assert asm.write_units(0, 2, data[:64]) == 0  # duplicate
    assert asm.duplicates == 2
    with pytest.raises(ValueError):
        asm.write_units(31, 1, b"x" * 32)  # last unit must be 8 bytes
    with pytest.raises(ValueError):
        asm.write_units(40, 1, b"x" * 32)
    results = ra.finalize_all()
    assert [r.status for r in results] == [FILE_OK, FILE_OK]
    assert results[0].path.read_bytes() == data
    assert results[1].path.read_bytes() == b""
    ra.close()
    assert not (tmp_path / ".airdrop-partial").exists()


def test_reassembly_hash_mismatch_never_placed(tmp_path: Path):
    data = b"A" * 100
    m = Manifest((ManifestEntry(0, "bad.bin", 100, hashlib.sha256(b"B" * 100).hexdigest(), 100, False, 0),))
    ra = Reassembly(tmp_path, m, 1, "ab")
    asm = ra.assembler(0)
    for u in range(asm.n_units):
        asm.write_units(u, 1, data[u * 32 : u * 32 + 32])
    res = ra.finalize_all()
    assert res[0].status == FILE_HASH_MISMATCH and res[0].path is None
    assert not (tmp_path / "bad.bin").exists()
    assert (tmp_path / ".airdrop-partial" / f"{m.files[0].sha256}.corrupt").exists()


def test_reassembly_compressed(tmp_path: Path):
    orig = b"hello " * 1000
    comp = zlib.compress(orig)
    m = Manifest((ManifestEntry(0, "h.txt", len(orig), hashlib.sha256(orig).hexdigest(), len(comp), True, 0),))
    ra = Reassembly(tmp_path, m, 1, "ab")
    asm = ra.assembler(0)
    asm.write_units(0, asm.n_units, comp) if asm.n_units <= 255 else None
    assert ra.finalize_all()[0].status == FILE_OK
    assert (tmp_path / "h.txt").read_bytes() == orig


def test_reassembly_compressed_bomb_rejected(tmp_path: Path):
    orig = b"\x00" * 100_000
    comp = zlib.compress(orig)
    # manifest claims a small size: decompression must stop and fail verification
    m = Manifest((ManifestEntry(0, "b", 10, hashlib.sha256(orig).hexdigest(), len(comp), True, 0),))
    ra = Reassembly(tmp_path, m, 1, "ab")
    ra.assembler(0).write_units(0, ra.assembler(0).n_units, comp)
    assert ra.finalize_all()[0].status == FILE_HASH_MISMATCH


def test_resume_sidecar_roundtrip(tmp_path: Path, rng):
    data = rng.integers(0, 256, 3200, dtype=np.uint8).tobytes()
    m = _manifest_for({"r.bin": data})
    ra = Reassembly(tmp_path, m, 7, "manifesthash")
    asm = ra.assembler(0)
    for u in (0, 1, 2, 50, 51, 99):
        asm.write_units(u, 1, data[u * 32 : u * 32 + 32])
    ra.close()
    side = resume.load_sidecar(tmp_path / ".airdrop-partial", m.files[0].sha256)
    assert side is not None and side.ranges == [(0, 3), (50, 2), (99, 1)]
    assert resume.manifest_seen(tmp_path / ".airdrop-partial", "manifesthash")
    ra2 = Reassembly(tmp_path, m, 8, "manifesthash")
    asm2 = ra2.assembler(0)
    assert asm2.resumed_units == 6
    assert asm2.missing_ranges()[0] == (3, 47)
    for u in range(100):
        asm2.write_units(u, 1, data[u * 32 : u * 32 + 32])
    assert ra2.finalize_all()[0].status == FILE_OK
    assert not (tmp_path / ".airdrop-partial" / f"{m.files[0].sha256}.json").exists()


def test_resume_mismatched_size_restarts(tmp_path: Path):
    data = b"z" * 64
    m = _manifest_for({"z": data})
    ra = Reassembly(tmp_path, m, 1, "h")
    ra.assembler(0).write_units(0, 1, data[:32])
    ra.close()
    other = Manifest((ManifestEntry(0, "z", 96, m.files[0].sha256, 96, False, 0),))
    ra2 = Reassembly(tmp_path, other, 2, "h2")
    assert ra2.assembler(0).resumed_units == 0
    ra2.close()


def test_bitmap_ranges():
    bm = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert resume.bitmap_to_ranges(bm) == [(0, 2), (4, 1), (6, 3)]
    assert resume.bitmap_to_ranges(bm, False) == [(2, 2), (5, 1)]
    assert (resume.ranges_to_bitmap([(0, 2), (4, 1), (6, 3)], 9) == bm).all()


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "f"
    p.write_bytes(os.urandom(3_000_000))
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_aliases_share_assembler(tmp_path: Path):
    data = b"same" * 20
    m = _manifest_for({"one.txt": data, "two.txt": data})
    ra = Reassembly(tmp_path, m, 1, "h")
    assert ra.assembler(0) is ra.assembler(1)
    asm = ra.assembler(0)
    asm.write_units(0, asm.n_units, data)
    res = ra.finalize_all()
    assert [r.status for r in res] == [FILE_OK, FILE_OK]
    assert (tmp_path / "one.txt").read_bytes() == data == (tmp_path / "two.txt").read_bytes()


def test_incomplete_status(tmp_path: Path):
    m = _manifest_for({"p": b"q" * 100})
    ra = Reassembly(tmp_path, m, 1, "h")
    ra.assembler(0).write_units(0, 1, b"q" * 32)
    assert ra.finalize_all()[0].status == FILE_INCOMPLETE
    assert UNIT_SIZE == 32
