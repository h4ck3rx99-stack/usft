"""Filename sanitisation for untrusted names received over the air.

Rules (§6): NFC-normalise; only the final path component is ever used and any
path syntax (``/``, ``\\``, ``..``, drive letters, UNC, NTFS alternate data
streams) is rejected; control characters and ``<>:"|?*`` are stripped; trailing
dots/spaces are removed; Windows reserved device names are renamed; the name is
limited to 180 UTF-16 code units preserving the extension; empty names become
``received_file``.  :func:`safe_destination` additionally resolves the final
path and asserts it is inside the destination directory, and picks a
non-existing ``name (n).ext`` variant instead of overwriting.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path

from ultrasonic_airdrop.errors import FileSafetyError

MAX_UTF16_UNITS = 180
DEFAULT_NAME = "received_file"
_RESERVED = (
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
    | {"COM¹", "COM²", "COM³", "LPT¹", "LPT²", "LPT³"}
)
_STRIP = re.compile(r'[<>:"|?*\x00-\x1f\x7f]')
_DRIVE = re.compile(r"^[A-Za-z]:")
# Bidirectional override / isolate controls can disguise extensions ("evil‮txt.exe").
_BIDI = re.compile("[‪-‮⁦-⁩‎‏؜]")


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def _truncate_utf16(s: str, limit: int) -> str:
    out = []
    used = 0
    for ch in s:
        n = _utf16_len(ch)
        if used + n > limit:
            break
        out.append(ch)
        used += n
    # do not leave a dangling combining mark at the cut
    while out and unicodedata.combining(out[-1]):
        out.pop()
    return "".join(out)


def sanitize_filename(name: str, strict: bool = True) -> str:
    """Return a safe single-component filename.

    With ``strict`` (the default for received manifests) names containing path
    syntax raise :class:`FileSafetyError`; otherwise they are reduced to their
    last component.
    """
    if not isinstance(name, str):
        raise FileSafetyError("name is not a string")
    name = unicodedata.normalize("NFC", name)
    has_path = "/" in name or "\\" in name or name in (".", "..") or bool(_DRIVE.match(name)) or ":" in name
    if has_path:
        if strict:
            raise FileSafetyError(f"path components are not allowed: {name!r}")
        name = re.split(r"[\\/]", name)[-1]
        name = name.split(":")[-1]
    name = _BIDI.sub("", name)
    name = _STRIP.sub("", name)
    name = "".join(ch for ch in name if unicodedata.category(ch) not in ("Cc", "Cs"))
    name = name.strip().rstrip(". ")
    if name in ("", ".", ".."):
        return DEFAULT_NAME
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    if not stem:  # dotfile like ".bashrc"
        stem, ext, dot = name, "", ""
    base = stem.split(".")[0].strip().upper()
    if base in _RESERVED:
        stem = "_" + stem
    ext_part = f".{ext}" if dot else ""
    if _utf16_len(ext_part) > 40:
        ext_part = _truncate_utf16(ext_part, 40)
    stem = _truncate_utf16(stem, MAX_UTF16_UNITS - _utf16_len(ext_part))
    result = (stem + ext_part).rstrip(". ")
    return result or DEFAULT_NAME


def _numbered(name: str, n: int) -> str:
    stem, dot, ext = name.rpartition(".")
    if not dot or not stem:
        stem, ext_part = name, ""
    else:
        ext_part = "." + ext
    suffix = f" ({n})"
    stem = _truncate_utf16(stem, MAX_UTF16_UNITS - _utf16_len(ext_part) - len(suffix))
    return f"{stem}{suffix}{ext_part}"


def ensure_inside(dest: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and assert it lies directly inside ``dest``."""
    root = dest.resolve()
    resolved = candidate.resolve()
    if resolved.parent != root:
        raise FileSafetyError(f"{candidate} escapes destination {root}")
    return resolved


def safe_destination(dest: Path, name: str, max_tries: int = 10_000) -> Path:
    """Reserve and return a non-existing path ``dest/name`` (or ``name (n).ext``).

    The file is created exclusively (empty) so concurrent writers cannot race us;
    the caller replaces it atomically with :func:`os.replace`.
    """
    clean = sanitize_filename(name, strict=False)
    dest.mkdir(parents=True, exist_ok=True)
    for n in range(max_tries):
        candidate_name = clean if n == 0 else _numbered(clean, n)
        candidate = ensure_inside(dest, dest / candidate_name)
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise FileSafetyError("could not find a free file name")
