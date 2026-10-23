"""Optional photo shrinking before sending (acoustic links carry only ~kB/s).

A photo is re-encoded as a JPEG no larger than ``target_bytes`` by reducing its
resolution and quality step by step.  EXIF orientation is applied and all
metadata (including GPS) is dropped.  Non-images and images that are already
small enough are passed through unchanged.  The receiver gets the smaller copy;
the original file is never modified.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

from ultrasonic_airdrop.errors import AirDropError
from ultrasonic_airdrop.logging_setup import kv

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
DEFAULT_TARGET_BYTES = 900 * 1024
_EDGES = (4096, 3200, 2560, 2048, 1600, 1280, 1024, 800, 640)
_QUALITIES = (88, 82, 76, 70, 62, 55)


@dataclass(frozen=True)
class ShrinkResult:
    """Outcome for one input file."""

    source: Path
    output: Path
    original_bytes: int
    new_bytes: int
    shrunk: bool


def pillow_available() -> bool:
    """True if Pillow can be imported."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def is_image(path: Path) -> bool:
    """Judged by file extension."""
    return path.suffix.lower() in IMAGE_SUFFIXES


def _encode(img: object, edge: int, quality: int) -> bytes:
    from PIL import Image

    assert isinstance(img, Image.Image)
    im = img.copy()
    im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue()


def shrink_image(path: Path, out_dir: Path, target_bytes: int = DEFAULT_TARGET_BYTES) -> ShrinkResult:
    """Write a JPEG copy of ``path`` no larger than ``target_bytes`` into ``out_dir`` (if needed)."""
    size = path.stat().st_size
    if size <= target_bytes or not is_image(path):
        return ShrinkResult(path, path, size, size, False)
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise AirDropError("Shrinking photos needs the Pillow package.", "Run: pip install Pillow") from exc
    try:
        with Image.open(path) as raw:
            img = ImageOps.exif_transpose(raw)
            img = img.convert("RGB")
    except Exception as exc:  # unreadable/unsupported image: send as is
        log.warning(kv("cannot read image; sending original", file=path.name, error=exc))
        return ShrinkResult(path, path, size, size, False)
    best: bytes | None = None
    longest = max(img.size)
    for edge in (e for e in _EDGES if e < longest or e == _EDGES[-1]):
        for q in _QUALITIES:
            data = _encode(img, edge, q)
            best = data if best is None or len(data) < len(best) else best
            if len(data) <= target_bytes:
                best = data
                break
        if best is not None and len(best) <= target_bytes:
            break
    assert best is not None
    if len(best) >= size:
        return ShrinkResult(path, path, size, size, False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_small.jpg"
    out.write_bytes(best)
    log.info(kv("image shrunk", file=path.name, before=size, after=len(best)))
    return ShrinkResult(path, out, size, len(best), True)


def shrink_images(paths: list[Path], out_dir: Path, target_bytes: int = DEFAULT_TARGET_BYTES) -> list[ShrinkResult]:
    """Shrink every large image in ``paths``; other files pass through."""
    return [shrink_image(p, out_dir, target_bytes) for p in paths]
