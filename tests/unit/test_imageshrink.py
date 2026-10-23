import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from ultrasonic_airdrop.files.imageshrink import shrink_image  # noqa: E402


def _photo(path, w=2400, h=1800, quality=98):
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(128, 60, (h, w, 3)), 0, 255).astype(np.uint8)
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90° when displayed
    exif[0x8825] = {1: "N", 2: (51.0, 30.0, 0.0)}  # GPS
    Image.fromarray(img).save(path, quality=quality, exif=exif)
    return path


def test_large_photo_is_shrunk_and_metadata_dropped(tmp_path):
    src = _photo(tmp_path / "big.jpg")
    assert src.stat().st_size > 1_000_000
    r = shrink_image(src, tmp_path / "out", target_bytes=300_000)
    assert r.shrunk and r.new_bytes <= 300_000 and r.output.name == "big_small.jpg"
    with Image.open(r.output) as im:
        assert im.width < im.height  # EXIF orientation applied
        assert not im.getexif()  # no EXIF/GPS carried over
    assert src.stat().st_size == r.original_bytes  # original untouched


def test_small_and_non_image_files_pass_through(tmp_path):
    small = tmp_path / "s.jpg"
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(small)
    assert not shrink_image(small, tmp_path / "out", 100_000).shrunk
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(b"%PDF" + bytes(2_000_000))
    r = shrink_image(doc, tmp_path / "out", 100_000)
    assert not r.shrunk and r.output == doc
