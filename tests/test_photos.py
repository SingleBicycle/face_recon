import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from facerecon.photos import export_clean_photo, list_photos, photo_info
from synthetic import checkerboard, save_photo


def test_photo_info_reports_size_camera_and_privacy_flags(tmp_path):
    path = save_photo(tmp_path / "a.jpg", np.stack([checkerboard()] * 3, -1),
                      datetime="2015:06:01 10:00:00", focal_mm=4.25)
    info = photo_info(path)
    assert (info["width"], info["height"]) == (640, 480)
    assert info["model"] == "TestCam"
    assert info["focal_length_mm"] == pytest.approx(4.25)
    assert info["has_datetime"] is True
    assert info["has_gps"] is False


def test_photo_without_exif_has_no_camera_info(tmp_path):
    info = photo_info(save_photo(tmp_path / "a.png"))
    assert info["focal_length_mm"] is None
    assert info["has_datetime"] is False


def test_sharpness_drops_when_image_is_blurred(tmp_path):
    sharp = save_photo(tmp_path / "sharp.png", checkerboard())
    blurred = save_photo(tmp_path / "blurred.png",
                         ndimage.gaussian_filter(checkerboard().astype(float), 4).astype(np.uint8))
    assert photo_info(sharp)["sharpness"] > 5 * photo_info(blurred)["sharpness"]


def test_clean_copy_is_upright_and_carries_no_metadata(tmp_path):
    stored = np.zeros((48, 64, 3), np.uint8)
    stored[:8, :8] = (255, 0, 0)  # red marker in the stored top-left corner
    src = save_photo(tmp_path / "raw.jpg", stored, datetime="2015:06:01 10:00:00", focal_mm=100,
                     gps=True, orientation=8)  # 8: stored rotated, display = rotate 90 deg CCW
    assert photo_info(src)["has_gps"]  # the fixture really carries GPS

    out = export_clean_photo(src, tmp_path / "clean" / "raw.jpg")

    info = photo_info(out)
    assert (info["width"], info["height"]) == (48, 64)
    assert not info["has_datetime"] and not info["has_gps"] and info["model"] is None
    with Image.open(out) as img:
        assert len(img.getexif()) == 0
        r, g, b = img.getpixel((3, 60))  # marker is now bottom-left
    assert r > 200 and g < 60 and b < 60


def test_list_photos_finds_images_case_insensitively_in_name_order(tmp_path):
    for name in ["b.JPG", "a.jpg", "c.png", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in list_photos(tmp_path)] == ["a.jpg", "b.JPG", "c.png"]
