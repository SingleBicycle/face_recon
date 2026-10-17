import numpy as np

from facerecon.face_parsing import face_mask
from facerecon.recon_vggt import confident_pixels
from facerecon.views import load_views
from synthetic import save_photo


def test_load_views_pads_portrait_photo_to_square_after_exif_rotation(tmp_path):
    stored = np.zeros((300, 600, 3), np.uint8)  # landscape as stored...
    path = save_photo(tmp_path / "p.jpg", stored, orientation=8)  # ...portrait when displayed

    [view] = load_views([path])

    assert view.shape == (518, 518, 3) and view.dtype == np.float32
    assert np.allclose(view[:, :100], 1.0)  # white padding left and right of the portrait
    assert np.allclose(view[:, -100:], 1.0)
    assert view[:, 200:300].max() < 0.1  # the (black) photo itself in the middle


def test_face_mask_keeps_facial_skin_and_features_only():
    labels = np.array([[0, 1, 2, 4], [5, 10, 13, 17]])  # bg, skin, nose, l_eye / r_eye, mouth, hair, neck
    np.testing.assert_array_equal(face_mask(labels), [[False, True, True, True], [True, True, False, False]])


def test_confident_pixels_keeps_masked_pixels_above_each_views_median():
    conf = np.array([[[1, 5, 9], [2, 6, 8]], [[3, 3, 3], [7, 7, 7]]], float)
    mask = np.ones((2, 2, 3), bool)
    mask[0, 0, 2] = False  # most confident pixel of view 0, but not face

    keep = confident_pixels(conf, mask, keep_quantile=0.5)

    # view 0: median of masked conf [1, 5, 2, 6, 8] is 5; view 1: median of [3, 3, 3, 7, 7, 7] is 5
    np.testing.assert_array_equal(keep, (conf >= 5) & mask)
