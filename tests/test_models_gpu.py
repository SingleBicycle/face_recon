"""Smoke tests for the pretrained-model wrappers (need CUDA and the cached weights)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_vggt_returns_points_confidence_and_cameras_in_the_first_camera_frame():
    from facerecon.recon_vggt import VGGTReconstructor

    views = np.random.default_rng(0).random((3, 518, 518, 3), dtype=np.float32)
    out = VGGTReconstructor()(views)

    assert out["points"].shape == (3, 518, 518, 3) and np.isfinite(out["points"]).all()
    assert out["conf"].shape == (3, 518, 518)
    assert out["extrinsic"].shape == (3, 3, 4) and out["intrinsic"].shape == (3, 3, 3)
    np.testing.assert_allclose(out["extrinsic"][0], np.eye(4)[:3], atol=1e-3)


def test_face_parser_labels_a_blank_view_as_background():
    from facerecon.face_parsing import FaceParser

    labels = FaceParser()(np.ones((2, 518, 518, 3), np.float32))

    assert labels.shape == (2, 518, 518)
    assert (labels == 0).mean() > 0.9
