"""Stage 2 baseline: multi-view photos -> 3D points with VGGT (feed-forward, uncalibrated)."""
from __future__ import annotations

import numpy as np


def confident_pixels(conf: np.ndarray, mask: np.ndarray, keep_quantile: float = 0.5) -> np.ndarray:
    """(S, H, W) bool: masked pixels whose confidence reaches their view's `keep_quantile` of the
    masked confidences. Index points / colors / labels with it to pool all views."""
    keep = np.zeros_like(mask, dtype=bool)
    for i in range(len(conf)):
        if mask[i].any():
            keep[i] = mask[i] & (conf[i] >= np.quantile(conf[i][mask[i]], keep_quantile))
    return keep


class VGGTReconstructor:
    """VGGT-1B: per-view depth, confidence and cameras in the first camera's frame (arbitrary scale)."""

    def __init__(self, device: str = "cuda", weights: str = "facebook/VGGT-1B"):
        from vggt.models.vggt import VGGT

        self.model = VGGT.from_pretrained(weights).to(device).eval()
        self.device = device

    def __call__(self, views: np.ndarray) -> dict:
        """`views`: (S, H, W, 3) float in [0, 1] from `load_views`.

        Returns points (S, H, W, 3) unprojected from depth (more accurate than VGGT's point head),
        conf (S, H, W), extrinsic (S, 3, 4) world->camera in OpenCV convention, intrinsic (S, 3, 3).
        """
        import torch
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        images = torch.from_numpy(views).permute(0, 3, 1, 2).to(self.device)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(self.device)[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.autocast("cuda", dtype=dtype):
                pred = self.model(images[None])
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"].float(), images.shape[-2:])
        extrinsic = extrinsic[0].cpu().numpy()
        intrinsic = intrinsic[0].cpu().numpy()
        depth = pred["depth"][0].float().cpu().numpy()
        return {
            "points": unproject_depth_map_to_point_map(depth, extrinsic, intrinsic).astype(np.float32),
            "conf": pred["depth_conf"][0].float().cpu().numpy(),
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }
