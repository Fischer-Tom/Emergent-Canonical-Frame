from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


from .structures import Camera, FrameRecord, SequenceSample

class MissingPointCloudError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Sequence-level
# ---------------------------------------------------------------------------

def normalize_sequence_to_bbox(
    sample: SequenceSample, q_low: float = 0.1, q_high: float = 0.9
) -> SequenceSample:
    """Compute a robust object bbox from the segmented point cloud and
    re-express points in the object's local frame (centered, unit extent).
    Writes ``sample.object_size`` and ``sample.obj_center``.
    """
    pc = sample.segmented_point_cloud
    if pc is None or getattr(pc, "xyz", None) is None:
        raise MissingPointCloudError(
            f"Sequence {sample.sequence_name!r} has no segmented point cloud"
        )

    xyz = pc.xyz
    ql = torch.quantile(xyz, q_low, dim=0)
    qh = torch.quantile(xyz, q_high, dim=0)
    mask = ((xyz >= ql) & (xyz <= qh)).all(dim=1)
    if not mask.any():
        raise MissingPointCloudError(
            f"Sequence {sample.sequence_name!r} point cloud has no inliers"
        )

    inliers = xyz[mask]
    lo = inliers.min(dim=0).values
    hi = inliers.max(dim=0).values
    size = hi - lo
    center = (hi + lo) * 0.5
    pc.xyz = (xyz - center) / torch.linalg.norm(size).clamp_min(1e-8)

    sample.object_size = size
    sample.obj_center = center
    return sample


# ---------------------------------------------------------------------------
# Frame-level: undistort + align
# ---------------------------------------------------------------------------

def undistort_frame(
    frame: FrameRecord,
    distortion_coeffs: torch.Tensor,
    alpha: float = 0.5,
) -> FrameRecord:
    """Undistort image+mask via OpenCV with COLMAP distortion; rewrite
    ``frame.camera`` with rectified intrinsics (still NDC-convention ``Camera``).

    ``distortion_coeffs``: 1D tensor of COLMAP distortion coefficients for this
    frame. ``frame.camera`` must be a ``Camera`` (batch=1) in NDC convention.
    ``frame.image_rgb`` must be set.
    """
    if frame.camera is None or frame.image_rgb is None:
        return frame

    img_np = frame.image_rgb.numpy()
    mask_np = None if frame.mask is None else frame.mask.numpy()
    h, w = img_np.shape[-2:]
    image_size_hw = torch.tensor([h, w])
    wh = (int(w), int(h))

    R_cv, T_cv, K_th = _opencv_from_camera(frame.camera, image_size_hw[None])
    K = K_th[0].numpy()
    distortion = np.asarray(distortion_coeffs).reshape(-1)

    new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, wh, alpha, wh)
    map_x, map_y = cv2.initUndistortRectifyMap(K, distortion, None, new_K, wh, cv2.CV_16SC2)

    img_u = cv2.remap(
        np.ascontiguousarray(img_np.transpose(1, 2, 0)), map_x, map_y, cv2.INTER_LINEAR
    )
    frame.image_rgb = torch.from_numpy(img_u.transpose(2, 0, 1).copy())
    frame.orig_size_hw = image_size_hw

    if mask_np is not None:
        mask_u = cv2.remap(
            np.ascontiguousarray(mask_np.transpose(1, 2, 0)), map_x, map_y, cv2.INTER_LINEAR
        )
        if mask_u.ndim == 2:
            mask_u = mask_u[None]
        else:
            mask_u = mask_u.transpose(2, 0, 1)
        frame.mask = torch.from_numpy(np.ascontiguousarray(mask_u))
    else:
        frame.mask = None

    frame.camera = _camera_from_opencv(
        R=R_cv,
        tvec=T_cv,
        camera_matrix=torch.tensor(new_K)[None],
        image_size=image_size_hw[None],
    )
    return frame


def align_point_cloud(point_cloud: Any, sequence_orm: Any) -> Any:
    """Apply the sequence similarity transform to a point cloud in-place."""
    aln = getattr(sequence_orm, "alignment", None) if sequence_orm is not None else None
    if aln is None or point_cloud is None:
        return point_cloud

    R = torch.as_tensor(aln.R, dtype=torch.float32)
    T = torch.as_tensor(aln.T, dtype=torch.float32)
    s = torch.as_tensor(aln.scale, dtype=torch.float32)

    if hasattr(point_cloud, "xyz") and point_cloud.xyz is not None:
        point_cloud.xyz = (point_cloud.xyz @ R + T) * s
    elif hasattr(point_cloud, "points") and point_cloud.points is not None:
        point_cloud.points = (point_cloud.points @ R + T) * s
    return point_cloud


def align_frame(frame: FrameRecord, sequence_orm: Any) -> FrameRecord:
    """Apply the sequence's similarity transform (R, T, scale) to camera +
    point cloud. No-op if the sequence has no alignment.
    """
    aln = getattr(sequence_orm, "alignment", None) if sequence_orm is not None else None
    if aln is None:
        return frame

    R = torch.as_tensor(aln.R, dtype=torch.float32)
    T = torch.as_tensor(aln.T, dtype=torch.float32)
    s = torch.as_tensor(aln.scale, dtype=torch.float32)

    cam = frame.camera
    if cam is not None:
        cam.R = R.transpose(-1, -2) @ cam.R
        cam.T = s * (cam.T - T @ cam.R)

    frame.segmented_point_cloud = align_point_cloud(frame.segmented_point_cloud, sequence_orm)
    return frame