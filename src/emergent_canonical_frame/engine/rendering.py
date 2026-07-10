import math
from dataclasses import dataclass
from typing import Any

import nvdiffrast.torch as dr
import torch
from einops import rearrange


@dataclass
class AnnotationResult:
    proxy_mask: torch.Tensor
    obj_xyz: torch.Tensor
    obj_idx: torch.Tensor
    cameras: Any  

    def to(self, device=None, non_blocking: bool = False) -> "AnnotationResult":
        def mv(t):
            return t.to(device=device, non_blocking=non_blocking) if isinstance(t, torch.Tensor) else t

        cams = self.cameras
        if cams is not None:
            cams = cams.to(device=device, non_blocking=non_blocking)
        return AnnotationResult(
            proxy_mask=mv(self.proxy_mask),
            obj_xyz=mv(self.obj_xyz),
            obj_idx=mv(self.obj_idx),
            cameras=cams,
        )


def normalize_coords(verts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    verts: (V, 3) in object coordinates.
    returns: (V, 3) NOCS in [0, 1]
    """
    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    scale = (vmax - vmin).clamp_min(eps)
    return (verts - vmin) / scale


class Renderer:
    def __init__(
        self,
        V,
        F,
        image_size,
        device=torch.device("cpu"),
        background_color=(1.0, 1.0, 1.0),
    ):
        """
        Initializes the nvdiffrast-based annotator.
        """
        self.device = device

        self.V_obj = torch.as_tensor(
            V, dtype=torch.float32, device=self.device
        )  # (V, 3)
        # nvdiffrast requires faces to be int32
        self.F = torch.as_tensor(F, dtype=torch.int32, device=self.device)  # (F, 3)

        self.size = (
            torch.max(self.V_obj, dim=0).values - torch.min(self.V_obj, dim=0).values
        )

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.bg_color = torch.tensor(
            background_color, dtype=torch.float32, device=self.device
        )
        self.ctx = (
            dr.RasterizeCudaContext(device=self.device)
            if self.device.type == "cuda"
            else dr.RasterizeGLContext(device=self.device)
        )

        # arbitrary in principle, useful for visualization 
        self.nocs0 = normalize_coords(self.V_obj)

    @torch.no_grad()
    def __call__(
        self, multi_frame, rescale_T=True, cameras_override=None
    ) -> AnnotationResult:
        src_cams = cameras_override if cameras_override is not None else multi_frame.cameras
        cameras = src_cams.clone()
        batch_size = len(cameras)

        seq_lengths = torch.tensor(multi_frame.seq_lengths, device=self.device)
        per_frame_obj_idx = torch.arange(
            len(seq_lengths), device=self.device
        ).repeat_interleave(seq_lengths)

        obj_sizes = multi_frame.obj_sizes.to(self.device)
        obj_centers = multi_frame.obj_centers.to(self.device)
        if obj_sizes.shape[0] == batch_size:
            sizes = obj_sizes
            centers = obj_centers
        elif obj_sizes.shape[0] == len(seq_lengths):
            sizes = obj_sizes[per_frame_obj_idx]
            centers = obj_centers[per_frame_obj_idx]
        else:
            raise ValueError(
                "obj_sizes/obj_centers must be either per-frame or per-sequence: "
                f"got {obj_sizes.shape[0]} rows for {batch_size} frames and "
                f"{len(seq_lengths)} sequences"
            )

        scales = torch.linalg.norm(sizes, dim=-1)
        V_render = self.V_obj[None, ...].expand(len(seq_lengths), -1, -1).repeat_interleave(seq_lengths, dim=0)

        Rc = torch.bmm(centers[:, None, :], cameras.R).squeeze(1)
        if rescale_T:
            cameras.T = (cameras.T + Rc) / scales[:, None]

        # Row-vector convention: v' = v @ M
        proj_matrix = cameras.full_projection_matrix()  # (B, 4, 4)

        V_homog = torch.cat(
            [V_render, torch.ones_like(V_render[..., :1])], dim=-1
        )
        V_clip = torch.bmm(V_homog, proj_matrix)  # (B, V, 4)

        # Invert Y and X axes to match OpenGL/nvdiffrast NDC conventions if coming from PyTorch3D
        # PyTorch3D's NDC is +X left, +Y up. OpenGL's is +X right, +Y up.
        # We negate X and Y to ensure rasterization matches PyTorch3D screen space exactly.
        V_clip[..., 0] *= -1.0
        V_clip[..., 1] *= -1.0

        # 2. Rasterize
        # Resolution is passed as [height, width]
        rast, rast_db = dr.rasterize(
            self.ctx, V_clip, self.F, resolution=list(self.image_size)
        )

        # Face ID mask (rast[..., 3] > 0 means a face was hit)
        mask = rast[..., 3] > 0


        obj_xyz, _ = dr.interpolate(self.V_obj.contiguous(), rast, self.F)

        # 6. Return AnnotationResult
        return AnnotationResult(
            proxy_mask=mask,
            obj_xyz=obj_xyz,
            obj_idx=per_frame_obj_idx,
            cameras=cameras,
        )

    @torch.no_grad()
    def rerender_aligned(
        self,
        multi_frame,
        R_obj: torch.Tensor,
        obj_id_per_frame: torch.Tensor = None,
        base_cameras=None,
    ) -> AnnotationResult:
        if obj_id_per_frame is None:
            seq_lengths = torch.tensor(multi_frame.seq_lengths, device=self.device)
            obj_id_per_frame = torch.arange(len(seq_lengths), device=self.device).repeat_interleave(seq_lengths)

        if base_cameras is None:
            annotations = getattr(multi_frame, "annotations", None)
            if annotations is not None:
                base_cameras = getattr(annotations, "cameras", None)
        if base_cameras is None:
            base_cameras = multi_frame.cameras

        cameras = base_cameras.clone()
        cameras.R = R_obj[obj_id_per_frame] @ cameras.R
        return self.__call__(
            multi_frame, rescale_T=False, cameras_override=cameras
        )
