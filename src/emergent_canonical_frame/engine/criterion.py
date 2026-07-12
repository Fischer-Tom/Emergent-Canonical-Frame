from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from emergent_canonical_frame.utils.alignment import (
    AlignmentResult,
    find_closest_vertex,
    l2_normalize,
)


class Criterion(nn.Module):
    """Loss calculation only. Alignment (Wahba/RANSAC/discretization) lives in
    utils/alignment.py; mesh geometry (V) is owned by the model."""

    def __init__(self, mask_loss_weight: float = 1.0):
        super().__init__()
        self.mask_loss_weight = float(mask_loss_weight)

    @torch.no_grad()
    def _teacher_distribution_bary(self, Vn: Tensor, Rv_flat: Tensor, k: int = 3, eps: float = 1e-12):
        device = Rv_flat.device
        dtype = Rv_flat.dtype
        N = Rv_flat.shape[0]
        M = Vn.shape[0]

        idx, bary = find_closest_vertex(Vn, Rv_flat, k=k, single=False)
        bary = bary / (bary.sum(dim=1, keepdim=True) + eps)
        q = torch.zeros(N, M, device=device, dtype=dtype)
        q.scatter_add_(1, idx, bary.to(dtype))
        return q / (q.sum(dim=1, keepdim=True) + eps)

    @torch.no_grad()
    def _teacher_distribution_top(self, Vn: Tensor, Rv_flat: Tensor, eps: float = 1e-12):
        device = Rv_flat.device
        dtype = Rv_flat.dtype
        N = Rv_flat.shape[0]
        M = Vn.shape[0]

        idx, _ = find_closest_vertex(Vn, Rv_flat, single=True)
        q = torch.zeros(N, M, device=device, dtype=dtype)
        q.scatter_add_(1, idx, torch.ones_like(idx).to(dtype))
        return q

    def compute_loss(
        self,
        V: Tensor,
        alignment: AlignmentResult,
        mask_logits: Tensor,
        gt_mask: Tensor,
        valid_region: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        Lf_student = alignment.Lf_student  # (B, M, H, W)
        Rv = alignment.Rv  # (B, H, W, 3)
        valid_mask = alignment.valid_mask  # (B, H, W)
        sched = alignment.sched
        B, M, H, W = Lf_student.shape
        valid = valid_mask.to(Lf_student.dtype)

        Vn = l2_normalize(V, dim=1)

        # Teacher target built densely over every pixel (no gather), masked at
        # reduction time only, so shapes stay fixed for torch.compile.
        Rv_flat = Rv.reshape(-1, 3)
        if sched["teacher"] == "top":
            q_flat = self._teacher_distribution_top(Vn, Rv_flat)
        else:
            q_flat = self._teacher_distribution_bary(Vn, Rv_flat, k=3)
        q = q_flat.view(B, H, W, M)

        logp = torch.log_softmax(Lf_student, dim=1).clamp(min=-100)  # (B, M, H, W)
        loss_kd_pp = -(q * logp.permute(0, 2, 3, 1)).sum(dim=-1)  # (B, H, W)

        counts = valid.sum(dim=(1, 2))
        sums = (loss_kd_pp * valid).sum(dim=(1, 2))
        nonempty = counts > 0
        loss_kd = (sums[nonempty] / counts[nonempty]).mean()

        entropy = -(torch.exp(logp) * logp).sum(dim=1)  # (B, H, W)
        n_valid = valid.sum().clamp_min(1.0)
        loss_entropy = (entropy * valid).sum() / n_valid

        loss_mask = self._edge_mask_loss(mask_logits, gt_mask, valid_region=valid_region)
        return {
            "loss_mesh": loss_kd,
            "loss_entropy": loss_entropy.detach(),
            "loss_mask": loss_mask,
            "loss_total": loss_kd + self.mask_loss_weight * loss_mask,
            "alignment_angle": alignment.alignment_angle.detach(),
            "ransac_inlier_ratio": alignment.ransac_inlier_ratio.detach(),
        }

    def forward(
        self,
        V: Tensor,
        alignment: AlignmentResult,
        mask_logits: Tensor,
        gt_mask: Tensor,
        valid_region: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        return self.compute_loss(V, alignment, mask_logits, gt_mask, valid_region=valid_region)

    def _get_mask_edges(self, mask: torch.Tensor, edge_width: int = 2) -> torch.Tensor:
        """
        Extract edge pixels from a binary mask using morphological operations.

        Args:
            mask: Binary mask of shape (B, H, W) or (B, 1, H, W)
            edge_width: Width of the edge band in pixels

        Returns:
            Edge mask of same shape as input, with 1s at edge pixels
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Use max pooling for dilation, -max(-x) for erosion
        kernel_size = 2 * edge_width + 1
        padding = edge_width

        # Dilate: expand the mask outward
        dilated = F.max_pool2d(mask.float(), kernel_size, stride=1, padding=padding)
        # Erode: shrink the mask inward
        eroded = -F.max_pool2d(-mask.float(), kernel_size, stride=1, padding=padding)

        # Edge = dilated - eroded (band around the boundary)
        edges = (dilated - eroded).clamp(0, 1)
        # Zero out edges at image boundaries (artifact from padding in pooling)
        # When mask=1 at image boundary, erosion shrinks it due to zero-padding,
        # creating spurious edges that cause the model to predict masks near black borders
        H, W = edges.shape[-2:]
        boundary_mask = torch.ones_like(edges)
        boundary_mask[..., :edge_width, :] = 0  # top
        boundary_mask[..., -edge_width:, :] = 0  # bottom
        boundary_mask[..., :, :edge_width] = 0  # left
        boundary_mask[..., :, -edge_width:] = 0  # right
        edges = edges * boundary_mask
        return edges.squeeze(1)  # (B, H, W)

    def _edge_mask_loss(
        self,
        pred_mask: torch.Tensor,
        gt_mask: torch.Tensor,
        edge_width: int = 2,
        interior_weight: float = 0.1,
        valid_region: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute mask loss focused on edges (sparse computation when interior_weight=0).

        Args:
            pred_mask: Predicted mask logits (B, 1, H, W)
            gt_mask: Ground truth binary mask (B, H, W)
            edge_width: Width of edge band in pixels
            interior_weight: Weight for non-edge pixels (0 = sparse edge-only)
            valid_region: (B, H, W) or (B, 1, H, W) bool/uint8 mask. Loss is only
                            computed where valid_region=1 (excludes padding & occlusions).

        Returns:
            Weighted mask loss (BCE + Dice on edges)
        """
        # Get edge regions from GT mask
        edge_mask = self._get_mask_edges(gt_mask, edge_width=edge_width)  # (B, H, W)
        edge_pixels = edge_mask > 0.5

        # Restrict to valid region (exclude padding and synthetic occlusions)
        if valid_region is not None:
            if valid_region.dim() == 4:
                valid_region = valid_region.squeeze(1)
            valid_bool = valid_region.bool()
            edge_pixels = edge_pixels & valid_bool

        pred_flat = pred_mask.squeeze(1)  # (B, H, W)
        gt_flat = gt_mask.float()
        edge_weights = edge_pixels.to(pred_flat.dtype)
        if valid_region is not None:
            edge_weights = edge_weights * valid_bool.to(pred_flat.dtype)

        # Dense throughout: mask via multiplication instead of boolean gather,
        # so shapes stay fixed regardless of how many edge pixels are present
        # (needed for torch.compile).
        if interior_weight > 0:
            bce = F.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction="none")
            weights = torch.where(edge_pixels, 1.0, interior_weight)
            if valid_region is not None:
                weights = weights * valid_bool.to(pred_flat.dtype)
            weighted_bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
        else:
            bce = F.binary_cross_entropy_with_logits(pred_flat, gt_flat, reduction="none")
            weighted_bce = (bce * edge_weights).sum() / edge_weights.sum().clamp_min(1.0)

        pred_sigmoid = torch.sigmoid(pred_flat)
        intersection = (pred_sigmoid * gt_flat * edge_weights).sum()
        union = (pred_sigmoid * edge_weights).sum() + (gt_flat * edge_weights).sum()
        dice = 1 - (2 * intersection + 1) / (union + 1)

        return weighted_bce + dice
