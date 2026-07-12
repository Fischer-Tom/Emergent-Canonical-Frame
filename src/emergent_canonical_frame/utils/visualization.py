from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def save_batch_visualization(
    path: Path,
    images: torch.Tensor,
    obj_xyz: torch.Tensor,
    valid_mask: torch.Tensor,
    v_min: torch.Tensor,
    v_max: torch.Tensor,
) -> None:
    """
    Dumps a per-view grid [image | rendered obj_xyz | valid_mask] to `path`.
    images: (N, 3, H, W) ImageNet-normalized. obj_xyz: (N, Hf, Wf, 3). valid_mask: (N, Hf, Wf) bool.
    """
    device = images.device
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    imgs = (images * std + mean).clamp(0.0, 1.0)

    Hf, Wf = obj_xyz.shape[1:3]
    imgs = F.interpolate(imgs, size=(Hf, Wf), mode="bilinear", align_corners=False)

    mask = valid_mask.float()
    xyz = (obj_xyz - v_min) / (v_max - v_min).clamp_min(1e-8)
    xyz = xyz.clamp(0.0, 1.0).permute(0, 3, 1, 2) * mask.unsqueeze(1)

    mask_rgb = mask.unsqueeze(1).expand(-1, 3, -1, -1)

    n = imgs.shape[0]
    triplets = torch.stack([imgs, xyz, mask_rgb], dim=1).reshape(n * 3, 3, Hf, Wf)
    grid = vutils.make_grid(triplets, nrow=3, padding=2)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, path)
