import math
import multiprocessing as mp
import torch
from dataclasses import dataclass
from typing import Tuple, List, Optional
from omegaconf.dictconfig import DictConfig

from torchvision.transforms import v2
from torchvision.transforms import functional as TVF

from emergent_canonical_frame.data.structures import Camera, SequenceSample


class SharedIteration:
    def __init__(self, iteration: int = 0):
        self._value = mp.Value("q", int(iteration))

    def get(self) -> int:
        with self._value.get_lock():
            return int(self._value.value)

    def set(self, iteration: int) -> None:
        with self._value.get_lock():
            self._value.value = int(iteration)


@dataclass
class SampleTransformCfg:
    size_hw: Tuple[int, int] = (384, 384)
    training: bool = True

    crop_to_mask: bool = True
    crop_padding: Tuple[float, float] = (0.1, 0.25)

    geometric_start_iter: int = 0
    recrop_p: float = 0.7
    recrop_scale: Tuple[float, float] = (0.9, 1.0)
    recrop_ratio: Tuple[float, float] = (0.9, 1.1)
    recrop_min_size: int = 48
    rotate_p: float = 0.3
    rotate_deg: float = 30.0

    occlusion_start_iter: int = 0
    patch_mask_p: float = 0.5
    patch_mask_num: Tuple[int, int] = (1, 2)
    patch_mask_size: Tuple[float, float] = (0.05, 0.30)

    blur_p: float = 0.4
    grayscale_p: float = 0.2
    jitter_brightness: float = 0.4
    jitter_contrast: float = 0.4
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.1

    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

 
class SampleTransform:
    def __init__(self, cfg: SampleTransformCfg, iteration_state: SharedIteration | None = None):
        self.cfg = cfg
        self.iteration_state = iteration_state or SharedIteration(0)

        if cfg.training:
            self.color_jitter = v2.Compose(
                [
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                cfg.jitter_brightness,
                                cfg.jitter_contrast,
                                cfg.jitter_saturation,
                                cfg.jitter_hue,
                            )
                        ],
                        p=0.8,
                    ),
                    v2.RandomGrayscale(p=cfg.grayscale_p),
                ]
            )
            self.blur = v2.RandomApply([v2.GaussianBlur(kernel_size=9, sigma=(0.1, 1.0))], p=cfg.blur_p)
            self.occluder = RandomPatchOcclusion(
                p=cfg.patch_mask_p, num_patches=cfg.patch_mask_num, patch_size=cfg.patch_mask_size,
            ) if cfg.patch_mask_p > 0 else None
        else:
            self.color_jitter = None
            self.blur = None
            self.occluder = None

    def set_iteration(self, iteration: int) -> None:
        self.iteration_state.set(iteration)


    def _geometric_enabled(self) -> bool:
        return self.cfg.training and self.iteration_state.get() >= self.cfg.geometric_start_iter

    def _occlusion_enabled(self) -> bool:
        return self.cfg.training and self.iteration_state.get() >= self.cfg.occlusion_start_iter

    @torch.no_grad()
    def __call__(self, sample: SequenceSample) -> SequenceSample:
        cfg = self.cfg
        Ht, Wt = cfg.size_hw
        if sample.image_rgb is None:
            return sample

        imgs = TVF.convert_image_dtype(sample.image_rgb, dtype=torch.float32)  # (K,C,H,W)
        masks = sample.masks.float() if sample.masks is not None else None
        geometric = self._geometric_enabled()
        occlusion = self._occlusion_enabled()

        out_imgs, out_masks, out_valid, out_orig = [], [], [], []
        crop_offsets, scales, pad_offsets, rot_degs = [], [], [], []

        for i in range(imgs.shape[0]):
            frame = imgs[i]
            mask = masks[i] if masks is not None else None

            if cfg.crop_to_mask and mask is not None:
                pad = torch.empty(()).uniform_(*cfg.crop_padding).item() if geometric else cfg.crop_padding[0]
                y1, x1, y2, x2 = _mask_bbox(mask, padding=pad)
                frame, mask = frame[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2]
                crop = (x1, y1)
            else:
                crop = (0, 0)

            if geometric and cfg.recrop_p > 0 and torch.rand(()).item() < cfg.recrop_p:
                Hc, Wc = frame.shape[-2:]
                y1r, x1r, y2r, x2r = _sample_recrop_hw(
                    Hc, Wc, cfg.recrop_scale, cfg.recrop_ratio, min_size=cfg.recrop_min_size,
                )
                frame = frame[:, y1r:y2r, x1r:x2r]
                if mask is not None:
                    mask = mask[:, y1r:y2r, x1r:x2r]
                crop = (crop[0] + x1r, crop[1] + y1r)

            pre_sz = torch.tensor(frame.shape[-2:], dtype=torch.float32)

            frame_4d, s, cam_off, (vH, vW) = _scale_to_fit_and_blur_pad(
                frame.unsqueeze(0), Ht, Wt, mode="bilinear"
            )
            frame = frame_4d.squeeze(0)
            valid = torch.zeros((1, Ht, Wt), dtype=torch.uint8)
            valid[:, cam_off[1]:cam_off[1] + vH, cam_off[0]:cam_off[0] + vW] = 1
            if mask is not None:
                mask = _scale_to_fit_and_zero_pad(mask.unsqueeze(0), Ht, Wt, "nearest")[0].squeeze(0)

            ang = 0.0
            if geometric and cfg.rotate_p > 0 and torch.rand(()).item() < cfg.rotate_p:
                ang = torch.empty(()).uniform_(-cfg.rotate_deg, cfg.rotate_deg).item()
                # Rotating leaves triangular corners uncovered; fill them with
                # the frame's mean color via alpha-composite instead of black.
                ch_mean = frame.mean(dim=(-2, -1), keepdim=True)
                rgba = torch.cat([frame, torch.ones_like(frame[:1])], dim=0)
                rgba = TVF.rotate(rgba, angle=ang, interpolation=TVF.InterpolationMode.BILINEAR, expand=False, fill=0.0)
                frame, cov = rgba[:3], rgba[3:4]
                frame = frame + (1.0 - cov) * ch_mean
                if mask is not None:
                    mask = TVF.rotate(mask, angle=ang, interpolation=TVF.InterpolationMode.NEAREST, expand=False, fill=0)
                valid = TVF.rotate(valid, angle=ang, interpolation=TVF.InterpolationMode.NEAREST, expand=False, fill=0)

            if occlusion and self.occluder is not None:
                content_box = (max(0, cam_off[0]), max(0, cam_off[1]), vW, vH)
                frame, keep = self.occluder(frame, content_box=content_box)
                valid = valid * keep.to(dtype=valid.dtype)

            if self.color_jitter is not None:
                frame = self.color_jitter(frame)
            if self.blur is not None:
                frame = self.blur(frame)

            out_imgs.append(frame)
            if mask is not None:
                out_masks.append(mask)
            out_valid.append(valid)
            out_orig.append(pre_sz)
            crop_offsets.append(crop)
            scales.append(s)
            pad_offsets.append(cam_off)
            rot_degs.append(ang)

        imgs_out = torch.stack(out_imgs, dim=0)
        imgs_out = TVF.normalize(imgs_out, mean=list(cfg.normalize_mean), std=list(cfg.normalize_std))
        masks_out = (torch.stack(out_masks, dim=0) > 0.5).to(torch.uint8) if out_masks else None
        valid_out = (torch.stack(out_valid, dim=0) > 0).to(torch.uint8)

        cams = sample.cameras
        _adjust_cameras_for_crop_scale_pad(cams, crop_offsets, scales, pad_offsets, (Ht, Wt))
        if any(a != 0.0 for a in rot_degs):
            Rz = _rotz_batch(torch.tensor(rot_degs, device=cams.R.device, dtype=cams.R.dtype))
            cams.R, cams.T = _rotate_on_spot(cams.R, cams.T, Rz)
            _rotate_principal_point_(cams, Rz)

        return SequenceSample(
            sequence_name=sample.sequence_name,
            frame_numbers=sample.frame_numbers,
            cameras=cams,
            image_rgb=imgs_out,
            masks=masks_out,
            valid_masks=valid_out,
            orig_sizes_hw=torch.stack(out_orig, dim=0),
            segmented_point_cloud=sample.segmented_point_cloud,
            object_size=sample.object_size,
            obj_center=sample.obj_center,
        )

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _rotz_batch(deg: torch.Tensor) -> torch.Tensor:
    rad = deg * (math.pi / 180.0)
    c, s = torch.cos(rad), torch.sin(rad)
    R = torch.zeros((deg.shape[0], 3, 3), device=deg.device, dtype=torch.float32)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def _rotate_on_spot(R: torch.Tensor, T: torch.Tensor, Rz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row-convention (`v_view = v_world @ R + T`) rotation about the camera center."""
    return R @ Rz, (T.unsqueeze(1) @ Rz).squeeze(1)


def _rotate_principal_point_(cams: Camera, Rz: torch.Tensor) -> None:
    if cams.principal_point is None:
        return
    R2 = Rz[:, :2, :2].to(device=cams.device, dtype=cams.principal_point.dtype)
    cams.principal_point = (cams.principal_point.unsqueeze(1) @ R2).squeeze(1)


def _mask_bbox(mask: torch.Tensor, padding: float = 0.0) -> Tuple[int, int, int, int]:
    if mask.dim() == 3:
        mask = mask.squeeze(0)
    H, W = mask.shape
    nz = mask.nonzero(as_tuple=True)
    if len(nz[0]) == 0:
        return 0, 0, H, W
    y1, y2 = nz[0].min().item(), nz[0].max().item()
    x1, x2 = nz[1].min().item(), nz[1].max().item()
    if padding > 0:
        pad_h, pad_w = int((y2 - y1) * padding), int((x2 - x1) * padding)
        y1, y2 = max(0, y1 - pad_h), min(H, y2 + pad_h)
        x1, x2 = max(0, x1 - pad_w), min(W, x2 + pad_w)
    return y1, x1, y2 + 1, x2 + 1


def _sample_recrop_hw(H, W, scale_range, ratio_range, min_size=32, num_tries=10):
    area = float(H * W)
    smin, smax = scale_range
    rmin, rmax = ratio_range
    for _ in range(num_tries):
        target_area = area * torch.empty(()).uniform_(smin, smax).item()
        ratio = math.exp(torch.empty(()).uniform_(math.log(rmin), math.log(rmax)).item())
        h = int(round(math.sqrt(target_area / ratio)))
        w = int(round(math.sqrt(target_area * ratio)))
        if min_size <= h <= H and min_size <= w <= W:
            y1 = int(torch.randint(0, H - h + 1, (1,)).item())
            x1 = int(torch.randint(0, W - w + 1, (1,)).item())
            return y1, x1, y1 + h, x1 + w
    h = w = max(min_size, int(round(min(H, W) * 0.9)))
    return (H - h) // 2, (W - w) // 2, (H - h) // 2 + h, (W - w) // 2 + w

def _scale_to_fit_and_zero_pad(x, Ht, Wt, mode):
    N, C, H, W = x.shape
    s = min(Ht / H, Wt / W)
    Hr, Wr = int(round(H * s)), int(round(W * s))
    ac = False if mode == "bilinear" else None
    x_fit = torch.nn.functional.interpolate(x, size=(Hr, Wr), mode=mode, align_corners=ac)
    out = torch.zeros(N, C, Ht, Wt, device=x.device, dtype=x.dtype)
    pt, pl = (Ht - Hr) // 2, (Wt - Wr) // 2
    out[:, :, pt:pt + Hr, pl:pl + Wr] = x_fit
    return out, s, (pl, pt), (Hr, Wr)

def _scale_to_fit_and_mean_pad(x, Ht, Wt, mode, fill):
    """Resize to fit inside (Ht,Wt) keeping aspect ratio; fill the border with
    a flat mean color instead of black, so the model doesn't see hard zero
    edges and (unlike blur-pad) no object content leaks into the border."""
    N, C, H, W = x.shape
    s = min(Ht / H, Wt / W)
    Hr, Wr = int(round(H * s)), int(round(W * s))
    ac = False if mode == "bilinear" else None
    x_fit = torch.nn.functional.interpolate(x, size=(Hr, Wr), mode=mode, align_corners=ac)
    fill_t = torch.tensor(fill, device=x.device, dtype=x.dtype).view(1, C, 1, 1)
    out = fill_t.expand(N, C, Ht, Wt).clone()
    pt, pl = (Ht - Hr) // 2, (Wt - Wr) // 2
    out[:, :, pt:pt + Hr, pl:pl + Wr] = x_fit
    return out, s, (pl, pt), (Hr, Wr)

def _scale_to_fit_and_blur_pad(x, Ht, Wt, mode="bilinear", bg_down: int = 16):
    N, C, H, W = x.shape
    s = min(Ht / H, Wt / W)
    Hr, Wr = int(round(H * s)), int(round(W * s))
    ac = False if mode == "bilinear" else None
    x_fit = torch.nn.functional.interpolate(x, size=(Hr, Wr), mode=mode, align_corners=ac)
    # Blurry background approximated by hard area-downsample then upsample.
    # Constant cost (independent of input size) and visually equivalent to a
    # wide-sigma gaussian for the purpose of hiding hard pad edges.
    Hs = max(1, Ht // bg_down)
    Ws = max(1, Wt // bg_down)
    x_small = torch.nn.functional.interpolate(x, size=(Hs, Ws), mode="area")
    x_bg = torch.nn.functional.interpolate(x_small, size=(Ht, Wt), mode=mode, align_corners=ac)
    pt, pl = (Ht - Hr) // 2, (Wt - Wr) // 2
    x_bg[:, :, pt:pt + Hr, pl:pl + Wr] = x_fit
    return x_bg, s, (pl, pt), (Hr, Wr)

class RandomPatchOcclusion(torch.nn.Module):
    """Blacks out a few random rectangles inside `content_box`, e.g. to
    simulate occluders. Returns the modified image and a `keep` mask."""

    def __init__(self, p=0.5, num_patches=(1, 2), patch_size=(0.06, 0.25), min_patch_px=4):
        super().__init__()
        self.p = float(p)
        self.num_patches = (int(num_patches[0]), int(num_patches[1]))
        self.patch_size = (float(patch_size[0]), float(patch_size[1]))
        self.min_patch_px = int(min_patch_px)

    def forward(self, img: torch.Tensor, content_box: Tuple[int, int, int, int]):
        C, H, W = img.shape
        keep = torch.ones((1, H, W), device=img.device, dtype=torch.uint8)
        x0, y0, bw, bh = content_box
        x0, y0 = max(0, min(x0, W - 1)), max(0, min(y0, H - 1))
        bw, bh = max(1, min(bw, W - x0)), max(1, min(bh, H - y0))
        if torch.rand(()).item() > self.p:
            return img, keep

        img = img.clone()
        num = int(torch.randint(self.num_patches[0], self.num_patches[1] + 1, (1,)).item())
        base = float(min(bw, bh))
        for _ in range(num):
            fh = torch.empty(()).uniform_(*self.patch_size).item()
            fw = torch.empty(()).uniform_(*self.patch_size).item()
            ph = min(max(self.min_patch_px, int(round(base * fh))), bh)
            pw = min(max(self.min_patch_px, int(round(base * fw))), bw)
            y = int(torch.randint(y0, y0 + max(1, bh - ph + 1), (1,)).item())
            x = int(torch.randint(x0, x0 + max(1, bw - pw + 1), (1,)).item())
            img[:, y:y + ph, x:x + pw] = 0
            keep[:, y:y + ph, x:x + pw] = 0
        return img, keep


def _adjust_cameras_for_crop_scale_pad(
    cams: Camera,
    crop_offsets: List[Tuple[int, int]],
    scales: List[float],
    padding_offsets: List[Tuple[int, int]],
    new_size_hw: Tuple[int, int],
) -> None:
    """Rewrite intrinsics in-place so the camera stays consistent with the
    crop -> rescale -> pad chain each frame went through."""
    if cams.focal_length is None or cams.principal_point is None or cams.image_size is None:
        H, W = new_size_hw
        cams.image_size = torch.tensor([H, W], dtype=torch.float32, device=cams.device).repeat(cams.R.shape[0], 1)
        return

    N, dev = cams.R.shape[0], cams.device
    scales_t = torch.tensor(scales, dtype=torch.float32, device=dev)
    pad_left_t = torch.tensor([p[0] for p in padding_offsets], dtype=torch.float32, device=dev)
    pad_top_t = torch.tensor([p[1] for p in padding_offsets], dtype=torch.float32, device=dev)
    crop_x_t = torch.tensor([c[0] for c in crop_offsets], dtype=torch.float32, device=dev)
    crop_y_t = torch.tensor([c[1] for c in crop_offsets], dtype=torch.float32, device=dev)

    old_sizes_wh = cams.image_size.flip(dims=[1])
    half_old = old_sizes_wh / 2.0
    rescale_old = half_old.min(dim=1, keepdim=True).values
    focal_px = cams.focal_length * rescale_old
    pp_px = half_old - cams.principal_point * rescale_old

    pp_px[:, 0] -= crop_x_t
    pp_px[:, 1] -= crop_y_t
    focal_px = focal_px * scales_t.unsqueeze(1)
    pp_px = pp_px * scales_t.unsqueeze(1)
    pp_px[:, 0] += pad_left_t
    pp_px[:, 1] += pad_top_t

    Ht, Wt = new_size_hw
    half_new = torch.tensor([Wt, Ht], dtype=torch.float32, device=dev).unsqueeze(0) / 2.0
    rescale_new = half_new.min(dim=1, keepdim=True).values

    cams.focal_length = focal_px / rescale_new
    cams.principal_point = (half_new - pp_px) / rescale_new
    cams.image_size = torch.tensor([Ht, Wt], dtype=torch.float32, device=dev).repeat(N, 1)

def transform_from_cfg(cfg: DictConfig, training: bool = True) -> SampleTransform:
    aug = cfg.augmentation

    tcfg = SampleTransformCfg(
        size_hw=tuple(cfg.image_size),
        training=training,
        **aug,
    )

    return SampleTransform(tcfg)