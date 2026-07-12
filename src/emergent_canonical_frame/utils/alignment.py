import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from einops import einsum
from torch import nn


_RANSAC_WAHBA_ITERS = 100
_RANSAC_WAHBA_SAMPLE_SIZE = 4
_RANSAC_WAHBA_INLIER_COS = math.cos(math.radians(25.0))
_RANSAC_WAHBA_EPS = 1e-6


@dataclass
class AlignmentResult:
    R_frame: torch.Tensor  # (B, 3, 3)
    R_obj: torch.Tensor  # (G, 3, 3)
    obj_id: torch.Tensor  # (B,)
    v: torch.Tensor  # (B, H, W, 3)
    Rv: torch.Tensor  # (B, H, W, 3)
    Lf_student: torch.Tensor  # (B, M, H, W)
    valid_mask: torch.Tensor  # (B, H, W)
    sched: dict
    alignment_angle: torch.Tensor
    ransac_inlier_ratio: torch.Tensor
    G: int


def _cosine(t, start, end):
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def _phase_value(t, vA, vB, vC):
    if t <= 0.3:
        x = 0.0 if not isinstance(vA, tuple) else (t / 0.3)
        return vA if not isinstance(vA, tuple) else _cosine(x, vA[0], vA[1])
    elif t <= 0.8:
        x = (t - 0.3) / 0.5
        return vB if not isinstance(vB, tuple) else _cosine(x, vB[0], vB[1])
    else:
        x = (t - 0.8) / 0.2
        return vC if not isinstance(vC, tuple) else _cosine(x, vC[0], vC[1])


def build_schedule(step: int, total_steps: int):
    t = min(max(step / max(1, total_steps - 1), 0.0), 1.0)
    T_f = _phase_value(t, (0.20, 0.17), (0.17, 0.13), (0.13, 0.10))
    return dict(T_f=T_f, t=t, teacher="top")


def l2_normalize(x, dim):
    return F.normalize(x, p=2, dim=dim, eps=1e-12)


def matching(V: torch.Tensor, logits: torch.Tensor):
    _, v_idx = logits.max(dim=1)
    return V[v_idx]


@torch.no_grad()
def find_closest_vertex(Vn: torch.Tensor, v3d: torch.Tensor, k: int = 3, single: bool = True):
    v3d_hat = l2_normalize(v3d, dim=1)
    sim = einsum(Vn, v3d_hat, "v i, b i -> v b")
    if single:
        idx = torch.argmax(sim, dim=0)
        return idx.unsqueeze(1), None
    idx = torch.topk(sim, k=k, dim=0).indices.permute(1, 0)
    nbrs = Vn[idx]
    bary_coords = invdist_weights(v3d_hat, nbrs)
    return idx, bary_coords


def invdist_weights(p: torch.Tensor, nbrs: torch.Tensor, eps: float = 1e-6):
    sims = (p.unsqueeze(1) * nbrs).sum(dim=-1)
    d = 2.0 * (1.0 - sims).clamp_min(0.0)
    inv = 1.0 / (d + eps)
    return inv / inv.sum(dim=-1, keepdim=True)


@torch.no_grad()
def wahba_batched(H: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Row-convention Procrustes: R = V @ diag(corr) @ U.T, batched over dim 0 of H.

    Invalid entries (valid_mask=False) fall back to identity via `where`
    instead of boolean indexing, so the op has no data-dependent shape.
    """
    G = H.shape[0]
    original_dtype = H.dtype

    with torch.amp.autocast(device_type="cuda", enabled=False):
        H_f32 = H.float()
        U, S, VT = torch.linalg.svd(H_f32, full_matrices=False)
        V = VT.transpose(-1, -2)
        det = torch.det(V @ U.transpose(-1, -2))
        sgn = det < 0.0
        corr = torch.ones((G, 3), device=H.device, dtype=torch.float32)
        corr[:, -1] = torch.where(sgn, -1.0, 1.0)
        Rf = V @ torch.diag_embed(corr) @ U.transpose(-1, -2)

    eye3 = torch.eye(3, device=H.device, dtype=torch.float32).expand(G, 3, 3)
    Rf = torch.where(valid_mask.view(G, 1, 1), Rf, eye3)
    return Rf.to(original_dtype)


@torch.no_grad()
def _wahba_matrix_from_points(
    v: torch.Tensor,
    v_match: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    outer = torch.einsum("ni,nj->nij", v, v_match)
    return (weights[:, None, None] * outer).sum(dim=0)


@torch.no_grad()
def wahba_ransac_dense(
    v: torch.Tensor,
    v_match: torch.Tensor,
    weights: torch.Tensor,
    obj_id: torch.Tensor,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust per-object Wahba over dense grids.

    The model and loss remain dense. Only the no-grad teacher estimator gathers
    the valid pixels belonging to each object, matching the original RANSAC
    semantics: 100 weighted four-point hypotheses followed by an inlier refit.
    """
    device = v.device
    if G == 0:
        return (
            torch.empty((0, 3, 3), device=device, dtype=torch.float32),
            torch.tensor(0.0, device=device),
        )

    R_out = torch.eye(3, device=device, dtype=torch.float32).expand(G, 3, 3).clone()
    inlier_ratio = torch.zeros(G, device=device, dtype=torch.float32)
    has_inlier_metric = torch.zeros(G, device=device, dtype=torch.bool)

    for g in range(G):
        frame_mask = obj_id == g
        if not bool(frame_mask.any()):
            continue

        vg = v[frame_mask].reshape(-1, 3).float()
        mg = v_match[frame_mask].reshape(-1, 3).float()
        wg = weights[frame_mask].reshape(-1).float().clamp_min(0.0)

        valid_global = wg.sum() > _RANSAC_WAHBA_EPS
        H_global = _wahba_matrix_from_points(vg, mg, wg).unsqueeze(0)
        R_global = wahba_batched(
            H_global,
            valid_mask=valid_global.view(1),
        )[0].float()

        positive = wg > _RANSAC_WAHBA_EPS
        n_positive = int(positive.sum().item())
        if n_positive == 0:
            R_out[g] = R_global
            continue

        vp = vg[positive]
        mp = mg[positive]
        wp = wg[positive]
        R_final = R_global

        if n_positive >= _RANSAC_WAHBA_SAMPLE_SIZE:
            probability = wp / wp.sum().clamp_min(_RANSAC_WAHBA_EPS)
            sample_idx = torch.stack(
                [
                    torch.multinomial(
                        probability,
                        _RANSAC_WAHBA_SAMPLE_SIZE,
                        replacement=False,
                    )
                    for _ in range(_RANSAC_WAHBA_ITERS)
                ],
                dim=0,
            )
            vs = vp[sample_idx]
            ms = mp[sample_idx]
            ws = wp[sample_idx]
            H_samples = (
                ws[:, :, None, None] * torch.einsum("ksi,ksj->ksij", vs, ms)
            ).sum(dim=1)
            R_candidates = wahba_batched(
                H_samples,
                valid_mask=ws.sum(dim=1) > _RANSAC_WAHBA_EPS,
            ).float()

            pred = torch.matmul(vp.unsqueeze(0), R_candidates.transpose(-1, -2))
            cos = (pred * mp.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
            inliers = cos >= _RANSAC_WAHBA_INLIER_COS
            consensus = (inliers.float() * wp.unsqueeze(0)).sum(dim=1)
            mean_cos = (cos * wp.unsqueeze(0)).sum(dim=1) / wp.sum().clamp_min(
                _RANSAC_WAHBA_EPS
            )
            best = (consensus + 1e-3 * mean_cos).argmax()
            best_inliers = inliers[best]

            if int(best_inliers.sum().item()) >= _RANSAC_WAHBA_SAMPLE_SIZE:
                H_inlier = _wahba_matrix_from_points(
                    vp[best_inliers],
                    mp[best_inliers],
                    wp[best_inliers],
                ).unsqueeze(0)
                R_final = wahba_batched(
                    H_inlier,
                    valid_mask=torch.ones(1, device=device, dtype=torch.bool),
                )[0].float()

        R_out[g] = R_final
        pred_final = vp @ R_final.transpose(0, 1)
        final_inliers = (pred_final * mp).sum(dim=-1).clamp(
            -1.0, 1.0
        ) >= _RANSAC_WAHBA_INLIER_COS
        inlier_ratio[g] = (final_inliers.float() * wp).sum() / wp.sum().clamp_min(
            _RANSAC_WAHBA_EPS
        )
        has_inlier_metric[g] = True

    if bool(has_inlier_metric.any()):
        mean_inlier_ratio = inlier_ratio[has_inlier_metric].mean()
    else:
        mean_inlier_ratio = torch.tensor(0.0, device=device)
    return R_out, mean_inlier_ratio


class CanonicalAligner(nn.Module):
    """Aligns dense per-pixel mesh-vertex predictions to a per-object canonical
    rotation via weighted Wahba/Procrustes.

    Operates on full image grids (logits, v_grid, valid_mask) rather than
    gathered/selected pixels, so tensor shapes are static across calls and the
    op is friendly to torch.compile.
    """

    def __init__(self, total_iters: int = None, use_ransac: bool = True):
        super().__init__()
        self.total_iters = total_iters
        self.use_ransac = bool(use_ransac)
        self.c_iter = 0

    def forward(
        self,
        V: torch.Tensor,  # (M, 3)
        logits: torch.Tensor,  # (B, M, H, W)
        v_grid: torch.Tensor,  # (B, H, W, 3)
        valid_mask: torch.Tensor,  # (B, H, W) bool
        obj_id: torch.Tensor,  # (B,) long
        n_objects: Optional[int] = None,
    ) -> AlignmentResult:
        device = logits.device
        assert self.total_iters is not None
        sched = build_schedule(self.c_iter, self.total_iters)

        temperature = float(sched["T_f"])
        # Preserve the model dtype and gradient path used by the dense loss.
        Lf_student = logits / temperature

        # Teacher alignment is no-grad and explicitly FP32. In particular,
        # normalized entropy close to one must not be quantized to exactly one
        # by BF16 before the confidence weight is squared.
        with (
            torch.no_grad(),
            torch.amp.autocast(device_type=device.type, enabled=False),
        ):
            Vn = l2_normalize(V.float(), dim=1)
            v = l2_normalize(v_grid.to(device=device).float(), dim=-1)
            Lf_alignment = logits.float() / temperature
            p_student = torch.softmax(Lf_alignment, dim=1)
            v_match = l2_normalize(
                torch.einsum("bmhw,mc->bhwc", p_student, Vn), dim=-1
            )

            M = logits.shape[1]
            logp = torch.log(p_student.clamp_min(1e-8))
            Hn = -(p_student * logp).sum(dim=1)
            Hn_norm = Hn / math.log(M)
            valid = valid_mask.to(dtype=torch.float32)
            w = (1.0 - Hn_norm).clamp(0.0, 1.0).pow(2.0) * valid

        obj_id = obj_id.to(torch.long)
        G = n_objects if n_objects is not None else int(obj_id.max().item()) + 1

        if self.use_ransac:
            R_obj, ransac_inlier_ratio = wahba_ransac_dense(
                v,
                v_match,
                w,
                obj_id,
                G,
            )
        else:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                v_f32 = v.float()
                v_match_f32 = v_match.float()
                w_f32 = w.float()

                H_frame = torch.einsum(
                    "bhwi,bhwj,bhw->bij", v_f32, v_match_f32, w_f32
                )
                weight_frame = w_f32.sum(dim=(1, 2))

                H_obj = torch.zeros(G, 3, 3, device=device, dtype=torch.float32)
                H_obj.index_add_(0, obj_id, H_frame)
                weight_obj = torch.zeros(G, device=device, dtype=torch.float32)
                weight_obj.index_add_(0, obj_id, weight_frame)

                R_obj = wahba_batched(H_obj, valid_mask=weight_obj > 1e-6)
            ransac_inlier_ratio = torch.tensor(0.0, device=device)

        R_frame = R_obj.float()[obj_id]
        Rv = torch.einsum("bhwi,bji->bhwj", v, R_frame)

        with torch.no_grad():
            cos_sim = (Rv.detach() * v_match.detach()).sum(dim=-1)
            alignment_angle = (cos_sim * valid).sum() / valid.sum().clamp_min(1.0)

        return AlignmentResult(
            R_frame=R_frame,
            R_obj=R_obj.float(),
            obj_id=obj_id,
            v=v,
            Rv=Rv,
            Lf_student=Lf_student,
            valid_mask=valid_mask,
            sched=sched,
            alignment_angle=alignment_angle.detach(),
            ransac_inlier_ratio=ransac_inlier_ratio.detach(),
            G=G,
        )


def align_sfm_poses(corr_dict, out_dict):
    R = corr_dict.R @ out_dict["R"].T
    return R
