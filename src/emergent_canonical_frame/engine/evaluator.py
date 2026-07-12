from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from einops import einsum
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emergent_canonical_frame.data.loader import build_loader
from emergent_canonical_frame.utils.pose import (
    estimate_T_n2c_for_frame,
    rot_err_deg,
    rot_err_deg_sym,
)


_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


def valid_rotations_cube(device, dtype, gravity: bool = False):
    Rs = []
    seen = set()
    eye = torch.eye(3, device=device, dtype=dtype)
    permutations = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    signs_set = (
        (1, 1, 1),
        (1, 1, -1),
        (1, -1, 1),
        (-1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (-1, -1, -1),
    )
    for perm in permutations:
        P = eye[:, perm]
        for signs in signs_set:
            S = torch.diag(torch.tensor(signs, device=device, dtype=dtype))
            R = P @ S
            if int(round(torch.det(R).item())) == 1:
                key = tuple(
                    (R.cpu().numpy() * 100).round().astype(int).flatten()
                )
                if key not in seen:
                    seen.add(key)
                    Rs.append(R)
    Rset = torch.stack(Rs, dim=0)
    if Rset.shape[0] != 24:
        raise RuntimeError(f"Expected 24 cube rotations, got {Rset.shape[0]}")
    if gravity:
        y_axis = torch.tensor(
            [0.0, 1.0, 0.0],
            device=device,
            dtype=dtype,
        )
        keep = torch.tensor(
            [bool(torch.allclose(R[:, 1], y_axis, atol=1e-6)) for R in Rset],
            device=device,
        )
        Rset = Rset[keep]
    return Rset


def conjugate_rotation_average(
    gt_Rs,
    pred_Rs,
    n_faces: int = 1,
    gravity: bool = False,
):
    best_mean_accuracy = float("-inf")
    R_align = torch.eye(3, device=gt_Rs.device, dtype=gt_Rs.dtype)
    Rset = valid_rotations_cube(gt_Rs.device, gt_Rs.dtype, gravity=gravity)
    rot_error_fn = (
        (lambda p, g: rot_err_deg_sym(p, g, n_faces))
        if n_faces != 1
        else rot_err_deg
    )
    invalid = torch.any(pred_Rs.isnan().flatten(1), dim=1)
    pred_Rs = pred_Rs[~invalid]
    gt_Rs = gt_Rs[~invalid]
    for R in Rset:
        rotated_pred_Rs = einsum(R, pred_Rs, "i j, b j k -> b i k")
        errors = [
            rot_error_fn(rT_gt, rT_est)
            for rT_gt, rT_est in zip(gt_Rs, rotated_pred_Rs)
        ]
        mean_accuracy = torch.stack(errors).lt(30).float().mean().item()
        if mean_accuracy > best_mean_accuracy:
            best_mean_accuracy = mean_accuracy
            R_align = R
    return R_align


def _procrustes(A, B, weights=None):
    N = A.shape[0]
    if weights is None:
        M = einsum(B, A, "n i j, n k j -> i k")
    else:
        w = weights.reshape(N, 1, 1)
        M = einsum(w * B, A, "n i j, n k j -> i k")

    U, _S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    if torch.det(R) < 0:
        U = U.clone()
        U[:, -1] *= -1
        R = U @ Vh
    return R


def align_rotation_sets(
    A: torch.Tensor,
    B: torch.Tensor,
    weights: torch.Tensor | None = None,
    n_faces: int = 1,
    gravity: bool = False,
):
    if A.shape != B.shape or A.shape[-2:] != (3, 3):
        raise ValueError(f"Expected matching rotation sets, got {A.shape}, {B.shape}")
    if n_faces == 1:
        return _procrustes(A, B, weights)

    Rset = valid_rotations_cube(A.device, A.dtype, gravity=gravity)
    rot_error_fn = (
        (lambda p, g: rot_err_deg_sym(p, g, n_faces))
        if n_faces != 1
        else rot_err_deg
    )
    invalid = torch.any(A.isnan().flatten(1), dim=1)
    A_valid = A[~invalid]
    B_valid = B[~invalid]
    best_R = torch.eye(3, device=A.device, dtype=A.dtype)
    best_acc = -1.0

    for S in Rset:
        A_sym = einsum(S, A_valid, "i j, n j k -> n i k")
        R_align = _procrustes(A_sym, B_valid, weights)
        aligned = einsum(R_align, A_sym, "i j, n j k -> n i k")
        errors = torch.stack(
            [rot_error_fn(b, a) for a, b in zip(B_valid, aligned)]
        )
        acc = errors.lt(30).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_R = R_align @ S
    return best_R


def load_symmetries_csv(path: str | None) -> dict[str, int]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, int] = {}
    for row in rows:
        category = row.get("category") or row.get("name") or row.get("class")
        value = (
            row.get("n_faces")
            or row.get("symmetry")
            or row.get("symmetries")
            or row.get("n_symmetries")
        )
        if category is not None and value not in (None, ""):
            out[str(category)] = int(value)
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def _dataset_cfg_for_name(cfg: DictConfig, name: str) -> DictConfig:
    if name in {"dataset", "default", "configured"}:
        return cfg.dataset
    path = _CONFIG_ROOT / "dataset" / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    return OmegaConf.load(path)


def _category_names(dataset, dataset_cfg: DictConfig) -> list[str]:
    configured = _as_list(
        dataset_cfg.get("categories", None) or dataset_cfg.get("cate", None)
    )
    if configured:
        return [str(category) for category in configured]
    dataset_categories = _as_list(getattr(dataset, "categories", None))
    if dataset_categories:
        return [str(category) for category in dataset_categories]
    return ["__all__"]


def _instantiate_dataset(
    dataset_cfg: DictConfig,
    split: str,
    category: str | None = None,
):
    if category is None or category == "__all__":
        return instantiate(dataset_cfg, split=split)
    field = "cate" if "cate" in dataset_cfg else "categories"
    category_cfg = OmegaConf.merge(dataset_cfg, {field: [category]})
    return instantiate(category_cfg, split=split)


def _empty_metrics():
    return {
        "median_ang_error": float("nan"),
        "median_ang_sym_error": float("nan"),
        "acc_ang_15": float("nan"),
        "acc_ang_sym_15": float("nan"),
        "acc_ang_30": float("nan"),
        "acc_ang_sym_30": float("nan"),
        "n_samples": 0,
        "avg_time_ms": 0.0,
    }


def _evaluate_category(
    *,
    cfg: DictConfig,
    model,
    renderer,
    dataset,
    category: str,
    n_faces: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, torch.Tensor, torch.Tensor]:
    loader_cfg = cfg.training
    eval_num_workers = cfg.evaluation.get("num_workers", None)
    if eval_num_workers is not None:
        loader_cfg = OmegaConf.merge(
            cfg.training, {"num_workers": int(eval_num_workers)}
        )

    loader = build_loader(
        dataset,
        loader_cfg,
        training=False,
        distributed=False,
    )
    all_pred_Rs = []
    all_gt_Rs = []
    timings: list[float] = []

    for batch in loader:
        batch = batch.to(device=device, non_blocking=True)
        with torch.no_grad():
            batch.annotations = renderer(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            inf_dict = model.infer(
                batch,
                batch.annotations,
                use_gt_mask=bool(cfg.evaluation.get("use_gt_mask", False)),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

        cameras = batch.cameras
        timings.extend([elapsed / len(cameras)] * len(cameras))
        feat_hw = inf_dict["feat_hw"]
        Tn2c_list = [
            estimate_T_n2c_for_frame(inf_dict, cameras, frame_idx, feat_hw)[0]
            for frame_idx in range(len(cameras))
        ]

        pred_Rs = cameras.R.clone()
        for frame_idx, Tn2c in enumerate(Tn2c_list):
            pred_Rs[frame_idx] = Tn2c[:3, :3].transpose(-1, -2)
        all_pred_Rs.append(pred_Rs.detach())
        all_gt_Rs.append(cameras.R.detach().clone())

    if not all_pred_Rs:
        empty = torch.empty(0)
        return _empty_metrics(), np.eye(3, dtype=np.float32), empty, empty

    gt_Rs = torch.cat(all_gt_Rs)
    pred_Rs = torch.cat(all_pred_Rs)
    representation = cfg.get("representation", {})
    gravity = bool(representation.get("gravity", False))
    mesh_type = str(representation.get("mesh_type", "cube"))
    if mesh_type == "sphere":
        R_align = align_rotation_sets(
            pred_Rs,
            gt_Rs,
            n_faces=n_faces,
            gravity=gravity,
        )
    else:
        R_align = conjugate_rotation_average(
            gt_Rs,
            pred_Rs,
            n_faces=n_faces,
            gravity=gravity,
        )

    aligned_pred_Rs = [R_align @ R for R in pred_Rs]
    angular_errors = [
        rot_err_deg(gt, pred)
        for gt, pred in zip(gt_Rs, aligned_pred_Rs)
    ]
    if n_faces == 1:
        symmetry_errors = angular_errors
    else:
        symmetry_errors = [
            rot_err_deg_sym(gt, pred, n_faces)
            for gt, pred in zip(gt_Rs, aligned_pred_Rs)
        ]

    angular_errors = torch.stack(angular_errors)
    symmetry_errors = torch.stack(symmetry_errors)
    angular_valid = angular_errors[~torch.isnan(angular_errors)]
    symmetry_valid = symmetry_errors[~torch.isnan(symmetry_errors)]
    avg_time_ms = 1000.0 * sum(timings) / len(timings) if timings else 0.0
    metrics = {
        "median_ang_error": (
            angular_valid.median().item() if len(angular_valid) else float("nan")
        ),
        "median_ang_sym_error": (
            symmetry_valid.median().item() if len(symmetry_valid) else float("nan")
        ),
        "acc_ang_15": (
            angular_valid.lt(15.0).float().mean().item()
            if len(angular_valid)
            else float("nan")
        ),
        "acc_ang_sym_15": (
            symmetry_valid.lt(15.0).float().mean().item()
            if len(symmetry_valid)
            else float("nan")
        ),
        "acc_ang_30": (
            angular_valid.lt(30.0).float().mean().item()
            if len(angular_valid)
            else float("nan")
        ),
        "acc_ang_sym_30": (
            symmetry_valid.lt(30.0).float().mean().item()
            if len(symmetry_valid)
            else float("nan")
        ),
        "n_samples": len(angular_errors),
        "avg_time_ms": avg_time_ms,
    }
    return (
        metrics,
        R_align.cpu().numpy(),
        angular_valid.cpu(),
        symmetry_valid.cpu(),
    )


def evaluate(
    cfg: DictConfig,
    model,
    renderer,
    device: torch.device,
    run_dir: Path,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    symmetries = load_symmetries_csv(
        cfg.evaluation.get("symmetries_csv", None)
    )
    dataset_names = _as_list(
        cfg.evaluation.get("datasets", None)
    ) or ["dataset"]
    split = str(cfg.evaluation.get("split", "val"))
    summary_rows: dict[str, dict[str, float]] = {}

    for dataset_name in dataset_names:
        print(f"\n{'=' * 60}")
        print(f"Evaluating dataset: {dataset_name}")
        print("=" * 60)

        dataset_cfg = _dataset_cfg_for_name(cfg, str(dataset_name))
        dataset_split = (
            split
            if str(dataset_name) in {"dataset", "default", "configured"}
            else str(dataset_cfg.get("split", split))
        )
        probe_dataset = None
        categories = _as_list(
            dataset_cfg.get("categories", None)
            or dataset_cfg.get("cate", None)
        )
        if not categories:
            probe_dataset = instantiate(dataset_cfg, split=dataset_split)
            categories = _category_names(probe_dataset, dataset_cfg)
        categories = [str(category) for category in categories] or ["__all__"]

        errors_per_category = {}
        alignments_per_category = {}
        all_angular_errors = None
        all_symmetry_errors = None

        for category in categories:
            dataset = (
                probe_dataset
                if probe_dataset is not None
                and (
                    hasattr(probe_dataset, "setup_samples")
                    or category == "__all__"
                )
                else _instantiate_dataset(
                    dataset_cfg,
                    dataset_split,
                    category,
                )
            )
            if hasattr(dataset, "setup_samples"):
                dataset.setup_samples([category])
            dataset_symmetries = getattr(dataset, "symmetries", {}) or {}
            n_faces = int(
                symmetries.get(category, dataset_symmetries.get(category, 1))
            )
            if n_faces == 0:
                symmetry_description = "continuous Y-axis"
            elif n_faces == 1:
                symmetry_description = "none"
            else:
                symmetry_description = f"{n_faces}-fold Y-axis"
            print(
                f"  Category: {category}, #sequences: {len(dataset)}, "
                f"symmetry: {symmetry_description}"
            )
            metrics, R_align, angular_valid, symmetry_valid = _evaluate_category(
                cfg=cfg,
                model=model,
                renderer=renderer,
                dataset=dataset,
                category=category,
                n_faces=n_faces,
                device=device,
            )
            errors_per_category[category] = metrics
            alignments_per_category[category] = R_align
            all_angular_errors = (
                torch.cat((all_angular_errors, angular_valid))
                if all_angular_errors is not None
                else angular_valid
            )
            all_symmetry_errors = (
                torch.cat((all_symmetry_errors, symmetry_valid))
                if all_symmetry_errors is not None
                else symmetry_valid
            )
            print(
                f"    Asymmetric:      median={metrics['median_ang_error']:.2f} deg  "
                f"Acc@15={metrics['acc_ang_15']:.3f}  "
                f"Acc@30={metrics['acc_ang_30']:.3f}"
            )
            print(
                "    Symmetry-aware:  "
                f"median={metrics['median_ang_sym_error']:.2f} deg  "
                f"Acc@15={metrics['acc_ang_sym_15']:.3f}  "
                f"Acc@30={metrics['acc_ang_sym_30']:.3f}"
            )
            print(
                f"    Avg inference time: {metrics['avg_time_ms']:.1f}ms/image"
            )

        alignment_path = run_dir / f"{dataset_name}_alignments.npz"
        np.savez(alignment_path, **alignments_per_category)
        print(f"  Saved alignment rotations to {alignment_path}")

        if all_angular_errors is None or len(all_angular_errors) == 0:
            global_row = _empty_metrics()
        else:
            global_row = {
                "median_ang_error": all_angular_errors.median().item(),
                "median_ang_sym_error": all_symmetry_errors.median().item(),
                "acc_ang_15": all_angular_errors.lt(15).float().mean().item(),
                "acc_ang_sym_15": (
                    all_symmetry_errors.lt(15).float().mean().item()
                ),
                "acc_ang_30": all_angular_errors.lt(30).float().mean().item(),
                "acc_ang_sym_30": (
                    all_symmetry_errors.lt(30).float().mean().item()
                ),
                "n_samples": len(all_angular_errors),
            }

        print("  Dataset-global metrics:")
        print(
            f"    Asymmetric:      median={global_row['median_ang_error']:.2f} deg  "
            f"Acc@15={global_row['acc_ang_15']:.3f}  "
            f"Acc@30={global_row['acc_ang_30']:.3f}"
        )
        print(
            "    Symmetry-aware:  "
            f"median={global_row['median_ang_sym_error']:.2f} deg  "
            f"Acc@15={global_row['acc_ang_sym_15']:.3f}  "
            f"Acc@30={global_row['acc_ang_sym_30']:.3f}"
        )

        category_df = pd.DataFrame.from_dict(
            errors_per_category,
            orient="index",
        )
        category_average = (
            category_df.drop(columns=["n_samples"], errors="ignore")
            .mean()
            .to_dict()
        )
        category_average["n_samples"] = category_df["n_samples"].sum()
        errors_per_category["__global__"] = global_row
        errors_per_category["__category_avg__"] = category_average

        output_df = pd.DataFrame.from_dict(
            errors_per_category,
            orient="index",
        )
        output_df.index.name = "category"
        csv_path = run_dir / f"{dataset_name}.csv"
        output_df.to_csv(csv_path)
        summary_rows[str(dataset_name)] = {
            key: category_average[key]
            for key in (
                "median_ang_error",
                "median_ang_sym_error",
                "acc_ang_15",
                "acc_ang_sym_15",
                "acc_ang_30",
                "acc_ang_sym_30",
                "n_samples",
            )
        }
        print(f"  Saved to {csv_path}")

    if summary_rows:
        summary_df = pd.DataFrame.from_dict(summary_rows, orient="index")
        summary_df.index.name = "dataset"
        summary_path = run_dir / "summary.csv"
        summary_df.to_csv(summary_path)
        print(f"\n{'=' * 60}")
        print(f"Summary saved to {summary_path}")
        print(summary_df.to_string())
