from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from emergent_canonical_frame.engine.evaluator import evaluate
from emergent_canonical_frame.engine.rendering import Renderer
from emergent_canonical_frame.utils import seed_everything


def _checkpoint_path(cfg: DictConfig) -> str:
    checkpoint = cfg.evaluation.get("checkpoint", None)
    if checkpoint is None and "checkpoint" in cfg:
        checkpoint = cfg.checkpoint
    if checkpoint is None:
        raise ValueError("Set evaluation.checkpoint to the checkpoint path")
    path = Path(to_absolute_path(str(checkpoint))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return str(path)


def _load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _cfg_from_checkpoint(
    base_cfg: DictConfig,
    checkpoint: dict,
) -> DictConfig:
    evaluation_cfg = OmegaConf.create(base_cfg.get("evaluation", {}))
    checkpoint_cfg = checkpoint.get("config", None)
    if checkpoint_cfg is None:
        cfg = OmegaConf.create(base_cfg)
    else:
        cfg = OmegaConf.create(checkpoint_cfg)
    return OmegaConf.merge(cfg, {"evaluation": evaluation_cfg})


def _model_state(checkpoint: dict) -> dict:
    state = checkpoint.get("model", None)
    if state is None:
        state = checkpoint.get("model_state_dict", None)
    if state is None:
        raise KeyError("Checkpoint has neither 'model' nor 'model_state_dict'")
    return state


def _save_eval_config(cfg: DictConfig, run_dir: Path) -> None:
    config_dir = run_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config=cfg,
        f=str(config_dir / "eval_config_resolved.yaml"),
        resolve=True,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="eval",
)
def main(base_cfg: DictConfig) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint_path = _checkpoint_path(base_cfg)
    checkpoint = _load_checkpoint(checkpoint_path)
    cfg = _cfg_from_checkpoint(base_cfg, checkpoint)
    cfg.evaluation.checkpoint = checkpoint_path

    seed = int(cfg.evaluation.get("seed", cfg.get("seed", 0)))
    seed_everything(seed, rank=0)
    cv2.setRNGSeed(seed)

    requested_device = str(cfg.evaluation.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    model = instantiate(cfg.model).to(device)
    model.load_state_dict(_model_state(checkpoint))
    model.backbone.lora_frozen = bool(checkpoint.get("lora_frozen", False))
    model.eval()

    feature_size = [
        int(size) // model.downsample_factor
        for size in cfg.training.image_size
    ]
    renderer = Renderer(
        model.V,
        model.F,
        feature_size,
        device=device,
    )

    checkpoint_stem = Path(checkpoint_path).stem
    output_root = Path(
        to_absolute_path(str(cfg.evaluation.get("output_root", "eval_logs")))
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{checkpoint_stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_eval_config(cfg, run_dir)
    print(f"Eval run directory: {run_dir}")

    evaluate(cfg, model, renderer, device, run_dir)

if __name__ == "__main__":
    main()