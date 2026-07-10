from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path

from emergent_canonical_frame.data.loader import build_loader
from emergent_canonical_frame.engine.criterion import Criterion
from emergent_canonical_frame.engine.rendering import (
    Renderer,
)
from emergent_canonical_frame.utils import seed_everything
from emergent_canonical_frame.utils.alignment import CanonicalAligner
from emergent_canonical_frame.utils.ddp_utils import (
    get_rank,
    get_world_size,
    init_ddp,
    is_dist_avail_and_initialized,
)
from emergent_canonical_frame.utils.logging import Logger
from emergent_canonical_frame.utils.optim import Optimizer


@dataclass
class TrainerState:
    iteration: int = 0
    epoch: int = 0
    lora_frozen: bool = False


def unwrap_model(model):
    model = getattr(model, "module", model)  # DistributedDataParallel
    model = getattr(model, "_orig_mod", model)  # torch.compile
    return model


def save_checkpoint(path: Path, model, optimizer: Optimizer, state: TrainerState):
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": state.iteration,
            "epoch": state.epoch,
            "lora_frozen": state.lora_frozen,
        },
        path,
    )


def train(cfg):
    state = TrainerState()

    device = init_ddp()
    distributed = is_dist_avail_and_initialized()
    world_size = get_world_size()
    rank = get_rank()
    seed_everything(cfg.seed, rank)
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log_dir = run_dir / cfg.logging.log
    checkpoint_dir = run_dir / cfg.logging.checkpoints
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(rank=rank, log_file=log_dir / "train.log" if rank == 0 else None)

    dataset = instantiate(cfg.dataset, split="train")
    dataloader = build_loader(
        dataset,
        cfg.training,
        training=True,
        world_size=world_size,
        distributed=distributed,
        rank=rank,
    )

    model = instantiate(cfg.model).to(device)
    if cfg.training.use_torch_compile:
        model = torch.compile(model)
    criterion = Criterion()
    total_iters = cfg.training.epochs * len(dataloader)
    optimizer = Optimizer(model, total_iters, cfg.optim)
    aligner = CanonicalAligner(total_iters=total_iters).to(device)

    # Render at feature resolution (backbone downsamples by model.downsample_factor)
    # so corr_dict lines up pixel-for-pixel with logits/mask_logits, no resampling needed.
    feature_size = [int(s) // model.downsample_factor for s in cfg.training.image_size]
    V, F_mesh = model.V, model.F
    renderer = Renderer(V, F_mesh, feature_size, device=device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index
        )

    for epoch in range(state.epoch, cfg.training.epochs):
        if distributed:
            dataloader.sampler.set_epoch(epoch)
        if hasattr(dataloader.dataset, "set_iteration"):
            dataloader.dataset.set_iteration(state.iteration)

        for batch in dataloader:
            state.iteration += 1
            if hasattr(dataloader.dataset, "set_iteration"):
                dataloader.dataset.set_iteration(state.iteration)
            aligner.c_iter = state.iteration
            batch = batch.to(device)
            batch_image = batch.image_rgb
            with torch.autocast(
                device_type="cuda",
                enabled=cfg.training.mixed_precision,
                dtype=torch.bfloat16,
            ):
                logits, mask_logits = model(batch_image)
            corr_dict = renderer(batch)

            B, M = logits.shape[:2]
            Hf, Wf = feature_size
            logits = logits.view(B, M, Hf, Wf)

            valid_region = (
                F.interpolate(batch.valid_masks, size=(Hf, Wf), mode="nearest")
                .squeeze(1)
                .bool()
            )
            alignment = aligner(
                V=V,
                logits=logits,
                v_grid=corr_dict.obj_xyz,
                valid_mask=corr_dict.proxy_mask & valid_region,
                obj_id=corr_dict.obj_idx,
            )
            frame_indices = torch.arange(len(batch.cameras), device=device)
            rerendered = renderer.rerender_aligned(
                batch,
                alignment.R_frame.float(),
                obj_id_per_frame=frame_indices,
                base_cameras=corr_dict.cameras,
            )
            alignment_for_loss = replace(
                alignment,
                v=rerendered.obj_xyz,
                Rv=rerendered.obj_xyz,
                valid_mask=rerendered.proxy_mask & valid_region,
            )
            with torch.autocast(
                device_type="cuda",
                enabled=cfg.training.mixed_precision,
                dtype=torch.bfloat16,
            ):
                loss_dict = criterion(
                    V,
                    alignment_for_loss,
                    mask_logits,
                    rerendered.proxy_mask,
                    valid_region=valid_region,
                )
            optimizer.zero_grad()
            loss_dict["loss_total"].backward()
            optimizer.step()

            if state.iteration % cfg.logging.log_interval == 0:
                logger.log_dict(loss_dict, step=state.iteration)

            if rank == 0 and state.iteration % cfg.training.checkpoint_interval == 0:
                name = (
                    f"iter_{state.iteration:07d}.pt"
                    if cfg.training.keep_interval_checkpoints
                    else "latest.pt"
                )
                save_checkpoint(checkpoint_dir / name, model, optimizer, state)
