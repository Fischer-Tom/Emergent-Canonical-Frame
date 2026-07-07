from dataclasses import dataclass
from logging import critical
from pathlib import Path
from termios import TIOCPKT_DOSTOP

import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path

from emergent_canonical_frame.data.loader import get_dataloader
from emergent_canonical_frame.engine.rendering import (
    Renderer,
)
from emergent_canonical_frame.utils import seed_everything
from emergent_canonical_frame.utils.alignment import align_sfm_poses
from emergent_canonical_frame.utils.ddp_utils import (
    get_rank,
    get_world_size,
    init_ddp,
    is_dist_avail_and_initialized,
)


@dataclass
class TrainerState:
    iteration: int = 0
    epoch: int = 0
    lora_frozen: bool = False


def train(cfg):
    state = TrainerState()

    device = init_ddp()
    distributed = is_dist_avail_and_initialized()
    world_size = get_world_size()
    rank = get_rank()
    seed_everything(cfg.seed, rank)
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    dataset = instantiate(cfg.dataset, split="train")
    dataloader = get_dataloader(
        dataset,
        cfg.batch_size,
        train=True,
        world_size=world_size,
        distributed=distributed,
        rank=rank,
    )

    model = instantiate(cfg.model)
    criterion = instantiate(cfg.criterion)
    optimizer = instantiate(cfg.optim, params=model.parameters())
    renderer = Renderer(model.V, model.F, cfg.image_size)

    for epoch in range(state.epoch, cfg.epochs):
        dataloader.set_epoch(epoch)

        for batch in dataloader:
            state.iteration += 1

            with torch.autocast(
                device_type="cuda", enabled=cfg.precision, dtype=torch.bfloat16
            ):
                out_dict = model(batch)

            corr_dict = renderer(batch)

            alignment_R = align_sfm_poses(corr_dict, out_dict)
            # Align SfM Poses with Model Output
            aligned_corr_dict = renderer.rerender_aligned(batch, alignment_R)
            # Compute Loss with Aligned Poses


            loss_dict = criterion(aligned_corr_dict, out_dict)
