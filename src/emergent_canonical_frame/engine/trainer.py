from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path

from emergent_canonical_frame.utils import seed_everything
from emergent_canonical_frame.utils.ddp_utils import (
    init_ddp,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
)

def train(cfg):

    device = init_ddp()
    distributed = is_dist_avail_and_initialized()
    world_size = get_world_size()
    rank = get_rank()

    seed_everything(cfg.seed, rank)
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    dataset = instantiate(cfg.dataset, split="train")


    # Step 2: Initialize the model
    # 
    # Step 3: Initialize the optimizer
    # 
    # Step 4: Initialize the dataloaders
    # 
    # Step 5: Train the model

    pass