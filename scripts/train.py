import hydra
from omegaconf import DictConfig
import torch


from emergent_canonical_frame.engine.trainer import train


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    train(cfg)


if __name__ == "__main__":
    main()