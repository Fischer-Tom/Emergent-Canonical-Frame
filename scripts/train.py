import hydra
from omegaconf import DictConfig

from emergent_canonical_frame.engine.trainer import train


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()