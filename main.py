"""Main script to run the experiments."""


# from src.model.ripsnet import DenseNestedTensors, PermopNestedTensors

# import logging

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def training(cfg: DictConfig) -> None:

    # - set the seed
    pl.seed_everything(cfg.seed)

    # - create the dataset
    # datamodule =
    # train_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)
    # val_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)
    # test_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)

    # - create the model


if __name__ == "__main__":
    training()
