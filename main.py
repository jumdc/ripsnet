"""Main script to run the experiments."""

from src.data.circles import SyntheticCircle
from src.model.ripsnet import DenseNestedTensors, PermopNestedTensors

import logging

import hydra 
import gudhi as gd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from gudhi.representations import DiagramSelector
from gudhi.representations import Landscape, PersistenceImage



@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def training(cfg: OmegaConf) -> None:   

    # - set the seed
    pl.seed_everything(cfg.seed)
    
    # - create the dataset 
    # datamodule = 
    train_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)
    val_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)
    test_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)

    # - create the model
    
    
    


if __name__ == "__main__":
    training()