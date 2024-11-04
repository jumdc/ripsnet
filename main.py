from src.datamodules.circles import SyntheticCircle
from src.model.ripsnet import DenseNestedTensors, PermopNestedTensors

import src.datamodules.circles

import logging
import gudhi as gd
import numpy as np
from tqdm import tqdm
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import pairwise_distances
from gudhi.representations import DiagramSelector
from gudhi.representations import Landscape, PersistenceImage




def run():
    # - create the dataset 
    # train_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)
    # val_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)
    # test_dataset = SyntheticCircle(N_points=100, N_noise=0, noisy=False)
    pass

    # - create the model





if __name__ == "__main__":
    run()