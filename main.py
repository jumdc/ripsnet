"""Main script to run the experiments.

TODO:
[ ] - callbacks
[ ] - augmentations
[x] - finish model
"""


# from src.model.ripsnet import DenseNestedTensors, PermopNestedTensors
from src.data.datamodule import Datamodule


from datetime import datetime

import hydra
import torch
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def training(cfg: DictConfig) -> None:

    # - set the seed
    pl.seed_everything(cfg.seed)
    for cv in range(cfg.cv):
        # - create the dataset
        datamodule = Datamodule(cfg)

        # - create the model
        model = hydra.utils.instantiate(
            cfg.model, cfg, convert="partial", _recursive_=False
        )

        # - create the trainer
        logger = None
        if cfg.log:
            name = f"{cfg.logger.name}_{datetime.now().strftime('%Y-%m-%d_%Hh%M')}"
            name_cv = f"{name}_cv{cv}"
            logger = hydra.utils.instantiate(cfg.logger, name=name_cv, group=name)
        trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
        trainer.fit(model, datamodule)

        # - test the model.
        # -- add a function here for that ??
        classification = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
        ripsnet = model.model
        ripsnet.eval()
        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []

        for batch in datamodule.train_dataloader():
            X, _, label = batch
            with torch.no_grad():
                train_classif = ripsnet(X)
            X_train.append(train_classif.detach().numpy())
            y_train.append(label.detach().numpy())

        for batch in datamodule.val_dataloader():
            X, _, label = batch
            with torch.no_grad():
                val_classif = ripsnet(X)
            X_val.append(val_classif.detach().numpy())
            y_val.append(label.detach().numpy())

        for batch in datamodule.test_dataloader():
            X, _, label = batch
            with torch.no_grad():
                test_classif = ripsnet(X)
            X_test.append(test_classif.detach().numpy())
            y_test.append(label.detach().numpy())

        X_train = np.concatenate(X_train, axis=0)
        X_val = np.concatenate(X_val, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        print(y_train)
        le = LabelEncoder().fit(y_train)
        y_clean_train = le.transform(y_train)
        y_clean_test = le.transform(y_test)

        classification.fit(X_train, y_clean_train)
        score_test = classification.score(X_test, y_clean_test)
        print(score_test)


if __name__ == "__main__":
    training()
