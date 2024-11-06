"""Main script to run the experiments.

TODO:
[ ] - callbacks
[ ] - augmentations
[x] - finish model
"""

from src.data.datamodule import Datamodule
from src.utils.helpers import SizeDatamodule, SanityCheckInput
from src.utils.plot import plot_cm

from datetime import datetime

import hydra
import torch
import wandb
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from xgboost import XGBClassifier
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.preprocessing import LabelEncoder


@hydra.main(version_base="1.2", config_path="configs", config_name="config.yaml")
def training(cfg: DictConfig) -> None:
    for cv in range(cfg.cv):
        callbacks = []
        # ----- RipsNet ----- #
        # - create the dataset
        pl.seed_everything(cfg.seed + cv)
        datamodule = Datamodule(cfg)
        callbacks = [
            EarlyStopping(
                monitor=cfg.monitor, patience=cfg.patience, min_delta=cfg.min_delta
            ),
            SizeDatamodule(),
            SanityCheckInput(),
        ]

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
        trainer = hydra.utils.instantiate(
            cfg.trainer, logger=logger, callbacks=callbacks
        )

        # ---- fit the model ---- #
        trainer.fit(model, datamodule)

        # ---- test the model. ---- #
        datamodule.setup(stage="test")  # check this -> need to have new datamodule
        classification = XGBClassifier(eval_metric="logloss", use_label_encoder=False)
        ripsnet = model.model
        ripsnet.eval()
        X_train, X_test, X_test_noise, y_train, y_test, y_test_noise = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        # - train set
        for batch in datamodule.train_dataloader():
            X, _, label = batch
            with torch.no_grad():
                train_classif = ripsnet(X)
            X_train.append(train_classif.detach().numpy())
            y_train.append(label.detach().numpy())

        # ----- test sets ----- #
        # -- no noise
        clean_datamodule, noisy_datamodule = datamodule.test_dataloader()
        for batch in clean_datamodule:
            X, _, label = batch
            with torch.no_grad():
                test_classif = ripsnet(X)
            X_test.append(test_classif.detach().numpy())
            y_test.append(label.detach().numpy())
        # -- w/ noise
        for batch in noisy_datamodule:
            X, _, label = batch
            with torch.no_grad():
                test_classif = ripsnet(X)
            X_test_noise.append(test_classif.detach().numpy())
            y_test_noise.append(label.detach().numpy())

        X_train = np.concatenate(X_train, axis=0)
        X_test = np.concatenate(X_test, axis=0)
        X_test_noise = np.concatenate(X_test_noise, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        y_test_noise = np.concatenate(y_test_noise, axis=0)

        le = LabelEncoder().fit(y_train)
        y_clean_train = le.transform(y_train)
        y_clean_test = le.transform(y_test)
        y_clean_test_noise = le.transform(y_test_noise)

        # ---- fit the test model ---- #
        classification.fit(X_train, y_clean_train)
        y_pred = classification.predict(X_test)
        y_pred_noise = classification.predict(X_test_noise)

        # --score
        score_test = classification.score(X_test, y_clean_test)
        score_test_noise = classification.score(X_test_noise, y_clean_test_noise)
        print(f"Test score: {score_test}, Test score noise: {score_test_noise}")
        # --confusion matrix
        cm = plot_cm(y_clean_test, y_pred, dataset=cfg.data._target_)
        cm_noise = plot_cm(
            y_clean_test_noise, y_pred_noise, dataset=cfg.data._target_, name="Noisy"
        )
        if cfg.log:
            model.logger.experiment.log(
                {
                    "test/test_no_noise": score_test,
                    "test/test_noise": score_test_noise,
                    "test/cm": wandb.Image(cm),
                    "test/cm_noise": wandb.Image(cm_noise),
                }
            )
        elif cfg.paths.name == "didion":
            cm.savefig("checks/cm.png")
            cm_noise.savefig("checks/cm_noise.png")


if __name__ == "__main__":
    training()
