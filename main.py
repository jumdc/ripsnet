"""Main script to run the experiments.

TODO:
[ ] - callbacks
[ ] - augmentations
[ ] - finish model
"""


# from src.model.ripsnet import DenseNestedTensors, PermopNestedTensors
from src.data.datamodule import Datamodule


from datetime import datetime

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl


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
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            # callbacks=callbacks,
        )
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    training()
