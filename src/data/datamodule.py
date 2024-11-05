

"""Multimodal datamodule."""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from copy import deepcopy


class Datamodule(LightningDataModule):
    """Multimodal datamodule class"""

    def __init__(self, cfg: dict, stage: str = "anchoring", fold: int = 0) -> None:
        """Initialization

        Parameters
        ----------
        cfg : dict
            cfg dict.
        """
        super().__init__()
        self.cfg = deepcopy(cfg)
        self.fold = fold
        self.stage = stage
        self.train_set, self.val_set, self.test_set = None, None, None

    def setup(self, stage=None):
        """Setup the datamodule."""
        if stage == "fit" or stage is None:
            pass
        #     self.train_set = instantiate(self.cfg["train_set"])
        #     hparams = self.train_set.hparams
        #     self.val_set = instantiate(self.cfg["val_set"], hparams=hparams)
        # if stage == "test" or stage is None:
        #     self.test_set = instantiate(self.cfg["test_set"])