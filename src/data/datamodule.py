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
            self.train_set = instantiate(self.cfg.data, stage="train")
            self.hparams = self.train_set.hparams
            self.val_set = instantiate(
                self.cfg.data, stage="train", hparams=self.hparams
            )

        if stage == "test" or stage is None:
            self.test_set = instantiate(
                self.cfg.data, stage="test", hparams=self.hparams
            )

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
