"""Multimodal datamodule."""

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from hydra.utils import instantiate


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
        self.cfg = cfg
        self.fold = fold
        self.stage = stage
        self.feature_hparams = None
        self.train_set, self.val_set, self.test_set = None, None, None

    def setup(self, stage=None):
        """Setup the datamodule."""
        if stage == "fit" or stage is None:
            self.train_set = instantiate(self.cfg.data, stage="train")
            self.feature_hparams = self.train_set.hparams
            self.val_set = instantiate(
                self.cfg.data, stage="val", hparams=self.feature_hparams
            )
        # - noise & no noise test set
        if stage == "test":
            # use a new test set for classif.
            self.train_set = instantiate(
                self.cfg.data, stage="train", hparams=self.feature_hparams, loss=None
            )
            self.test_set = instantiate(
                self.cfg.data,
                stage="test",
                hparams=self.feature_hparams,
                num_points_noisy=0,
                loss=None,
            )
            self.noisy_test_set = instantiate(
                self.cfg.data, stage="test", hparams=self.feature_hparams, loss=None
            )

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
            # custom_collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        """Test dataloader."""
        clean = DataLoader(
            self.test_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
        )
        noisy = DataLoader(
            self.noisy_test_set,
            batch_size=self.cfg.data.batch_size,
            shuffle=False,
            num_workers=self.cfg.machine.num_workers,
            drop_last=False,
        )
        return [clean, noisy]
