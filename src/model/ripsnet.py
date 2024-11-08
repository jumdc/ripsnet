"""Model class for RipsNet.

Use nested tensors to handle point clouds with varying number of points.
"""

# from src.model.components import DenseNestedTensors, PermopNestedTensors,
from src.model.components import Permop
from src.utils.plot import plot_reconstruction

from typing import Any

import wandb
import pytorch_lightning as pl
from torch import nn
from hydra.utils import instantiate


class RipsNet(pl.LightningModule):
    """Model class for RipsNet."""

    def __init__(self, cfg: Any, *args, **kwargs) -> None:
        """Initialization of the RipsNet model."""
        super().__init__()
        self.cfg = cfg
        self.model = nn.Sequential(
            nn.Linear(2, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            Permop(),
            # DenseNestedTensors(30, last_dim=2, use_bias=True),
            # DenseNestedTensors(20, last_dim=30, use_bias=True),
            # DenseNestedTensors(10, last_dim=20, use_bias=True),
            # PermopNestedTensors(),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, cfg.model.output_dim),
            nn.Sigmoid(),
        )
        # self.loss = nn.MSELoss()
        self.loss = instantiate(cfg.loss)
        self.mutiview = False if isinstance(self.loss, nn.MSELoss) else True

        self.torch_output = []

    def forward(self, x):
        """Forward pass for the RipsNet."""
        return self.model(x)

    def multiview_forward(self, x):
        """Forward pass for the RipsNet."""
        return [self.model(x[0]), self.model(x[1])]

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, feature, _ = batch

        feature_hat = self.model(X) if not self.mutiview else self.multiview_forward(X)
        loss = self.loss(feature_hat, feature)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, feature, _ = batch
        # feature_hat = self.model(X)
        feature_hat = self.model(X) if not self.mutiview else self.multiview_forward(X)
        loss = self.loss(feature_hat, feature)
        self.log("val_loss", loss)
        if batch_idx == 0 and self.current_epoch == self.cfg.trainer.max_epochs - 1:
            fig = plot_reconstruction(feature, feature_hat)
            if isinstance(self.logger, pl.loggers.WandbLogger):
                self.logger.experiment.log({"test/reconstruction": wandb.Image(fig)})
            elif self.cfg.paths.name == "didion":
                fig.savefig("checks/reconstruction.png")
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        X, feature, _ = batch
        feature_hat = self.model(X)
        loss = self.loss(feature_hat, feature)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = instantiate(self.cfg.optimizer, self.parameters())
        return optimizer
