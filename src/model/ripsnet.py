from src.model.components import DenseNestedTensors, PermopNestedTensors

from typing import Any
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
            DenseNestedTensors(30, last_dim=2, use_bias=True),
            DenseNestedTensors(20, last_dim=30, use_bias=True),
            DenseNestedTensors(10, last_dim=20, use_bias=True),
            PermopNestedTensors(),
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, cfg.model.output_dim),
            nn.Sigmoid(),
        )
        self.loss = nn.MSELoss()
        self.torch_output = []

    def forward(self, x):
        """Forward pass for the RipsNet."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step."""
        X, feature, _ = batch
        feature_hat = self.model(X)
        loss = self.loss(feature_hat, feature)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        X, feature, _ = batch
        feature_hat = self.model(X)
        loss = self.loss(feature_hat, feature)
        self.log("val_loss", loss)
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
