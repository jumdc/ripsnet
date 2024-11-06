"""Criterion module for the model."""

from torch import nn


class MultiviewMSE(nn.Module):
    def __init__(self):
        super(MultiviewMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, feature_hat, feature):
        return (
            self.mse(feature_hat[0], feature) / 2
            + self.mse(feature_hat[1], feature) / 2
        )
