"""Torch implementation for RipsNet.

Adapted from the original
implementation in TensorFlow: https://github.com/hensel-f/ripsnet


Use nested_tensors to handle point clouds with varying number of points.
"""


import torch
import torch.nn as nn


class DenseNestedTensors(nn.Module):
    def __init__(self, units, last_dim, use_bias=True, activation="ReLU"):
        super(DenseNestedTensors, self).__init__()
        self.activation = getattr(nn, activation)()
        self.layer = nn.Linear(last_dim, units)

    def forward(self, inputs):
        """Forward pass for the dense layer."""
        outputs = self.layer(inputs)
        outputs = self.activation(outputs)
        return outputs


class PermopNestedTensors(nn.Module):
    def __init__(self):
        super(PermopNestedTensors, self).__init__()

    def forward(self, inputs):
        """Forward pass for the permutation operator."""
        # -- we pad with 0 - to convert nested tensors to tensors
        out = torch.nested.to_padded_tensor(inputs, padding=0)
        # -- we pad with 0  - identity operator for sum
        out = torch.sum(out, dim=1, keepdim=False)
        return out


class Permop(nn.Module):
    def __init__(self):
        super(Permop, self).__init__()

    def forward(self, inputs):
        out = torch.sum(inputs, dim=1, keepdim=False)
        return out
