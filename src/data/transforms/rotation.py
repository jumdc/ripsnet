import torch
import math
import numpy as np


class Rotation2d:
    def __init__(self, angle_range=(-90, 90)):
        self.angle_range = angle_range

    def __call__(self, point_cloud):
        # - choose the angle
        angle = torch.FloatTensor(1).uniform_(*self.angle_range)
        # - create the rotation matrix
        rotation = torch.tensor(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        return point_cloud @ rotation.T


class Rotation3d:
    def __init__(self, angle_range=(-90, 90)):
        self.angle_range = angle_range

    def __call__(self, point_cloud):
        # - choose the angle
        angle = torch.FloatTensor(1).uniform_(*self.angle_range)
        # - choose the axis
        axis = np.random.randint(0, 3)
        # - create the rotation matrix
        if axis == 0:
            rotation = torch.tensor(
                [
                    [1, 0, 0],
                    [0, math.cos(angle), -math.sin(angle)],
                    [0, math.sin(angle), math.cos(angle)],
                ]
            )
        elif axis == 1:
            rotation = torch.tensor(
                [
                    [math.cos(angle), 0, math.sin(angle)],
                    [0, 1, 0],
                    [-math.sin(angle), 0, math.cos(angle)],
                ]
            )
        else:
            rotation = torch.tensor(
                [
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle), math.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        return point_cloud @ rotation.T
