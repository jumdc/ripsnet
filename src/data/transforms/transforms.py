"""Custom transformations for point clouds."""


from torchvision import transforms

from src.data.transforms.rotation import Rotation2d
from src.data.transforms.translation import Translation


class MultiViewIsometricTransform:
    """Apply a list of isometric transformations."""

    def __init__(self, rotation_range, translation_range):
        self.transform_1 = transforms.Compose(
            [Rotation2d(rotation_range), Translation(translation_range)]
        )
        self.transform_2 = transforms.Compose(
            [Rotation2d(rotation_range), Translation(translation_range)]
        )

    def __call__(self, point_cloud):
        return self.transform_1(point_cloud), self.transform_2(point_cloud)
