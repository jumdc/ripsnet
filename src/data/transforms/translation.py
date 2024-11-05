import torch


class Translation:
    """Isometric translation of the point cloud.
    Moves every point of a point cloud by the same distance in a given direction
    """

    def __init__(self, translation_range=(-0.1, 0.1)):
        self.translation_range = translation_range

    def __call__(self, point_cloud):
        dim = point_cloud.size(-1)
        translation = torch.FloatTensor(dim).uniform_(*self.translation_range)
        return point_cloud + translation
