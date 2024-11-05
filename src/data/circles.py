"""Dataset for synthetic circle data."""

from src.data.utils.create_circles import create_multiple_circles
from src.data.utils.tda_features import alpha_pi

from torch.utils.data import Dataset


class SyntheticCircle(Dataset):
    """Dataset for synthetic circle data."""

    def __init__(
        self,
        num_points_train,
        N_noise,
        noisy=False,
        transform=None,
        stage="train",
        hparams=None,
    ):
        self.N_points = num_points_train
        self.N_noise = N_noise
        self.noisy = noisy
        self.data, self.labels = create_multiple_circles(self.N_points, self.N_noise)

        # here add the transforms is need be
        self.featurization, self.hparams = alpha_pi(self.data, hparams=hparams)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # nested tensor for data !!
        return self.data[idx], self.featurization[idx], self.labels[idx]
