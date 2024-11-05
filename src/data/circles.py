"""Dataset for synthetic circle data."""

from src.data.utils.create_circles import create_multiple_circles
from src.data.utils.tda_features import alpha_pi

import torch
from torch.utils.data import Dataset


class SyntheticCircle(Dataset):
    """Dataset for synthetic circle data."""

    def __init__(
        self,
        size_train,
        size_test,
        size_val,
        num_points,
        num_points_noisy,
        transform=None,
        stage="train",
        hparams=None,
        *args,
        **kwargs
    ):
        super(SyntheticCircle, self).__init__()
        size = (
            size_train
            if stage == "train"
            else (size_test if stage == "test" else size_val)
        )

        self.data, self.labels = create_multiple_circles(
            size=size,
            num_points=num_points,
            noisy=True if num_points_noisy > 0 else False,
            num_points_noise=num_points_noisy,
        )

        # - TDA features
        self.featurization, self.hparams = alpha_pi(self.data, hparams=hparams)
        # - do the transform is necessary.
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # pc = torch.nested.nested_tensor(self.data[idx], dtype=torch.float32)
        pc = torch.tensor(self.data[idx], dtype=torch.float32)
        feature = torch.tensor(self.featurization[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return pc, feature, label
