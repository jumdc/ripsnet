"""Dataset for synthetic circle data."""

from src.data.utils.create_circles import create_multiple_circles
from src.data.utils.tda_features import alpha_pi

from torch.utils.data import Dataset

class SyntheticCircle(Dataset):
    """Dataset for synthetic circle data."""

    def __init__(self, N_points, N_noise, noisy=False, transform=None, stage='train'):
        self.N_points = N_points
        self.N_noise = N_noise
        self.noisy = noisy
        data, self.labels = create_multiple_circles(self.N_points, self.N_noise)
        featurazation, self.hparams = alpha_pi(data, stage=stage)
        self.transform = transform
        # add the featurization here. 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = (self.data[idx] 
                if self.transform is None 
                else self.transform(self.data[idx]))
        return data, self.labels[idx]