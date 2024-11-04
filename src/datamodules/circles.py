"""Dataset for synthetic circle data."""

from src.utils.create_circles import create_multiple_circles

from torch.utils.data import Dataset

class SyntheticCircle(Dataset):
    def __init__(self, N_points, N_noise, noisy=False, transform=None):
        self.N_points = N_points
        self.N_noise = N_noise
        self.noisy = noisy
        self.data, self.labels = create_multiple_circles(self.N_points, self.N_noise)
        self.transform = transform
        # add the featurization here. 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = (self.data[idx] 
                if self.transform is None 
                else self.transform(self.data[idx]))
        return data, self.labels[idx]