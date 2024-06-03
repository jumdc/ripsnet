import pyrootutils
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True)




## Define the dataloader
class PointCloudDatamodule(LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        