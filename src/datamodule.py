import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
        self.train = None
        self.val = None

    def setup(self, stage: str | None = None) -> None:
        self.train = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
        self.val = MNIST(self.data_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size)
