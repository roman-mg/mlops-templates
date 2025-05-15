import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, save_dir: str, batch_size: int, num_workers: int) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def setup(self, stage: str | None = None) -> None:
        self.train = MNIST(self.save_dir, train=True, download=True, transform=self.transform)
        self.val = MNIST(self.save_dir, train=False, download=True, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
