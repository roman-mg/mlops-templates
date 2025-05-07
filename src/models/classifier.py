import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.optimizer import Optimizer


class LitClassifier(pl.LightningModule):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 128, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.size(0), -1))

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore[attr-defined]
