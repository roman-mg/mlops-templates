import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn


class MLP(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float, lr_decay: float) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore[attr-defined]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.get("lr_decay", 1.0))

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
