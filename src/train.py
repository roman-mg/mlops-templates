import logging

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from datamodule import MNISTDataModule
from models.mlp import MLP


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))

    model = MLP(cfg.model.input_dim, cfg.model.hidden_dim, cfg.model.output_dim, cfg.trainer.lr)
    data = MNISTDataModule()

    logger = CSVLogger("./logs")  # temporary logger

    trainer = Trainer(max_epochs=cfg.trainer.max_epochs, logger=logger)
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    train()
