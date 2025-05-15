import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from utils.callbacks import ArtifactLoggerCallback


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        logger=instantiate(cfg.logger),
        callbacks=[instantiate(cfg.checkpoint), ArtifactLoggerCallback()],
    )
    trainer.fit(instantiate(cfg.model), datamodule=instantiate(cfg.datamodule))


if __name__ == "__main__":
    train()
