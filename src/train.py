import logging

import hydra
from clearml import Task
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    load_dotenv()
    logging.info(OmegaConf.to_yaml(cfg))

    if cfg.logger.get("_target_"):
        logger = instantiate(cfg.logger)
    else:
        # TODO: now it's only ClearML, in future it can be extended
        _ = Task.init(**cfg.logger)
        logger = None

    seed_everything(cfg.seed)
    trainer = Trainer(
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        logger=logger,
        callbacks=[instantiate(cfg.checkpoint)],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
    )
    trainer.fit(instantiate(cfg.model), datamodule=instantiate(cfg.datamodule))


if __name__ == "__main__":
    train()
