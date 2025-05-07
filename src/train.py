from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

from datamodule import MNISTDataModule
from models.classifier import LitClassifier

if __name__ == "__main__":
    model = LitClassifier()
    data = MNISTDataModule()

    logger = CSVLogger("./logs")  # temporary logger

    trainer = Trainer(max_epochs=5, logger=logger)
    trainer.fit(model, datamodule=data)
