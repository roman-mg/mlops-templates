import os

import mlflow
import pandas as pd
import soundfile as sf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger


class ArtifactLoggerCallback(Callback):
    def __init__(self, extra_artifacts: list[str] = None):
        super().__init__()
        self.extra_artifacts = extra_artifacts or []
        for path in self.extra_artifacts:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        df = pd.DataFrame([trainer.callback_metrics])

        for path in self.extra_artifacts:
            match os.path.splitext(path)[1].lower():
                case ".csv":
                    df.to_csv(path, index=False)
                case ".json":
                    df.to_json(path, orient="records", lines=True)
                case ".txt":
                    with open(path, "w") as f:
                        f.write(df.to_string(index=False))
                case ".wav" | ".flac" | ".ogg":
                    audio_data, sample_rate = pl_module.generate_audio()
                    sf.write(path, audio_data, sample_rate)
                case _:
                    pass

        if isinstance(trainer.logger, TensorBoardLogger):
            for k, v in trainer.callback_metrics.items():
                if isinstance(v, (int, float)):
                    trainer.logger.experiment.add_scalar(k, v, global_step=trainer.global_step)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict) -> None:
        checkpoint_cb = next((cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
        if not checkpoint_cb:
            return

        checkpoint_path = checkpoint_cb.best_model_path if checkpoint_cb else None
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        if isinstance(trainer.logger, MLFlowLogger):
            if mlflow.active_run():
                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
                for file_path in self.extra_artifacts:
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, artifact_path="artifacts")
        elif isinstance(trainer.logger, WandbLogger):
            run = trainer.logger.experiment
            run.save(checkpoint_path, base_path=os.getcwd())
            for file_path in self.extra_artifacts:
                if os.path.exists(file_path):
                    run.save(file_path, base_path=os.getcwd())
        elif isinstance(trainer.logger, TensorBoardLogger):
            pass
        else:
            pass
