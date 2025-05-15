import os

import mlflow
import pandas as pd
import soundfile as sf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger, WandbLogger


class RemoteModelCheckpoint(ModelCheckpoint):
    def __init__(self, extra_artifacts: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.extra_artifacts = extra_artifacts or []
        for path in self.extra_artifacts:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)

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
                    # TODO: extend to audio-project example
                    audio_data, sample_rate = pl_module.generate_audio()  # type: ignore[operator]
                    sf.write(path, audio_data, sample_rate)
                case _:
                    pass

        if isinstance(trainer.logger, TensorBoardLogger):
            for k, v in trainer.callback_metrics.items():
                if isinstance(v, int | float):
                    trainer.logger.experiment.add_scalar(k, v, global_step=trainer.global_step)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)

        path = self.last_model_path or self.best_model_path
        if not path or not os.path.exists(path):
            return

        if isinstance(trainer.logger, MLFlowLogger):
            mlflow.set_tracking_uri(trainer.logger._tracking_uri)
            with mlflow.start_run(run_id=trainer.logger.run_id):
                for file_path in self.extra_artifacts:
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, artifact_path="artifacts", run_id=trainer.logger.run_id)
        elif isinstance(trainer.logger, WandbLogger):
            run = trainer.logger.experiment
            for file_path in self.extra_artifacts:
                if os.path.exists(file_path):
                    run.save(file_path, base_path=os.getcwd())
        elif isinstance(trainer.logger, TensorBoardLogger):
            pass
        else:
            pass
