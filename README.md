# mlops-templates
A modular MLOps project template that integrates multiple tools for experiment tracking, model versioning, and pipeline orchestration.

## TODO
### MLFlow
- `mlflow.log_artifact()` does not work, probably `sftp` needs to be used
### WnB
- test `wnb` with API KEY
### DVC
- check is it possible to use `sftp` as `dvc remote` instead of `ssh`
### Clearml
- deploy and test
### Airflow
- add some dags & pipelines
### Other
- add Kubeflow example
- CI/CD? maybe some k8s example
- add inference/deployment
- add support of audio project additionally to `MNIST`
- test deeply `src/utils/callbacks.py`