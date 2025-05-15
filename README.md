# mlops-templates
A modular MLOps project template that integrates multiple tools for experiment tracking, model versioning, and pipeline orchestration.

## TODO
### MLFlow
- fix docker-compose file: issue with container paths
- `mlflow.log_artifact()` does not work, probably `sftp` needs to be used
### WnB
- test `wnb` with API KEY
### DVC
- use `sftp` as `dvc remote`
### Other
- add clearml examples
- add airflow examples
- CI/CD? maybe some k8s example
- add different model deployments
- add support of audio project additionally to `MNIST`
- test `src/utils/callbacks.py`