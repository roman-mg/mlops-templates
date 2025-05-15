from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


with DAG('mnist_pipeline') as dag:
    dvc_pull = BashOperator(
        task_id='pull_data',
        bash_command='dvc pull'
    )

    train = BashOperator(
        task_id='train_model',
        bash_command='python train.py'
    )

    dvc_pull >> train
