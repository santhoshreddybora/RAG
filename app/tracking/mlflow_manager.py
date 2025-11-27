import mlflow
from app.logger import logging
import os

class MLflowManager:
    def __init__(self):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment("Clinical-RAG")

    def start_run(self, run_name):
        mlflow.start_run(run_name=run_name)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_text(self, text, name="artifact.txt"):
        with open(name, "w") as f:
            f.write(text)
        mlflow.log_artifact(name)

    def end_run(self):
        mlflow.end_run()
