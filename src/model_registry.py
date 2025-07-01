# src/model_registry.py

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from config import MLFLOW_CONFIG

def register_model(model, model_name: str, accuracy: float):
    """
    Loggea el modelo en MLflow, lo registra en Model Registry
    y lo promueve a 'Production'.
    """
    # Setup tracking
    mlflow.set_tracking_uri(MLFLOW_CONFIG["tracking_uri"])
    mlflow.set_experiment(MLFLOW_CONFIG["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Register in registry
        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, model_name)

        # Ponerlo en producción
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True
        )
