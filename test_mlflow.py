import mlflow
from mlflow import log_metric, log_param, log_artifact, start_run

mlflow.set_tracking_uri("http://localhost:5000")  # Indica a MLflow a qué servidor de seguimiento (Tracking Server) debe conectarse para registrar y consultar información sobre los experimentos

if __name__ == '__main__':
    print("Iniciando ejecución...") 
    with start_run():
        log_param("threshold", 3)
    
        log_metric("timestamp", 1000)

        log_artifact("produced-dataset.csv")
