import mlflow
from mlflow import log_metric, log_param, log_artifact, start_run, set_tags

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mi_primer_experimento")

if __name__ == '__main__':
    print("Iniciando ejecución...")  
    with start_run():
        log_param("threshold", 3)
    
        log_metric("timestamp", 1000)

        log_artifact("produced-dataset.csv")

        set_tags({
            "author": "Jhony",
            "stage": "development",
            "version": "v1.0"
            })