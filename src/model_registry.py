def register_model(model, model_name, n_estimators, accuracy):
    import mlflow

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Mi primer Modelo")

    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")