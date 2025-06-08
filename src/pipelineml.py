# filepath: /ml-pipeline-project/ml-pipeline-project/src/pipelineml.py
import pandas as pd
from data_loader import load_client_data
from data_loader import load_reqs_data
from data_preparation import clean_and_impute
from model_trainer import train_model
from model_registry import register_model

# PRUEBA PREPROCESS

def main():

    print("1. Cargando datos...")
    df_clientes = load_client_data()
    df_reqs = load_reqs_data()

    print("2. Limpieza e imputación...")
    df_clientes = clean_and_impute(df_clientes)
    df_reqs = clean_and_impute(df_reqs)

    print("4. Entrenando modelo...")
    # Opción 1: Usar tu función train_model() existente
    train_model()  # Ejecuta el modelo
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train model
    model, accuracy = train_model(X_train, y_train, X_test, y_test)
    n_estimators = model.n_estimators
    model_name = "RandomForestClassifier"

    # Register model with MLflow
    register_model(model, model_name, n_estimators, accuracy)
    """
if __name__ == "__main__":
    main()