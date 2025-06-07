# filepath: /ml-pipeline-project/ml-pipeline-project/src/pipelineml.py
import pandas as pd
from data_loader import load_data
from data_preparation import prepare_data
from model_trainer import train_model
from model_registry import register_model

def main():
    
    # Load data - TRAIN (entrenamiento) y OOT (evaluación)
    print("📊 Cargando datos de entrenamiento...")
    train_data = load_data("train")  # DataFrame con target ATTRITION
    
    print("\n📊 Cargando datos de evaluación (Out of Time)...")
    oot_data = load_data("oot")  # DataFrame SIN target (para predicción)
    
    print(f"\n✅ Datos cargados:")
    print(f"   🎯 Train: {train_data.shape} (con target)")
    print(f"   🎯 OOT: {oot_data.shape} (sin target)")

    # Prepare data - Solo datos de entrenamiento
    X_train, X_test, y_train, y_test = prepare_data(train_data)

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # Train model
    model, accuracy = train_model(X_train, y_train, X_test, y_test)
    n_estimators = model.n_estimators
    model_name = "RandomForestClassifier"

    # Register model with MLflow
    register_model(model, model_name, n_estimators, accuracy)

if __name__ == "__main__":
    main()