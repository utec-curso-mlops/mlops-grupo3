# src/pipelineml.py

import click
from pathlib import Path
import pandas as pd

from config import PATH_CONFIG, MERGE_CONFIG, MLFLOW_CONFIG
from data_loader import load_data
from data_preparation import prepare_data
from model_trainer import train_model
from model_registry import register_model

@click.command()
@click.option(
    "--tune",
    is_flag=True,
    default=False,
    help="Si se pasa, activa hyperparameter tuning con Optuna"
)
def main(tune):
    # 1) Carga de datos
    print("📊 Cargando TRAIN y OOT …")
    df_train = load_data("train")
    df_oot   = load_data("oot")
    print(f"   TRAIN: {df_train.shape},  OOT: {df_oot.shape}")

    # 2) Prepara X/y y split
    X_train, X_test, y_train, y_test = prepare_data(df_train)

    # 3) Entrena
    model, acc = train_model(X_train, y_train, X_test, y_test, tune)
    print(f"🏆 Accuracy en TEST: {acc:.4f}")

    # 4) Registra en MLflow + Registry
    register_model(model, MLFLOW_CONFIG["model_name"], acc)
    print("✅ Modelo registrado en MLflow Registry")

    # 5) Scoring OOT
    print("📊 Scoring OOT …")
    drop_cols = MERGE_CONFIG["join_keys"] + [MERGE_CONFIG["target_column"]]
    X_oot = df_oot.drop(columns=drop_cols, errors="ignore")
    X_oot = pd.get_dummies(X_oot)
    # Asegurar mismas columnas que X_train
    for c in X_train.columns:
        if c not in X_oot.columns:
            X_oot[c] = 0
    X_oot = X_oot[X_train.columns]
    preds = model.predict(X_oot)

    out = df_oot[MERGE_CONFIG["join_keys"]].copy()
    out["prediction"] = preds
    out_path = Path(PATH_CONFIG["data_folder"])/PATH_CONFIG["output_subfolder"]/"oot_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"✅ OOT guardado en: {out_path}")

if __name__ == "__main__":
    main()
