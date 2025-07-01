# src/config.py

# Configuración de cómo están dispuestos los datos en tu repo
DATA_CONFIG = {
    "train": {
        "clientes": "train_clientes_sample.csv",
        "requerimientos": "train_requerimientos_sample.csv"
    },
    "oot": {
        "clientes": "oot_clientes_sample.csv",
        "requerimientos": "oot_requerimientos_sample.csv"
    }
}

# Cómo hacemos el merge y cuál es la columna target
MERGE_CONFIG = {
    "join_keys": ["ID_CORRELATIVO", "CODMES"],
    "join_type": "left",
    "target_column": "ATTRITION"
}

# Rutas relativas en tu repo
PATH_CONFIG = {
    "data_folder": "data",          # carpeta padre de 'in' y 'out'
    "input_subfolder": "in",
    "output_subfolder": "out"
}

# Configuración de MLflow (tracking + registry)
MLFLOW_CONFIG = {
    "tracking_uri": "sqlite:///mlruns.db",
    "experiment_name": "bank_attrition",
    "model_name": "bank_attrition_model"
}
