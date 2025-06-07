# config.py - Configuración centralizada del proyecto

# Configuración de archivos de datos
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

# Configuración de columnas para el cruce
MERGE_CONFIG = {
    "join_keys": ["ID_CORRELATIVO", "CODMES"],  # Columnas para hacer el join
    "join_type": "left",  # Tipo de join
    "target_column": "ATTRITION"  # Columna objetivo
}

# Configuración de división train/test
SPLIT_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True  # Mantener proporción de clases
}

# Configuración de rutas
PATH_CONFIG = {
    "data_folder": "data",
    "input_subfolder": "in",
    "output_subfolder": "out"
}