# src/data_loader.py

__path__ = []  # para que importlib.util.find_spec no falle buscando submódulos

import pandas as pd
from pathlib import Path
from config import PATH_CONFIG, DATA_CONFIG, MERGE_CONFIG

def _get_data_path(table: str, data_type: str) -> Path:
    base = Path(PATH_CONFIG["data_folder"])
    sub  = PATH_CONFIG["input_subfolder"]
    fname = DATA_CONFIG[data_type][table]
    return base / sub / fname

def load_individual_dataset(data_type: str, table: str) -> pd.DataFrame:
    """
    Carga uno de los CSV de clientes o requerimientos,
    según sea 'train' u 'oot'.
    """
    path = _get_data_path(table, data_type)
    return pd.read_csv(path)

def load_and_merge_datasets(data_type: str) -> pd.DataFrame:
    """
    Carga y hace el merge de clientes + requerimientos
    con los keys y tipo de join configurado.
    """
    df_c = load_individual_dataset(data_type, "clientes")
    df_r = load_individual_dataset(data_type, "requerimientos")
    keys = MERGE_CONFIG["join_keys"]
    how  = MERGE_CONFIG.get("join_type", "left")
    return pd.merge(df_c, df_r, on=keys, how=how)

def load_data(data_type: str = "train") -> pd.DataFrame:
    """
    Carga todo el dataset (train u oot) ya mergeado.
    """
    if data_type not in DATA_CONFIG:
        raise ValueError(f"data_type={data_type} inválido")
    return load_and_merge_datasets(data_type)
