#!/usr/bin/env python3
# src/test_data_loader_completo.py

import sys
from pathlib import Path
import pandas as pd

# Asegurar que src/ esté en PYTHONPATH
current_dir = Path.cwd()
if current_dir.name != "src":
    sys.path.insert(0, str(current_dir / "src"))

def test_config_import():
    from config import DATA_CONFIG, MERGE_CONFIG, PATH_CONFIG
    assert isinstance(DATA_CONFIG, dict)
    assert isinstance(PATH_CONFIG, dict)
    # Verificar keys mínimas
    assert "train" in DATA_CONFIG and "oot" in DATA_CONFIG
    assert "join_keys" in MERGE_CONFIG and "target_column" in MERGE_CONFIG
    print("✅ config.py OK")

def test_data_loader_import():
    from data_loader import load_data, load_individual_dataset, load_and_merge_datasets
    # Todas deben existir
    assert callable(load_data)
    assert callable(load_individual_dataset)
    assert callable(load_and_merge_datasets)
    # split_train_test NO debe existir
    from importlib import util
    assert util.find_spec("data_loader.split_train_test") is None
    print("✅ data_loader.py OK")

def test_individual_loading():
    from data_loader import load_individual_dataset
    for fn in [
        "train_clientes_sample.csv",
        "train_requerimientos_sample.csv",
        "oot_clientes_sample.csv",
        "oot_requerimientos_sample.csv"
    ]:
        df = load_individual_dataset(fn)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
    print("✅ carga individual OK")

def test_merge_functionality():
    from data_loader import load_and_merge_datasets
    df_tr  = load_and_merge_datasets("train")
    df_oot = load_and_merge_datasets("oot")
    # Debe contener la columna target
    assert "ATTRITION" in df_tr.columns
    # OOT no debe tenerla
    assert "ATTRITION" not in df_oot.columns
    print("✅ merge datasets OK")

if __name__ == "__main__":
    # Ejecutar todos los tests
    test_config_import()
    test_data_loader_import()
    test_individual_loading()
    test_merge_functionality()
    print("🎉 Todos los tests pasaron.")
