# src/data_preparation.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from config import MERGE_CONFIG

def prepare_data(df: pd.DataFrame):
    """
    - Separa X / y
    - Imputa numéricos con mediana
    - Codifica categóricos con one-hot
    - Stratified train/test split
    """
    keys   = MERGE_CONFIG["join_keys"]
    target = MERGE_CONFIG["target_column"]

    # Separa
    y = df[target]
    X = df.drop(columns=keys + [target], errors="ignore")

    # Imputa numéricos
    num_cols = X.select_dtypes(include="number").columns
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # Rellena categóricos faltantes y dummies
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    X[cat_cols] = X[cat_cols].fillna("missing")
    X = pd.get_dummies(X, columns=cat_cols)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
