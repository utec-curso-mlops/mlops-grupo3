import pandas as pd
import numpy as np

def clean_and_impute(df):
    """
    Realiza limpieza de valores nulos con imputación basada en el tipo de variable:
    - Numéricas continuas: mediana
    - Binarias (0/1): moda
    - Categóricas: moda
    """
    df_clean = df.copy()

    # Reemplazar strings no informativos por NaN
    df_clean.replace(["", "-", "NA", "null"], np.nan, inplace=True)

    for col in df_clean.columns:
        if df_clean[col].isnull().sum() == 0:
            continue  # No imputar si no hay nulos

        dtype = df_clean[col].dtype
        valores_unicos = df_clean[col].dropna().unique()

        # Numéricas
        if dtype in ['int64', 'float64']:
            if sorted(valores_unicos.tolist()) in [[0, 1], [1, 0]]:
                # Binaria
                imputado = df_clean[col].mode()[0]
                df_clean[col] = df_clean[col].fillna(imputado)
            else:
                # Continua
                imputado = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(imputado)

        # Categóricas
        elif dtype == 'object' or dtype.name == 'category':
            imputado = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(imputado)

        # Otros tipos pueden extenderse si es necesario

    return df_clean



def prepare_data(df):
    from sklearn.model_selection import train_test_split

    # Imputar todos los valores nulos en variables numéricas por la mediana de cada variable
    df = df.fillna(df.median(numeric_only=True))

    # Identificar todas las columnas categóricas u object
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Crear variables dummies para todas las columnas categóricas
    df_dummies = pd.get_dummies(df, columns=cat_cols)
    X = df_dummies.drop(columns=["TARGET"])
    y = df_dummies["TARGET"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    return X_train, X_test, y_train, y_test
    
   