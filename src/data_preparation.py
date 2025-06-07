def prepare_data(df):
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df = df.drop(columns="SK_ID_CURR")
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