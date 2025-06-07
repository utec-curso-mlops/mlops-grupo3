import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow.xgboost
import xgboost as xgb
import numpy as np


mlflow.set_tracking_uri("http://localhost:5000")

# Cargar datos
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target, random_state=123)

# Mi primer experimento con mlflow.autolog()
mlflow.set_experiment("experiment_autolog")

with mlflow.start_run():
    mlflow.autolog() # Activa autologging para todas las librerías compatibles

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)    
    rf.fit(X_train, y_train)

# Experimento con mlflow.sklearn.autolog()
mlflow.set_experiment("experiment_sklearn_autolog")

with mlflow.start_run():
    mlflow.sklearn.autolog() 

    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)    
    rf.fit(X_train, y_train)
    
    # Calcular RMSE en validación
    y_pred = rf.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Registrar RMSE en MLflow
    mlflow.log_metric("validation_rmse", rmse_test)


# Experimento con mlflow.sklearn.autolog()
mlflow.set_experiment("experiment_xgboost_autolog")

with mlflow.start_run():
    # Activar autologging para XGBoost
    mlflow.xgboost.autolog()

    # Entrenar el modelo
    xgboost = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=3, learning_rate=0.1)
    xgboost.fit(X_train, y_train)
    
    # Calcular RMSE en validación
    y_pred = xgboost.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    # Registrar RMSE en MLflow
    mlflow.log_metric("validation_rmse", rmse_test)
