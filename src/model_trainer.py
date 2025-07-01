# src/model_trainer.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import optuna
from typing import Tuple

def train_model(X_train, y_train, X_test, y_test, tune: bool = False) -> Tuple[RandomForestClassifier, float]:
    """
    Entrena un RandomForest. Si tune=True, corre Optuna para buscar hiperparámetros.
    Devuelve (modelo_final, accuracy_test).
    """
    if tune:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42
            }
            m = RandomForestClassifier(**params)
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            return accuracy_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_params = study.best_params
    else:
        best_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 1,
            "random_state": 42
        }

    # Re-entrena con los mejores parámetros
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return model, acc
