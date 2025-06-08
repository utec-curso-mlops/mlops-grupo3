import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from data_preparation import get_prepared_dataset

def train_model():
    # 1. Obtener el dataset procesado directamente
    df = get_prepared_dataset()

    # 2. Separar features y target
    X = df.drop(columns=["attrition", "customer_id"])  # Ajusta si tu target tiene otro nombre
    y = df["attrition"]

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluación
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("=== ROC AUC ===")
    print(roc_auc_score(y_test, y_proba))

    # 6. Guardar modelo
    joblib.dump(model, "model_random_forest.pkl")

if __name__ == "__main__":
    train_model()
