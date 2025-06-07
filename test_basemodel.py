import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("data/in/application_data.csv")

# Separar variable objetivo y features
X = df.drop(["SK_ID_CURR", "TARGET"], axis=1)
y = df["TARGET"]

# Vista general del target
print("Shape de X:", X.shape)
print("Distribución de y:", y.value_counts())

# Identificar variables categóricas
cat_cols = X.select_dtypes(include=["object"]).columns

# Codificar variables categóricas
for col in cat_cols:
    X[col] = X[col].fillna("missing")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Imputar valores faltantes en variables numéricas
num_cols = X.select_dtypes(include=["number"]).columns
imputer = SimpleImputer(strategy="median")
X[num_cols] = imputer.fit_transform(X[num_cols])

# Separar en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("RandomForestClassifier_BestModel")
with mlflow.start_run(run_name="RandomForestClassifier"):
    # Entrenar modelo inicial con todas las variables
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Importancia de variables usando permutation_importance
    result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    importances = result.importances_mean
    mask = importances > 0
    filtered_idx = importances.argsort()[::-1][mask[importances.argsort()[::-1]]]
    filtered_features = X.columns[filtered_idx]

    # Filtrar datasets solo con variables importantes
    X_train_filtered = X_train[filtered_features]
    X_test_filtered = X_test[filtered_features]

    # Entrenar modelo final solo con variables importantes
    clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_final.fit(X_train_filtered, y_train)

    # Predicciones y métricas
    y_pred = clf_final.predict(X_test_filtered)
    y_proba = clf_final.predict_proba(X_test_filtered)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log de métricas
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("precision", report["1"]["precision"])
    mlflow.log_metric("recall", report["1"]["recall"])
    mlflow.log_metric("f1-score", report["1"]["f1-score"])

    # Graficar y guardar la curva ROC usando RocCurveDisplay
    roc_display = RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.savefig("roc_curve.png")
    plt.close()
    mlflow.log_artifact("roc_curve.png")

    # Graficar y guardar la importancia de variables (solo positivas)
    filtered_importances = importances[filtered_idx]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(filtered_importances)), filtered_importances, align="center")
    plt.xticks(range(len(filtered_importances)), filtered_features, rotation=90)
    plt.title("Permutation Feature Importance (solo positivas)")
    plt.tight_layout()
    plt.savefig("permutation_feature_importance.png")
    plt.close()
    mlflow.log_artifact("permutation_feature_importance.png")

    # Ejemplo de entrada para la firma
    input_example = X_test_filtered.iloc[:5]

    # Log del modelo (esto guarda flavor sklearn y python_function)
    mlflow.sklearn.log_model(
        clf_final,
        "model",
        input_example=input_example
    )

    # Evaluación automática con mlflow.evaluate
    eval_results = mlflow.evaluate(
        model="runs:/{}/model".format(mlflow.active_run().info.run_id),
        data=X_test_filtered.assign(TARGET=y_test),  # DataFrame con features y target
        targets="TARGET",
        model_type="classifier",
        evaluators="default"
    )
    print("Resultados de mlflow.evaluate:", eval_results.metrics)
