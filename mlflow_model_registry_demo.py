"""
MLflow Model Registry: Ejemplo y explicación paso a paso
-------------------------------------------------------

Este script muestra cómo registrar, versionar, promover y administrar modelos usando el Model Registry de MLflow.
Incluye referencias a la documentación oficial y recursos avanzados.
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Entrenamiento y log de un modelo en MLflow
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
clf = RandomForestClassifier(n_estimators=10, random_state=42)

experiment_name = "RandomForestClassifier_IrisModel"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name="rf_model_registry_demo") as run:
    clf.fit(X_train, y_train)
    input_example = X_train[:5]
    # Inferir la firma del modelo
    signature = infer_signature(X_train, clf.predict(X_train))
    mlflow.sklearn.log_model(
        clf,
        "model",
        input_example=input_example,
        signature=signature
    )
    run_id = run.info.run_id

# 2. Registrar el modelo en el Model Registry
# Documentación: https://mlflow.org/docs/latest/model-registry.html#registering-models
model_name = "iris_model_testing"
model_uri = f"runs:/{run_id}/model"

# El registro crea una nueva versión del modelo bajo el nombre especificado
result = mlflow.register_model(model_uri, model_name)
print(f"Modelo registrado como '{model_name}', versión: {result.version}")

# 3. Consultar y administrar modelos con MlflowClient
# Documentación: https://mlflow.org/docs/latest/python_api/mlflow.client.html
client = MlflowClient()

# Listar todas las versiones del modelo
for mv in client.search_model_versions(f"name='{model_name}'"):
    print(f"Versión: {mv.version}, estado: {mv.current_stage}, run_id: {mv.run_id}")

# 4. Marcar el estado del modelo usando tags (recomendado desde MLflow 2.9.0)
client.set_model_version_tag(
    name=model_name,
    version=result.version,
    key="estado",
    value="staging"
)
print(f"Versión {result.version} marcada como 'staging' usando un tag.")

# 5. Asignar el alias "staging" a la versión deseada
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=result.version
)
print(f"Alias 'staging' asignado a la versión {result.version}.")

# 6. Añadir descripción y comentarios
client.update_model_version(
    name=model_name,
    version=result.version,
    description="RandomForest entrenado sobre Iris dataset. Ejemplo de Model Registry."
)

# Recursos adicionales:
# - MLflow Registry REST API: https://mlflow.org/docs/latest/rest-api.html#modelregistry
# - Ejemplo de CI/CD con Model Registry: https://mlflow.org/docs/latest/model-registry.html#ci-cd-workflows

"""
Resumen:
- El Model Registry permite versionar, promover y administrar modelos en entornos colaborativos.
- Es posible automatizar flujos de trabajo de despliegue y gobernanza de modelos.
- La integración con MLflow Tracking y Projects facilita la trazabilidad y reproducibilidad.

Referencias:
- Documentación oficial: https://mlflow.org/docs/latest/model-registry.html
- API de Model Registry: https://mlflow.org/docs/latest/python_api/mlflow.client.html#mlflow.client.MlflowClient
- Ejemplo avanzado: https://mlflow.org/docs/latest/tutorials-and-examples/tutorial.html#register-the-model
"""