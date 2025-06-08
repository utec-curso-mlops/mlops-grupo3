from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression #Modelo de regresión logística
from sklearn.svm import SVC #Modelo de Clasificador de vectores de soporte
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train, X_test, y_test):
    """
    Entrena varios modelos y selecciona el que tenga mayor accuracy.
    
    Parámetros:
    - X_train, y_train: datos de entrenamiento
    - X_test, y_test: datos de prueba

    Retorna:
    - clf: el mejor modelo entrenado
    - accuracy: accuracy del modelo seleccionado
    """
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=50), #Bosque Aleatorio con 50 árbols de decisión
        'LogisticRegression': LogisticRegression(max_iter=1000), #Regresión logística con más interacciones para estabilidad
        'SVC': SVC() #Vectores de Soportes con parámetros por defecto
    }

    best_model = None #Variable que guardará el mejor modelo
    best_score = 0 #Variable que guardará el score del modelo elegido

    for name, model in models.items():
        model.fit(X_train, y_train) #Entrenamiento del modelo
        y_pred = model.predict(X_test) #Predicción del modelo
        acc = accuracy_score(y_test, y_pred) #Cálculo del score
        print(f"{name} accuracy: {acc:.4f}") #Impresión del score


        if acc > best_score:
            best_score = acc #Actualización del score
            best_model = model #Actualización del modelo


    # Para que el nombre coincida con la interfaz original
    clf = best_model
    accuracy = best_score 
    return clf, accuracy
