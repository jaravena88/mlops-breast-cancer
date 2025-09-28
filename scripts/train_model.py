# scripts/train_model.py
# -*- coding: utf-8 -*-

"""
Entrenamiento del modelo Breast Cancer.
- Preprocesa el dataset.
- Entrena un modelo de Logistic Regression (modelo simple e interpretable).
- Guarda artefactos: model.pkl, metadata.json, feature_names.json.

Justificación: Logistic Regression es adecuado para dataset pequeño, rápido de entrenar
y fácil de interpretar (importante en problemas de salud).
"""

# Importar librerías necesarias
import os
import json
from datetime import datetime
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Función para entrenar y guardar modelo
def train_and_save_model():
    """
    Orquesta el pipeline completo de entrenamiento y guardado de un modelo.

    Esta función carga un dataset, lo limpia, entrena un modelo de regresión
    logística, evalúa su rendimiento y finalmente guarda el modelo entrenado
    junto con sus metadatos y la lista de características utilizadas.

    Parámetros:
    - Ninguno

    Retorno:
    - Ninguno

    Efectos:
    - Carga data desde data/data.csv.
    - Preprocesa: elimina columnas innecesarias y codifica 'diagnosis'.
    - Entrena modelo.
    - Guarda modelo en models/model.pkl.
    - Guarda metadatos en models/metadata.json.
    - Guarda lista de columnas en models/feature_names.json.
    - Imprime métricas en consola.
    """
    # Define la ruta del directorio actual y la raíz del proyecto.
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Construye la ruta completa al archivo del dataset.
    data_path = os.path.join(project_root, 'data', 'data.csv')

    # --- Carga del dataset ---
    # Verifica si el archivo del dataset existe en la ruta especificada.
    if not os.path.exists(data_path):
        # Si no existe, imprime un error y termina la ejecución de la función.
        print(f"❌ Error: No se encontró el dataset en: {data_path}")
        return

    # Carga el archivo CSV en un DataFrame de pandas.
    df = pd.read_csv(data_path)
    print("✅ Dataset cargado exitosamente.")

    # --- Limpieza y Validación ---
    # Define columnas a eliminar si existen en el DataFrame.
    cols_to_drop = [c for c in ['id', 'Unnamed: 32'] if c in df.columns]
    # Elimina las columnas no deseadas del DataFrame.
    df = df.drop(columns = cols_to_drop, errors = 'ignore')

    # Verifica la presencia de la columna objetivo 'diagnosis'.
    if 'diagnosis' not in df.columns:
        # Si no se encuentra, imprime un error y termina la ejecución.
        print("❌ Error: La columna 'diagnosis' no está presente en el CSV.")
        return

    # --- Preprocesamiento ---
    # Inicializa un codificador de etiquetas.
    label_encoder = LabelEncoder()
    # Transforma la columna 'diagnosis' a valores numéricos (ej. 'M'->1, 'B'->0).
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

    # Separa las características (X) de la variable objetivo (y).
    X = df.drop('diagnosis', axis = 1)
    y = df['diagnosis']

    # --- División de Datos ---
    # Divide los datos en conjuntos de entrenamiento y prueba (80/20).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.20, random_state = 42, stratify = y
    )

    # --- Entrenamiento del Modelo ---
    # Instancia un modelo de Regresión Logística con parámetros definidos.
    model = LogisticRegression(random_state = 42, max_iter = 10000)
    # Entrena (ajusta) el modelo con los datos de entrenamiento.
    model.fit(X_train, y_train)

    # --- Evaluación ---
    # Realiza predicciones sobre el conjunto de prueba.
    y_pred = model.predict(X_test)
    # Calcula la métrica de exactitud (accuracy).
    accuracy = accuracy_score(y_test, y_pred)
    # Genera un reporte de clasificación completo (precisión, recall, f1-score).
    report = classification_report(
        y_test, y_pred, target_names = ['Benigno', 'Maligno']
    )

    # Imprime los resultados de la evaluación en la consola.
    print("\n🚀 Entrenamiento completado.\n")
    print(f"✅ Accuracy: {accuracy:.4f}\n")
    print("📋 Classification Report:\n", report)

    # --- Guardado de Artefactos ---
    # Define la ruta del directorio para guardar los modelos.
    models_dir = os.path.join(project_root, 'models')
    # Crea el directorio si no existe; 'exist_ok=True' evita un error si ya existe.
    os.makedirs(models_dir, exist_ok = True)

    # Guarda el objeto del modelo entrenado en un archivo .pkl.
    model_path = os.path.join(models_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Modelo guardado en: {os.path.relpath(model_path)}\n")

    # Crea un diccionario con los metadatos del entrenamiento.
    metadata = {
        'model_type': 'Logistic Regression',
        'training_date': datetime.now().isoformat(),
        'accuracy': float(accuracy),
        'metrics': classification_report(y_test, y_pred, output_dict = True)
    }
    # Define la ruta para el archivo de metadatos.
    metadata_path = os.path.join(models_dir, 'metadata.json')
    # Escribe el diccionario de metadatos en un archivo JSON.
    with open(metadata_path, 'w', encoding = 'utf-8') as f:
        json.dump(metadata, f, indent = 4, ensure_ascii = False)
    print(f"💾 Metadatos guardados en: {os.path.relpath(metadata_path)}\n")

    # Guarda la lista de nombres de las características en el orden esperado por el modelo.
    feature_names = list(X.columns)
    feat_path = os.path.join(models_dir, 'feature_names.json')
    with open(feat_path, 'w', encoding = 'utf-8') as f:
        json.dump(feature_names, f, indent = 2, ensure_ascii = False)
    print(f"💾 Columnas del modelo guardadas en: {os.path.relpath(feat_path)}\n")

# Punto de entrada cuando se ejecuta el script directamente
if __name__ == '__main__':
    train_and_save_model()