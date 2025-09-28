# app/app.py
# -*- coding: utf-8 -*-

"""
API REST de inferencia para el modelo de Breast Cancer.

Rutas:
- GET  /            -> status del servicio
- GET  /health      -> healthcheck simple
- POST /predict     -> recibe {"instances": [ {...}, {...} ] } y responde predicciones

Ejecución local:
    export FLASK_APP=app.app:app
    flask run --host=0.0.0.0 --port=5000

Ejecución con Gunicorn (producción local):
    gunicorn -w 2 -b 0.0.0.0:5000 app.app:app
"""
# Importar librerías necesarias
import json
import os
from typing import List, Dict, Any, Tuple
import joblib
import numpy as np
from flask import Flask, jsonify, request
import logging
import sys

# Función para configurar el logger
def get_logger(name: str) -> logging.Logger:
    """
    Crea o recupera un logger configurado con salida estándar y formato personalizado.

    Parámetros:
    - name (str): Nombre del logger, usualmente relacionado con el módulo o componente.

    Retorna:
    - logging.Logger: Instancia de logger lista para usar.

    Efectos:
    - Si el logger aún no tiene handlers, se le agrega un StreamHandler a stdout.
    - El formato incluye fecha/hora, nivel de log, nombre y mensaje.
    - Se desactiva la propagación para evitar duplicación de logs.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)                              # Nivel por defecto
        handler = logging.StreamHandler(sys.stdout)                # Handler → stdout
        fmt = logging.Formatter(                                   # Formato de log
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.propagate = False                                   # No propagar a root
    return logger


# Función para validar las instancias
def validate_instances(instances: Any, feature_names: List[str]) -> Tuple[bool, str]:
    """
    Valida que las instancias recibidas sean una lista de diccionarios con
    exactamente las columnas esperadas y valores numéricos.

    Parámetros:
    - instances (Any): Lista de instancias a validar (cada una debe ser un dict con feature:value).
    - feature_names (List[str]): Nombres de características esperadas.

    Retorna:
    - Tuple[bool, str]: (True, "") si todo es válido.
    - Tuple[bool, str]: (False, mensaje_error) si hay algún problema.

    Reglas de validación:
    - 'instances' debe ser lista no vacía.
    - Cada elemento debe ser un dict.
    - Cada dict debe contener exactamente las columnas en feature_names (ni más ni menos).
    - Los valores deben ser numéricos (int, float o bool).
    """
    if not isinstance(instances, list) or len(instances) == 0:
        return False, "⚠️ 'instances' debe ser una lista no vacía."

    for i, row in enumerate(instances):
        if not isinstance(row, dict):
            return False, f"⚠️ Elemento {i} de 'instances' no es un objeto JSON (dict)."

        missing = [c for c in feature_names if c not in row]
        extras = [c for c in row.keys() if c not in feature_names]
        if missing:
            return False, f"⚠️ Faltan columnas en la instancia {i}: {missing}"
        if extras:
            return False, f"⚠️ Columnas no reconocidas en la instancia {i}: {extras}"

        # Validar tipos de valores
        for c in feature_names:
            v = row[c]
            if not isinstance(v, (int, float, bool)):
                return False, f"⚠️ Valor no numérico para '{c}' en instancia {i}: {v}"

    return True, ""


# Crear app y logger
app = Flask(__name__)               # Inicialización de Flask
logger = get_logger(__name__)       # Inicialización de logger

# Rutas de artefactos
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))   # Ruta raíz del proyecto (un nivel arriba del archivo actual)
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')   # Carpeta donde se guardan los modelos
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pkl')  # Ruta del archivo del modelo entrenado
FEATURES_PATH = os.path.join(MODELS_DIR, 'feature_names.json')  # Ruta del archivo JSON con los nombres de las características
METADATA_PATH = os.path.join(MODELS_DIR, 'metadata.json')   # Ruta del archivo JSON con la metadata del modelo

# Cargar artefactos al iniciar
"""
Este bloque de código valida si el modelo entrenado existe en la ruta definida. 
Si no existe, se inicializan variables vacías y se muestra una advertencia. 
Si existe, carga el modelo y los nombres de las características, registrando la información en logs.
"""
if not os.path.exists(MODEL_PATH):
    # Caso: el modelo aún no ha sido entrenado ni guardado
    logger.warning("⚠️ El modelo no existe todavía. Ejecuta 'python scripts/train_model.py' primero.")
    MODEL = None
    FEATURE_NAMES: List[str] = []
else:
    # Caso: el modelo ya existe, se procede a cargarlo
    MODEL = joblib.load(MODEL_PATH)
    
    # Carga el archivo JSON con los nombres de las características
    with open(FEATURES_PATH, 'r', encoding = 'utf-8') as f:
        FEATURE_NAMES = json.load(f)
    
    # Log de confirmación con número de características cargadas
    logger.info(f"✅ Modelo y columnas cargados. n_features = {len(FEATURE_NAMES)}")

# Define el endpoint raíz
@app.get("/")

# Función para el endpoint raíz
def root():
    """
    Proporciona un endpoint raíz que informa el estado de la API.

    Esta función actúa como un chequeo de salud para la API, indicando si
    el servicio está en funcionamiento y si el modelo de machine learning
    ha sido cargado. También carga y muestra metadatos si están disponibles.

    Parámetros:
    - Ninguno.

    Retorna:
    - tuple: Una tupla que contiene un objeto Flask Response con una carga útil
            JSON y un código de estado HTTP 200. El JSON incluye el estado
            del servicio, un mensaje, un booleano que indica si el modelo
            está cargado y un diccionario con metadatos.

    Efectos:
    - Lee el archivo ubicado en la ruta METADATA_PATH si este existe.
    """
    meta = {}  # Diccionario para almacenar los metadatos del modelo si existen

    # Verificar si el archivo de metadatos existe en la ruta configurada
    if os.path.exists(METADATA_PATH):
        # Abrir el archivo JSON de metadatos en modo lectura
        with open(METADATA_PATH, "r", encoding = "utf-8") as f:
            # Cargar los metadatos desde el archivo
            meta = json.load(f)

    # Construir y devolver la respuesta JSON con estado, mensaje, si el modelo está cargado y los metadatos
    return jsonify({
        "status": "ok",                                # Estado de la API
        "message": "Breast Cancer API up & running",   # Mensaje informativo
        "model_loaded": MODEL is not None,             # Indica si el modelo está cargado en memoria
        "metadata": meta                               # Metadatos del modelo (o vacío si no hay)
    }), 200  # Código HTTP de respuesta OK


# Define el endpoint de healthcheck
@app.get("/health")

# Función para el endpoint de healthcheck
def health():
    """
    Verifica el estado de salud del modelo de machine learning.

    Este endpoint se utiliza para comprobar si el modelo predictivo ha sido
    cargado correctamente en la aplicación. Devuelve un estado de error si
    el modelo no está disponible.

    Parámetros:
    - Ninguno.

    Retorna:
    - tuple: Una tupla que contiene un objeto Flask Response con una carga útil
            JSON y un código de estado HTTP. Será un estado 'healthy' con
            código 200 si el modelo está cargado, o un estado 'error' con
            código 500 si no lo está.

    Efectos:
    - Ninguno
    """
    # Comprueba si la variable global del modelo no ha sido inicializada (es None).
    if MODEL is None:
        # Si el modelo no está cargado, devuelve un error y el código 500 (Error Interno del Servidor).
        return jsonify({"status": "error", "detail": "Modelo no cargado"}), 500
    
    # Si el modelo está cargado, devuelve un estado saludable y el código 200 (OK).
    return jsonify({"status": "healthy"}), 200


# Define el endpoint de predicción
@app.post("/predict")

# Función para el endpoint de predicción
def predict():
    """
    Realiza predicciones utilizando el modelo de ML cargado.

    Este endpoint recibe una o más instancias en formato JSON, las valida,
    las procesa y utiliza el modelo para generar predicciones y, si es
    posible, las probabilidades asociadas a cada predicción.

    Parámetros:
    - Ninguno (los datos se leen del cuerpo de la solicitud HTTP).

    Retorna:
    - tuple: Una tupla que contiene un objeto Flask Response y un código de estado HTTP.
        - 200: Si la predicción es exitosa.
        - 400: Si el JSON de entrada es inválido o no pasa la validación.
        - 500: Si el modelo no está cargado o si ocurre un error inesperado.

    Efectos:
    - Lee el cuerpo de la solicitud HTTP.
    - Registra mensajes informativos, de advertencia o de error en un logger.
    """
    # Verifica si el modelo ha sido cargado en memoria.
    if MODEL is None:
        # Si no, devuelve un error 500 indicando que el servidor no está listo.
        return jsonify({"❌ Error": "Modelo no cargado en el servidor"}), 500

    # Obtiene el payload JSON de la solicitud; 'silent=True' evita un error si el parseo falla.
    payload = request.get_json(silent = True)
    # Verifica que el payload no sea nulo y que contenga la clave 'instances'.
    if not payload or 'instances' not in payload:
        # Devuelve un error 400 (Bad Request) si el formato del JSON es incorrecto.
        return jsonify({"❌ Error": "JSON inválido. Debe incluir 'instances'."}), 400

    # Extrae la lista de instancias a predecir del payload.
    instances = payload['instances']
    # Llama a una función auxiliar para validar la estructura y los datos de las instancias.
    is_ok, msg = validate_instances(instances, FEATURE_NAMES)
    # Si la validación falla...
    if not is_ok:
        # Registra una advertencia con el mensaje de error de la validación.
        logger.warning(f"⚠️ Validación fallida: {msg}")
        # Devuelve el mensaje de error al cliente con un código 400.
        return jsonify({"❌ Error": msg}), 400

    # Inicia un bloque try-except para capturar errores durante la inferencia.
    try:
        # Construye la matriz de características (X) asegurando el orden correcto de las columnas.
        X = [[row[c] for c in FEATURE_NAMES] for row in instances]
        # Convierte la lista de listas a un array de NumPy con tipo de dato float.
        X = np.asarray(X, dtype = float)

        # Realiza la predicción y convierte el resultado a una lista estándar de Python.
        preds = MODEL.predict(X).tolist()
        
        # Si el modelo tiene el método 'predict_proba', obtiene las probabilidades.
        proba = MODEL.predict_proba(X).tolist() if hasattr(MODEL, "predict_proba") else None

        # Registra un mensaje informativo indicando que las predicciones se generaron.
        logger.info(f"✅ Predicciones generadas para {len(instances)} instancia(s).")
        # Devuelve las predicciones y probabilidades en formato JSON con un estado 200 (OK).
        return jsonify({"predictions": preds, "proba": proba}), 200

    # Si ocurre cualquier excepción durante el proceso de predicción
    except Exception as e:
        # Registra la excepción completa (con traceback) para depuración.
        logger.exception("❌ Error durante la inferencia")
        # Devuelve un error genérico 500 al cliente, incluyendo el mensaje de la excepción.
        return jsonify({"❌ Error": f"Error en predicción: {str(e)}"}), 500


# Punto de entrada para ejecutar con `python -m app.app`
if __name__ == "__main__":
    # ¡Útil en desarrollo local! Para producción usa Gunicorn o Docker.
    app.run(host = "0.0.0.0", port = 5000, debug = False)