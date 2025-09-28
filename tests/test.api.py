# tests/test_api.py
# -*- coding: utf-8 -*-

"""
Pruebas básicas del API con el test client de Flask.
"""

# Importar librerías necesarias
import json
import os
import pytest
from app.app import app, FEATURE_NAMES, MODEL

# Decorador para fixture
@pytest.fixture(scope = "module")

# Función fixture para el cliente de pruebas
def client():
    """
    Define un fixture de pytest para el cliente de pruebas de Flask.

    Este fixture crea un cliente de pruebas para la aplicación Flask (`app`).
    El cliente puede ser utilizado en las funciones de prueba para simular
    peticiones HTTP a los endpoints de la aplicación sin necesidad de
    levantar un servidor real.

    Parámetros:
    - Ninguno

    Retorna:
    - flask.testing.FlaskClient: Una instancia del cliente de pruebas de Flask.
    
    Efectos:
    - Ninguno
    """
    # Crea un cliente de pruebas para la app de Flask usando un gestor de contexto.
    with app.test_client() as c:
        # 'yield' entrega el cliente 'c' a la función de prueba que lo solicita.
        # El contexto se cierra automáticamente después de que la prueba termina.
        yield c


# Función de prueba raíz
def test_root_ok(client):
    """
    Prueba que el endpoint raíz ("/") responde correctamente.

    Esta prueba utiliza el cliente de Flask para hacer una petición GET a la
    ruta raíz. Verifica que el código de estado de la respuesta sea 200 (OK)
    y que el cuerpo de la respuesta JSON contenga un campo "status" con el
    valor "ok".

    Parámetros:
    - client (flask.testing.FlaskClient): El fixture del cliente de pruebas de Flask.

    Retorna:
    - Ninguno

    Excepciones:
    - AssertionError: Si el código de estado no es 200 o el contenido del JSON no es el esperado.
    
    Efectos:
    - Ninguno
    """
    # Envía una petición GET al endpoint raíz ("/") usando el cliente de prueba.
    resp = client.get("/")
    
    # Verifica que el código de estado de la respuesta sea 200 (OK).
    assert resp.status_code == 200
    
    # Parsea la respuesta JSON a un diccionario de Python.
    data = resp.get_json()
    
    # Verifica que el diccionario contenga la clave "status" y su valor sea "ok".
    assert "status" in data and data["status"] == "ok"


# Función de prueba de healthcheck
def test_health(client):
    """
    Prueba el endpoint de salud ('/health') en diferentes escenarios.

    Esta prueba verifica que el endpoint '/health' se comporte correctamente
    dependiendo de si el modelo de machine learning global (`MODEL`) ha sido
    cargado o no.

    - Si `MODEL` no está cargado, espera un código de estado 500.
    - Si `MODEL` está cargado, espera un código de estado 200.

    Parámetros:
    - client (flask.testing.FlaskClient): El fixture del cliente de pruebas de Flask.

    Retorna:
    - Ninguno

    Excepciones:
    - AssertionError: Si el código de estado de la respuesta no coincide con el
                    esperado según el estado del modelo.

    Efectos:
    - Ninguno
    """
    # Envía una petición GET al endpoint de salud.
    resp = client.get("/health")

    # Escenario 1: El modelo NO está cargado.
    if MODEL is None:
        # Verifica que la API responda con un error 500 (Error Interno del Servidor).
        assert resp.status_code == 500
    # Escenario 2: El modelo SÍ está cargado.
    else:
        # Verifica que la API responda con un estado 200 (OK), indicando que está saludable.
        assert resp.status_code == 200


# Función para probar endpoint predicción
def test_predict_happy_path(client):
    """
    Prueba el "camino feliz" del endpoint de predicción ('/predict').

    Esta prueba simula una solicitud de predicción exitosa. Se salta si el
    modelo o los nombres de las características no están cargados. De lo
    contrario, construye una carga útil (payload) válida, la envía al
    endpoint '/predict' y verifica que la respuesta sea un 200 OK y que
    contenga las claves 'predictions' y 'proba'.

    Parámetros:
    - client (flask.testing.FlaskClient): El fixture del cliente de pruebas de Flask.

    Retorna:
    - Ninguno

    Excepciones:
    - AssertionError: Si el código de estado de la respuesta no es 200 o si
                    el cuerpo de la respuesta JSON no contiene las claves esperadas.

    Efectos:
    - Puede omitir la ejecución de la prueba si no se cumplen las condiciones iniciales.
    """
    # Si el modelo o la lista de características no están disponibles, omite esta prueba.
    if MODEL is None or len(FEATURE_NAMES) == 0:
        pytest.skip("⚠️ Modelo/feature_names no disponibles para test_predict_happy_path")

    # Crea una instancia de datos de ejemplo con valores arbitrarios (0.0).
    instance = {c: 0.0 for c in FEATURE_NAMES}
    # Envuelve la instancia en el formato de payload que espera la API.
    payload = {"instances": [instance]}

    # Envía una petición POST al endpoint '/predict' con el payload en formato JSON.
    resp = client.post(
        "/predict",
        data = json.dumps(payload),
        content_type = "application/json"
    )
    
    # Verifica que el código de estado sea 200 (OK), indicando éxito.
    assert resp.status_code == 200
    
    # Parsea la respuesta JSON a un diccionario.
    data = resp.get_json()
    
    # Comprueba que la clave 'predictions' exista en la respuesta.
    assert "predictions" in data
    
    # Comprueba que la clave 'proba' exista en la respuesta.
    # Nota: su valor podría ser nulo si el modelo no lo soporta, pero la clave debe estar.
    assert "proba" in data


# Función de prueba predicción con payload inválido
def test_predict_bad_payload(client):
    """
    Prueba el endpoint de predicción ('/predict') con un payload inválido.

    Esta prueba envía una solicitud al endpoint '/predict' con un cuerpo
    JSON que no sigue la estructura esperada (carece de la clave 'instances').
    Verifica que la API maneje correctamente este error, respondiendo con un
    código de estado 400 (Bad Request).

    Parámetros:
    - client (flask.testing.FlaskClient): El fixture del cliente de pruebas de Flask.

    Retorna:
    - Ninguno.

    Excepciones:
    - AssertionError: Si el código de estado de la respuesta no es 400.
    
    Efectos:
    - Ninguno.
    """
    # Envía una petición POST con un payload JSON inválido (falta la clave 'instances').
    resp = client.post(
        "/predict",
        data = json.dumps({"foo": "bar"}),
        content_type = "application/json"
    )
    
    # Verifica que la API responda con el código 400 (Bad Request), como se esperaba.
    assert resp.status_code == 400