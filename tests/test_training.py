# tests/test_training.py
# -*- coding: utf-8 -*-

"""
Prueba de entrenamiento: verifica que se generen los artefactos.
"""

# Importar librerías necesarias
import os
import json
import joblib
import subprocess
import sys

# Función de prueba entrenamiento
def test_training_script_runs_and_creates_artifacts():
    """
    Prueba la ejecución completa del script de entrenamiento y la creación de artefactos.

    Esta función de prueba ejecuta el script 'train_model.py' en un subproceso
    para simular una ejecución real. Luego, verifica que el script se complete
    sin errores y que los archivos de salida esperados (modelo, metadatos y
    nombres de características) se hayan creado correctamente. Finalmente,
    realiza una validación básica de los artefactos generados.

    Args:
    - Ninguno

    Retorna:
    - Ninguno

    Excepciones:
    - AssertionError: Si el script de entrenamiento falla, si alguno de los
                        artefactos no se crea, o si los artefactos no tienen
                        el formato básico esperado.

    Efectos:
    - Ejecuta el script 'scripts/train_model.py', lo cual puede crear o
        sobrescribir archivos en el directorio 'models/'.
    - Lee los artefactos generados del disco para validarlos.
    """
    # Ejecuta el script de entrenamiento en un proceso separado para aislar la prueba.
    result = subprocess.run(
        [sys.executable, "scripts/train_model.py"],  # Comando a ejecutar.
        capture_output = True,  # Captura la salida estándar y el error estándar.
        text = True  # Decodifica la salida y el error como texto.
    )
    # Verifica que el script terminó con un código de salida 0 (éxito).
    assert result.returncode == 0, f"Entrenamiento falló: {result.stderr}"

    # --- Verificación de la existencia de artefactos ---
    # Comprueba que el archivo del modelo serializado fue creado.
    assert os.path.exists("models/model.pkl"), "No se creó models/model.pkl"
    # Comprueba que el archivo de metadatos fue creado.
    assert os.path.exists("models/metadata.json"), "No se creó models/metadata.json"
    # Comprueba que el archivo con los nombres de las características fue creado.
    assert os.path.exists("models/feature_names.json"), "No se creó models/feature_names.json"

    # --- Carga y validación básica de los artefactos ---
    # Carga el modelo desde el archivo para verificar que no esté corrupto.
    model = joblib.load("models/model.pkl")
    # Asegura que el objeto cargado tiene un método 'predict', una propiedad clave de un modelo.
    assert hasattr(model, "predict"), "El modelo no tiene método predict"

    # Abre y lee el archivo JSON que contiene los nombres de las características.
    with open("models/feature_names.json", "r", encoding = "utf-8") as f:
        # Carga el contenido del archivo JSON.
        feats = json.load(f)
    # Verifica que el contenido sea una lista y que no esté vacía.
    assert isinstance(feats, list) and len(feats) > 0, "feature_names.json inválido"