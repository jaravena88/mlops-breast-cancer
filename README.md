# Breast Cancer API – MLOps Módulo 10 (Windows 11)

## 📌 Descripción
Este proyecto corresponde a la **Evaluación Modular N°10 – MLOps en la Nube**.  
Se desarrolla un flujo completo que incluye:  

- Entrenamiento de un modelo de **clasificación** para diagnóstico de cáncer de mama (dataset Breast Cancer).  
- Exposición del modelo como **API REST con Flask**.  
- **Contenerización con Docker Desktop** para despliegue reproducible y escalable.  
- Flujo de **CI/CD con GitHub Actions** para automatizar pruebas, construcción y despliegue de la imagen.  

---

## 📂 Estructura del repositorio

```
mlops-breast-cancer/
├─ app/
│  ├─ __init__.py
│  ├─ app.py
│  └─ sample_payload.json
├─ data/
│  └─ data.csv
├─ models/
│  ├─ model.pkl
│  ├─ metadata.json
│  └─ feature_names.json
├─ scripts/
│  └─ train_model.py
├─ tests/
│  ├─ test_training.py
│  └─ test_api.py
├─ .github/
│  └─ workflows/
│     └─ ci.yml
├─ .dockerignore
├─ Dockerfile
├─ requirements.txt
├─ README.md
└─ Makefile
```

---

## ⚙️ Requisitos en Windows 11
- **Python 3.10.11**  
- **Docker Desktop** instalado y en ejecución  
- **PowerShell** (para ejecutar los comandos)

---

## ▶️ Pasos para ejecución local en Windows

### 1. Instalar dependencias
Abrir **PowerShell** en la carpeta del proyecto:
```powershell
pip install -r requirements.txt
```

### 2. Entrenar el modelo
Esto generará `models/model.pkl`, `metadata.json` y `feature_names.json`:
```powershell
python scripts/train_model.py
```

### 3. Levantar la API
Con Flask (desarrollo):
```powershell
$env:FLASK_APP="app.app:app"
flask run --host=0.0.0.0 --port=5000
```

⚠️ **Nota:** Gunicorn no funciona en Windows. Para un servidor de producción usar Docker.

### 4. Probar endpoints

En PowerShell usar **Invoke-RestMethod**:

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/" -Method GET
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST `
  -ContentType "application/json" -InFile "app/sample_payload.json"
```

---

## 🐳 Ejecución con Docker Desktop

1. Entrenar modelo en el host:
```powershell
python scripts/train_model.py
```

2. Construir imagen:
```powershell
docker build -t breast-cancer-api:latest .
```

3. Ejecutar contenedor:
```powershell
docker run --rm -p 5000:5000 --name bc-api breast-cancer-api:latest
```

4. Probar con `Invoke-RestMethod`.

En PowerShell usar **Invoke-RestMethod**:

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/" -Method GET
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST `
  -ContentType "application/json" -InFile "app/sample_payload.json"
```

---

## ✅ Pruebas automáticas

Se incluyen pruebas con **pytest**:

- `tests/test_training.py`: verifica que el entrenamiento cree los artefactos.  
- `tests/test_api.py`: valida los endpoints básicos de la API.  

Ejecutar:
```powershell
pytest -q
```

---

## 🔄 CI/CD con GitHub Actions

El workflow `.github/workflows/ci.yml` realiza:

1. Instalación de dependencias.  
2. Entrenamiento para generar artefactos.  
3. Ejecución de pruebas (`pytest`).  
4. Construcción de la imagen Docker.  
5. (Opcional) Push a GitHub Container Registry (GHCR).  
6. Smoke test del contenedor (`GET /`).  

---

## 📑 Notas finales

- Los nombres de las columnas esperadas están en `models/feature_names.json`.  
- El modelo es de tipo **Logistic Regression**.  
- Las salidas del endpoint `/predict` son:  
  ```json
  {
    "predictions": [0, 1],
    "proba": [[0.85, 0.15], [0.20, 0.80]]
  }
  ```
  Donde `0 = Benigno`, `1 = Maligno`.  

---