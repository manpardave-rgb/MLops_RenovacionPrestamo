Proyecto Final del curso MLOps — Pipeline end-to-end para predicción de renovación de línea de crédito bancaria.
# MLOps — Renovación de Préstamo (Banco Wiesse)

Flujo end-to-end de pre-producción y despliegue para el modelo de propensión
de renovación de préstamo, adaptado del caso trabajado en el notebook
`Caso_Renovación_de_Préstamo.ipynb` y siguiendo la plantilla de la Unidad GCP
del curso de MLOps (Docker Compose + quality gate + MLflow + deploy con
rollback).

## Estructura del proyecto

```
RenovacionPrestamo/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          <- pipeline de integración y despliegue continuo
├── api/                       <- servicio FastAPI
│   ├── app.py                 <- endpoints /, /health, /predecir
│   ├── predictor.py           <- carga modelo.pkl y predice (singleton)
│   ├── schemas.py             <- modelos Pydantic (entrada/salida)
│   └── __init__.py
├── notebooks/                 <- entregables de ciencia de datos
│   └── Caso Renovación de Préstamo.ipynb <- modelo y lógica base provista por el Data Scientist
├── src/
│   ├── preprocessing.py       <- feature engineering (rename, log1p, imputación, one-hot, Cluster K-Means)
│   ├── generate_data.py       <- usa el CSV real si existe; si no, genera uno sintético
│   ├── train_pipeline.py      <- undersampling + GridSearchCV RandomForest -> artifacts/
│   ├── validate_model.py      <- quality gate (recall >= umbral)
│   └── manage_versions.py     <- Model Registry (Staging/Production)
├── tests/                     <- pytest (unitarios + smoke)
│   ├── test_data.py  test_model.py  test_pipeline.py
│   └── smoke/test_smoke.py    <- verifica el stack ya levantado
├── data/
│   └── Dataset_Renovacion_prestamo.csv   <- dataset real
├── .dockerignore
├── .env.example                <- plantilla de variables de entorno requeridas
├── .gitignore                  <- excluye .coverage, pycache, .ipynb_checkpoints, etc.
├── deploy.sh                   <- despliegue + rollback
├── docker-compose.preprod.yml  <- stack de 3 servicios (mlflow, trainer, api)
├── Dockerfile                  <- imagen de la API
├── Dockerfile.trainer          <- imagen del trainer
├── Makefile                    <- atajos (make preprod-up, make smoke...)
├── README.md                   <- documentación del proyecto
└── requirements.txt            <- dependencias de Python
```

## Diferencias intencionales frente al notebook

El notebook es exploratorio; este repo lo convierte en un pipeline
reproducible y apto para servir en producción. Cambios deliberados:

1. **Imputación determinista.** El notebook imputaba 3 variables
   (`Uso_TrimLinea_LOG`, `Uso_Linea_LOG`, `Meses_oferta`) con un muestreo
   aleatorio uniforme. En este repo se reemplaza por la **media de train**
   guardada en `artifacts/preprocess.json`, para que un mismo cliente reciba
   siempre el mismo score en `/predecir` (determinismo, ver
   `tests/smoke/test_smoke.py::test_api_prediccion_es_determinista`).
2. **Preprocesamiento reutilizable.** `src/preprocessing.py` centraliza
   rename, capping, log1p, imputación, one-hot y el feature `Cluster`
   (K-Means K=3), y lo usan tanto `train_pipeline.py` (fit) como
   `api/predictor.py` (transform de un registro nuevo), garantizando que
   entrenamiento e inferencia apliquen exactamente la misma lógica.
3. **Modelo productivo:** Random Forest + GridSearchCV + undersampling,
   optimizando F1 de la clase positiva (la versión más completa que se
   validó en el notebook). Si prefieres SMOTE en vez de undersampling,
   el cambio se hace en `src/train_pipeline.py::undersample()`.
4. **Grid reducido** en `PARAM_GRID` (vs. las 27 combinaciones del
   notebook) para que el trainer termine en minutos dentro del contenedor;
   la métrica de selección (F1 de clase positiva) es la misma.

## Métricas actuales (dataset real, 87,556 filas)

Con el grid y el umbral definidos en este repo:

| Métrica          | Valor  |
|-------------------|--------|
| Recall (clase 1)  | 0.61   |
| Precision (clase 1)| 0.06  |
| F1 (clase 1)      | 0.11   |
| Accuracy          | 0.62   |
| Umbral quality gate (recall) | >= 0.55 |

El dataset tiene solo ~4% de clientes que renuevan, por lo que, como ya
identificaste en el EDA del notebook, ningún modelo logra recall alto y
precisión alta simultáneamente con las features actuales. El quality gate
está calibrado sobre el recall porque el caso de negocio prioriza no perder
clientes con propensión real a renovar.

## Cómo correr localmente (sin Docker)

```bash
pip install -r requirements.txt

# Entrenar (usa data/Dataset_Renovacion_prestamo.csv si existe)
python src/train_pipeline.py

# Quality gate
python src/validate_model.py

# Levantar la API
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Probar
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predecir -H "Content-Type: application/json" -d @ejemplo_cliente.json
```

## Cómo correr el stack completo (Docker Compose)

```bash
make preprod-up        # mlflow -> trainer -> api
make preprod-ps        # estado de los 3 servicios
make preprod-logs      # logs en vivo
make smoke             # smoke tests contra el stack levantado
make preprod-down      # bajar el stack y borrar volúmenes
```

En GitHub Codespaces, los puertos 8000 (API) y 5000 (MLflow) se reenvían
automáticamente en la pestaña **PORTS**.

## Despliegue con rollback

```bash
bash deploy.sh v1.0.0
# o: make deploy VERSION=v1.0.0
```

`deploy.sh` construye las imágenes, levanta el stack, corre los smoke
tests y, si fallan, hace rollback automático a la versión anterior.

## Gestión de versiones del modelo (MLflow Model Registry)

```bash
docker compose -f docker-compose.preprod.yml up -d mlflow
python src/manage_versions.py
# o: make versions
```

## Preparado para GCP

El mismo `Dockerfile` / `Dockerfile.trainer` que se construyen aquí se
publican tal cual en **Artifact Registry** y se despliegan en **Cloud
Run** o **GKE**, sin tocar el código. En un despliegue real, los secretos
(`SECRET_KEY`, credenciales) se inyectan vía **Secret Manager**, nunca se
suben al repositorio.