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
│       └── monitoreo_drift.yml         
├── api/                       <- servicio FastAPI
│   ├── app.py                 <- endpoints /, /health, /predecir
│   ├── predictor.py           <- carga modelo.pkl y predice (singleton)
│   ├── schemas.py              <- modelos Pydantic (entrada/salida)
│   └── __init__.py
├── notebooks/                 <- entregables de ciencia de datos
│   └── Caso Renovación de Préstamo.ipynb <- modelo y lógica base provista por el Data Scientist
├── src/
│   ├── preprocessing.py       <- feature engineering (rename, log1p, imputación, one-hot, Cluster K-Means)
│   ├── generate_data.py       <- usa el CSV real si existe; si no, genera uno sintético
│   ├── train_pipeline.py      <- undersampling + GridSearchCV RandomForest -> artifacts/
│   ├── validate_model.py      <- quality gate (recall >= umbral)
│   ├── manage_versions.py     <- Model Registry (Staging/Production)
│   └── monitoring/            <- monitoreo de drift en producción (EvidentlyAI)
│       ├── config.py                  <- rutas, features monitoreadas, umbrales de alerta
│       ├── 01_preparar_datos.py       <- separa referencia (pasado) vs producción (lote reciente) por MES
│       ├── 02_evaluar_baseline.py     <- predice con modelo.pkl y mide F1 en ambos lotes
│       ├── 03_reporte_drift.py        <- 3 reportes HTML EvidentlyAI (drift, calidad, performance)
│       ├── 04_pipeline_monitoreo.py   <- alertas + quality gate (exit 1 si CRÍTICO)
│       ├── 05_visualizacion_drift.py  <- histogramas KS + gráfico PSI
│       └── run_monitoreo.py           <- orquestador: ejecuta los 5 pasos
├── tests/                     <- pytest (unitarios + smoke)
│   ├── test_data.py  test_model.py  test_pipeline.py  test_monitoreo.py
│   └── smoke/test_smoke.py    <- verifica el stack ya levantado
├── data/
│   └── Dataset_Renovacion_prestamo.csv   <- dataset real (no se sube a Docker, ver .dockerignore)
├── reportes/                   <- reportes HTML/PNG/JSON del monitoreo de drift (generados)
├── Dockerfile                 <- imagen de la API
├── Dockerfile.trainer         <- imagen del trainer
├── docker-compose.preprod.yml <- stack de 3 servicios (mlflow, trainer, api)
├── deploy.sh                  <- despliegue + rollback
├── Makefile                   <- atajos (make preprod-up, make smoke, make monitoreo...)
├── .env.example / .env.preprod
├── .dockerignore / .gitignore
└── requirements.txt
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

## Monitoreo de drift en producción (EvidentlyAI)

> **Nota de dependencias:** el monitoreo usa EvidentlyAI, que vive en
> `requirements-monitoring.txt` (no en `requirements.txt`) para que las
> imágenes Docker de la API y del trainer sigan livianas — no lo necesitan
> en runtime. Instálalo con `make install-monitoring` antes de correr lo
> de abajo, o `pip install -r requirements-monitoring.txt`.

El modelo se entrena con datos de `2015-01` a `2015-06`. Para vigilar que
siga siendo confiable cuando llegan clientes de meses más recientes, el
módulo `src/monitoring/` compara ese periodo de **referencia** contra los
meses `2015-07` a `2015-09` como lote de **producción** (corte real por
`MES`, no drift simulado), y mide dos cosas:

1. **Data drift**: ¿cambió la distribución de las variables de entrada
   (uso de línea, saldo, edad, región, etc.) respecto a con qué se
   entrenó el modelo? (EvidentlyAI + PSI + test de Kolmogórov-Smirnov)
2. **Degradación de performance**: ¿cayó el F1 del modelo en el lote
   reciente respecto al F1 con el que fue validado?

```bash
# Pipeline completo (5 pasos) — requiere haber corrido `make train` antes,
# o lo entrena automáticamente si no encuentra artifacts/modelo.pkl
make monitoreo
# o con nombre de lote: make monitoreo LOTE=2015_Q3

# Pasos individuales
make monitor-paso1   # separa referencia vs producción por MES
make monitor-paso2   # evalúa el modelo ya entrenado en ambos lotes
make monitor-paso3   # genera reportes HTML interactivos (EvidentlyAI)
make monitor-paso4   # alertas + quality gate (falla si el estado es CRÍTICO)
make monitor-paso5   # gráficos comparativos KS + PSI
```

Salidas en `reportes/`:

| Archivo | Contenido |
|---|---|
| `01_data_drift.html` | Drift por feature (interactivo) |
| `02_data_quality.html` | Nulos, duplicados, outliers |
| `03_model_performance.html` | Degradación de F1/Recall/Precision |
| `04_distribuciones_comparativas.png` | Histogramas referencia vs producción con KS/PSI |
| `05_psi_barras.png` | PSI por feature con umbrales de alerta/crítico |
| `<timestamp>_<lote>_resumen.json` | Resumen de alertas para integraciones/CI |

**Umbrales de alerta** (`src/monitoring/config.py`): más de 30% de
features con drift, o una caída de F1 mayor a 10% (alerta) / 15%
(crítico, dispara `sys.exit(1)` para detener el pipeline de CI/CD).

**Automatización**: `.github/workflows/monitoreo_drift.yml` corre este
pipeline automáticamente cada **lunes a las 8am UTC**, además de en cada
push que toque `src/monitoring/` y de forma manual desde la pestaña
*Actions* de GitHub. Los reportes quedan disponibles como artefactos
descargables del workflow (30 días de retención).

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