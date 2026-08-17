"""
app.py — API FastAPI para predicción de ataque cardiaco.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import cargar_artifact


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | API | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger("API")


MODEL_PATH = Path("artifacts/model.joblib")

app = FastAPI(
    title="API Riesgo Ataque Cardiaco",
    version="1.0.0",
    description="Modelo ML para predicción de riesgo de ataque cardiaco"
)

artifact = None


class PacienteInput(BaseModel):

    Genero: str = Field(
        ...,
        examples=["Hombre"]
    )

    Edad: float = Field(
        ...,
        examples=[67]
    )

    Flag_hipertension: int = Field(
        ...,
        examples=[1]
    )

    Flag_problem_cardiaco: int = Field(
        ...,
        examples=[0]
    )

    Estados_civil: str = Field(
        ...,
        examples=["Si"]
    )

    Tipo_trabajo: str = Field(
        ...,
        examples=["Empresa_privada"]
    )

    Zona_residencia: str = Field(
        ...,
        examples=["Urbano"]
    )

    Promedio_nivel_glucosa: float = Field(
        ...,
        examples=[180.5]
    )

    IMC: float | None = Field(
        None,
        examples=[28.2]
    )

    Flag_fumador: str | None = Field(
        None,
        examples=["Nunca_fuma"]
    )


@app.on_event("startup")
def startup_event():

    """
    Carga artifact al iniciar la API.
    """

    global artifact

    log.info('Inicializando API')

    if not MODEL_PATH.exists():

        log.error('Modelo no encontrado en: %s',MODEL_PATH)

        return

    log.info('Cargando artifact del modelo')
    artifact = cargar_artifact(MODEL_PATH)
    log.info('Modelo cargado correctamente')


@app.get("/")
def healthcheck():

    """
    Endpoint de healthcheck.
    """

    log.info(
        'Healthcheck ejecutado'
    )

    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        "mensaje": "CI/CD Funcionando correctamente"
    }


@app.post("/predict")
def predict(
    paciente: PacienteInput
):

    """
    Endpoint de predicción.
    """

    log.info(
        'Nueva solicitud de predicción recibida'
    )

    if artifact is None:

        log.error(
            'Modelo no disponible'
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo no encontrado. "
                "Ejecuta main.py antes de desplegar."
            )
        )

    try:

        df_raw = pd.DataFrame([
            paciente.model_dump()
        ])

        log.info(
            'Aplicando feature engineering'
        )

        fe = artifact[
            "feature_engineer"
        ]

        selected_features = artifact[
            "selected_features"
        ]

        model = artifact[
            "model"
        ]

        threshold = artifact.get(
            "threshold",
            0.5
        )

        df_features = fe.transform(
            df_raw
        )

        X = df_features[
            selected_features
        ]

        log.info(
            'Generando predicción'
        )

        proba = float(
            model.predict_proba(X)[:, 1][0]
        )

        pred = int(
            proba > threshold
        )

        resultado = {
            "prediccion_ataque_cardiaco": pred,
            "probabilidad_ataque_cardiaco": round(
                proba,
                4
            ),
            "threshold": threshold,
            "features_usadas": selected_features,
        }

        log.info(
            'Predicción generada correctamente'
        )

        return resultado

    except Exception as e:

        log.exception(
            'Error durante predicción'
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )