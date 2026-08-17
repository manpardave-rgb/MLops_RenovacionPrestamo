"""api/predictor.py — Carga el modelo.pkl y ejecuta predicciones."""
import json
import logging
import pickle
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_PATH = Path("artifacts/modelo.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

UMBRAL = 0.50  # score >= 0.50 → RENUEVA

# src/ debe estar en el path para poder importar preprocessing.transform_one
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class Predictor:
    """Singleton que carga el modelo una vez y sirve predicciones."""

    def __init__(self):
        self.modelo = None
        self.metricas = {}
        self.stats = None

    def cargar(self) -> None:
        """Carga modelo, métricas y estadísticas de preprocesamiento. Llamar UNA vez en startup."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo no encontrado: {MODEL_PATH}. "
                "Asegúrate de que el servicio trainer completó exitosamente."
            )
        with open(MODEL_PATH, "rb") as f:
            self.modelo = pickle.load(f)

        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                self.metricas = json.load(f)

        from preprocessing import load_stats
        self.stats = load_stats()

        log.info("Modelo cargado: %s", type(self.modelo).__name__)
        log.info("Recall en entrenamiento: %.4f", self.metricas.get("recall", 0))

    def predecir(self, datos: dict) -> dict:
        """Recibe dict con los campos crudos del cliente y retorna la predicción."""
        if self.modelo is None:
            raise RuntimeError("Modelo no cargado. Llama a cargar() primero.")

        from preprocessing import transform_one
        X = transform_one(datos, self.stats)

        proba = float(self.modelo.predict_proba(X)[0][1])
        return {
            "score_riesgo": round(proba, 4),
            "decision": "RENUEVA" if proba >= UMBRAL else "NO RENUEVA",
            "probabilidad_renovacion": round(proba, 4),
            "umbral_usado": UMBRAL,
            "modelo": type(self.modelo).__name__,
        }


# Instancia global — singleton
predictor = Predictor()
