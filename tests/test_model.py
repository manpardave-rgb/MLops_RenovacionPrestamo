"""tests/test_model.py — Tests unitarios del modelo entrenado.

Requiere que artifacts/modelo.pkl y artifacts/metrics.json ya existan
(ejecutar `make train` antes de correr estos tests).
"""
import json
import pickle
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MODEL_PATH = Path("artifacts/modelo.pkl")
METRICS_PATH = Path("artifacts/metrics.json")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Ejecuta 'make train' antes de correr estos tests"
)


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


def test_modelo_tiene_predict_proba():
    """El modelo debe soportar predict_proba (necesario para el score)."""
    model = _load_model()
    assert hasattr(model, "predict_proba")


def test_metricas_cumplen_quality_gate():
    """El recall registrado debe superar el umbral mínimo del quality gate."""
    metrics = _load_metrics()
    assert metrics["recall"] >= metrics["recall_minimo"]


def test_prediccion_sobre_registro_sintetico():
    """El modelo debe predecir sobre un registro nuevo usando las stats de train ya guardadas."""
    from generate_data import generate
    from preprocessing import load_stats, transform_one

    stats = load_stats()
    fila = generate(n=1).iloc[0].to_dict()
    X_one = transform_one(fila, stats)

    model = _load_model()
    proba = model.predict_proba(X_one[stats["feature_order"]])[0][1]
    assert 0.0 <= proba <= 1.0
