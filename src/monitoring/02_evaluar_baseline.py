"""src/monitoring/02_evaluar_baseline.py — Mide degradación de performance.

Usa el MISMO modelo.pkl y el MISMO preprocesamiento (artifacts/preprocess.json,
src/preprocessing.py) que ya sirven en producción vía api/predictor.py — no
entrena un modelo nuevo. Genera predicciones sobre REFERENCIA y PRODUCCIÓN y
compara el F1 de la clase positiva (FLAG_VENTA=1) para medir cuánto se
degrada el modelo en el lote más reciente.

Prerrequisitos:
  - python src/train_pipeline.py           (genera artifacts/modelo.pkl)
  - python src/monitoring/01_preparar_datos.py

Ejecutar: python src/monitoring/02_evaluar_baseline.py
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
from config import (  # noqa: E402
    BASELINE_PATH, MODEL_PATH, PROD_PATH, PROD_PRED_PATH, REF_PATH,
    REF_PRED_PATH, TARGET,
)
from preprocessing import transform_df, load_stats  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | BASELINE | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

UMBRAL = 0.50  # mismo umbral que api/predictor.py


def calcular_metricas(y_true, y_pred, nombre: str) -> dict:
    m = {
        "f1": round(f1_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
    }
    log.info("[%s] F1=%.4f | Recall=%.4f | Precision=%.4f | Acc=%.4f",
             nombre, m["f1"], m["recall"], m["precision"], m["accuracy"])
    return m


def predecir_lote(df: pd.DataFrame, modelo, stats: dict) -> pd.DataFrame:
    """Aplica el preprocesamiento de producción y predice un lote completo."""
    X = transform_df(df, stats)
    proba = modelo.predict_proba(X)[:, 1]
    df = df.copy().reset_index(drop=True)
    df["prediction_proba"] = proba
    df["prediction"] = (proba >= UMBRAL).astype(int)
    return df


if __name__ == "__main__":
    for p in [REF_PATH, PROD_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} no encontrado. Ejecuta: python src/monitoring/01_preparar_datos.py")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} no encontrado. Ejecuta: python src/train_pipeline.py")

    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)
    stats = load_stats()
    log.info("Modelo cargado: %s", type(modelo).__name__)

    df_ref = pd.read_csv(REF_PATH)
    df_prod = pd.read_csv(PROD_PATH)
    log.info("Referencia: %d filas | Producción: %d filas", len(df_ref), len(df_prod))

    df_ref_pred = predecir_lote(df_ref, modelo, stats)
    df_prod_pred = predecir_lote(df_prod, modelo, stats)

    m_ref = calcular_metricas(df_ref_pred[TARGET], df_ref_pred["prediction"], "REFERENCIA")
    m_prod = calcular_metricas(df_prod_pred[TARGET], df_prod_pred["prediction"], "PRODUCCIÓN")
    decay = (m_ref["f1"] - m_prod["f1"]) / m_ref["f1"] * 100 if m_ref["f1"] > 0 else 0.0

    print("\n" + "=" * 50)
    print(" DEGRADACIÓN DE PERFORMANCE — Renovación de Préstamo")
    print("=" * 50)
    print(f" F1 Referencia : {m_ref['f1']:.4f}")
    print(f" F1 Producción : {m_prod['f1']:.4f}")
    print(f" Degradación   : {decay:.1f}%")
    estado = "⚠ ALERTA" if decay > 10 else "✓ OK"
    print(f" Estado        : {estado}")

    baseline = {
        "referencia": m_ref,
        "produccion": m_prod,
        "decay_pct": round(decay, 1),
        "estado": "ALERTA" if decay > 10 else "OK",
        "f1_umbral_alerta": 0.10,
        "f1_umbral_reentrenar": 0.15,
    }
    ARTIFACTS_DIR = MODEL_PATH.parent
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    log.info("Baseline guardado: %s", BASELINE_PATH)

    df_ref_pred.to_csv(REF_PRED_PATH, index=False)
    df_prod_pred.to_csv(PROD_PRED_PATH, index=False)
    log.info(
        "Datasets con predicciones guardados para EvidentlyAI (%s, %s)",
        REF_PRED_PATH,
        PROD_PRED_PATH)

    print("\n✓ Paso 2 completado. Siguiente: python src/monitoring/03_reporte_drift.py")
