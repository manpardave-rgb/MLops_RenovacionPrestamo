"""src/monitoring/04_pipeline_monitoreo.py — Monitoreo automatizado con alertas.

Ejecuta el ciclo completo de monitoreo del modelo de Renovación de Préstamo:
  1. Carga referencia y producción (ya con predicciones, ver script 02).
  2. Detecta data drift con EvidentlyAI sobre las features crudas.
  3. Mide degradación de performance (F1 decay) contra el baseline.
  4. Genera alertas si se superan los umbrales configurados en config.py.
  5. Guarda un resumen JSON + HTML con timestamp en reportes/.
  6. Quality gate: sys.exit(1) si el estado es CRÍTICO (para CI/CD).

Uso:
    python src/monitoring/04_pipeline_monitoreo.py                # lote actual
    python src/monitoring/04_pipeline_monitoreo.py octubre_2015   # con nombre de lote

Integración con GitHub Actions (cron semanal):
    schedule: - cron: '0 8 * * 1'  # lunes 8am UTC
    (ver .github/workflows/monitoreo_drift.yml)
"""
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
from config import (  # noqa: E402
    DRIFT_SHARE_UMBRAL, F1_DECAY_CRITICO, F1_DECAY_UMBRAL, FEATURES_MONITOR,
    MODEL_PATH, PROD_PRED_PATH, REF_PRED_PATH, REPORTS_DIR, TARGET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | MONITOR | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def ejecutar_monitoreo(nombre_lote: str = "lote_actual") -> dict:
    """Ejecuta el pipeline completo de monitoreo y retorna el resumen."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    alertas = []
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    for p in [REF_PRED_PATH, PROD_PRED_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos 1 y 2 primero.")

    df_ref = pd.read_csv(REF_PRED_PATH)
    df_prod = pd.read_csv(PROD_PRED_PATH)
    log.info("Lote '%s' | Ref: %d | Prod: %d filas", nombre_lote, len(df_ref), len(df_prod))

    # ── Data Drift ────────────────────────────────────────────────────────
    log.info("Ejecutando análisis de data drift...")
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=df_ref[FEATURES_MONITOR],
        current_data=df_prod[FEATURES_MONITOR],
    )
    html_path = REPORTS_DIR / f"{timestamp}_{nombre_lote}_drift.html"
    snapshot.save_html(str(html_path))

    resultado = snapshot.dict()
    drift_metrics = resultado["metrics"][0]["value"]
    drift_share = drift_metrics.get("share", 0.0)
    drift_n = drift_metrics.get("count", 0)
    drift_total = len(FEATURES_MONITOR)
    drift_detected = drift_share > 0

    if drift_share > DRIFT_SHARE_UMBRAL:
        alertas.append({
            "tipo": "DATA_DRIFT_CRITICO",
            "detalle": f"{drift_share * 100:.0f}% features con drift ({drift_n}/{drift_total})",
            "accion": "REENTRENAR modelo con datos nuevos",
        })
        log.warning("ALERTA DRIFT: %.0f%% features con drift", drift_share * 100)
    else:
        log.info("Drift OK: %.0f%% features con drift", drift_share * 100)

    # ── Performance Decay ────────────────────────────────────────────────
    log.info("Calculando degradación de performance...")
    if MODEL_PATH.exists() and "prediction" in df_ref.columns:
        f1_ref = f1_score(df_ref[TARGET], df_ref["prediction"], pos_label=1, zero_division=0)
        f1_prod = f1_score(df_prod[TARGET], df_prod["prediction"], pos_label=1, zero_division=0)
        decay = (f1_ref - f1_prod) / f1_ref if f1_ref > 0 else 0

        if decay > F1_DECAY_CRITICO:
            alertas.append({
                "tipo": "PERFORMANCE_CRITICA",
                "detalle": f"F1 cayó {decay * 100:.1f}%: {f1_ref:.3f} → {f1_prod:.3f}",
                "accion": "REENTRENAR URGENTE — degradación supera umbral crítico",
            })
            log.error("CRÍTICO: F1 cayó %.1f%% (%.3f → %.3f)", decay * 100, f1_ref, f1_prod)
        elif decay > F1_DECAY_UMBRAL:
            alertas.append({
                "tipo": "PERFORMANCE_DEGRADADA",
                "detalle": f"F1 cayó {decay * 100:.1f}%: {f1_ref:.3f} → {f1_prod:.3f}",
                "accion": "EVALUAR reentrenamiento — monitorear en próximos ciclos",
            })
            log.warning("ALERTA: F1 cayó %.1f%% (%.3f → %.3f)", decay * 100, f1_ref, f1_prod)
        else:
            log.info("Performance OK: F1 decay = %.1f%%", decay * 100)
    else:
        f1_ref = f1_prod = decay = 0.0
        log.warning("modelo.pkl o columna 'prediction' no encontrados — se salta performance")

    # ── Resumen ───────────────────────────────────────────────────────────
    estado = "OK"
    if any(a["tipo"].endswith("CRITICO") or a["tipo"].endswith("CRITICA") for a in alertas):
        estado = "CRITICO"
    elif alertas:
        estado = "ALERTA"

    resumen = {
        "timestamp": timestamp,
        "lote": nombre_lote,
        "drift_detectado": drift_detected,
        "drift_features": f"{drift_share * 100:.0f}%",
        "drift_n": drift_n,
        "drift_total": drift_total,
        "f1_ref": round(f1_ref, 4),
        "f1_prod": round(f1_prod, 4),
        "decay_pct": round(decay * 100, 1),
        "estado": estado,
        "alertas": alertas,
        "reporte_html": str(html_path),
    }

    json_path = REPORTS_DIR / f"{timestamp}_{nombre_lote}_resumen.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    return resumen


if __name__ == "__main__":
    nombre_lote = sys.argv[1] if len(sys.argv) > 1 else "lote_actual"

    print("\n" + "=" * 55)
    print(f" PIPELINE DE MONITOREO — Renovación de Préstamo — {nombre_lote}")
    print("=" * 55)

    resumen = ejecutar_monitoreo(nombre_lote)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    print("\n" + "=" * 55)
    print(f" ESTADO: {resumen['estado']}")
    print("=" * 55)
    if resumen["alertas"]:
        for alerta in resumen["alertas"]:
            print(f" ⚠ {alerta['tipo']}: {alerta['detalle']}")
            print(f"   → {alerta['accion']}")
    else:
        print(" ✓ Sin alertas — modelo estable en producción")

    if resumen["estado"] == "CRITICO":
        log.error("Quality gate FALLIDO — estado CRITICO detectado")
        sys.exit(1)

    print("\n✓ Paso 4 completado. Siguiente: python src/monitoring/05_visualizacion_drift.py")
