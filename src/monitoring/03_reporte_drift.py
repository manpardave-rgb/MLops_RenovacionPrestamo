"""src/monitoring/03_reporte_drift.py — Reportes HTML interactivos con EvidentlyAI.

Genera 3 reportes sobre las features crudas del caso "Renovación de
Préstamo" (las mismas 19 columnas que recibe el endpoint /predecir):

  01_data_drift.html        — drift de distribución en cada feature
  02_data_quality.html      — calidad de datos: nulos, duplicados, outliers
  03_model_performance.html — degradación de F1/Recall/Precision del modelo

Prerrequisito: python src/monitoring/02_evaluar_baseline.py

Ejecutar: python src/monitoring/03_reporte_drift.py
"""
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
from config import (  # noqa: E402
    FEATURES_MONITOR, PROD_PRED_PATH, REF_PRED_PATH, REPORTS_DIR, TARGET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | DRIFT | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def cargar_datos():
    for p in [REF_PRED_PATH, PROD_PRED_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} no encontrado. Ejecuta: python src/monitoring/02_evaluar_baseline.py")
    return pd.read_csv(REF_PRED_PATH), pd.read_csv(PROD_PRED_PATH)


def generar_reporte_drift(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> dict:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=df_ref[FEATURES_MONITOR],
        current_data=df_prod[FEATURES_MONITOR],
    )
    snapshot.save_html(str(REPORTS_DIR / "01_data_drift.html"))
    log.info("Reporte HTML: reportes/01_data_drift.html")

    resultado = snapshot.dict()
    resumen = resultado["metrics"][0]["value"]
    drift_info = {
        "drift_detectado": resumen.get("count", 0) > 0,
        "features_con_drift": int(resumen.get("count", 0)),
        "share_drifted": round(resumen.get("share", 0.0), 4),
        "total_features": len(FEATURES_MONITOR),
    }
    log.info("Drift detectado: %s | Features con drift: %d/%d (%.0f%%)",
             drift_info["drift_detectado"], drift_info["features_con_drift"],
             drift_info["total_features"], drift_info["share_drifted"] * 100)
    return drift_info


def generar_reporte_calidad(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> None:
    from evidently import Report
    from evidently.presets import DataSummaryPreset

    report = Report(metrics=[DataSummaryPreset()])
    snapshot = report.run(
        reference_data=df_ref[FEATURES_MONITOR],
        current_data=df_prod[FEATURES_MONITOR],
    )
    snapshot.save_html(str(REPORTS_DIR / "02_data_quality.html"))
    log.info("Reporte HTML: reportes/02_data_quality.html")


def generar_reporte_performance(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> None:
    from evidently import Report, Dataset
    from evidently.presets import ClassificationPreset
    from evidently.core.datasets import DataDefinition, BinaryClassification

    cols = FEATURES_MONITOR + [TARGET, "prediction"]
    df_ref_eval = df_ref[cols].rename(columns={TARGET: "target"}).copy()
    df_prod_eval = df_prod[cols].rename(columns={TARGET: "target"}).copy()
    df_ref_eval["target"] = df_ref_eval["target"].astype(int)
    df_prod_eval["target"] = df_prod_eval["target"].astype(int)
    df_ref_eval["prediction"] = df_ref_eval["prediction"].astype(int)
    df_prod_eval["prediction"] = df_prod_eval["prediction"].astype(int)

    data_definition = DataDefinition(
        classification=[BinaryClassification(target="target", prediction_labels="prediction")]
    )
    ref_dataset = Dataset.from_pandas(df_ref_eval, data_definition=data_definition)
    prod_dataset = Dataset.from_pandas(df_prod_eval, data_definition=data_definition)

    report = Report(metrics=[ClassificationPreset()])
    snapshot = report.run(reference_data=ref_dataset, current_data=prod_dataset)
    snapshot.save_html(str(REPORTS_DIR / "03_model_performance.html"))
    log.info("Reporte HTML: reportes/03_model_performance.html")


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    df_ref, df_prod = cargar_datos()
    log.info("Datos cargados — Ref: %d | Prod: %d filas", len(df_ref), len(df_prod))

    print("\n[1/3] Generando reporte de Data Drift...")
    drift_info = generar_reporte_drift(df_ref, df_prod)

    print("\n[2/3] Generando reporte de Data Quality...")
    generar_reporte_calidad(df_ref, df_prod)

    print("\n[3/3] Generando reporte de Model Performance...")
    generar_reporte_performance(df_ref, df_prod)

    resumen_path = REPORTS_DIR / "drift_summary.json"
    with open(resumen_path, "w") as f:
        json.dump(drift_info, f, indent=2)

    print("\n" + "=" * 50)
    print(" REPORTES GENERADOS")
    print("=" * 50)
    print(" reportes/01_data_drift.html")
    print(" reportes/02_data_quality.html")
    print(" reportes/03_model_performance.html")
    print(" reportes/drift_summary.json")
    print(f"\n Drift detectado   : {drift_info['drift_detectado']}")
    print(f" Features con drift: {drift_info['features_con_drift']}/{drift_info['total_features']} "
          f"({drift_info['share_drifted'] * 100:.0f}%)")
    print("\n✓ Paso 3 completado. Siguiente: python src/monitoring/04_pipeline_monitoreo.py")
