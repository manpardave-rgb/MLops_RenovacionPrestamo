"""src/monitoring/config.py — Configuración compartida del monitoreo de drift.

Centraliza rutas, columnas monitoreadas y umbrales de alerta para que los
5 scripts del pipeline de monitoreo (01..05) y `run_monitoreo.py` usen
exactamente los mismos valores.
"""
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # raíz del repo
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reportes"

DATASET_PATH = DATA_DIR / "Dataset_Renovacion_prestamo.csv"
REF_PATH = DATA_DIR / "monitor_referencia.csv"
PROD_PATH = DATA_DIR / "monitor_produccion.csv"
REF_PRED_PATH = DATA_DIR / "monitor_ref_con_pred.csv"
PROD_PRED_PATH = DATA_DIR / "monitor_prod_con_pred.csv"

MODEL_PATH = ARTIFACTS_DIR / "modelo.pkl"
BASELINE_PATH = ARTIFACTS_DIR / "monitor_baseline.json"

# ── Columna target y corte temporal ─────────────────────────────────────────
TARGET = "FLAG_VENTA"

# El dataset trae la columna MES (formato AAAAMM). En vez de simular drift
# artificialmente, usamos un corte temporal real: los meses más antiguos son
# la REFERENCIA (con la que se entrenó/valida el modelo) y los meses más
# recientes son la "PRODUCCIÓN" (el lote nuevo que llega y que hay que
# monitorear). Ajusta MES_CORTE si tu dataset cubre otro rango de meses.
MES_CORTE = 201506  # <= MES_CORTE -> referencia | > MES_CORTE -> producción

# ── Columnas monitoreadas (features crudas, mismas que recibe /predecir) ────
FEATURES_MONITOR = [
    "LINEA_RENOVADO", "PLAZO_RENOVADO", "USO_LINEA_TOTAL_TC_T2",
    "USO_TRIM_LINEA_BBVA", "NR_ENTIDADES_TOTAL_T2",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12", "SDO_CONSUMO_T2",
    "RESENCIA_OFERTA_PLD_RENOVADO", "Ahorro_Sldo_Bco_T1",
    "PConsumo_Sldo_Bco_T1", "SDO_BCO_tot_sm_pasivo_Bco_6M", "EDAD",
    "SEXO", "EST_CIVIL", "ANTIGUEDAD_MES", "REGION",
    "FLAG_LIMA_PROVINCIA", "SUELDO_ESTIMADO",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD",
]

# Subconjunto numérico usado en las visualizaciones KS/PSI (script 05)
FEATURES_NUMERICAS_VIZ = [
    "USO_LINEA_TOTAL_TC_T2", "USO_TRIM_LINEA_BBVA", "SDO_CONSUMO_T2",
    "EDAD", "SUELDO_ESTIMADO", "ANTIGUEDAD_MES", "Ahorro_Sldo_Bco_T1",
    "PConsumo_Sldo_Bco_T1",
]

# ── Umbrales de alerta ───────────────────────────────────────────────────────
DRIFT_SHARE_UMBRAL = 0.30      # > 30% de features con drift -> ALERTA
F1_DECAY_UMBRAL = 0.10         # > 10% caída en F1 -> ALERTA
F1_DECAY_CRITICO = 0.15        # > 15% caída en F1 -> CRÍTICO (reentrenar)
PSI_UMBRAL_ALERTA = 0.10
PSI_UMBRAL_CRITICO = 0.25
