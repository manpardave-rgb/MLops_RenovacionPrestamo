"""src/monitoring/run_monitoreo.py — Orquestador del pipeline de monitoreo.

Ejecuta en secuencia los 5 pasos del monitoreo de drift para el modelo de
Renovación de Préstamo. Si no encuentra un modelo ya entrenado
(artifacts/modelo.pkl), corre primero `src/train_pipeline.py` para poder
medir performance real, no solo drift de datos.

Uso desde la terminal:
    python src/monitoring/run_monitoreo.py                  # lote con nombre por defecto
    python src/monitoring/run_monitoreo.py octubre_2015      # con nombre de lote específico

Equivale a ejecutar en secuencia:
    python src/monitoring/01_preparar_datos.py
    python src/monitoring/02_evaluar_baseline.py
    python src/monitoring/03_reporte_drift.py
    python src/monitoring/04_pipeline_monitoreo.py <lote>
    python src/monitoring/05_visualizacion_drift.py
"""
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
from config import ARTIFACTS_DIR, DATA_DIR, MODEL_PATH, REPORTS_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | ORQUESTADOR | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MONITORING_DIR = Path(__file__).resolve().parent
ROOT_DIR = MONITORING_DIR.parent.parent

SCRIPTS = [
    ("Paso 1: Separar referencia vs producción (corte temporal)",
     MONITORING_DIR / "01_preparar_datos.py"),
    ("Paso 2: Evaluar baseline con el modelo ya entrenado",
     MONITORING_DIR / "02_evaluar_baseline.py"),
    ("Paso 3: Generar reportes HTML EvidentlyAI", MONITORING_DIR / "03_reporte_drift.py"),
    ("Paso 4: Pipeline de monitoreo con alertas (quality gate)",
     MONITORING_DIR / "04_pipeline_monitoreo.py"),
    ("Paso 5: Visualizaciones KS + PSI", MONITORING_DIR / "05_visualizacion_drift.py"),
]


def ejecutar(nombre: str, script: Path, extra_args: list = None) -> bool:
    inicio = time.time()
    log.info(">>> %s", nombre)
    cmd = [sys.executable, str(script)] + (extra_args or [])
    r = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=False)
    dur = round(time.time() - inicio, 2)
    if r.returncode == 0:
        log.info("<<< OK: %s (%.2f s)", nombre, dur)
        return True
    log.error("XXX FALLO: %s (código: %d)", nombre, r.returncode)
    return False


if __name__ == "__main__":
    nombre_lote = sys.argv[1] if len(sys.argv) > 1 else "lote_actual"

    log.info("=" * 55)
    log.info(" MONITOREO DE DRIFT — Renovación de Préstamo | %s", nombre_lote)
    log.info("=" * 55)

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    if not MODEL_PATH.exists():
        log.warning(
            "No se encontró %s. Entrenando el modelo primero (src/train_pipeline.py)...",
            MODEL_PATH)
        ok = ejecutar(
            "Paso 0: Entrenar modelo (prerrequisito)",
            ROOT_DIR / "src" / "train_pipeline.py")
        if not ok:
            log.error("No se pudo entrenar el modelo. Pipeline de monitoreo detenido.")
            sys.exit(1)

    resumen = []
    for nombre, script in SCRIPTS:
        extra = [nombre_lote] if "04_pipeline_monitoreo" in script.name else []
        ok = ejecutar(nombre, script, extra)
        resumen.append((nombre, ok))
        if not ok:
            log.error("Pipeline detenido. Corrige el error y reintenta.")
            sys.exit(1)

    log.info("=" * 55)
    log.info(" MONITOREO COMPLETADO")
    log.info("=" * 55)
    for nombre, ok in resumen:
        log.info("  [%s] %s", "OK" if ok else "XX", nombre)

    log.info("")
    log.info("  Reportes HTML  : reportes/01_data_drift.html")
    log.info("                   reportes/02_data_quality.html")
    log.info("                   reportes/03_model_performance.html")
    log.info("  Visualizaciones: reportes/04_distribuciones_comparativas.png")
    log.info("                   reportes/05_psi_barras.png")
    log.info("  Resumen JSON   : reportes/<timestamp>_<lote>_resumen.json")
