"""src/monitoring/01_preparar_datos.py — Prepara REFERENCIA vs PRODUCCIÓN.

A diferencia de un laboratorio con drift simulado artificialmente, aquí se
usa un corte TEMPORAL real sobre el dataset de Renovación de Préstamo:

  - REFERENCIA : clientes de los meses más antiguos (MES <= MES_CORTE).
                 Es el mismo universo de datos con el que se entrena/valida
                 el modelo en `src/train_pipeline.py`.
  - PRODUCCIÓN : clientes de los meses más recientes (MES > MES_CORTE).
                 Simula el "lote nuevo" que llegaría en producción y que
                 hay que monitorear en busca de drift.

Si `data/Dataset_Renovacion_prestamo.csv` no existe, reutiliza el mismo
generador sintético que usa el resto del proyecto (src/generate_data.py)
para que el pipeline de monitoreo se pueda ejercitar de punta a punta sin
el dataset real.

Ejecutar: python src/monitoring/01_preparar_datos.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
from config import DATASET_PATH, MES_CORTE, REF_PATH, PROD_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | DATOS | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def cargar_dataset() -> pd.DataFrame:
    """Carga el CSV real; si no existe, genera uno sintético de respaldo."""
    if DATASET_PATH.exists():
        log.info("Usando dataset real: %s", DATASET_PATH)
        return pd.read_csv(DATASET_PATH, sep=";")

    log.warning("No se encontró %s. Generando dataset sintético de respaldo...", DATASET_PATH)
    from generate_data import generate
    DATASET_PATH.parent.mkdir(exist_ok=True, parents=True)
    df = generate()
    df.to_csv(DATASET_PATH, sep=";", index=False)
    return df


def separar_referencia_produccion(df: pd.DataFrame, mes_corte: int = MES_CORTE):
    """Separa el dataset en referencia (pasado) y producción (lote reciente)."""
    df_ref = df[df["MES"] <= mes_corte].reset_index(drop=True)
    df_prod = df[df["MES"] > mes_corte].reset_index(drop=True)
    return df_ref, df_prod


if __name__ == "__main__":
    REF_PATH.parent.mkdir(exist_ok=True, parents=True)

    df = cargar_dataset()
    log.info("Dataset: %d filas, %d columnas | meses: %s a %s",
             *df.shape, df["MES"].min(), df["MES"].max())

    df_ref, df_prod = separar_referencia_produccion(df)

    if len(df_prod) == 0:
        log.warning(
            "No hay filas con MES > %d. Usando el 25%% más reciente como producción.",
            MES_CORTE,
        )
        df_sorted = df.sort_values("MES")
        corte_idx = int(len(df_sorted) * 0.75)
        df_ref = df_sorted.iloc[:corte_idx].reset_index(drop=True)
        df_prod = df_sorted.iloc[corte_idx:].reset_index(drop=True)

    df_ref.to_csv(REF_PATH, index=False)
    df_prod.to_csv(PROD_PATH, index=False)

    log.info(
        "Referencia : %s filas (meses %s-%s) | tasa FLAG_VENTA=1: %.2f%%",
        len(df_ref), df_ref["MES"].min(), df_ref["MES"].max(),
        df_ref["FLAG_VENTA"].mean() * 100,
    )
    log.info(
        "Producción : %s filas (meses %s-%s) | tasa FLAG_VENTA=1: %.2f%%",
        len(df_prod), df_prod["MES"].min(), df_prod["MES"].max(),
        df_prod["FLAG_VENTA"].mean() * 100,
    )

    print("\nDatasets guardados:")
    print(f"  Referencia : {REF_PATH}  ({len(df_ref)} filas)")
    print(f"  Producción : {PROD_PATH} ({len(df_prod)} filas)")
    print("\n✓ Paso 1 completado. Siguiente: python src/monitoring/02_evaluar_baseline.py")
