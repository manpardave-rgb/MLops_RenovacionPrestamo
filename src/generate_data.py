"""src/generate_data.py — Provee el dataset de "Renovación de Préstamo".

En pre-producción, este script NO inventa datos: si existe el CSV real
(montado como volumen o copiado en el build), lo usa tal cual. Si no
existe (por ejemplo, para probar el stack sin exponer datos reales),
genera un dataset SINTÉTICO con la misma estructura de columnas y una
tasa de conversión similar (~4%), para que el resto del pipeline
(entrenamiento, quality gate, API) se pueda ejercitar de punta a punta.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path("data/Dataset_Renovacion_prestamo.csv")
N = 20000
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DATA | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REGIONES = [
    "CALLAO", "CENTRO", "LIMA BALNEARIO", "LIMA CENTRO", "LIMA ESTE",
    "LIMA MODERNA", "LIMA NORTE", "LIMA PROVINCIA", "LIMA SUR", "NORTE",
    "OESTE", "ORIENTE", "SIERRA CENTRAL", "SUR",
]


def generate(n: int = N, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Genera un dataset sintético con la misma estructura que el original."""
    rng = np.random.default_rng(random_state)
    n1 = int(n * 0.04)
    n0 = n - n1

    def base(n_rows, renueva: int):
        shift = 1.0 if renueva else 0.0
        return {
            "MES": rng.choice(range(201501, 201513), n_rows),
            "CLIENTE": rng.integers(1, 10**6, n_rows),
            "LINEA_RENOVADO": rng.gamma(3, 1500 + 400 * shift, n_rows),
            "PLAZO_RENOVADO": rng.choice([6, 12, 18, 24, 36], n_rows),
            "FLAG_VENTA": renueva,
            "USO_LINEA_TOTAL_TC_T2": np.clip(rng.normal(0.35 + 0.15 * shift, 0.2, n_rows), 0, 1),
            "USO_TRIM_LINEA_BBVA": np.clip(rng.normal(0.30 + 0.15 * shift, 0.2, n_rows), 0, 1),
            "NR_ENTIDADES_TOTAL_T2": rng.integers(1, 8, n_rows),
            "DIFF_NRO_ENTIDA_TOTALES_T2_T12": rng.integers(-3, 3, n_rows),
            "SDO_CONSUMO_T2": rng.gamma(2, 5000 + 2000 * shift, n_rows),
            "RESENCIA_OFERTA_PLD_RENOVADO": rng.choice(
                list(range(1, 24)) + [np.nan] * 5, n_rows),
            "Ahorro_Sldo_Bco_T1": rng.normal(3000, 4000, n_rows),
            "PConsumo_Sldo_Bco_T1": rng.gamma(2, 4000 + 3000 * shift, n_rows) - 500,
            "SDO_BCO_tot_sm_pasivo_Bco_6M": rng.gamma(2, 3000, n_rows),
            "EDAD": np.clip(rng.normal(38, 12, n_rows), 18, 80),
            "SEXO": rng.choice(["M", "F"], n_rows),
            "EST_CIVIL": rng.choice(["S", "C", "D", "V", "U"], n_rows),
            "ANTIGUEDAD_MES": np.clip(rng.normal(36, 24, n_rows), 1, 300),
            "REGION": rng.choice(REGIONES, n_rows),
            "FLAG_LIMA_PROVINCIA": rng.integers(0, 2, n_rows),
            "SUELDO_ESTIMADO": rng.gamma(3, 1200, n_rows),
            "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": np.clip(rng.normal(2.5, 3, n_rows), 0, None),
        }

    df0 = pd.DataFrame(base(n0, 0))
    df1 = pd.DataFrame(base(n1, 1))
    df = pd.concat([df0, df1], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


if __name__ == "__main__":
    if DATA_PATH.exists():
        log.info("Dataset ya existe en %s, no se genera uno sintético.", DATA_PATH)
    else:
        log.info("No se encontró %s. Generando dataset sintético de respaldo...", DATA_PATH)
        DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
        df = generate()
        df.to_csv(DATA_PATH, sep=";", index=False)
        log.info("Dataset sintético generado: %s | tasa FLAG_VENTA=1: %.2f%%",
                  df.shape, df.FLAG_VENTA.mean() * 100)
