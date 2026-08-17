"""src/preprocessing.py — Preprocesamiento del caso "Renovación de Préstamo".

Reproduce, de forma determinista y reutilizable, la limpieza y feature
engineering desarrollada en el notebook original (renombrado de columnas,
capping de negativos, transformación logarítmica, imputación, one-hot
encoding y clustering K-Means) para que TRAIN e INFERENCIA usen exactamente
la misma lógica.

Diferencia deliberada frente al notebook: en el notebook, 3 variables
(Uso_TrimLinea_LOG, Uso_Linea_LOG, Meses_oferta) se imputaban con un
muestreo aleatorio uniforme (media ± desviación). Eso no es reproducible
en producción (un mismo cliente podría recibir un score distinto en cada
llamada). Aquí se reemplaza por imputación determinista con la MEDIA
calculada en el set de entrenamiento y guardada en artifacts/preprocess.json.
El resto de la lógica (capping, log1p, imputación por mediana/moda,
one-hot, K-Means con K=3 sobre las 3 variables más correlacionadas) es
fiel al notebook.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")
PREPROCESS_PATH = ARTIFACTS / "preprocess.json"
KMEANS_PATH = ARTIFACTS / "kmeans.pkl"

# Columna target y columnas identificadoras (no predictivas)
TARGET = "FLAG_VENTA"
ID_COLS = ["MES", "CLIENTE"]

RENAME_MAP = {
    "LINEA_RENOVADO": "Linea_Renovado",
    "PLAZO_RENOVADO": "Plazo_Renovado",
    "USO_LINEA_TOTAL_TC_T2": "Uso_Linea",
    "USO_TRIM_LINEA_BBVA": "Uso_TrimLinea",
    "NR_ENTIDADES_TOTAL_T2": "Nro_Entidades",
    "DIFF_NRO_ENTIDA_TOTALES_T2_T12": "Dif_Entidades",
    "SDO_CONSUMO_T2": "Saldo_Consumo",
    "RESENCIA_OFERTA_PLD_RENOVADO": "Meses_oferta",
    "Ahorro_Sldo_Bco_T1": "Ahorro",
    "PConsumo_Sldo_Bco_T1": "Prestamo_vigente",
    "SDO_BCO_tot_sm_pasivo_Bco_6M": "Promed_6Mdeuda",
    "FLAG_LIMA_PROVINCIA": "Flag_LimProv",
    "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": "Deuda_Cubierta%",
}

# Variables con capping de negativos -> 0 antes del log1p
CAPPING_COLS = ["Ahorro", "Prestamo_vigente", "Promed_6Mdeuda"]

# Variables transformadas con log1p (mismo listado que el notebook)
LOG_TRANSFORM_COLS = [
    "Uso_Linea", "Uso_TrimLinea", "Saldo_Consumo", "SUELDO_ESTIMADO",
    "ANTIGUEDAD_MES", "Linea_Renovado", "Ahorro", "Prestamo_vigente",
    "Promed_6Mdeuda", "Deuda_Cubierta%",
]

# Imputación determinista con la MEDIA de train (reemplaza el sampling
# aleatorio del notebook, ver docstring del módulo)
MEAN_IMPUTE_COLS = ["Uso_TrimLinea_LOG", "Uso_Linea_LOG", "Meses_oferta"]

# Imputación con la MEDIANA de train (igual que el notebook)
MEDIAN_IMPUTE_COLS = [
    "Saldo_Consumo_LOG", "SUELDO_ESTIMADO_LOG", "ANTIGUEDAD_MES_LOG", "EDAD",
]

CATEGORICAL_COLS = ["REGION", "SEXO", "EST_CIVIL"]

# Columnas originales que se eliminan una vez que existe su versión _LOG
ORIGINAL_COLS_TO_DROP = [
    "Uso_Linea", "Uso_TrimLinea", "Saldo_Consumo", "SUELDO_ESTIMADO",
    "ANTIGUEDAD_MES", "Linea_Renovado", "Ahorro", "Prestamo_vigente",
    "Promed_6Mdeuda", "Deuda_Cubierta%",
]

# Variables usadas para el clustering K-Means (las 3 con mayor correlación
# con FLAG_VENTA según el análisis del notebook)
CLUSTERING_FEATURES = ["Uso_TrimLinea_LOG", "Prestamo_vigente_LOG", "Uso_Linea_LOG"]
N_CLUSTERS = 3
RANDOM_STATE = 42


def _rename_and_cap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP)
    for col in CAPPING_COLS:
        if col in df.columns:
            df[col] = np.maximum(0, df[col])
    return df


def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
    for col in LOG_TRANSFORM_COLS:
        if col in df.columns:
            df[f"{col}_LOG"] = np.log1p(df[col])
    return df


def fit_transform(df_raw: pd.DataFrame) -> "tuple[pd.DataFrame, pd.DataFrame, dict]":
    """Ajusta el preprocesamiento sobre datos de ENTRENAMIENTO.

    Retorna (X, y, stats) donde `stats` contiene todo lo necesario para
    reproducir la misma transformación en inferencia (medias, medianas,
    modas, columnas one-hot y el modelo KMeans ya entrenado).
    """
    df = df_raw.copy()
    df = _rename_and_cap(df)
    df = _log_transform(df)

    stats: dict = {"mean_impute": {}, "median_impute": {}, "mode_impute": {}}

    for col in MEAN_IMPUTE_COLS:
        mean_val = float(df[col].mean())
        stats["mean_impute"][col] = mean_val
        df[col] = df[col].fillna(mean_val)

    for col in MEDIAN_IMPUTE_COLS:
        median_val = float(df[col].median())
        stats["median_impute"][col] = median_val
        df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        mode_val = df[col].mode()[0]
        stats["mode_impute"][col] = mode_val
        df[col] = df[col].fillna(mode_val)

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False, dtype=int)
    onehot_cols = sorted(c for c in df_encoded.columns
                          if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_COLS))
    stats["onehot_columns"] = onehot_cols

    X_cluster = df_encoded[CLUSTERING_FEATURES].fillna(df_encoded[CLUSTERING_FEATURES].median())
    kmeans = KMeans(n_clusters=N_CLUSTERS, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X_cluster)
    df_encoded["Cluster"] = kmeans.labels_

    existing_drop = [c for c in ORIGINAL_COLS_TO_DROP if c in df_encoded.columns]
    df_final = df_encoded.drop(columns=existing_drop)

    y = df_final[TARGET]
    drop_cols = [TARGET] + [c for c in ID_COLS if c in df_final.columns]
    X = df_final.drop(columns=drop_cols)

    stats["feature_order"] = X.columns.tolist()

    ARTIFACTS.mkdir(exist_ok=True)
    import pickle
    with open(KMEANS_PATH, "wb") as f:
        pickle.dump(kmeans, f)
    with open(PREPROCESS_PATH, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info("Preprocesamiento ajustado: %d features finales", len(stats["feature_order"]))

    return X, y, stats


def load_stats() -> dict:
    with open(PREPROCESS_PATH) as f:
        return json.load(f)


def transform_one(datos: dict, stats: Optional[dict] = None) -> pd.DataFrame:
    """Aplica el MISMO preprocesamiento a un registro nuevo (inferencia).

    `datos` debe traer las columnas originales del dataset crudo (antes del
    renombrado), con los mismos nombres que produce la API/el cliente.
    """
    import pickle

    if stats is None:
        stats = load_stats()

    df = pd.DataFrame([datos])
    df = _rename_and_cap(df)
    df = _log_transform(df)

    for col, mean_val in stats["mean_impute"].items():
        if col in df.columns:
            df[col] = df[col].fillna(mean_val)
    for col, median_val in stats["median_impute"].items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    for col, mode_val in stats["mode_impute"].items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)

    df_encoded = pd.get_dummies(df, columns=[c for c in CATEGORICAL_COLS if c in df.columns],
                                 drop_first=False, dtype=int)

    # Asegurar que existan TODAS las columnas one-hot vistas en train
    for col in stats["onehot_columns"]:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    with open(KMEANS_PATH, "rb") as f:
        kmeans = pickle.load(f)
    X_cluster = df_encoded[CLUSTERING_FEATURES]
    df_encoded["Cluster"] = kmeans.predict(X_cluster)

    existing_drop = [c for c in ORIGINAL_COLS_TO_DROP if c in df_encoded.columns]
    df_final = df_encoded.drop(columns=existing_drop)

    # Reordenar/completar columnas exactamente como en entrenamiento
    for col in stats["feature_order"]:
        if col not in df_final.columns:
            df_final[col] = 0
    X = df_final[stats["feature_order"]]
    return X
