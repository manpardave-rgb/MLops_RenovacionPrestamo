"""tests/test_pipeline.py — Tests unitarios del preprocesamiento.

IMPORTANTE: fit_transform() escribe artifacts/kmeans.pkl y
artifacts/preprocess.json como parte de su contrato (son necesarios para
que la API reproduzca el mismo preprocesamiento en inferencia). Por eso
TODOS los tests de este archivo corren en un directorio temporal aislado
(monkeypatch.chdir), para no sobrescribir los artefactos reales del
modelo entrenado con datos de producción.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import generate  # noqa: E402
from preprocessing import fit_transform, transform_one  # noqa: E402


@pytest.fixture(autouse=True)
def _sandbox(tmp_path, monkeypatch):
    """Aísla cada test en un directorio temporal con su propia carpeta artifacts/."""
    monkeypatch.chdir(tmp_path)  # se restaura automáticamente al terminar el test
    (tmp_path / "artifacts").mkdir()


def test_fit_transform_no_deja_nulos():
    """Tras el preprocesamiento no deben quedar valores nulos en X."""
    df = generate(n=2000)
    X, y, stats = fit_transform(df)
    assert X.isna().sum().sum() == 0


def test_fit_transform_elimina_columnas_originales_log():
    """Las columnas originales con versión _LOG no deben quedar en X."""
    df = generate(n=1000)
    X, y, stats = fit_transform(df)
    assert "Uso_Linea" not in X.columns
    assert "Uso_Linea_LOG" in X.columns


def test_fit_transform_agrega_cluster():
    """El feature 'Cluster' (K-Means) debe existir y tener 3 valores posibles."""
    df = generate(n=2000)
    X, y, stats = fit_transform(df)
    assert "Cluster" in X.columns
    assert set(X["Cluster"].unique()) <= {0, 1, 2}


def test_transform_one_mismas_columnas_que_train():
    """La transformación de un registro nuevo debe producir las mismas columnas que en train."""
    df = generate(n=2000)
    X, y, stats = fit_transform(df)

    fila = df.iloc[0].to_dict()
    X_one = transform_one(fila, stats)

    assert list(X_one.columns) == stats["feature_order"]
    assert X_one.isna().sum().sum() == 0
