"""tests/test_data.py — Tests unitarios del dataset y la carga de datos."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generate_data import generate  # noqa: E402


def test_generate_shape():
    """El dataset sintético debe tener las 22 columnas esperadas."""
    df = generate(n=1000)
    assert df.shape[0] == 1000
    assert "FLAG_VENTA" in df.columns


def test_generate_target_is_binary():
    """FLAG_VENTA solo debe tomar valores 0 y 1."""
    df = generate(n=1000)
    assert set(df["FLAG_VENTA"].unique()) <= {0, 1}


def test_generate_es_desbalanceado():
    """El dataset debe simular el desbalance real (~4% de clase positiva)."""
    df = generate(n=5000)
    tasa = df["FLAG_VENTA"].mean()
    assert 0.01 < tasa < 0.10


def test_generate_columnas_categoricas_validas():
    """SEXO y EST_CIVIL deben tener categorías válidas."""
    df = generate(n=500)
    assert set(df["SEXO"].unique()) <= {"M", "F"}
    assert df["REGION"].notna().all()
