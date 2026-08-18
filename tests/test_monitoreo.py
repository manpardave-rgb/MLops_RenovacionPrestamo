"""tests/test_monitoreo.py — Tests unitarios del pipeline de monitoreo de drift."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MONITORING = Path(__file__).parent.parent / "src" / "monitoring"
sys.path.insert(0, str(MONITORING.parent))  # src/
sys.path.insert(0, str(MONITORING))         # src/monitoring/


def _load(filename, alias):
    """Carga un módulo cuyo nombre empieza con dígito usando importlib."""
    spec = importlib.util.spec_from_file_location(alias, MONITORING / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_datos = _load("01_preparar_datos.py", "preparar_datos")
_viz = _load("05_visualizacion_drift.py", "visualizacion_drift")
calcular_psi = _viz.calcular_psi
clasificar_psi = _viz.clasificar_psi


# ── Tests de separación referencia/producción ────────────────────────────────

def test_separar_por_mes():
    """separar_referencia_produccion debe respetar el corte de MES."""
    df = pd.DataFrame({
        "MES": [201501, 201502, 201507, 201509],
        "FLAG_VENTA": [0, 1, 0, 1],
    })
    df_ref, df_prod = _datos.separar_referencia_produccion(df, mes_corte=201506)
    assert len(df_ref) == 2
    assert len(df_prod) == 2
    assert df_ref["MES"].max() <= 201506
    assert df_prod["MES"].min() > 201506


def test_separar_produccion_vacia_no_rompe():
    """Si no hay filas después del corte, debe retornar producción vacía (no error)."""
    df = pd.DataFrame({"MES": [201501, 201502], "FLAG_VENTA": [0, 1]})
    df_ref, df_prod = _datos.separar_referencia_produccion(df, mes_corte=201506)
    assert len(df_ref) == 2
    assert len(df_prod) == 0


# ── Tests de métricas PSI ─────────────────────────────────────────────────────

def test_psi_misma_distribucion():
    """PSI de dos distribuciones idénticas debe ser ~0."""
    serie = pd.Series(np.random.default_rng(0).normal(10, 1, 500))
    psi = calcular_psi(serie, serie)
    assert psi < 0.05, f"PSI esperado ~0, obtenido {psi}"


def test_psi_distribuciones_distintas():
    """PSI de distribuciones muy distintas debe ser > 0.25 (crítico)."""
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(10, 1, 500))
    prod = pd.Series(rng.normal(14, 1, 500))
    psi = calcular_psi(ref, prod)
    assert psi > 0.25, f"PSI esperado > 0.25, obtenido {psi}"


def test_psi_clasificacion():
    """clasificar_psi debe retornar el estado correcto según los umbrales."""
    assert clasificar_psi(0.05)[0] == "ESTABLE"
    assert clasificar_psi(0.15)[0] == "ALERTA"
    assert clasificar_psi(0.30)[0] == "CRÍTICO"


# ── Test de integración del pipeline de monitoreo (con modelo dummy) ────────

def test_pipeline_monitoreo_genera_resumen(tmp_path, monkeypatch):
    """ejecutar_monitoreo debe generar un resumen con las claves esperadas."""
    for d in ("data", "reportes", "artifacts"):
        (tmp_path / d).mkdir()

    rng = np.random.default_rng(42)
    n = 300

    def lote(n_rows):
        return pd.DataFrame({
            "LINEA_RENOVADO": rng.gamma(3, 1500, n_rows),
            "PLAZO_RENOVADO": rng.choice([6, 12, 24], n_rows),
            "USO_LINEA_TOTAL_TC_T2": rng.uniform(0, 1, n_rows),
            "USO_TRIM_LINEA_BBVA": rng.uniform(0, 1, n_rows),
            "NR_ENTIDADES_TOTAL_T2": rng.integers(1, 8, n_rows),
            "DIFF_NRO_ENTIDA_TOTALES_T2_T12": rng.integers(-3, 3, n_rows),
            "SDO_CONSUMO_T2": rng.gamma(2, 5000, n_rows),
            "RESENCIA_OFERTA_PLD_RENOVADO": rng.integers(1, 24, n_rows),
            "Ahorro_Sldo_Bco_T1": rng.normal(3000, 4000, n_rows),
            "PConsumo_Sldo_Bco_T1": rng.gamma(2, 4000, n_rows),
            "SDO_BCO_tot_sm_pasivo_Bco_6M": rng.gamma(2, 3000, n_rows),
            "EDAD": rng.normal(38, 12, n_rows).clip(18, 80),
            "SEXO": rng.choice(["M", "F"], n_rows),
            "EST_CIVIL": rng.choice(["S", "C"], n_rows),
            "ANTIGUEDAD_MES": rng.normal(36, 24, n_rows).clip(1, 300),
            "REGION": rng.choice(["LIMA CENTRO", "NORTE"], n_rows),
            "FLAG_LIMA_PROVINCIA": rng.integers(0, 2, n_rows),
            "SUELDO_ESTIMADO": rng.gamma(3, 1200, n_rows),
            "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": rng.uniform(0, 3, n_rows),
            "FLAG_VENTA": rng.choice([0, 1], n_rows, p=[0.96, 0.04]),
        })

    df_ref = lote(n)
    df_prod = lote(n)
    # Predicciones simuladas (no requiere modelo.pkl real para este test)
    df_ref["prediction"] = df_ref["FLAG_VENTA"]
    df_prod["prediction"] = df_prod["FLAG_VENTA"]
    df_ref["prediction_proba"] = df_ref["FLAG_VENTA"].astype(float)
    df_prod["prediction_proba"] = df_prod["FLAG_VENTA"].astype(float)

    monkeypatch.chdir(tmp_path)
    df_ref.to_csv(tmp_path / "data" / "monitor_ref_con_pred.csv", index=False)
    df_prod.to_csv(tmp_path / "data" / "monitor_prod_con_pred.csv", index=False)

    # Reapuntar las rutas del módulo de config al tmp_path
    _pipeline = _load("04_pipeline_monitoreo.py", "pipeline_monitoreo")
    _pipeline.REF_PRED_PATH = tmp_path / "data" / "monitor_ref_con_pred.csv"
    _pipeline.PROD_PRED_PATH = tmp_path / "data" / "monitor_prod_con_pred.csv"
    _pipeline.REPORTS_DIR = tmp_path / "reportes"
    _pipeline.MODEL_PATH = tmp_path / "artifacts" / "modelo.pkl"  # no existe -> salta performance

    resumen = _pipeline.ejecutar_monitoreo("test_lote")

    assert "estado" in resumen
    assert "drift_detectado" in resumen
    assert "alertas" in resumen
    assert "timestamp" in resumen
    assert resumen["estado"] in ("OK", "ALERTA", "CRITICO")
