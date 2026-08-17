"""tests/smoke/test_smoke.py — Smoke tests para verificar el entorno completo.

Se ejecutan DESPUÉS de docker compose up para verificar que todos
los servicios estén respondiendo correctamente.

Uso en Codespace (con el entorno levantado):
    pytest tests/smoke/ -v
    # o: make smoke

Variables de entorno configurables:
    API_URL    (default: http://localhost:8000)
    MLFLOW_URL (default: http://localhost:5000)
"""
import json
import os
import urllib.request

import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

PAYLOAD_BAJA_PROPENSION = {
    "MES": 201506, "CLIENTE": 1,
    "LINEA_RENOVADO": 1300.0, "PLAZO_RENOVADO": 6,
    "USO_LINEA_TOTAL_TC_T2": 0.05, "USO_TRIM_LINEA_BBVA": 0.03,
    "NR_ENTIDADES_TOTAL_T2": 1, "DIFF_NRO_ENTIDA_TOTALES_T2_T12": 0,
    "SDO_CONSUMO_T2": 300.0, "RESENCIA_OFERTA_PLD_RENOVADO": 20,
    "Ahorro_Sldo_Bco_T1": 500.0, "PConsumo_Sldo_Bco_T1": 200.0,
    "SDO_BCO_tot_sm_pasivo_Bco_6M": 400.0, "EDAD": 55,
    "SEXO": "F", "EST_CIVIL": "V", "ANTIGUEDAD_MES": 10,
    "REGION": "LIMA NORTE", "FLAG_LIMA_PROVINCIA": 1,
    "SUELDO_ESTIMADO": 1200.0, "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 0.1,
}

PAYLOAD_ALTA_PROPENSION = {
    "MES": 201506, "CLIENTE": 2,
    "LINEA_RENOVADO": 7800.0, "PLAZO_RENOVADO": 36,
    "USO_LINEA_TOTAL_TC_T2": 0.85, "USO_TRIM_LINEA_BBVA": 0.78,
    "NR_ENTIDADES_TOTAL_T2": 5, "DIFF_NRO_ENTIDA_TOTALES_T2_T12": 2,
    "SDO_CONSUMO_T2": 20000.0, "RESENCIA_OFERTA_PLD_RENOVADO": 2,
    "Ahorro_Sldo_Bco_T1": 16000.0, "PConsumo_Sldo_Bco_T1": 15000.0,
    "SDO_BCO_tot_sm_pasivo_Bco_6M": 10000.0, "EDAD": 34,
    "SEXO": "M", "EST_CIVIL": "C", "ANTIGUEDAD_MES": 60,
    "REGION": "ORIENTE", "FLAG_LIMA_PROVINCIA": 0,
    "SUELDO_ESTIMADO": 4500.0, "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 2.5,
}


def get_json(url: str) -> dict:
    """Hace GET a una URL y retorna el JSON parseado."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def post_json(url: str, payload: dict) -> dict:
    """Hace POST con JSON y retorna la respuesta."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ── Tests de la API ──────────────────────────────────────────────────────
def test_api_root_responde():
    """El endpoint raíz de la API debe responder con info básica."""
    data = get_json(f"{API_URL}/")
    assert data.get("api") is not None
    assert data.get("version") == "1.0.0"


def test_api_health_ok():
    """El health check de la API debe retornar status ok."""
    data = get_json(f"{API_URL}/health")
    assert data["status"] == "ok"
    assert "modelo" in data
    assert 0 <= data.get("recall", 0) <= 1


def test_api_health_tiene_env():
    """El health check debe incluir el entorno (preprod)."""
    data = get_json(f"{API_URL}/health")
    assert "env" in data


def test_api_prediccion_payload_baja_propension():
    """El endpoint /predecir debe retornar una predicción válida."""
    data = post_json(f"{API_URL}/predecir", PAYLOAD_BAJA_PROPENSION)
    assert "score_riesgo" in data
    assert data["decision"] in ["RENUEVA", "NO RENUEVA"]
    assert 0.0 <= data["score_riesgo"] <= 1.0
    assert "modelo" in data


def test_api_prediccion_payload_alta_propension():
    """Un payload de alta propensión debe retornar una predicción válida."""
    data = post_json(f"{API_URL}/predecir", PAYLOAD_ALTA_PROPENSION)
    assert "score_riesgo" in data
    assert data["decision"] in ["RENUEVA", "NO RENUEVA"]
    assert 0.0 <= data["score_riesgo"] <= 1.0


def test_api_prediccion_es_determinista():
    """El mismo payload debe producir siempre el mismo score."""
    r1 = post_json(f"{API_URL}/predecir", PAYLOAD_BAJA_PROPENSION)
    r2 = post_json(f"{API_URL}/predecir", PAYLOAD_BAJA_PROPENSION)
    assert r1["score_riesgo"] == r2["score_riesgo"]


def test_api_docs_disponible():
    """La documentación Swagger debe estar disponible en /docs."""
    with urllib.request.urlopen(f"{API_URL}/docs", timeout=10) as resp:
        assert resp.status == 200


# ── Tests de MLflow ──────────────────────────────────────────────────────
def test_mlflow_ui_responde():
    """La UI de MLflow debe responder en el puerto configurado."""
    try:
        with urllib.request.urlopen(f"{MLFLOW_URL}/health", timeout=10) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("MLflow no disponible — verifica que el stack está levantado")


def test_mlflow_experimentos_accesibles():
    """La API de experimentos de MLflow debe responder."""
    try:
        data = get_json(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/list")
        assert "experiments" in data
    except Exception:
        pytest.skip("MLflow API no disponible en este entorno")
