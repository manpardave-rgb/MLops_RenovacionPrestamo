"""src/validate_model.py — Valida que las métricas superen el umbral mínimo.

Quality gate: compara el recall de la clase positiva (FLAG_VENTA=1,
"el cliente renueva") contra el umbral mínimo (RECALL_MIN). Si no lo
supera, sale con código 1 y el pipeline se detiene: no se construye la
imagen Docker y la API nunca arranca con un modelo que no cumple el
mínimo aceptable de negocio.
"""
import json
import sys
from pathlib import Path

METRICS_PATH = Path("artifacts/metrics.json")


def validate(metrics_path: Path = METRICS_PATH) -> None:
    """Valida métricas. Lanza SystemExit(1) si falla el quality gate."""
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} no existe. Ejecuta train_pipeline.py primero.")
        sys.exit(1)

    with open(metrics_path) as f:
        m = json.load(f)

    umbral = m.get("recall_minimo", 0.55)
    recall = m.get("recall", 0.0)
    f1 = m.get("f1", 0.0)
    precision = m.get("precision", 0.0)

    print("=" * 50)
    print(" QUALITY GATE — VALIDACIÓN DE MÉTRICAS")
    print(" Caso: Renovación de Préstamo (FLAG_VENTA)")
    print("=" * 50)
    print(f" Recall (clase 1)    : {recall:.4f} (umbral: >= {umbral})")
    print(f" Precision (clase 1) : {precision:.4f}")
    print(f" F1-Score (clase 1)  : {f1:.4f}")
    print(f" Accuracy            : {m.get('accuracy', 0):.4f}")

    if recall < umbral:
        print(f"\nFALLO: Recall {recall:.4f} < umbral {umbral}")
        print(" El pipeline CI/CD se detiene. No se construye Docker.")
        sys.exit(1)

    print(f"\nAPROBADO: Recall {recall:.4f} >= {umbral}")
    print(" Pipeline CI/CD continúa al build de Docker.")


if __name__ == "__main__":
    validate()
