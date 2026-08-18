"""src/monitoring/05_visualizacion_drift.py — Visualización comparativa KS + PSI.

Genera gráficos propios (matplotlib) para identificar qué variables del
caso "Renovación de Préstamo" tienen mayor drift entre referencia y
producción. Complementa los reportes interactivos de EvidentlyAI (script 03).

Ejecutar: python src/monitoring/05_visualizacion_drift.py
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/
from config import (  # noqa: E402
    FEATURES_MONITOR, FEATURES_NUMERICAS_VIZ, PROD_PRED_PATH,
    PSI_UMBRAL_ALERTA, PSI_UMBRAL_CRITICO, REF_PRED_PATH, REPORTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | VIZ | %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COLORS = {"ref": "#2196F3", "prod": "#DC2626"}


def calcular_psi(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (PSI)."""
    ref = ref.dropna()
    prod = prod.dropna()
    breakpoints = np.linspace(min(ref.min(), prod.min()), max(ref.max(), prod.max()), bins + 1)
    ref_pct = np.histogram(ref, bins=breakpoints)[0] / len(ref)
    prod_pct = np.histogram(prod, bins=breakpoints)[0] / len(prod)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)
    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def clasificar_psi(psi: float) -> tuple:
    """Clasifica PSI según umbrales estándar de la industria."""
    if psi < PSI_UMBRAL_ALERTA:
        return "ESTABLE", "#059669"
    if psi < PSI_UMBRAL_CRITICO:
        return "ALERTA", "#D97706"
    return "CRÍTICO", "#DC2626"


def grafico_distribuciones(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> str:
    """Histogramas comparativos ref vs producción (2x4 subplots)."""
    n_vars = len(FEATURES_NUMERICAS_VIZ)
    n_cols = 4
    n_rows = (n_vars + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    fig.suptitle(
        "Referencia vs Producción — Renovación de Préstamo\n"
        "Distribución de variables por corte temporal (MES)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    axes = axes.flatten()

    for i, col in enumerate(FEATURES_NUMERICAS_VIZ):
        ax = axes[i]
        ref_vals = df_ref[col].dropna()
        prod_vals = df_prod[col].dropna()

        ax.hist(ref_vals, bins=30, alpha=0.6, label="Referencia", color=COLORS["ref"], density=True)
        ax.hist(
            prod_vals,
            bins=30,
            alpha=0.6,
            label="Producción",
            color=COLORS["prod"],
            density=True)

        ks_stat, p_val = ks_2samp(ref_vals, prod_vals)
        psi = calcular_psi(ref_vals, prod_vals)
        estado_ks = "DRIFT" if p_val < 0.05 else "OK"
        estado_psi, c = clasificar_psi(psi)

        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=10.5)
        ax.set_xlabel(
            f"KS={ks_stat:.3f} p={p_val:.3f} [{estado_ks}]  |  PSI={psi:.3f} [{estado_psi}]",
            fontsize=8, color=c,
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        log.info("%-30s KS=%.3f p=%.4f PSI=%.3f [%s]", col, ks_stat, p_val, psi, estado_psi)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_path = str(REPORTS_DIR / "04_distribuciones_comparativas.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def grafico_psi_barras(df_ref: pd.DataFrame, df_prod: pd.DataFrame) -> str:
    """Gráfico de barras PSI para todas las features numéricas monitoreadas."""
    features = [
        f for f in FEATURES_MONITOR
        if pd.api.types.is_numeric_dtype(df_ref[f])
        and pd.api.types.is_numeric_dtype(df_prod[f])
    ]

    psis = [calcular_psi(df_ref[f], df_prod[f]) for f in features]
    colores = [clasificar_psi(p)[1] for p in psis]

    fig, ax = plt.subplots(figsize=(15, 5.5))
    bars = ax.bar(range(len(features)), psis, color=colores, alpha=0.85, edgecolor="white")

    for bar, v in zip(bars, psis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=7.5)
    ax.axhline(y=PSI_UMBRAL_ALERTA, color="#D97706", linestyle="--", alpha=0.7,
               label=f"PSI={PSI_UMBRAL_ALERTA} (alerta)")
    ax.axhline(y=PSI_UMBRAL_CRITICO, color="#DC2626", linestyle="--", alpha=0.7,
               label=f"PSI={PSI_UMBRAL_CRITICO} (crítico)")
    ax.set_title("Population Stability Index (PSI) — Renovación de Préstamo",
                 fontweight="bold", fontsize=12)
    ax.set_ylabel("PSI")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out_path = str(REPORTS_DIR / "05_psi_barras.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


if __name__ == "__main__":
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    for p in [REF_PRED_PATH, PROD_PRED_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos 1 y 2 primero.")

    df_ref = pd.read_csv(REF_PRED_PATH)
    df_prod = pd.read_csv(PROD_PRED_PATH)
    log.info("Datos cargados — Ref: %d | Prod: %d filas", len(df_ref), len(df_prod))

    print("\n[1/2] Generando gráfico de distribuciones comparativas...")
    path1 = grafico_distribuciones(df_ref, df_prod)
    print(f"  Guardado: {path1}")

    print("\n[2/2] Generando gráfico PSI por feature...")
    path2 = grafico_psi_barras(df_ref, df_prod)
    print(f"  Guardado: {path2}")

    print("\n" + "=" * 55)
    print(" VISUALIZACIONES GENERADAS")
    print("=" * 55)
    print(f" {path1}")
    print(f" {path2}")
    print("\n✓ Paso 5 completado — Pipeline de monitoreo finalizado")
