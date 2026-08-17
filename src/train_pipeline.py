"""src/train_pipeline.py — Entrena el modelo de Renovación de Préstamo.

Reproduce el flujo validado en el notebook (Caso_Renovación_de_Préstamo):
  1. Preprocesamiento (src/preprocessing.py): rename, capping, log1p,
     imputación, one-hot, feature de Cluster (K-Means K=3).
  2. Split train/test estratificado (70/30).
  3. Undersampling de la clase mayoritaria en TRAIN (FLAG_VENTA=0),
     igual que el notebook.
  4. GridSearchCV sobre RandomForestClassifier, optimizando F1 de la
     clase positiva (recall y precisión de "sí renueva" son las métricas
     que importan en este negocio: dataset con ~4% de conversión).
  5. Guarda modelo.pkl y metrics.json en artifacts/.

Uso:
    python src/generate_data.py     # solo si no existe data real
    python src/train_pipeline.py
"""
import json
import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np  # noqa: F401
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, make_scorer,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split

from generate_data import DATA_PATH
from preprocessing import fit_transform

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "modelo.pkl"
METRICS_PATH = ARTIFACTS / "metrics.json"

TEST_SIZE = 0.30
RANDOM_STATE = 42
RECALL_MIN = 0.55  # quality gate — recall mínimo de la clase "renueva" (FLAG_VENTA=1)

# Grid reducido respecto del notebook (que usaba 27 combinaciones con cv=3
# sobre ~87k filas) para que el trainer termine en minutos dentro del
# contenedor; la lógica de selección (F1 de la clase positiva) es la misma.
PARAM_GRID = {
    "n_estimators": [100, 150],
    "max_depth": [None, 10],
    "min_samples_leaf": [1, 4],
    "class_weight": ["balanced"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | TRAIN | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Carga el dataset (real si está montado, sintético si no)."""
    if not DATA_PATH.exists():
        from generate_data import generate
        log.info("Dataset no encontrado, generando uno sintético de respaldo...")
        DATA_PATH.parent.mkdir(exist_ok=True, parents=True)
        generate().to_csv(DATA_PATH, sep=";", index=False)
    return pd.read_csv(DATA_PATH, sep=";")


def undersample(X_train: pd.DataFrame, y_train: pd.Series):
    """Undersampling de la clase mayoritaria (igual que el notebook)."""
    df_train = pd.concat([X_train, y_train], axis=1)
    target = y_train.name
    count_0, count_1 = df_train[target].value_counts()
    df_0 = df_train[df_train[target] == 0].sample(count_1, random_state=RANDOM_STATE)
    df_1 = df_train[df_train[target] == 1]
    df_under = pd.concat([df_0, df_1], axis=0)
    return df_under.drop(columns=[target]), df_under[target]


def train(df: pd.DataFrame) -> dict:
    """Preprocesa, aplica undersampling, corre GridSearchCV y guarda artefactos."""
    X, y, _stats = fit_transform(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train_u, y_train_u = undersample(X_train, y_train)

    scoring = {
        "roc_auc": make_scorer(roc_auc_score),
        "recall_pos": make_scorer(recall_score, pos_label=1),
        "precision_pos": make_scorer(precision_score, pos_label=1),
        "f1_pos": make_scorer(f1_score, pos_label=1),
    }

    mlflow_uri = mlflow.get_tracking_uri()
    log.info("MLflow tracking URI: %s", mlflow_uri)
    mlflow.set_experiment("RenovacionPrestamo")

    with mlflow.start_run(run_name="rf_gridsearch_undersampling"):
        rf = RandomForestClassifier(random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=rf,
            param_grid=PARAM_GRID,
            scoring=scoring,
            refit="f1_pos",
            cv=3,
            n_jobs=-1,
            verbose=1,
        )
        log.info("Iniciando GridSearchCV sobre %d filas de entrenamiento (undersampled)...", len(X_train_u))
        grid.fit(X_train_u, y_train_u)
        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        metricas = {
            "f1": round(f1_score(y_test, y_pred, pos_label=1), 4),
            "recall": round(recall_score(y_test, y_pred, pos_label=1), 4),
            "precision": round(precision_score(y_test, y_pred, pos_label=1), 4),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_pred), 4),
            "params": grid.best_params_,
            "recall_minimo": RECALL_MIN,
            "n_features": X.shape[1],
            "n_train_rows": len(X_train_u),
        }

        log.info("F1=%.4f | Recall=%.4f | Precision=%.4f | Acc=%.4f",
                  metricas["f1"], metricas["recall"], metricas["precision"], metricas["accuracy"])
        log.info("Matriz de confusión:\n%s", confusion_matrix(y_test, y_pred))
        log.info("Reporte de clasificación:\n%s", classification_report(y_test, y_pred, digits=3))

        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({k: v for k, v in metricas.items()
                             if isinstance(v, (int, float))})
        mlflow.sklearn.log_model(best_model, artifact_path="modelo",
                                  registered_model_name="RenovacionPrestamo")

    ARTIFACTS.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metricas, f, indent=2)
    log.info("Artefactos guardados en %s", ARTIFACTS)
    return metricas


if __name__ == "__main__":
    df = load_data()
    log.info("Dataset: %d filas, %d columnas", *df.shape)
    metricas = train(df)
    print(json.dumps(metricas, indent=2))
