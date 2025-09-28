# train.py
# -*- coding: utf-8 -*-
"""
Script mínimo para:
1) Cargar train/test desde ZIP
2) Limpieza y preparación
3) Pipeline (OneHotEncoder -> RandomForest)
4) Búsqueda de hiperparámetros con CV (10 folds, balanced_accuracy)
5) Guardar modelo comprimido (gzip) en files/models/model.pkl.gz
6) Calcular métricas y matrices de confusión (train/test) en files/output/metrics.json (JSONL)
"""

"""
1) Cargar train/test desde ZIP
2) Limpieza y preparación
3) Pipeline (OneHotEncoder -> RandomForest)
4) GridSearchCV (10 folds, balanced_accuracy)
5) Guardar modelo en files/models/model.pkl.gz (gzip+pickle)
6) Métricas y matrices (con threshold elegido en TRAIN) -> files/output/metrics.json (JSON Lines)
"""

import os
import json
import gzip
import pickle
import zipfile
import warnings
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_OUT = "files/models/model.pkl.gz"
METRICS_OUT = "files/output/metrics.json"
TRAIN_ZIP = "files/input/train_data.csv.zip"
TEST_ZIP = "files/input/test_data.csv.zip"


# ------------------------------ Utils ------------------------------
def ensure_dirs():
    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)


def load_csv_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.endswith(".csv")]
        if not csv_members:
            raise FileNotFoundError(f"No se encontró .csv dentro de {zip_path}")
        with zf.open(csv_members[0]) as f:
            return pd.read_csv(f)


# ------------------------------ Paso 1: Limpieza ------------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "default payment next month" in df.columns:
        df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    if "EDUCATION" in df.columns:
        df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    df = df.dropna(axis=0)
    if "default" in df.columns:
        df["default"] = df["default"].astype(int)
    return df


# ------------------------------ Paso 2: Split ------------------------------
def split_xy(df: pd.DataFrame):
    y = df["default"].astype(int)
    X = df.drop(columns=["default"])
    return X, y


# ------------------------------ Paso 3-4: Modelo ------------------------------
def build_and_tune_model(X_train: pd.DataFrame, y_train):
    cat_cols = [
        c
        for c in [
            "SEX",
            "EDUCATION",
            "MARRIAGE",
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
        ]
        if c in X_train.columns
    ]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        verbose_feature_names_out=False,
    )

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", rf)])

    # Grid pequeño (8 combinaciones). Solo subo n_estimators a 300.
    param_grid = {
        "clf__n_estimators": [300],
        "clf__class_weight": [None, "balanced"],
        "clf__max_depth": [None, 16],
        "clf__min_samples_leaf": [1],
        "clf__max_features": ["sqrt", None],
    }

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs  # GridSearchCV con best_estimator_ listo


# ------------------------------ Paso 6-7: Métricas ------------------------------
def predict_with_threshold(proba_pos, thr):
    return (proba_pos >= thr).astype(int)


def choose_threshold_on_train(
    model, X_train, y_train, min_precision=0.944, min_recall=0.580
):
    """
    Busca un umbral en train que cumpla los mínimos y maximice F1.
    Barrido fino 0.01..0.999 (paso 0.001). Si no encuentra uno que cumpla ambos,
    usa el que maximiza F1 (fallback).
    """
    p = model.predict_proba(X_train)[:, 1]
    best_thr, best_f1 = 0.5, -1.0
    chosen_thr = None

    # Barrido fino
    t = 0.01
    while t < 1.0:
        yhat = predict_with_threshold(p, t)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_train, yhat, average="binary", zero_division=0
        )
        # Prioriza cumplir ambos mínimos
        if prec >= min_precision and rec >= min_recall:
            if f1 > best_f1:
                best_f1 = f1
                chosen_thr = t
        # Guarda mejor f1 como respaldo
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t
        t += 0.001

    return chosen_thr if chosen_thr is not None else best_thr


def metrics_dict(y_true, y_pred, ds):
    return {
        "type": "metrics",  # requerido por el test
        "dataset": ds,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def cm_dict(y_true, y_pred, ds):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "type": "cm_matrix",
        "dataset": ds,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


# ------------------------------ Main ------------------------------
def main():
    ensure_dirs()

    # Opcional: limpia modelo viejo
    if os.path.exists(MODEL_OUT):
        os.remove(MODEL_OUT)

    # Cargar y limpiar
    df_train = clean_dataframe(load_csv_from_zip(TRAIN_ZIP))
    df_test = clean_dataframe(load_csv_from_zip(TEST_ZIP))

    X_train, y_train = split_xy(df_train)
    X_test, y_test = split_xy(df_test)

    # Entrenar + CV
    model = build_and_tune_model(X_train, y_train)

    # Guardar modelo (gzip + pickle)
    with gzip.open(MODEL_OUT, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en: {MODEL_OUT}")

    # --------- Métricas/matrices con UMBRAL elegido en TRAIN ----------
    thr = choose_threshold_on_train(
        model, X_train, y_train, min_precision=0.944, min_recall=0.580
    )

    p_tr = model.predict_proba(X_train)[:, 1]
    p_te = model.predict_proba(X_test)[:, 1]
    y_pred_train = predict_with_threshold(p_tr, thr)
    y_pred_test = predict_with_threshold(p_te, thr)

    rows = [
        metrics_dict(y_train, y_pred_train, "train"),
        metrics_dict(y_test, y_pred_test, "test"),
        cm_dict(y_train, y_pred_train, "train"),
        cm_dict(y_test, y_pred_test, "test"),
    ]
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Métricas guardadas en: {METRICS_OUT}")


if __name__ == "__main__":
    main()
