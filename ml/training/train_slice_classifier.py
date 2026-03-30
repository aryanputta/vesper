#!/usr/bin/env python3
"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
train_slice_classifier.py – Train XGBoost slice assignment classifier.

Usage (from project root):
    python ml/training/train_slice_classifier.py \
        --data data/processed/features.csv \
        --model ml/models/slice_classifier.pkl
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLS: list[str] = [
    # Telemetry rolling features
    "speed_mean_5s",
    "speed_std_5s",
    "accel_mean_5s",
    "accel_max_5s",
    "brake_mean_5s",
    "brake_max_5s",
    "battery_temp_mean_5s",
    "battery_temp_slope_5s",
    "obstacle_dist_min_5s",
    "obstacle_dist_slope_5s",
    "sensor_conf_mean_5s",
    "congestion_event_count_10s",
    # Network slice features
    "urllc_latency_mean_5s",
    "urllc_latency_slope_5s",
    "urllc_utilization",
    "urllc_loss_rate",
    "embb_latency_mean_5s",
    "embb_utilization",
    "mmtc_utilization",
    # Derived / composite features
    "urgency_score",
    "retransmission_proxy",
    # Instant passthrough from raw data (used as feature)
    "emergency_flag",
]

TARGET_COL = "slice_label"
CLASS_NAMES = ["MMTC", "EMBB", "URLLC"]

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the processed feature CSV and return (X, y, full_df).

    Parameters
    ----------
    path:
        Path to the processed features CSV (output of FeatureBuilder).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix containing only FEATURE_COLS columns.
    y : pd.Series
        Integer-encoded target labels (0=MMTC, 1=EMBB, 2=URLLC).
    df : pd.DataFrame
        Full original dataframe (for debugging / extra context).
    """
    print(f"Loading data from {path} …")
    df = pd.read_csv(path)

    # Ensure required columns are present; fill missing features with 0
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} feature columns not found, filling with 0: {missing}")
        for col in missing:
            df[col] = 0.0

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in {path}")

    X = df[FEATURE_COLS].astype(np.float32)
    y = df[TARGET_COL].astype(int)

    # Replace any remaining NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"  Loaded {len(df):,} rows | Features: {X.shape[1]} | Classes: {y.value_counts().to_dict()}")
    return X, y, df


def train(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train the XGBoost slice classifier with 5-fold stratified cross-validation.

    Parameters
    ----------
    X_train:
        Training feature matrix.
    y_train:
        Training labels.

    Returns
    -------
    xgb.XGBClassifier
        Model trained on the full (X_train, y_train) set.
    """
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        objective="multi:softprob",
        num_class=3,
    )

    print("\n--- 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s: list[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

        fold_model = xgb.XGBClassifier(**params)
        fold_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = fold_model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="weighted")
        fold_f1s.append(f1)
        print(f"  Fold {fold_idx}/5 – Weighted F1: {f1:.4f}")

    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)
    print(f"  CV Weighted F1: {mean_f1:.4f} ± {std_f1:.4f}\n")

    # Train final model on full training data
    print("Training final model on full training split …")
    t0 = time.perf_counter()
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_train, y_train, verbose=False)
    print(f"  Training completed in {time.perf_counter() - t0:.2f}s")

    return final_model


def evaluate(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute comprehensive evaluation metrics and print a nicely formatted report.

    Parameters
    ----------
    model:
        Trained XGBClassifier.
    X_test:
        Test feature matrix.
    y_test:
        Ground-truth test labels.

    Returns
    -------
    dict with keys: classification_report, confusion_matrix, macro_f1,
                    weighted_f1, roc_auc, latency_p50_us, latency_p95_us,
                    latency_p99_us.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n--- Classification Report ---")
    cr = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    print(cr)

    macro_f1    = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # ROC-AUC (one-vs-rest, multiclass)
    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        print(f"ROC-AUC (OvR macro): {roc_auc:.4f}")
    except ValueError as exc:
        print(f"ROC-AUC: could not compute – {exc}")
        roc_auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix ---")
    header = f"{'':>10s}" + "".join(f"{n:>8s}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[i]:>10s}" + "".join(f"{v:>8d}" for v in row)
        print(row_str)

    print("\n--- Inference Latency Benchmark (10,000 single-sample predictions) ---")
    sample_row = X_test.iloc[[0]]   # keep as DataFrame to avoid shape issues
    n_bench    = 10_000
    latencies_us: list[float] = []

    for _ in range(n_bench):
        t0 = time.perf_counter()
        model.predict_proba(sample_row)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

    p50  = float(np.percentile(latencies_us, 50))
    p95  = float(np.percentile(latencies_us, 95))
    p99  = float(np.percentile(latencies_us, 99))
    mean = float(np.mean(latencies_us))

    print(f"  Mean:  {mean:.1f} µs")
    print(f"  p50:   {p50:.1f} µs")
    print(f"  p95:   {p95:.1f} µs")
    print(f"  p99:   {p99:.1f} µs")

    return {
        "classification_report": cr,
        "confusion_matrix": cm.tolist(),
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc":     roc_auc,
        "latency_p50_us": p50,
        "latency_p95_us": p95,
        "latency_p99_us": p99,
    }


def shap_analysis(model: xgb.XGBClassifier, X_test: pd.DataFrame) -> None:
    """
    Compute SHAP values using TreeExplainer and print the top-10 features
    by mean absolute SHAP value across all classes.

    Saves a JSON file of per-feature SHAP importance to ml/models/ for use
    by the inference API.

    Parameters
    ----------
    model:
        Trained XGBClassifier.
    X_test:
        Test set features (a random subsample is used for speed).
    """
    print("\n--- SHAP Analysis ---")

    # Use a subsample for speed (max 2000 rows)
    n_shap = min(2000, len(X_test))
    X_shap = X_test.sample(n=n_shap, random_state=42) if len(X_test) > n_shap else X_test

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)   # shape: (n_classes, n_samples, n_features)

    # Aggregate: mean |SHAP| across all classes and samples
    if isinstance(shap_values, list):
        # Older shap returns list[array] of shape (n_samples, n_features) per class
        abs_shap = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=(0, 1))
    else:
        # Newer shap returns (n_samples, n_features, n_classes) or (n_samples, n_features)
        if shap_values.ndim == 3:
            abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Build sorted feature importance dict
    feature_importance: dict[str, float] = {
        feat: float(val)
        for feat, val in zip(FEATURE_COLS, abs_shap)
    }
    sorted_imp = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)

    print(f"  (Computed over {n_shap} samples)")
    print("\n  Top 10 features by mean |SHAP|:")
    for rank, (feat, val) in enumerate(sorted_imp[:10], start=1):
        print(f"  {rank:>2d}. {feat:<35s} {val:.5f}")

    # Save SHAP importance JSON for the inference API
    shap_path = Path(__file__).parent.parent / "models" / "slice_classifier_shap.json"
    shap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shap_path, "w") as fh:
        json.dump(
            {
                "feature_importance": feature_importance,
                "sorted_importance": sorted_imp,
                "feature_cols": FEATURE_COLS,
                "n_shap_samples": n_shap,
            },
            fh,
            indent=2,
        )
    print(f"\n  SHAP importance saved to {shap_path}")


def save_model(model: xgb.XGBClassifier, path: str) -> None:
    """
    Persist the trained model to disk using joblib.

    Also saves a companion JSON file listing the feature columns so the
    inference layer can perform schema validation at load time.

    Parameters
    ----------
    model:
        Trained XGBClassifier.
    path:
        Destination .pkl path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_path, compress=3)
    print(f"\nModel saved to {out_path}")

    # Companion feature list JSON
    feat_path = out_path.with_suffix(".features.json")
    with open(feat_path, "w") as fh:
        json.dump({"feature_cols": FEATURE_COLS, "target_col": TARGET_COL, "classes": CLASS_NAMES}, fh, indent=2)
    print(f"Feature list saved to {feat_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train VESPER slice assignment classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/processed/features.csv",
        help="Path to processed features CSV",
    )
    parser.add_argument(
        "--model",
        default="ml/models/slice_classifier.pkl",
        help="Output model pickle path",
    )
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split fraction")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_path    = project_root / args.data
    model_path   = project_root / args.model

    X, y, df = load_data(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model   = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    shap_analysis(model, X_test)
    save_model(model, str(model_path))

    print("\n--- Training Summary ---")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"  ROC-AUC (OvR):   {metrics['roc_auc']:.4f}")
    print(f"  Latency p50:     {metrics['latency_p50_us']:.1f} µs")
    print(f"  Latency p99:     {metrics['latency_p99_us']:.1f} µs")


if __name__ == "__main__":
    main()
