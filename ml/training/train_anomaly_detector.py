#!/usr/bin/env python3
"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
train_anomaly_detector.py – Train Isolation Forest anomaly detector.

Trains on normal (scenario='normal') samples only; evaluates on injected
anomaly samples (scenario='anomaly_attack') to report precision/recall/F1.

Features: 8 network-focused metrics that deviate measurably under attack.

Usage (from project root):
    python ml/training/train_anomaly_detector.py \
        --data data/processed/features.csv \
        --model ml/models/anomaly_detector.pkl
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# 8 network metric features used for anomaly detection
FEATURE_COLS: list[str] = [
    "urllc_latency_ms",    # raw URLLC latency (spikes under attack)
    "urllc_jitter_ms",     # derived: rolling std of urllc_latency_ms
    "urllc_loss_rate",     # packet loss (elevated under attack)
    "embb_latency_ms",     # eMBB latency (cross-slice side effects)
    "embb_loss_rate",      # eMBB packet loss (derived proxy)
    "mmtc_latency_ms",     # mMTC latency proxy
    "urllc_utilization",   # URLLC slice utilisation
    "embb_utilization",    # eMBB slice utilisation
]

# Isolation Forest hyperparameters
IF_N_ESTIMATORS  = 200
IF_CONTAMINATION = 0.02   # ~2% of training data expected to be noisy edge cases
IF_RANDOM_STATE  = 42
IF_MAX_SAMPLES   = "auto"


def _derive_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Derive additional anomaly-detection features not present in the raw CSV.

    Adds:
      urllc_jitter_ms  – rolling std of urllc_latency_ms (per ev_id)
      embb_loss_rate   – proxy via embb_utilization-based heuristic
      mmtc_latency_ms  – proxy derived from mmtc_utilization
    """
    out_parts: list[pd.DataFrame] = []

    for ev_id, grp in df.groupby("ev_id", sort=False):
        g = grp.sort_values("timestamp").copy()

        # URLLC jitter: rolling standard deviation of raw latency
        if "urllc_latency_ms" in g.columns:
            g["urllc_jitter_ms"] = (
                g["urllc_latency_ms"]
                .rolling(window=window, min_periods=3)
                .std()
                .fillna(0.0)
            )
        else:
            g["urllc_jitter_ms"] = 0.0

        # eMBB packet loss proxy: higher utilisation → higher loss estimate
        # Under attack, eMBB utilisation spikes erratically
        if "embb_utilization" in g.columns:
            g["embb_loss_rate"] = (g["embb_utilization"].clip(0.0, 1.0) ** 2) * 0.02
        else:
            g["embb_loss_rate"] = 0.0

        # mMTC latency proxy: rough inverse of mmtc_utilization slack
        if "mmtc_utilization" in g.columns:
            # At 0% util → ~5 ms; at 100% util → ~50 ms
            g["mmtc_latency_ms"] = 5.0 + g["mmtc_utilization"].clip(0.0, 1.0) * 45.0
        else:
            g["mmtc_latency_ms"] = 10.0

        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def load_and_split(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the processed CSV and split into three subsets:

    * ``train_df``  – normal scenario rows only (Isolation Forest training)
    * ``eval_normal`` – held-out normal rows (negative class in evaluation)
    * ``eval_anomaly`` – anomaly_attack scenario rows (positive class)

    Returns
    -------
    (train_df, eval_normal, eval_anomaly)
    """
    print(f"Loading data from {path} …")
    df = pd.read_csv(path)

    # Derive jitter / proxy features
    df = _derive_features(df)

    # Fill any missing feature columns
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"  Warning: feature '{col}' missing – filled with 0")
            df[col] = 0.0

    normal_mask  = df["scenario"] == "normal"
    anomaly_mask = df["scenario"] == "anomaly_attack"

    normal_df  = df[normal_mask].copy()
    anomaly_df = df[anomaly_mask].copy()

    # 80% normal for training, 20% for evaluation
    cutoff = int(len(normal_df) * 0.80)
    normal_df = normal_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df    = normal_df.iloc[:cutoff]
    eval_normal = normal_df.iloc[cutoff:]

    print(f"  Normal rows (train):  {len(train_df):,}")
    print(f"  Normal rows (eval):   {len(eval_normal):,}")
    print(f"  Anomaly rows (eval):  {len(anomaly_df):,}")

    return train_df, eval_normal, anomaly_df


def train(
    train_df: pd.DataFrame,
) -> tuple[IsolationForest, StandardScaler]:
    """
    Fit IsolationForest on the training set (normal scenarios only).

    Also fits a StandardScaler so raw scores can be normalised to [0, 1]
    consistently at inference time.

    Returns
    -------
    (model, scaler)
    """
    X_train = train_df[FEATURE_COLS].astype(np.float32)
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"\nFitting StandardScaler on {len(X_train):,} normal samples …")
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(X_train)

    print(f"Training IsolationForest (n_estimators={IF_N_ESTIMATORS}, contamination={IF_CONTAMINATION}) …")
    t0    = time.perf_counter()
    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        max_samples=IF_MAX_SAMPLES,
        random_state=IF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)
    print(f"  Training completed in {time.perf_counter() - t0:.2f}s")

    return model, scaler


def evaluate(
    model: IsolationForest,
    scaler: StandardScaler,
    eval_normal: pd.DataFrame,
    eval_anomaly: pd.DataFrame,
) -> dict:
    """
    Evaluate the detector on a balanced normal + anomaly set.

    IsolationForest convention: ``predict`` returns +1 for normal, -1 for anomaly.
    We convert to binary labels: 0=normal, 1=anomaly.

    Parameters
    ----------
    model:
        Fitted IsolationForest.
    scaler:
        Fitted StandardScaler (applied before prediction).
    eval_normal:
        Hold-out normal samples.
    eval_anomaly:
        Anomaly_attack samples.

    Returns
    -------
    dict with precision, recall, f1, auc_approx.
    """
    def _prepare(df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURE_COLS].astype(np.float32)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return scaler.transform(X)

    X_normal  = _prepare(eval_normal)
    X_anomaly = _prepare(eval_anomaly)

    # Downsample normal eval to match anomaly count (balanced evaluation)
    n_anomaly = len(X_anomaly)
    if len(X_normal) > n_anomaly:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_normal), size=n_anomaly, replace=False)
        X_normal = X_normal[idx]

    X_eval    = np.vstack([X_normal, X_anomaly])
    y_true    = np.array([0] * len(X_normal) + [1] * len(X_anomaly))

    # IsolationForest: +1 = normal (inlier), -1 = anomaly (outlier)
    if_preds  = model.predict(X_eval)
    y_pred    = (if_preds == -1).astype(int)   # 1 = anomaly detected

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print("\n--- Anomaly Detection Evaluation ---")
    print(f"  Eval samples – Normal: {len(X_normal):,}  |  Anomaly: {len(X_anomaly):,}")
    print(classification_report(
        y_true, y_pred,
        target_names=["Normal", "Anomaly"],
        digits=4,
    ))
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # Score distribution analysis
    scores_normal  = -model.score_samples(X_normal)   # higher = more anomalous
    scores_anomaly = -model.score_samples(X_anomaly)
    print(f"\n  Anomaly score (normal)  – mean: {scores_normal.mean():.4f}  std: {scores_normal.std():.4f}")
    print(f"  Anomaly score (anomaly) – mean: {scores_anomaly.mean():.4f}  std: {scores_anomaly.std():.4f}")

    # Inference latency
    print("\n--- Inference Latency Benchmark (10,000 single-sample) ---")
    sample = X_eval[[0]]
    lats: list[float] = []
    for _ in range(10_000):
        t0 = time.perf_counter()
        model.predict(sample)
        lats.append((time.perf_counter() - t0) * 1e6)

    p50 = float(np.percentile(lats, 50))
    p95 = float(np.percentile(lats, 95))
    p99 = float(np.percentile(lats, 99))
    print(f"  p50: {p50:.1f} µs  |  p95: {p95:.1f} µs  |  p99: {p99:.1f} µs")

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "latency_p50_us": p50,
        "latency_p95_us": p95,
        "latency_p99_us": p99,
    }


def save_model(
    model: IsolationForest,
    scaler: StandardScaler,
    path: str,
) -> None:
    """Persist (model, scaler) bundle and companion JSON metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save both model and scaler together as a dict for easy loading
    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, out_path, compress=3)
    print(f"\nModel bundle saved to {out_path}")

    feat_path = out_path.with_suffix(".features.json")
    with open(feat_path, "w") as fh:
        json.dump(
            {
                "feature_cols": FEATURE_COLS,
                "n_estimators": IF_N_ESTIMATORS,
                "contamination": IF_CONTAMINATION,
                "train_scenario": "normal",
                "eval_scenario": "anomaly_attack",
                "if_convention": "+1=normal, -1=anomaly",
            },
            fh,
            indent=2,
        )
    print(f"Feature metadata saved to {feat_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train VESPER Isolation Forest anomaly detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",  default="data/processed/features.csv",     help="Processed CSV")
    parser.add_argument("--model", default="ml/models/anomaly_detector.pkl",  help="Output pkl path")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    data_path    = project_root / args.data
    model_path   = project_root / args.model

    train_df, eval_normal, eval_anomaly = load_and_split(str(data_path))

    model, scaler = train(train_df)

    metrics = evaluate(model, scaler, eval_normal, eval_anomaly)
    save_model(model, scaler, str(model_path))

    print("\n--- Training Summary ---")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Latency p99: {metrics['latency_p99_us']:.1f} µs")


if __name__ == "__main__":
    main()
