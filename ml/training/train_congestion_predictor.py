#!/usr/bin/env python3
"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
train_congestion_predictor.py – Train XGBoost congestion score predictor.

Predicts the network congestion score 3 seconds (30 samples @ 10 Hz) ahead.

Congestion score formula:
    congestion_score = (
        urllc_utilization * 0.5
        + embb_utilization  * 0.3
        + min(urllc_loss_rate * 100, 1.0) * 0.2
    )

Usage (from project root):
    python ml/training/train_congestion_predictor.py \
        --data data/processed/features.csv \
        --model ml/models/congestion_predictor.pkl
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
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# Prediction horizon: 3 seconds at 10 Hz = 30 samples ahead
HORIZON_SAMPLES = 30

# Input features for congestion prediction
FEATURE_COLS: list[str] = [
    # Rolling slice metrics
    "urllc_latency_mean_5s",
    "urllc_latency_slope_5s",
    "urllc_utilization",
    "urllc_loss_rate",
    "embb_latency_mean_5s",
    "embb_utilization",
    "mmtc_utilization",
    "congestion_event_count_10s",
    "retransmission_proxy",
    # EV telemetry aggregates (proxy for upcoming network load)
    "speed_mean_5s",
    "speed_std_5s",
    "accel_mean_5s",
    "accel_max_5s",
    "brake_mean_5s",
    "brake_max_5s",
    "urgency_score",
    "sensor_conf_mean_5s",
    "obstacle_dist_min_5s",
]

TARGET_COL = "congestion_score"


def compute_congestion_score(df: pd.DataFrame) -> pd.Series:
    """
    Derive a scalar congestion score in [0, 1] from slice utilisation and loss.

    Formula (from spec):
        congestion_score = (
            urllc_utilization * 0.5
            + embb_utilization  * 0.3
            + min(urllc_loss_rate * 100, 1.0) * 0.2
        )

    Parameters
    ----------
    df:
        DataFrame containing urllc_utilization, embb_utilization, urllc_loss_rate.

    Returns
    -------
    pd.Series of float32, clamped to [0, 1].
    """
    loss_component = (df["urllc_loss_rate"] * 100.0).clip(upper=1.0)
    score = (
        df["urllc_utilization"] * 0.5
        + df["embb_utilization"] * 0.3
        + loss_component * 0.2
    )
    return score.clip(0.0, 1.0).astype(np.float32)


def load_and_prepare(path: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load processed CSV, compute congestion score, and create the future-horizon target.

    Parameters
    ----------
    path:
        Path to processed features CSV.

    Returns
    -------
    X:
        Feature matrix (current-time features).
    y:
        Target congestion score HORIZON_SAMPLES steps in the future.
    df:
        Full dataframe (for reference).
    """
    print(f"Loading data from {path} …")
    df = pd.read_csv(path)

    # Fill any missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
            print(f"  Warning: feature '{col}' missing – filled with 0")

    # Compute instantaneous congestion score
    df["congestion_score"] = compute_congestion_score(df)

    # Create the future target: shift congestion_score backward by HORIZON_SAMPLES
    # (each row's target is what congestion will look like 3 s later)
    # We group by ev_id to avoid cross-EV leakage
    df["congestion_score_future"] = (
        df.groupby("ev_id")["congestion_score"]
          .shift(-HORIZON_SAMPLES)
    )

    # Drop rows where the future target is NaN (end of each EV's window)
    df_valid = df.dropna(subset=["congestion_score_future"]).copy()
    print(f"  Loaded {len(df):,} rows → {len(df_valid):,} usable after horizon shift")

    # Replace inf/nan in features
    X = df_valid[FEATURE_COLS].astype(np.float32)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_valid["congestion_score_future"].astype(np.float32)

    print(f"  Target range: [{y.min():.4f}, {y.max():.4f}]  mean={y.mean():.4f}")
    return X, y, df_valid


def train(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBRegressor:
    """
    Train XGBRegressor with same hyperparams as the classifier counterpart.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training targets (congestion scores, 3 s ahead).

    Returns
    -------
    Fitted XGBRegressor.
    """
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        objective="reg:squarederror",
    )

    print("\nTraining XGBRegressor …")
    t0    = time.perf_counter()
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    print(f"  Training completed in {time.perf_counter() - t0:.2f}s")
    return model


def evaluate(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """
    Compute MAE, RMSE, and R² on the hold-out test set.

    Also benchmarks single-sample inference latency.

    Parameters
    ----------
    model:
        Fitted XGBRegressor.
    X_test, y_test:
        Hold-out evaluation data.

    Returns
    -------
    dict of metric names → values.
    """
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)

    print("\n--- Congestion Predictor Metrics ---")
    print(f"  MAE:  {mae:.5f}")
    print(f"  RMSE: {rmse:.5f}")
    print(f"  R²:   {r2:.5f}")

    # Feature importance (gain-based)
    print("\n--- Top 10 Features by XGBoost Gain ---")
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (feat, gain) in enumerate(sorted_imp[:10], start=1):
        print(f"  {rank:>2d}. {feat:<35s} {gain:.2f}")

    # Inference latency
    print("\n--- Inference Latency Benchmark (10,000 single-sample) ---")
    sample_row    = X_test.iloc[[0]]
    latencies_us: list[float] = []
    for _ in range(10_000):
        t0 = time.perf_counter()
        model.predict(sample_row)
        latencies_us.append((time.perf_counter() - t0) * 1e6)

    p50 = float(np.percentile(latencies_us, 50))
    p95 = float(np.percentile(latencies_us, 95))
    p99 = float(np.percentile(latencies_us, 99))
    print(f"  p50: {p50:.1f} µs  |  p95: {p95:.1f} µs  |  p99: {p99:.1f} µs")

    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "latency_p50_us": p50, "latency_p95_us": p95, "latency_p99_us": p99,
    }


def save_model(model: xgb.XGBRegressor, path: str) -> None:
    """Save model and companion feature-list JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_path, compress=3)
    print(f"\nModel saved to {out_path}")

    feat_path = out_path.with_suffix(".features.json")
    with open(feat_path, "w") as fh:
        json.dump(
            {
                "feature_cols": FEATURE_COLS,
                "target_col": TARGET_COL,
                "horizon_samples": HORIZON_SAMPLES,
                "horizon_seconds": HORIZON_SAMPLES / 10.0,
                "congestion_formula": (
                    "urllc_utilization * 0.5"
                    " + embb_utilization * 0.3"
                    " + min(urllc_loss_rate * 100, 1.0) * 0.2"
                ),
            },
            fh,
            indent=2,
        )
    print(f"Feature metadata saved to {feat_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train VESPER congestion score predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",  default="data/processed/features.csv",     help="Processed CSV")
    parser.add_argument("--model", default="ml/models/congestion_predictor.pkl", help="Output pkl path")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split fraction")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    data_path    = project_root / args.data
    model_path   = project_root / args.model

    X, y, df = load_and_prepare(str(data_path))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    model   = train(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    save_model(model, str(model_path))

    print("\n--- Training Summary ---")
    print(f"  MAE:  {metrics['mae']:.5f}")
    print(f"  RMSE: {metrics['rmse']:.5f}")
    print(f"  R²:   {metrics['r2']:.5f}")
    print(f"  Latency p99: {metrics['latency_p99_us']:.1f} µs")


if __name__ == "__main__":
    main()
