"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
feature_builder.py – Offline feature engineering from raw telemetry CSV.

Reads the raw simulator output CSV, groups by ev_id, computes rolling
statistics and derived features, and saves the processed dataset ready
for model training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Rolling window sizes in samples (10 Hz sampling rate)
WINDOW_2S  =  20   # 2-second window
WINDOW_5S  =  50   # 5-second window
WINDOW_10S = 100   # 10-second window

# Minimum number of valid samples required to compute a window stat;
# windows with fewer samples are filled with NaN and then forward-filled.
MIN_PERIODS = 5

# Integer encoding for the three slice classes
SLICE_LABEL_MAP = {"MMTC": 0, "EMBB": 1, "URLLC": 2}
SLICE_INT_TO_STR = {v: k for k, v in SLICE_LABEL_MAP.items()}

# Congestion event threshold
CONGESTION_UTIL_THRESHOLD = 0.80


def _rolling_slope(series: pd.Series, window: int, min_periods: int = MIN_PERIODS) -> pd.Series:
    """
    Compute the slope of a linear fit over a rolling window.

    Uses numpy.polyfit(degree=1) on each window.  Returns NaN when the
    window does not contain enough valid (non-NaN) observations.

    Parameters
    ----------
    series:
        Input time series.
    window:
        Number of samples in the rolling window.
    min_periods:
        Minimum number of non-NaN observations required.

    Returns
    -------
    pd.Series of slopes, same index as *series*.
    """
    def _slope(y: np.ndarray) -> float:
        # y is the actual window slice passed by pandas rolling.apply (raw=True).
        # Its length is <= window; build x to match y's length.
        n = len(y)
        x = np.arange(n, dtype=float)
        valid = ~np.isnan(y)
        if valid.sum() < min_periods:
            return np.nan
        try:
            return float(np.polyfit(x[valid], y[valid], 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    return series.rolling(window=window, min_periods=min_periods).apply(_slope, raw=True)


class FeatureBuilder:
    """
    Offline feature engineering pipeline for VESPER telemetry data.

    Typical usage
    -------------
    .. code-block:: python

        fb = FeatureBuilder()
        raw = fb.load_raw("data/raw/training_data.csv")
        feat = fb.build_rolling_features(raw)
        feat = fb.encode_target(feat)
        fb.save_processed(feat, "data/processed/features.csv")
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        config_path:
            Path to ``feature_config.yaml``.  Defaults to the YAML file
            in the same directory as this module.
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "feature_config.yaml")

        with open(config_path, "r") as fh:
            self.config = yaml.safe_load(fh)

        logger.info("FeatureBuilder initialised with config: %s", config_path)

    def load_raw(self, path: str) -> pd.DataFrame:
        """
        Load raw telemetry CSV.

        Parameters
        ----------
        path:
            Path to the raw CSV file (output of ``generate_dataset.py``).

        Returns
        -------
        pd.DataFrame with all raw columns, sorted by timestamp within each ev_id.
        """
        logger.info("Loading raw data from %s", path)
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.sort_values(["ev_id", "timestamp"]).reset_index(drop=True)
        logger.info("Loaded %d rows, %d EVs", len(df), df["ev_id"].nunique())
        return df

    def build_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all engineered features by grouping on ev_id and applying
        rolling statistics.

        Features computed
        -----------------
        Telemetry rolling features (window = WINDOW_5S unless stated):
          speed_mean_5s, speed_std_5s
          accel_mean_5s, accel_max_5s
          brake_mean_5s, brake_max_5s
          battery_temp_mean_5s, battery_temp_slope_5s
          obstacle_dist_min_5s, obstacle_dist_slope_5s
          sensor_conf_mean_5s

        Network slice features (instant + rolling):
          urllc_latency_mean_5s, urllc_latency_slope_5s
          urllc_utilization (instant passthrough)
          urllc_loss_rate (instant passthrough)
          embb_latency_mean_5s
          embb_utilization (instant passthrough)
          mmtc_utilization (instant passthrough)

        Derived features:
          urgency_score (passthrough)
          retransmission_proxy = urllc_loss_rate * urllc_throughput_mbps
          congestion_event_count_10s

        Parameters
        ----------
        df:
            Raw telemetry dataframe (output of ``load_raw``).

        Returns
        -------
        pd.DataFrame with all original columns plus engineered feature columns.
        """
        logger.info("Building rolling features for %d rows …", len(df))
        result_parts: list[pd.DataFrame] = []

        for ev_id, grp in df.groupby("ev_id", sort=False):
            grp = grp.sort_values("timestamp").copy()
            feat = self._compute_ev_features(grp)
            result_parts.append(feat)

        out = pd.concat(result_parts, ignore_index=True)

        # Forward-fill NaN from the first-window edges, then fill remaining with 0
        feature_cols = [c for c in out.columns if c not in df.columns or c == "urgency_score"]
        out[feature_cols] = out[feature_cols].ffill().fillna(0.0)

        logger.info("Rolling feature build complete. Output shape: %s", out.shape)
        return out

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the ``slice_label`` column as integer 0/1/2 (MMTC/EMBB/URLLC).

        Accepts both string labels ("MMTC", "EMBB", "URLLC") and integer labels.
        Adds a ``slice_label_str`` column preserving the original string for
        human readability.

        Parameters
        ----------
        df:
            Dataframe containing a ``slice_label`` column.

        Returns
        -------
        pd.DataFrame with ``slice_label`` as int64.
        """
        if "slice_label" not in df.columns:
            raise ValueError("DataFrame must contain a 'slice_label' column")

        if df["slice_label"].dtype == object:
            # String labels – map to ints
            df = df.copy()
            df["slice_label_str"] = df["slice_label"]
            df["slice_label"] = df["slice_label"].map(SLICE_LABEL_MAP)
        else:
            df = df.copy()
            df["slice_label"] = df["slice_label"].astype(int)
            df["slice_label_str"] = df["slice_label"].map(SLICE_INT_TO_STR)

        invalid = df["slice_label"].isna().sum()
        if invalid > 0:
            logger.warning("%d rows had unknown slice_label values (set to NaN)", invalid)

        df["slice_label"] = df["slice_label"].astype("Int64")
        return df

    def save_processed(self, df: pd.DataFrame, path: str) -> None:
        """
        Persist the processed feature dataframe to CSV.

        Creates parent directories automatically.

        Parameters
        ----------
        df:
            Processed feature dataframe.
        path:
            Output CSV path.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info("Saved processed features to %s (%d rows, %d cols)", out_path, len(df), len(df.columns))

    def _compute_ev_features(self, grp: pd.DataFrame) -> pd.DataFrame:
        """Compute all rolling features for a single EV's data chunk."""
        g = grp.copy()
        w  = WINDOW_5S
        w2 = WINDOW_2S
        w10 = WINDOW_10S

        g["speed_mean_5s"] = g["speed_kmh"].rolling(w, min_periods=MIN_PERIODS).mean()
        g["speed_std_5s"]  = g["speed_kmh"].rolling(w, min_periods=MIN_PERIODS).std()

        g["accel_mean_5s"] = g["acceleration_ms2"].rolling(w, min_periods=MIN_PERIODS).mean()
        g["accel_max_5s"]  = g["acceleration_ms2"].rolling(w, min_periods=MIN_PERIODS).max()

        g["brake_mean_5s"] = g["brake_intensity"].rolling(w, min_periods=MIN_PERIODS).mean()
        g["brake_max_5s"]  = g["brake_intensity"].rolling(w, min_periods=MIN_PERIODS).max()

        g["battery_temp_mean_5s"]  = g["battery_temp_celsius"].rolling(w, min_periods=MIN_PERIODS).mean()
        g["battery_temp_slope_5s"] = _rolling_slope(g["battery_temp_celsius"], w)

        g["obstacle_dist_min_5s"]   = g["obstacle_distance_m"].rolling(w, min_periods=MIN_PERIODS).min()
        g["obstacle_dist_slope_5s"] = _rolling_slope(g["obstacle_distance_m"], w)

        g["sensor_conf_mean_5s"] = g["sensor_confidence"].rolling(w, min_periods=MIN_PERIODS).mean()

        if "urllc_latency_ms" in g.columns:
            g["urllc_latency_mean_5s"]  = g["urllc_latency_ms"].rolling(w, min_periods=MIN_PERIODS).mean()
            g["urllc_latency_slope_5s"] = _rolling_slope(g["urllc_latency_ms"], w)
        else:
            g["urllc_latency_mean_5s"]  = 0.0
            g["urllc_latency_slope_5s"] = 0.0

        # Instant passthrough columns (already present from simulator)
        for col in ["urllc_utilization", "urllc_loss_rate", "embb_utilization", "mmtc_utilization"]:
            if col not in g.columns:
                g[col] = 0.0

        if "embb_latency_ms" in g.columns:
            g["embb_latency_mean_5s"] = g["embb_latency_ms"].rolling(w, min_periods=MIN_PERIODS).mean()
        else:
            g["embb_latency_mean_5s"] = 0.0

        # urgency_score – passthrough (computed in simulator)
        if "urgency_score" not in g.columns:
            g["urgency_score"] = 0.0

        # retransmission_proxy
        if "urllc_loss_rate" in g.columns and "urllc_throughput_mbps" in g.columns:
            g["retransmission_proxy"] = g["urllc_loss_rate"] * g["urllc_throughput_mbps"]
        elif "urllc_loss_rate" in g.columns:
            g["retransmission_proxy"] = g["urllc_loss_rate"] * 10.0  # assume 10 Mbps baseline
        else:
            g["retransmission_proxy"] = 0.0

        # congestion_event_count_10s: rolling count of samples where any slice util > 0.8
        congestion_indicator = (
            (g.get("urllc_utilization", pd.Series(0.0, index=g.index)) > CONGESTION_UTIL_THRESHOLD) |
            (g.get("embb_utilization",  pd.Series(0.0, index=g.index)) > CONGESTION_UTIL_THRESHOLD) |
            (g.get("mmtc_utilization",  pd.Series(0.0, index=g.index)) > CONGESTION_UTIL_THRESHOLD)
        ).astype(float)

        g["congestion_event_count_10s"] = (
            congestion_indicator.rolling(w10, min_periods=1).sum()
        )

        return g


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="VESPER offline feature builder")
    parser.add_argument("--input",  default="data/raw/training_data.csv",    help="Input raw CSV")
    parser.add_argument("--output", default="data/processed/features.csv",   help="Output processed CSV")
    parser.add_argument("--config", default=None, help="Path to feature_config.yaml")
    args = parser.parse_args()

    fb = FeatureBuilder(config_path=args.config)
    raw  = fb.load_raw(args.input)
    feat = fb.build_rolling_features(raw)
    feat = fb.encode_target(feat)
    fb.save_processed(feat, args.output)
    print(f"Done. Saved {len(feat)} rows with {len(feat.columns)} columns to {args.output}")
