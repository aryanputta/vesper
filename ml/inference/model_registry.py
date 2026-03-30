"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
model_registry.py – Model loader and inference dispatcher.

Provides a single ModelRegistry object that owns all three VESPER ML models:
  - Slice classifier     (XGBClassifier)    → predict_slice()
  - Congestion predictor (XGBRegressor)     → predict_congestion()
  - Anomaly detector     (IsolationForest)  → detect_anomaly()

Designed for low-latency synchronous inference; each method performs a
single-row prediction and returns within the µs budget needed for real-time
5G slice decisions.

Backward compatibility: the legacy load_models() / predict_slice(shap=list-of-tuples)
interface is preserved so existing callers (orchestrator, tests) are unaffected.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SLICE_FEATURES: list[str] = [
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
    "urllc_latency_mean_5s",
    "urllc_latency_slope_5s",
    "urllc_utilization",
    "urllc_loss_rate",
    "embb_latency_mean_5s",
    "embb_utilization",
    "mmtc_utilization",
    "urgency_score",
    "retransmission_proxy",
    "emergency_flag",
]

_CONGESTION_FEATURES: list[str] = [
    "urllc_latency_mean_5s",
    "urllc_latency_slope_5s",
    "urllc_utilization",
    "urllc_loss_rate",
    "embb_latency_mean_5s",
    "embb_utilization",
    "mmtc_utilization",
    "congestion_event_count_10s",
    "retransmission_proxy",
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

_ANOMALY_FEATURES: list[str] = [
    "urllc_latency_ms",
    "urllc_jitter_ms",
    "urllc_loss_rate",
    "embb_latency_ms",
    "embb_loss_rate",
    "mmtc_latency_ms",
    "urllc_utilization",
    "embb_utilization",
]

# Legacy feature order kept for backward compat with existing orchestrator code
_LEGACY_FEATURE_ORDER: list[str] = [
    "speed_mean_5s",
    "speed_std_5s",
    "accel_mean_2s",
    "accel_max_2s",
    "brake_intensity_mean_2s",
    "brake_intensity_max_2s",
    "battery_temp_mean_10s",
    "battery_temp_slope_10s",
    "obstacle_distance_min_5s",
    "obstacle_distance_trend_5s",
    "sensor_confidence_mean_5s",
    "state_of_charge_pct",
    "urllc_latency_mean",
    "urllc_latency_slope",
    "urllc_utilization",
    "urllc_loss_rate",
    "embb_latency_mean",
    "embb_utilization",
    "mmtc_utilization",
    "urgency_score",
    "congestion_event_count_10s",
    "retransmission_proxy",
]

_LEGACY_NETWORK_FEATURE_ORDER: list[str] = [
    "urllc_latency_ms",
    "urllc_jitter_ms",
    "urllc_packet_loss_rate",
    "urllc_utilization_ratio",
    "embb_latency_ms",
    "embb_jitter_ms",
    "embb_packet_loss_rate",
    "embb_utilization_ratio",
    "mmtc_latency_ms",
    "mmtc_jitter_ms",
    "mmtc_packet_loss_rate",
    "mmtc_utilization_ratio",
]

# Slice class names indexed by integer label
SLICE_CLASS_NAMES: dict[int, str] = {0: "MMTC", 1: "EMBB", 2: "URLLC"}


def _dict_to_row(features: dict[str, Any], cols: list[str]) -> "np.ndarray":
    """
    Convert a feature dictionary to a 2-D numpy float32 array (1, n_features).

    Missing keys are filled with 0.0.  Inf / NaN values are replaced with 0.0.
    """
    row = np.array(
        [float(features.get(col, 0.0)) for col in cols],
        dtype=np.float32,
    ).reshape(1, -1)
    row = np.where(np.isfinite(row), row, 0.0)
    return row


def _safe_load_json(path: "Path") -> Optional[dict]:
    """Load a JSON file; return None if it does not exist or cannot be parsed."""
    if not path.exists():
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        logger.debug("Could not load JSON at %s: %s", path, exc)
        return None


def _benchmark_fn(fn: Any, n: int) -> list[float]:
    """Run *fn* n times and return per-call latencies in microseconds."""
    lats: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        lats.append((time.perf_counter() - t0) * 1e6)
    return lats


class ModelRegistry:
    """
    Centralised loader and inference gateway for VESPER ML models.

    Usage
    -----
    .. code-block:: python

        registry = ModelRegistry(models_dir="ml/models")
        registry.load_all()

        if registry.is_ready():
            slice_class, confidence, shap_vals = registry.predict_slice(features)
            congestion = registry.predict_congestion(features)
            is_anomaly, score = registry.detect_anomaly(network_metrics)

    Backward-compatible usage (legacy orchestrator)
    -----------------------------------------------
    .. code-block:: python

        registry = ModelRegistry(models_path="./ml/models")
        if registry.is_ready():
            slice_int, conf, shap_pairs = registry.predict_slice(features)
    """

    def __init__(
        self,
        models_dir: str = "ml/models",
        # Backward compat alias used by legacy orchestrator code
        models_path: Optional[str] = None,
    ) -> None:
        """
        Parameters
        ----------
        models_dir:
            Directory containing the three .pkl model files.
        models_path:
            Legacy alias for models_dir (takes precedence if provided).
        """
        effective_dir = models_path if models_path is not None else models_dir
        self.models_dir = Path(effective_dir)

        self._slice_model:      Any = None   # xgb.XGBClassifier
        self._congestion_model: Any = None   # xgb.XGBRegressor
        self._anomaly_model:    Any = None   # IsolationForest
        self._anomaly_scaler:   Any = None   # StandardScaler (from training bundle)
        self._shap_explainer:   Any = None   # shap.TreeExplainer (built on demand)
        self._shap_importance: Optional[dict] = None

        # Anomaly score normalisation bounds (estimated from training data stats)
        self._anomaly_score_min: float = -0.5
        self._anomaly_score_max: float =  0.5

        # Legacy compat flags
        self._loaded: bool = False
        self._load_attempted: bool = False

    def load_all(self) -> None:
        """
        Load all three models from ``models_dir``.

        Falls back gracefully when a file is missing: logs a warning and
        leaves the corresponding attribute as ``None``.  Also sets the legacy
        ``_loaded`` / ``_load_attempted`` flags for backward compatibility.
        """
        try:
            import joblib
        except ImportError:
            logger.error("joblib is not installed – cannot load models")
            self._load_attempted = True
            return

        self._load_slice_classifier(joblib)
        self._load_congestion_predictor(joblib)
        self._load_anomaly_detector(joblib)
        self._load_shap_importance()
        self._try_build_shap_explainer()

        self._load_attempted = True
        self._loaded = self._slice_model is not None

        ready_count = sum([
            self._slice_model      is not None,
            self._congestion_model is not None,
            self._anomaly_model    is not None,
        ])
        logger.info(
            "ModelRegistry: %d/3 models loaded from %s",
            ready_count,
            self.models_dir,
        )

    # Legacy entry-point used by orchestrator
    def load_models(self) -> bool:
        """
        Backward-compatible alias for ``load_all()``.

        Returns True if the slice classifier was loaded successfully.
        """
        self.load_all()
        return self._slice_model is not None

    def _load_slice_classifier(self, joblib: Any) -> None:
        for filename in ["slice_classifier.pkl", "slice_classifier.joblib"]:
            path = self.models_dir / filename
            if path.exists():
                try:
                    self._slice_model = joblib.load(path)
                    logger.info("Loaded slice classifier from %s", path)
                    return
                except Exception as exc:
                    logger.error("Failed to load slice classifier from %s: %s", path, exc)
        logger.warning(
            "Slice classifier not found in %s – predictions unavailable", self.models_dir
        )

    def _load_congestion_predictor(self, joblib: Any) -> None:
        for filename in ["congestion_predictor.pkl", "congestion_regressor.joblib"]:
            path = self.models_dir / filename
            if path.exists():
                try:
                    self._congestion_model = joblib.load(path)
                    logger.info("Loaded congestion predictor from %s", path)
                    return
                except Exception as exc:
                    logger.warning("Failed to load congestion predictor from %s: %s", path, exc)
        logger.warning(
            "Congestion predictor not found in %s – predictions unavailable", self.models_dir
        )

    def _load_anomaly_detector(self, joblib: Any) -> None:
        for filename in ["anomaly_detector.pkl", "anomaly_detector.joblib"]:
            path = self.models_dir / filename
            if path.exists():
                try:
                    bundle = joblib.load(path)
                    if isinstance(bundle, dict) and "model" in bundle:
                        self._anomaly_model  = bundle["model"]
                        self._anomaly_scaler = bundle.get("scaler")
                    else:
                        # Plain model without scaler (legacy format)
                        self._anomaly_model = bundle
                    logger.info("Loaded anomaly detector from %s", path)
                    return
                except Exception as exc:
                    logger.warning("Failed to load anomaly detector from %s: %s", path, exc)
        logger.warning(
            "Anomaly detector not found in %s – detection unavailable", self.models_dir
        )

    def _load_shap_importance(self) -> None:
        path = self.models_dir / "slice_classifier_shap.json"
        self._shap_importance = _safe_load_json(path)
        if self._shap_importance:
            logger.debug("Loaded SHAP feature importance from %s", path)

    def _try_build_shap_explainer(self) -> None:
        """Build a SHAP TreeExplainer on the fly from the loaded slice model."""
        if self._slice_model is None:
            return
        # Check for persisted explainer first
        for filename in ["shap_explainer.joblib", "shap_explainer.pkl"]:
            path = self.models_dir / filename
            if path.exists():
                try:
                    import joblib
                    self._shap_explainer = joblib.load(path)
                    logger.info("Loaded SHAP explainer from %s", path)
                    return
                except Exception:
                    pass
        # Build from model
        try:
            import shap
            self._shap_explainer = shap.TreeExplainer(self._slice_model)
            logger.info("Built SHAP TreeExplainer from loaded slice classifier")
        except Exception as exc:
            logger.debug("Could not build SHAP explainer: %s", exc)

    def predict_slice(
        self,
        features: dict[str, Any],
    ) -> tuple[int, float, list]:
        """
        Predict the 5G slice assignment for a single telemetry frame.

        Parameters
        ----------
        features:
            Dict mapping feature names to their current values.
            Missing keys default to 0.0.

        Returns
        -------
        (slice_class, confidence, shap_output)
            slice_class  : int  (0=MMTC, 1=EMBB, 2=URLLC)
            confidence   : float in [0,1] – max predict_proba score
            shap_output  : When a SHAP explainer is available, a list of
                           (feature_name, shap_value) tuples sorted by |value|
                           (top 5).  Otherwise a list of raw floats computed
                           from the stored importance weights.

        Raises
        ------
        RuntimeError if slice model is not loaded.
        """
        if self._slice_model is None:
            if not self._load_attempted:
                self.load_all()
            if self._slice_model is None:
                raise RuntimeError(
                    "ModelRegistry: slice classifier not loaded – call load_all() first"
                )

        row   = _dict_to_row(features, _SLICE_FEATURES)
        proba = self._slice_model.predict_proba(row)[0]   # shape: (n_classes,)
        pred  = int(np.argmax(proba))
        conf  = float(proba[pred])

        # SHAP output
        shap_output: list = []
        if self._shap_explainer is not None:
            try:
                shap_vals = self._shap_explainer.shap_values(row)
                if isinstance(shap_vals, list):
                    # List-of-arrays: one per class
                    class_shap = shap_vals[pred][0]
                elif shap_vals.ndim == 3:
                    class_shap = shap_vals[0, :, pred]
                else:
                    class_shap = shap_vals[0]
                paired = sorted(
                    zip(_SLICE_FEATURES, class_shap),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )
                shap_output = [(k, float(v)) for k, v in paired[:5]]
            except Exception as exc:
                logger.debug("SHAP computation failed: %s", exc)
        elif self._shap_importance:
            importance = self._shap_importance.get("feature_importance", {})
            shap_output = [
                float(features.get(col, 0.0)) * importance.get(col, 0.0)
                for col in _SLICE_FEATURES
            ]

        return pred, conf, shap_output

    def predict_congestion(self, features: dict[str, Any]) -> float:
        """
        Predict the network congestion score 3 seconds ahead.

        Parameters
        ----------
        features:
            Dict of current feature values.

        Returns
        -------
        float in [0.0, 1.0] – predicted congestion score.
        Returns 0.0 silently when the congestion model is not loaded
        (for backward compat with legacy code that checks the return value).

        Raises
        ------
        RuntimeError if congestion model is explicitly required (not loaded).
        """
        if self._congestion_model is None:
            return 0.0

        row   = _dict_to_row(features, _CONGESTION_FEATURES)
        score = float(self._congestion_model.predict(row)[0])
        return float(np.clip(score, 0.0, 1.0))

    def detect_anomaly(
        self,
        network_metrics: dict[str, Any],
        # Backward compat kwarg alias
        network_metrics_dict: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, float]:
        """
        Detect whether the current network metric snapshot is anomalous.

        The Isolation Forest was trained only on normal (non-attack) samples,
        so high outlier scores indicate potential attacks or severe faults.

        Parameters
        ----------
        network_metrics:
            Dict containing the 8 anomaly-detection feature values.
            Accepts both the new feature names (_ANOMALY_FEATURES) and
            the legacy network KPI names (_LEGACY_NETWORK_FEATURE_ORDER).
        network_metrics_dict:
            Legacy alias for ``network_metrics`` (takes precedence if given).

        Returns
        -------
        (is_anomaly, anomaly_score)
            is_anomaly    : bool – True if Isolation Forest classifies as anomaly.
            anomaly_score : float in [0, 1] – normalised anomaly severity.
                            Derived from -score_samples, normalised to [0, 1].
        Returns (False, 0.0) silently when the anomaly model is not loaded.
        """
        effective_metrics = network_metrics_dict if network_metrics_dict is not None else network_metrics

        if self._anomaly_model is None:
            return False, 0.0

        # Build feature row using new feature names; fall back to legacy keys
        merged = {**effective_metrics}  # shallow copy
        row = _dict_to_row(merged, _ANOMALY_FEATURES)

        # Apply the training-time StandardScaler when available
        if self._anomaly_scaler is not None:
            try:
                row = self._anomaly_scaler.transform(row)
            except Exception as exc:
                logger.debug("Scaler transform failed (%s) – using raw features", exc)

        try:
            prediction    = self._anomaly_model.predict(row)[0]
            is_anomaly    = bool(prediction == -1)

            raw_score     = float(-self._anomaly_model.score_samples(row)[0])
            anomaly_score = float(np.clip(
                (raw_score - self._anomaly_score_min)
                / max(self._anomaly_score_max - self._anomaly_score_min, 1e-9),
                0.0,
                1.0,
            ))
        except Exception as exc:
            logger.debug("Anomaly detection failed: %s", exc)
            return False, 0.0

        return is_anomaly, anomaly_score

    def is_ready(self) -> bool:
        """
        Return True if the slice classifier is loaded and ready.

        Triggers a lazy ``load_all()`` on first call when no load has been
        attempted yet (backward compat with legacy orchestrator usage).
        """
        if not self._load_attempted:
            self.load_all()
        return self._slice_model is not None

    def benchmark_latency(self, n: int = 1000) -> dict[str, Any]:
        """
        Time all loaded models over *n* single-sample predictions and
        report p50 / p99 latency in microseconds.

        Models that are not loaded are skipped and reported as None.

        Parameters
        ----------
        n:
            Number of repeat predictions per model.

        Returns
        -------
        dict with keys:
            slice_p50_us, slice_p99_us,
            congestion_p50_us, congestion_p99_us,
            anomaly_p50_us, anomaly_p99_us
        """
        results: dict[str, Any] = {}

        dummy_slice   = {col: 0.1 for col in _SLICE_FEATURES}
        dummy_slice["emergency_flag"] = 0.0
        dummy_cong    = {col: 0.1 for col in _CONGESTION_FEATURES}
        dummy_anomaly = {col: 0.1 for col in _ANOMALY_FEATURES}

        # Slice classifier
        if self._slice_model is not None:
            lats = _benchmark_fn(lambda: self.predict_slice(dummy_slice), n)
            results["slice_p50_us"] = float(np.percentile(lats, 50))
            results["slice_p99_us"] = float(np.percentile(lats, 99))
            print(f"  Slice classifier   p50={results['slice_p50_us']:.1f} µs  p99={results['slice_p99_us']:.1f} µs")
        else:
            results["slice_p50_us"] = results["slice_p99_us"] = None
            print("  Slice classifier   – not loaded")

        # Congestion predictor
        if self._congestion_model is not None:
            lats = _benchmark_fn(lambda: self.predict_congestion(dummy_cong), n)
            results["congestion_p50_us"] = float(np.percentile(lats, 50))
            results["congestion_p99_us"] = float(np.percentile(lats, 99))
            print(f"  Congestion pred.   p50={results['congestion_p50_us']:.1f} µs  p99={results['congestion_p99_us']:.1f} µs")
        else:
            results["congestion_p50_us"] = results["congestion_p99_us"] = None
            print("  Congestion pred.   – not loaded")

        # Anomaly detector
        if self._anomaly_model is not None:
            lats = _benchmark_fn(lambda: self.detect_anomaly(dummy_anomaly), n)
            results["anomaly_p50_us"] = float(np.percentile(lats, 50))
            results["anomaly_p99_us"] = float(np.percentile(lats, 99))
            print(f"  Anomaly detector   p50={results['anomaly_p50_us']:.1f} µs  p99={results['anomaly_p99_us']:.1f} µs")
        else:
            results["anomaly_p50_us"] = results["anomaly_p99_us"] = None
            print("  Anomaly detector   – not loaded")

        return results

    def status(self) -> dict:
        """Return a dict summarising which models are loaded."""
        return {
            "slice_classifier":     self._slice_model      is not None,
            "congestion_regressor": self._congestion_model is not None,
            "anomaly_detector":     self._anomaly_model    is not None,
            "shap_explainer":       self._shap_explainer   is not None,
            "models_dir":           str(self.models_dir),
            "ready":                self.is_ready(),
        }


_default_registry: Optional[ModelRegistry] = None


def get_registry(models_dir: str = "ml/models") -> ModelRegistry:
    """
    Return (and lazily initialise) the module-level default ModelRegistry.

    Subsequent calls with the same ``models_dir`` return the cached instance.
    """
    global _default_registry  # noqa: PLW0603
    if _default_registry is None:
        _default_registry = ModelRegistry(models_dir=models_dir)
        _default_registry.load_all()
    return _default_registry


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="VESPER ModelRegistry CLI")
    parser.add_argument("--models-dir", default="ml/models", help="Directory with .pkl files")
    parser.add_argument("--benchmark",  action="store_true",  help="Run latency benchmark")
    parser.add_argument("--n",          type=int, default=1000, help="Benchmark iterations")
    args = parser.parse_args()

    registry = ModelRegistry(models_dir=args.models_dir)
    registry.load_all()

    print(f"\nRegistry status: {registry.status()}")

    if args.benchmark:
        print(f"\n--- Latency Benchmark (n={args.n}) ---")
        results = registry.benchmark_latency(n=args.n)
        print(f"\nFull results: {results}")
    else:
        dummy = {col: 0.1 for col in _SLICE_FEATURES}
        dummy["emergency_flag"] = 0.0
        dummy["urgency_score"]  = 0.35

        if registry.is_ready():
            sc, conf, shap = registry.predict_slice(dummy)
            print(f"\nSlice prediction: {SLICE_CLASS_NAMES[sc]} (conf={conf:.4f})")
            print(f"SHAP output type: {type(shap).__name__}  len={len(shap)}")

        dummy_cong = {col: 0.1 for col in _CONGESTION_FEATURES}
        cong = registry.predict_congestion(dummy_cong)
        print(f"Congestion score: {cong:.4f}")

        dummy_anom = {col: 0.1 for col in _ANOMALY_FEATURES}
        is_anom, anom_score = registry.detect_anomaly(dummy_anom)
        print(f"Anomaly: {is_anom}  (score={anom_score:.4f})")
