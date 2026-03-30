"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
urgency_scorer.py – Composite urgency scoring with optional C-extension acceleration.

The scorer combines five safety signals into a single urgency value in [0, 1].
An optional compiled C extension (urgency.so) is loaded via ctypes for
edge-node deployments where latency budget is tight; if the .so is absent or
fails to load the pure-Python path is used transparently.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Optional

from ..networking.message_schema import EVState, Priority, TelemetryMessage

logger = logging.getLogger(__name__)


_c_compute: Optional[ctypes.CFUNCTYPE] = None  # type: ignore[type-arg]

def _try_load_c_extension() -> None:
    """
    Attempt to load urgency.so from ``edge/urgency_scorer/`` relative to the
    repository root.  Sets ``_c_compute`` to the loaded function on success,
    leaves it as ``None`` on any failure.

    Expected C signature::

        double compute_urgency(
            double safety_flag_score,
            double obstacle_score,
            double brake_score,
            double battery_score,
            double sensor_score,
            double w_safety, double w_obstacle, double w_brake,
            double w_battery, double w_sensor,
            double state_multiplier
        );
    """
    global _c_compute  # noqa: PLW0603

    # Resolve repository root: go up from this file's location
    repo_root = Path(__file__).resolve().parents[4]  # …/backend/app/core/ → repo root
    so_path = repo_root / "edge" / "urgency_scorer" / "urgency.so"

    if not so_path.exists():
        logger.debug("urgency.so not found at %s – using Python scorer", so_path)
        return

    try:
        lib = ctypes.CDLL(str(so_path))
        fn = lib.compute_urgency
        fn.restype = ctypes.c_double
        fn.argtypes = [ctypes.c_double] * 11
        # Smoke-test: all-zero inputs → expect 0.0
        result = fn(0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.25, 0.20, 0.15, 0.10, 1.0)
        if not isinstance(result, float):
            raise TypeError(f"Unexpected return type from urgency.so: {type(result)}")
        _c_compute = fn  # type: ignore[assignment]
        logger.info("Loaded C urgency scorer from %s", so_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to load urgency.so (%s) – falling back to Python", exc)
        _c_compute = None


_try_load_c_extension()



class UrgencyScorer:
    """
    Computes a composite urgency score in [0.0, 1.0] for a telemetry frame.

    Weights
    -------
    W_SAFETY_FLAG : 0.30 – hard emergency signal
    W_OBSTACLE    : 0.25 – proximity to obstacles
    W_BRAKE       : 0.20 – brake intensity
    W_BATTERY     : 0.15 – thermal condition
    W_SENSOR      : 0.10 – sensor suite confidence

    State multipliers amplify the weighted sum when the vehicle is already in
    an elevated state, reflecting that the same physical signal is more urgent
    when context is already alarming.
    """

    W_SAFETY_FLAG: float = 0.30
    W_OBSTACLE: float = 0.25
    W_BRAKE: float = 0.20
    W_BATTERY: float = 0.15
    W_SENSOR: float = 0.10

    _STATE_MULTIPLIER: dict[EVState, float] = {
        EVState.EMERGENCY: 1.5,
        EVState.CRITICAL: 1.3,
        EVState.CAUTION: 1.1,
        EVState.RECOVERY: 1.0,
        EVState.NORMAL: 1.0,
    }

    # Battery temperature thresholds for scoring
    _BATTERY_TEMP_BASE: float = 40.0   # °C – below this → score 0
    _BATTERY_TEMP_RANGE: float = 40.0  # °C – span from base to fully critical

    # Obstacle distance cap
    _OBSTACLE_CAP_M: float = 50.0      # beyond this → score 0

    def compute(self, msg: TelemetryMessage, ev_state: EVState) -> float:
        """
        Compute urgency score for *msg* given the current *ev_state*.

        Uses the C extension when available; falls back to Python otherwise.

        Parameters
        ----------
        msg:
            Current telemetry frame.
        ev_state:
            Current state from the EVStateMachine (provides multiplier).

        Returns
        -------
        float
            Urgency score in [0.0, 1.0].
        """
        safety_flag_score, obstacle_score, brake_score, battery_score, sensor_score = (
            self._component_scores(msg)
        )
        multiplier = self._STATE_MULTIPLIER.get(ev_state, 1.0)

        if _c_compute is not None:
            raw = _c_compute(
                safety_flag_score,
                obstacle_score,
                brake_score,
                battery_score,
                sensor_score,
                self.W_SAFETY_FLAG,
                self.W_OBSTACLE,
                self.W_BRAKE,
                self.W_BATTERY,
                self.W_SENSOR,
                multiplier,
            )
        else:
            raw = self._python_compute(
                safety_flag_score,
                obstacle_score,
                brake_score,
                battery_score,
                sensor_score,
                multiplier,
            )

        return min(1.0, max(0.0, raw))

    def classify_urgency(self, score: float) -> Priority:
        """
        Map a continuous urgency score to a discrete Priority level.

        Thresholds
        ----------
        score >= 0.75 → CRITICAL
        score >= 0.50 → HIGH
        score >= 0.25 → MEDIUM
        else          → LOW
        """
        if score >= 0.75:
            return Priority.CRITICAL
        if score >= 0.50:
            return Priority.HIGH
        if score >= 0.25:
            return Priority.MEDIUM
        return Priority.LOW

    def _component_scores(
        self, msg: TelemetryMessage
    ) -> tuple[float, float, float, float, float]:
        """
        Derive the five raw component scores from the telemetry message.

        Returns
        -------
        (safety_flag_score, obstacle_score, brake_score, battery_score, sensor_score)
        All values in [0.0, 1.0].
        """
        # Safety flag
        safety_flag_score: float = 1.0 if msg.emergency_flag else 0.0

        # Obstacle proximity
        if msg.obstacle_distance_m < self._OBSTACLE_CAP_M:
            obstacle_score = max(0.0, 1.0 - msg.obstacle_distance_m / self._OBSTACLE_CAP_M)
        else:
            obstacle_score = 0.0

        # Brake intensity (already normalised to [0, 1])
        brake_score: float = float(msg.brake_intensity)

        # Battery temperature
        battery_raw = (msg.battery_temp_celsius - self._BATTERY_TEMP_BASE) / self._BATTERY_TEMP_RANGE
        battery_score: float = max(0.0, min(1.0, battery_raw))

        # Sensor confidence (inverted: low confidence → high score)
        sensor_score: float = 1.0 - float(msg.sensor_confidence)

        return safety_flag_score, obstacle_score, brake_score, battery_score, sensor_score

    def _python_compute(
        self,
        safety_flag_score: float,
        obstacle_score: float,
        brake_score: float,
        battery_score: float,
        sensor_score: float,
        multiplier: float,
    ) -> float:
        """Pure-Python weighted sum with state multiplier."""
        weighted = (
            self.W_SAFETY_FLAG * safety_flag_score
            + self.W_OBSTACLE * obstacle_score
            + self.W_BRAKE * brake_score
            + self.W_BATTERY * battery_score
            + self.W_SENSOR * sensor_score
        )
        return weighted * multiplier

    def score_components(
        self, msg: TelemetryMessage, ev_state: EVState
    ) -> dict[str, float]:
        """
        Return a breakdown of each component and the final score.

        Useful for explainability / debugging.
        """
        sf, obs, brk, bat, sen = self._component_scores(msg)
        multiplier = self._STATE_MULTIPLIER.get(ev_state, 1.0)
        final = self.compute(msg, ev_state)
        return {
            "safety_flag_score": sf,
            "obstacle_score": obs,
            "brake_score": brk,
            "battery_score": bat,
            "sensor_score": sen,
            "state_multiplier": multiplier,
            "weighted_sum_pre_clamp": self._python_compute(sf, obs, brk, bat, sen, multiplier),
            "final_urgency_score": final,
            "priority": self.classify_urgency(final).name,
        }
