"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
feature_extractor.py – Rolling-window feature extractor for the ML slice manager.

Each vehicle has its own EVFeatureBuffer.  The buffer stores raw telemetry
frames in a fixed-size ring buffer (300 samples @ 10 Hz = 30 s of history).
``extract_features`` computes a 22-feature vector consumed by the XGBoost
slice-selection model and the SHAP explainer.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..networking.message_schema import NetworkMetrics, SliceType, TelemetryMessage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


_BUFFER_MAXLEN: int = 300       # 30 s × 10 Hz
_WINDOW_10HZ_5S: int = 50       # 5-second window at 10 Hz
_WINDOW_10HZ_2S: int = 20       # 2-second window at 10 Hz
_WINDOW_10HZ_10S: int = 100     # 10-second window at 10 Hz
_CONGESTION_THRESHOLD: float = 0.8   # utilisation above this counts as congestion



class EVFeatureBuffer:
    """
    Per-vehicle rolling-window feature extractor.

    Attributes
    ----------
    ev_id : str
        The vehicle this buffer belongs to.
    _buffer : deque[TelemetryMessage]
        Ring buffer capped at ``_BUFFER_MAXLEN`` frames.
    """

    def __init__(self, ev_id: str) -> None:
        self.ev_id = ev_id
        self._buffer: deque[TelemetryMessage] = deque(maxlen=_BUFFER_MAXLEN)

    def update(self, msg: TelemetryMessage) -> None:
        """Append a new telemetry frame to the ring buffer."""
        if msg.ev_id != self.ev_id:
            raise ValueError(
                f"EVFeatureBuffer({self.ev_id!r}) received message for {msg.ev_id!r}"
            )
        self._buffer.append(msg)

    def extract_features(
        self,
        network_metrics: dict[str, NetworkMetrics],
        urgency_score: float = 0.0,
    ) -> dict[str, float]:
        """
        Compute the 22-feature vector used by the VESPER ML model.

        Parameters
        ----------
        network_metrics:
            Mapping of slice name (``"URLLC"``, ``"EMBB"``, ``"MMTC"``) to the
            most-recent ``NetworkMetrics`` snapshot for that slice.
        urgency_score:
            Latest urgency score from ``UrgencyScorer.compute()``.

        Returns
        -------
        dict[str, float]
            22-entry feature dictionary.  Missing slices are represented by
            sentinel values (0.0 for rates, 999.0 for latency).
        """
        if not self._buffer:
            logger.warning("EVFeatureBuffer(%s): buffer empty, returning zero features", self.ev_id)
            return self._zero_features(urgency_score)

        latest = self._buffer[-1]

        # ── Speed (5 s) ──────────────────────────────────────────────────────
        speed_w = self._window_values("speed_kmh", _WINDOW_10HZ_5S)
        speed_mean_5s = float(np.mean(speed_w)) if speed_w else 0.0
        speed_std_5s = float(np.std(speed_w)) if speed_w else 0.0

        # ── Acceleration (2 s) ───────────────────────────────────────────────
        accel_w = self._window_values("acceleration_ms2", _WINDOW_10HZ_2S)
        accel_mean_2s = float(np.mean(accel_w)) if accel_w else 0.0
        accel_max_2s = float(np.max(np.abs(accel_w))) if accel_w else 0.0

        # ── Brake intensity (2 s) ────────────────────────────────────────────
        brake_w = self._window_values("brake_intensity", _WINDOW_10HZ_2S)
        brake_intensity_mean_2s = float(np.mean(brake_w)) if brake_w else 0.0
        brake_intensity_max_2s = float(np.max(brake_w)) if brake_w else 0.0

        # ── Battery temperature (10 s) ────────────────────────────────────────
        temp_w = self._window_values("battery_temp_celsius", _WINDOW_10HZ_10S)
        battery_temp_mean_10s = float(np.mean(temp_w)) if temp_w else 0.0
        battery_temp_slope_10s = self._slope(temp_w)

        # ── Obstacle distance (5 s) ───────────────────────────────────────────
        obs_w = self._window_values("obstacle_distance_m", _WINDOW_10HZ_5S)
        obstacle_distance_min_5s = float(np.min(obs_w)) if obs_w else 0.0
        obstacle_distance_trend_5s = self._slope(obs_w)

        # ── Sensor confidence (5 s) ───────────────────────────────────────────
        conf_w = self._window_values("sensor_confidence", _WINDOW_10HZ_5S)
        sensor_confidence_mean_5s = float(np.mean(conf_w)) if conf_w else 1.0

        # ── Latest state of charge ────────────────────────────────────────────
        state_of_charge_pct = float(latest.state_of_charge_pct)

        # ── Network metrics per slice ─────────────────────────────────────────
        urllc = network_metrics.get(SliceType.URLLC.value) or network_metrics.get(SliceType.URLLC)
        embb = network_metrics.get(SliceType.EMBB.value) or network_metrics.get(SliceType.EMBB)
        mmtc = network_metrics.get(SliceType.MMTC.value) or network_metrics.get(SliceType.MMTC)

        urllc_latency_mean = float(urllc.latency_ms) if urllc else 999.0
        urllc_latency_slope = 0.0      # single-point snapshot; slope is 0
        urllc_utilization = float(urllc.utilization_ratio) if urllc else 0.0
        urllc_loss_rate = float(urllc.packet_loss_rate) if urllc else 0.0

        embb_latency_mean = float(embb.latency_ms) if embb else 999.0
        embb_utilization = float(embb.utilization_ratio) if embb else 0.0

        mmtc_utilization = float(mmtc.utilization_ratio) if mmtc else 0.0

        # ── Congestion event count (10 s) ────────────────────────────────────
        # Count telemetry frames in the last 10 s where at least one slice was
        # congested.  We approximate by checking if network_metrics show any
        # slice > threshold (since we have only the latest snapshot, we count
        # the frames that arrived after the congestion watermark was last set).
        congestion_event_count_10s = self._count_congestion_events(
            network_metrics, _WINDOW_10HZ_10S
        )

        # ── Retransmission proxy ──────────────────────────────────────────────
        # Estimated bytes that had to be re-sent = loss_rate × throughput (Mbps)
        # This is a proxy; actual retransmission count is not exposed.
        if urllc:
            retransmission_proxy = urllc.packet_loss_rate * urllc.throughput_mbps
        else:
            retransmission_proxy = 0.0

        return {
            # Speed
            "speed_mean_5s": speed_mean_5s,
            "speed_std_5s": speed_std_5s,
            # Acceleration
            "accel_mean_2s": accel_mean_2s,
            "accel_max_2s": accel_max_2s,
            # Brake
            "brake_intensity_mean_2s": brake_intensity_mean_2s,
            "brake_intensity_max_2s": brake_intensity_max_2s,
            # Battery temperature
            "battery_temp_mean_10s": battery_temp_mean_10s,
            "battery_temp_slope_10s": battery_temp_slope_10s,
            # Obstacle
            "obstacle_distance_min_5s": obstacle_distance_min_5s,
            "obstacle_distance_trend_5s": obstacle_distance_trend_5s,
            # Sensor
            "sensor_confidence_mean_5s": sensor_confidence_mean_5s,
            # SoC
            "state_of_charge_pct": state_of_charge_pct,
            # URLLC slice metrics
            "urllc_latency_mean": urllc_latency_mean,
            "urllc_latency_slope": urllc_latency_slope,
            "urllc_utilization": urllc_utilization,
            "urllc_loss_rate": urllc_loss_rate,
            # eMBB slice metrics
            "embb_latency_mean": embb_latency_mean,
            "embb_utilization": embb_utilization,
            # mMTC slice metrics
            "mmtc_utilization": mmtc_utilization,
            # Derived / composite
            "urgency_score": float(urgency_score),
            "congestion_event_count_10s": float(congestion_event_count_10s),
            "retransmission_proxy": retransmission_proxy,
        }

    def _window(self, n: int) -> list[TelemetryMessage]:
        """Return the last ``n`` TelemetryMessage objects from the buffer."""
        buf = list(self._buffer)
        return buf[-n:] if len(buf) >= n else buf

    def _window_values(self, attr: str, n: int) -> list[float]:
        """Return the last ``n`` numeric values for a given telemetry attribute."""
        return [float(getattr(msg, attr)) for msg in self._window(n)]

    def _slope(self, values: list[float]) -> float:
        """
        Estimate the linear trend (slope) of a time series via numpy polyfit.

        Parameters
        ----------
        values:
            Ordered list of scalar measurements (oldest first).

        Returns
        -------
        float
            Slope in units-per-sample, or 0.0 if ``values`` has fewer than
            2 distinct points.
        """
        if len(values) < 2:
            return 0.0
        try:
            x = np.arange(len(values), dtype=np.float64)
            coeffs = np.polyfit(x, values, 1)
            return float(coeffs[0])
        except (np.linalg.LinAlgError, ValueError):
            logger.debug("polyfit failed for %s, returning 0.0", self.ev_id)
            return 0.0

    def _count_congestion_events(
        self,
        network_metrics: dict[str, NetworkMetrics],
        window: int,
    ) -> int:
        """
        Count the number of frames in the last ``window`` samples where at
        least one known slice exceeded the congestion utilisation threshold.

        Because we do not store historical network metrics per-frame, we use
        the current snapshot: if any slice is congested right now we check how
        many frames in the rolling window arrived during a congestion period by
        counting how many sequential telemetry frames overlap with the metric
        timestamp.  As a practical approximation we simply count how many of
        the last ``window`` telemetry frames would be marked congested based on
        the latest network metrics (i.e. we assume network state is roughly
        constant over the window).
        """
        slices = [
            network_metrics.get(SliceType.URLLC.value) or network_metrics.get(SliceType.URLLC),
            network_metrics.get(SliceType.EMBB.value) or network_metrics.get(SliceType.EMBB),
            network_metrics.get(SliceType.MMTC.value) or network_metrics.get(SliceType.MMTC),
        ]
        any_congested = any(
            s is not None and s.utilization_ratio > _CONGESTION_THRESHOLD for s in slices
        )
        if not any_congested:
            return 0

        # Count frames in the last `window` period whose timestamp falls within
        # the congestion window (defined as any frame more recent than the
        # oldest metric timestamp we have).
        frames = self._window(window)
        if not frames:
            return 0

        # Estimate congestion start time: use the earliest metric timestamp
        metric_timestamps = [
            s.timestamp for s in slices if s is not None
        ]
        if not metric_timestamps:
            return len(frames)

        congestion_since = min(metric_timestamps)
        return sum(1 for f in frames if f.timestamp >= congestion_since)

    def _zero_features(self, urgency_score: float) -> dict[str, float]:
        """Return a zeroed feature dict for empty-buffer edge case."""
        return {
            "speed_mean_5s": 0.0,
            "speed_std_5s": 0.0,
            "accel_mean_2s": 0.0,
            "accel_max_2s": 0.0,
            "brake_intensity_mean_2s": 0.0,
            "brake_intensity_max_2s": 0.0,
            "battery_temp_mean_10s": 0.0,
            "battery_temp_slope_10s": 0.0,
            "obstacle_distance_min_5s": 0.0,
            "obstacle_distance_trend_5s": 0.0,
            "sensor_confidence_mean_5s": 1.0,
            "state_of_charge_pct": 0.0,
            "urllc_latency_mean": 999.0,
            "urllc_latency_slope": 0.0,
            "urllc_utilization": 0.0,
            "urllc_loss_rate": 0.0,
            "embb_latency_mean": 999.0,
            "embb_utilization": 0.0,
            "mmtc_utilization": 0.0,
            "urgency_score": float(urgency_score),
            "congestion_event_count_10s": 0.0,
            "retransmission_proxy": 0.0,
        }

    def buffer_size(self) -> int:
        """Current number of frames in the ring buffer."""
        return len(self._buffer)

    def latest(self) -> Optional[TelemetryMessage]:
        """Return the most-recent telemetry frame, or None if buffer is empty."""
        return self._buffer[-1] if self._buffer else None
