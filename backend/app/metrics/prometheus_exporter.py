"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
prometheus_exporter.py – Prometheus metrics definitions and exporter.

All VESPER metrics are defined here as module-level objects so they can be
imported and used from any part of the application without risk of duplicate
registration.

Metric catalogue
----------------
vesper_messages_total               Counter   labels: slice, priority, event_type
vesper_slice_latency_ms             Histogram labels: slice
vesper_packet_loss_ratio            Gauge     labels: slice
vesper_slice_utilization            Gauge     labels: slice
vesper_reassignment_total           Counter   (no labels)
vesper_safety_event_response_ms     Histogram (no labels)
vesper_ml_inference_latency_ms      Histogram (no labels)
vesper_active_evs                   Gauge     (no labels)
vesper_ev_state                     Gauge     labels: ev_id, state
vesper_queue_depth                  Gauge     labels: priority
vesper_anomaly_detections_total     Counter   (no labels)
vesper_rule_triggers_total          Counter   labels: rule_id
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    start_http_server,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions (module-level singletons)
# ---------------------------------------------------------------------------

vesper_messages_total = Counter(
    "vesper_messages_total",
    "Total number of telemetry messages processed, labelled by slice, priority, and event type.",
    labelnames=["slice", "priority", "event_type"],
)

vesper_slice_latency_ms = Histogram(
    "vesper_slice_latency_ms",
    "End-to-end decision latency in milliseconds per slice.",
    labelnames=["slice"],
    buckets=[0.5, 1, 2, 5, 10, 20, 50, 100, 500],
)

vesper_packet_loss_ratio = Gauge(
    "vesper_packet_loss_ratio",
    "Current packet loss ratio for each network slice.",
    labelnames=["slice"],
)

vesper_slice_utilization = Gauge(
    "vesper_slice_utilization",
    "Current capacity utilisation ratio (0–1) for each network slice.",
    labelnames=["slice"],
)

vesper_reassignment_total = Counter(
    "vesper_reassignment_total",
    "Total number of slice re-assignments triggered by admission control downgrading.",
)

vesper_safety_event_response_ms = Histogram(
    "vesper_safety_event_response_ms",
    "End-to-end response latency for safety events in milliseconds.",
    buckets=[1, 2, 5, 10, 20, 50],
)

vesper_ml_inference_latency_ms = Histogram(
    "vesper_ml_inference_latency_ms",
    "ML model inference latency in milliseconds (slice prediction + congestion + anomaly).",
    buckets=[0.1, 0.5, 1, 2, 5, 10],
)

vesper_active_evs = Gauge(
    "vesper_active_evs",
    "Number of EVs currently tracked by the orchestrator.",
)

vesper_ev_state = Gauge(
    "vesper_ev_state",
    "Current FSM state for each EV (1.0 if in this state, 0.0 otherwise).",
    labelnames=["ev_id", "state"],
)

vesper_queue_depth = Gauge(
    "vesper_queue_depth",
    "Current number of messages in the priority queue per priority level.",
    labelnames=["priority"],
)

vesper_anomaly_detections_total = Counter(
    "vesper_anomaly_detections_total",
    "Total number of network anomalies detected by the Isolation Forest model.",
)

vesper_rule_triggers_total = Counter(
    "vesper_rule_triggers_total",
    "Total number of deterministic rule triggers, labelled by rule ID.",
    labelnames=["rule_id"],
)

# ---------------------------------------------------------------------------
# MetricsExporter class
# ---------------------------------------------------------------------------

_EV_STATES = ["NORMAL", "CAUTION", "CRITICAL", "EMERGENCY", "RECOVERY"]


class MetricsExporter:
    """
    Façade over the Prometheus metric objects defined above.

    The orchestrator, slice manager, and other components call these
    methods to record observations.  The actual Prometheus client objects
    are module-level singletons; this class simply provides a typed API
    with sensible defaults.

    Thread safety
    -------------
    Prometheus client objects are internally thread-safe.  The only state
    maintained by this class is ``_server_started`` which is protected by
    a threading.Lock.
    """

    def __init__(self) -> None:
        self._server_started: bool = False
        self._start_lock: threading.Lock = threading.Lock()

        # Cache of known EV states so we can zero-out old state gauges
        self._ev_states: dict[str, str] = {}  # ev_id → current state name

    # ------------------------------------------------------------------
    # Message recording
    # ------------------------------------------------------------------

    def record_message(
        self,
        slice: str,
        priority: str,
        event_type: str,
        latency_ms: float,
        delivered: bool,
    ) -> None:
        """
        Record a processed telemetry message.

        Parameters
        ----------
        slice:
            Slice name (e.g. ``"URLLC"``).
        priority:
            Priority level name (e.g. ``"CRITICAL"``).
        event_type:
            Event type string (e.g. ``"BRAKE_ALERT"``).
        latency_ms:
            End-to-end decision latency in milliseconds.
        delivered:
            Whether the message was admitted (True) or rejected (False).
        """
        vesper_messages_total.labels(
            slice=slice,
            priority=priority,
            event_type=event_type,
        ).inc()

        vesper_slice_latency_ms.labels(slice=slice).observe(latency_ms)

    # ------------------------------------------------------------------
    # Safety events
    # ------------------------------------------------------------------

    def record_safety_event(self, response_latency_ms: float) -> None:
        """Record a safety event and its end-to-end response latency."""
        vesper_safety_event_response_ms.observe(response_latency_ms)

    # ------------------------------------------------------------------
    # ML inference
    # ------------------------------------------------------------------

    def record_ml_inference(self, latency_ms: float) -> None:
        """Record an ML inference round-trip latency observation."""
        vesper_ml_inference_latency_ms.observe(latency_ms)

    # ------------------------------------------------------------------
    # Anomalies
    # ------------------------------------------------------------------

    def record_anomaly(self) -> None:
        """Increment the anomaly detection counter."""
        vesper_anomaly_detections_total.inc()

    # ------------------------------------------------------------------
    # Rule triggers
    # ------------------------------------------------------------------

    def record_rule_trigger(self, rule_id: str) -> None:
        """Increment the rule trigger counter for the given rule ID."""
        vesper_rule_triggers_total.labels(rule_id=rule_id).inc()

    # ------------------------------------------------------------------
    # Slice state
    # ------------------------------------------------------------------

    def update_slice_state(
        self,
        slice: str,
        utilization: float,
        loss_ratio: float,
    ) -> None:
        """
        Update the utilisation and packet-loss gauges for a slice.

        Parameters
        ----------
        slice:
            Slice name (e.g. ``"URLLC"``).
        utilization:
            Current utilisation ratio in [0, 1].
        loss_ratio:
            Current packet loss ratio in [0, 1].
        """
        vesper_slice_utilization.labels(slice=slice).set(utilization)
        vesper_packet_loss_ratio.labels(slice=slice).set(loss_ratio)

    # ------------------------------------------------------------------
    # EV state
    # ------------------------------------------------------------------

    def update_ev_state(self, ev_id: str, state: str) -> None:
        """
        Update the EV state gauge for ``ev_id``.

        Sets the gauge for the new state to 1.0 and zeroes all other states
        for this vehicle.  This gives a clean time-series for per-EV state
        tracking in Grafana.

        Parameters
        ----------
        ev_id:
            Vehicle identifier string.
        state:
            New state name (must be one of ``_EV_STATES``).
        """
        previous = self._ev_states.get(ev_id)
        if previous is not None and previous != state:
            # Zero out the previous state gauge for this EV
            try:
                vesper_ev_state.labels(ev_id=ev_id, state=previous).set(0.0)
            except Exception:
                pass

        try:
            vesper_ev_state.labels(ev_id=ev_id, state=state).set(1.0)
        except Exception:
            pass

        self._ev_states[ev_id] = state

        # Update the active EV count
        vesper_active_evs.set(len(self._ev_states))

    # ------------------------------------------------------------------
    # Queue depths
    # ------------------------------------------------------------------

    def update_queue_depths(self, depths: dict[str, int]) -> None:
        """
        Update the queue depth gauges.

        Parameters
        ----------
        depths:
            Mapping of priority name to queue depth, e.g.
            ``{"CRITICAL": 0, "HIGH": 3, "MEDIUM": 12, "LOW": 45}``.
        """
        for priority, depth in depths.items():
            try:
                vesper_queue_depth.labels(priority=priority).set(depth)
            except Exception as exc:
                logger.debug("Failed to update queue depth gauge for %s: %s", priority, exc)

    # ------------------------------------------------------------------
    # Reassignments
    # ------------------------------------------------------------------

    def record_reassignment(self) -> None:
        """Increment the slice reassignment counter."""
        vesper_reassignment_total.inc()

    # ------------------------------------------------------------------
    # Prometheus HTTP server
    # ------------------------------------------------------------------

    def start_server(self, port: int = 8001) -> None:
        """
        Start the Prometheus metrics HTTP server on ``port``.

        This method is idempotent – calling it more than once has no effect.
        The server runs in a background daemon thread managed by the
        ``prometheus_client`` library.

        Parameters
        ----------
        port:
            TCP port on which to expose the ``/metrics`` scrape endpoint.
            Defaults to 8001.
        """
        with self._start_lock:
            if self._server_started:
                logger.debug("Prometheus HTTP server already running on port %d", port)
                return
            try:
                start_http_server(port)
                self._server_started = True
                logger.info("Prometheus metrics server started on port %d", port)
            except OSError as exc:
                logger.error(
                    "Failed to start Prometheus server on port %d: %s", port, exc
                )


# ---------------------------------------------------------------------------
# Module-level default exporter instance
# ---------------------------------------------------------------------------

# Convenience singleton – import and use directly:
#   from backend.app.metrics.prometheus_exporter import default_exporter
#   default_exporter.record_message(...)
default_exporter = MetricsExporter()
