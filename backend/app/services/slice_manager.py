"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
slice_manager.py – Tracks the health and current metrics of each 5G slice.

``SliceManager`` maintains rolling-window statistics (last 100 measurements)
for latency, packet loss, and throughput per slice, and provides slice
selection advice based on priority and current slice health.

This implementation also preserves backward-compatible helpers used by the
REST API and orchestrator (``update_slice_metrics``, ``get_current_metrics``,
``get_all_metrics``, ``get_utilization_dict``, ``get_metrics_flat_dict``).
"""

from __future__ import annotations

import collections
import logging
import time
from typing import Optional

from backend.app.networking.message_schema import NetworkMetrics, Priority, SliceType

logger = logging.getLogger(__name__)

_ROLLING_WINDOW = 100   # number of per-measurement samples kept per slice
_HISTORY_MAXLEN = 1000  # number of NetworkMetrics snapshots kept per slice



class _SliceWindow:
    """
    Maintains the last N latency samples and delivery outcomes for one slice
    plus an optional rolling history of full ``NetworkMetrics`` snapshots.
    """

    def __init__(self, slice_type: SliceType, window: int = _ROLLING_WINDOW) -> None:
        self.slice_type = slice_type
        self._window = window

        # Rolling measurement deques
        self._latencies: collections.deque[float] = collections.deque(maxlen=window)
        self._delivered: collections.deque[bool] = collections.deque(maxlen=window)
        self._payload_bytes: collections.deque[int] = collections.deque(maxlen=window)
        self._sample_timestamps: collections.deque[float] = collections.deque(maxlen=window)

        # Externally pushed utilisation (from network monitor or simulator)
        self._utilization: float = 0.0

        # Full NetworkMetrics snapshot history (for REST API)
        self._history: collections.deque[NetworkMetrics] = collections.deque(
            maxlen=_HISTORY_MAXLEN
        )
        self._latest_snapshot: Optional[NetworkMetrics] = None

    def record(self, latency_ms: float, delivered: bool, payload_bytes: int) -> None:
        self._latencies.append(latency_ms)
        self._delivered.append(delivered)
        self._payload_bytes.append(payload_bytes)
        self._sample_timestamps.append(time.time())

        # Derive a new utilisation estimate from the rolling loss rate
        loss = self.packet_loss_rate
        # Heuristic: each 1% of loss adds 10% utilisation on top of current value.
        # Clamp to [0, 1].
        derived = min(1.0, self._utilization + loss * 0.1)
        self._utilization = derived

    def update_utilization(self, util: float) -> None:
        self._utilization = max(0.0, min(1.0, util))

    def ingest_snapshot(self, metrics: NetworkMetrics) -> None:
        """Accept a full ``NetworkMetrics`` snapshot (from external monitor)."""
        self._latest_snapshot = metrics
        self._history.append(metrics)
        self._utilization = metrics.utilization_ratio

    @property
    def avg_latency_ms(self) -> float:
        if not self._latencies:
            # Fall back to snapshot if no rolling samples yet
            if self._latest_snapshot:
                return self._latest_snapshot.latency_ms
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    @property
    def jitter_ms(self) -> float:
        n = len(self._latencies)
        if n < 2:
            if self._latest_snapshot:
                return self._latest_snapshot.jitter_ms
            return 0.0
        mean = self.avg_latency_ms
        variance = sum((x - mean) ** 2 for x in self._latencies) / (n - 1)
        return variance ** 0.5

    @property
    def packet_loss_rate(self) -> float:
        if not self._delivered:
            if self._latest_snapshot:
                return self._latest_snapshot.packet_loss_rate
            return 0.0
        lost = sum(1 for d in self._delivered if not d)
        return lost / len(self._delivered)

    @property
    def throughput_mbps(self) -> float:
        if len(self._sample_timestamps) < 2:
            if self._latest_snapshot:
                return self._latest_snapshot.throughput_mbps
            return 0.0
        elapsed = self._sample_timestamps[-1] - self._sample_timestamps[0]
        if elapsed <= 0:
            return 0.0
        total_bytes = sum(self._payload_bytes)
        return (total_bytes * 8) / (elapsed * 1_000_000)

    @property
    def utilization(self) -> float:
        return self._utilization

    def to_network_metrics(self) -> NetworkMetrics:
        return NetworkMetrics(
            slice_type=self.slice_type,
            latency_ms=round(self.avg_latency_ms, 3),
            jitter_ms=round(self.jitter_ms, 3),
            packet_loss_rate=round(self.packet_loss_rate, 6),
            throughput_mbps=round(self.throughput_mbps, 3),
            utilization_ratio=round(self._utilization, 4),
            queue_depth=0,
            timestamp=time.time(),
        )

    def get_history(self, limit: int = _HISTORY_MAXLEN) -> list[NetworkMetrics]:
        history = list(self._history)
        return history[-limit:]



class SliceManager:
    """
    Maintains current ``NetworkMetrics`` for each slice and answers queries
    about which slice best serves a given priority level.

    API surface
    -----------
    Measurement ingestion
        ``update_metrics(slice_type, latency_ms, delivered, payload_bytes)``
            Record a single forwarded-message outcome.
        ``update_slice_metrics(metrics: NetworkMetrics)``
            Ingest a full snapshot from the network monitor (1 Hz).

    Queries
        ``get_metrics(slice_type) -> NetworkMetrics``
        ``get_all_metrics() -> dict[SliceType, NetworkMetrics]``
        ``get_healthiest_slice_for_priority(priority) -> SliceType``
        ``is_slice_stressed(slice_type) -> bool``

    Backward-compatible helpers (used by orchestrator / REST API)
        ``get_current_metrics``, ``get_all_metrics_as_dict``,
        ``get_utilization_dict``, ``get_metrics_flat_dict``,
        ``get_slice_history``, ``record_decision``, ``get_stats``
    """

    def __init__(self, settings=None) -> None:
        self._settings = settings
        self._windows: dict[SliceType, _SliceWindow] = {
            st: _SliceWindow(st) for st in SliceType
        }
        self._decision_counts: dict[SliceType, int] = {s: 0 for s in SliceType}
        self._total_decisions: int = 0

        # Seed with healthy baselines
        self._initialise_defaults()

    def _initialise_defaults(self) -> None:
        """Seed each slice with a synthetic healthy baseline ``NetworkMetrics``."""
        now = time.time()
        defaults: dict[SliceType, NetworkMetrics] = {
            SliceType.URLLC: NetworkMetrics(
                slice_type=SliceType.URLLC,
                latency_ms=1.2,
                jitter_ms=0.4,
                packet_loss_rate=0.00005,
                throughput_mbps=18.0,
                utilization_ratio=0.1,
                queue_depth=2,
                timestamp=now,
            ),
            SliceType.EMBB: NetworkMetrics(
                slice_type=SliceType.EMBB,
                latency_ms=12.0,
                jitter_ms=4.0,
                packet_loss_rate=0.0008,
                throughput_mbps=95.0,
                utilization_ratio=0.2,
                queue_depth=5,
                timestamp=now,
            ),
            SliceType.MMTC: NetworkMetrics(
                slice_type=SliceType.MMTC,
                latency_ms=70.0,
                jitter_ms=25.0,
                packet_loss_rate=0.004,
                throughput_mbps=0.8,
                utilization_ratio=0.15,
                queue_depth=10,
                timestamp=now,
            ),
        }
        for st, metrics in defaults.items():
            self._windows[st].ingest_snapshot(metrics)
            self._windows[st].update_utilization(metrics.utilization_ratio)

    def update_metrics(
        self,
        slice_type: SliceType,
        latency_ms: float,
        delivered: bool,
        payload_bytes: int,
    ) -> None:
        """
        Record a new outcome measurement for ``slice_type``.

        The rolling window keeps the last 100 samples; older samples are
        automatically discarded.

        Parameters
        ----------
        slice_type:
            The slice that carried this transmission.
        latency_ms:
            Measured one-way latency in milliseconds.
        delivered:
            Whether the packet was successfully delivered.
        payload_bytes:
            Payload size in bytes (used for throughput estimation).
        """
        self._windows[slice_type].record(latency_ms, delivered, payload_bytes)
        logger.debug(
            "SliceManager.update_metrics: %s lat=%.2f ms delivered=%s payload=%d B",
            slice_type.value,
            latency_ms,
            delivered,
            payload_bytes,
        )

    def update_slice_metrics(self, metrics: NetworkMetrics) -> None:
        """
        Ingest a fresh ``NetworkMetrics`` snapshot (from monitor, ~1 Hz).

        Updates the rolling history and the cached utilisation value.
        """
        st = metrics.slice_type
        self._windows[st].ingest_snapshot(metrics)
        logger.debug(
            "SliceManager.update_slice_metrics: %s lat=%.2f util=%.2f loss=%.4f",
            st.value,
            metrics.latency_ms,
            metrics.utilization_ratio,
            metrics.packet_loss_rate,
        )

    def record_decision(self, slice_type: SliceType) -> None:
        """Increment the decision counter for a slice."""
        self._decision_counts[slice_type] += 1
        self._total_decisions += 1

    def get_metrics(self, slice_type: SliceType) -> NetworkMetrics:
        """Return the current ``NetworkMetrics`` snapshot for ``slice_type``."""
        snap = self._windows[slice_type]._latest_snapshot
        if snap is not None:
            return snap
        return self._windows[slice_type].to_network_metrics()

    def get_all_metrics(self) -> dict[SliceType, NetworkMetrics]:
        """Return current ``NetworkMetrics`` for all three slices."""
        return {st: self.get_metrics(st) for st in SliceType}

    # Backward-compatible alias
    def get_current_metrics(self, slice_type: SliceType) -> Optional[NetworkMetrics]:
        return self.get_metrics(slice_type)

    def get_all_metrics_as_dict(self) -> dict[str, NetworkMetrics]:
        """Return metrics keyed by slice name string (for feature extraction)."""
        return {st.value: self.get_metrics(st) for st in SliceType}

    def get_healthiest_slice_for_priority(self, priority: Priority) -> SliceType:
        """
        Recommend the most appropriate slice for the given priority level.

        Selection policy
        ----------------
        CRITICAL
            Pick the *lowest-latency* slice whose utilisation is <= 0.95.
            If all slices are saturated, fall back to URLLC.
        HIGH
            Use URLLC if its utilisation < 0.85, otherwise EMBB.
        MEDIUM
            Use EMBB if its utilisation < 0.85, otherwise the slice with
            the lowest current utilisation.
        LOW
            Always MMTC.
        """
        if priority == Priority.LOW:
            return SliceType.MMTC

        if priority == Priority.MEDIUM:
            embb_util = self._windows[SliceType.EMBB].utilization
            if embb_util < 0.85:
                return SliceType.EMBB
            return self._lowest_utilisation_slice()

        if priority == Priority.HIGH:
            urllc_util = self._windows[SliceType.URLLC].utilization
            if urllc_util < 0.85:
                return SliceType.URLLC
            return SliceType.EMBB

        # CRITICAL – lowest latency slice that is not over-saturated
        candidates = [
            (self._windows[st].avg_latency_ms, st)
            for st in SliceType
            if self._windows[st].utilization <= 0.95
        ]
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        # All saturated – URLLC is still the safest bet
        return SliceType.URLLC

    def is_slice_stressed(self, slice_type: SliceType) -> bool:
        """
        Return ``True`` if the slice is under stress.

        Conditions: utilisation > 0.85 OR packet loss rate > 1%.
        """
        window = self._windows[slice_type]
        return window.utilization > 0.85 or window.packet_loss_rate > 0.01

    def get_utilization_dict(self) -> dict[str, float]:
        """
        Return flat utilisation dict keyed as used by the rules engine.

        Keys: ``urllc_utilization``, ``embb_utilization``, ``mmtc_utilization``.
        """
        return {
            f"{st.value.lower()}_utilization": self._windows[st].utilization
            for st in SliceType
        }

    def get_metrics_flat_dict(self) -> dict[str, float]:
        """Return a flat dict of all current slice KPIs for anomaly detection."""
        flat: dict[str, float] = {}
        for st in SliceType:
            m = self.get_metrics(st)
            prefix = st.value.lower()
            flat[f"{prefix}_latency_ms"] = m.latency_ms
            flat[f"{prefix}_jitter_ms"] = m.jitter_ms
            flat[f"{prefix}_packet_loss_rate"] = m.packet_loss_rate
            flat[f"{prefix}_utilization_ratio"] = m.utilization_ratio
            flat[f"{prefix}_throughput_mbps"] = m.throughput_mbps
            flat[f"{prefix}_queue_depth"] = float(m.queue_depth)
        return flat

    def get_slice_history(
        self, slice_type: SliceType, limit: int = _HISTORY_MAXLEN
    ) -> list[NetworkMetrics]:
        """Return up to ``limit`` historical snapshots (oldest first)."""
        return self._windows[slice_type].get_history(limit)

    def get_slice_metrics_series(
        self, slice_type: SliceType, limit: int = 100
    ) -> list[NetworkMetrics]:
        return self.get_slice_history(slice_type, limit)

    def _lowest_utilisation_slice(self) -> SliceType:
        return min(SliceType, key=lambda st: self._windows[st].utilization)

    def get_stats(self) -> dict:
        """Return a summary dict for the /health endpoint."""
        slices_status = {}
        for st in SliceType:
            m = self.get_metrics(st)
            slices_status[st.value] = {
                "latency_ms": m.latency_ms,
                "utilization": m.utilization_ratio,
                "packet_loss_rate": m.packet_loss_rate,
                "queue_depth": m.queue_depth,
                "decisions_routed": self._decision_counts.get(st, 0),
                "is_stressed": self.is_slice_stressed(st),
            }
        return {
            "slices": slices_status,
            "total_decisions": self._total_decisions,
        }
