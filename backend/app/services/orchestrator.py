"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
orchestrator.py – Core orchestration loop.

SliceOrchestrator is the heart of VESPER.  It ingests TelemetryMessages from
the AsyncPriorityQueue and runs the full processing pipeline:

    ingest → deduplicate → state-machine → features → urgency → ML →
    rules-engine → admission-control → decision → metrics → alerting

Each pipeline execution is timed; decision latencies are tracked in a rolling
list for the /stats endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from backend.app.networking.message_schema import (
    EventType,
    EVState,
    Priority,
    SliceDecision,
    SliceType,
    TelemetryMessage,
)
from backend.app.networking.priority_queue import AsyncPriorityQueue
from backend.app.core.feature_extractor import EVFeatureBuffer
from backend.app.core.state_machine import EVStateMachine
from backend.app.core.urgency_scorer import UrgencyScorer
from backend.app.core.event_deduplicator import EventDeduplicator
from backend.app.policies.rules_engine import RulesEngine
from backend.app.policies.admission_controller import AdmissionController
from backend.app.services.slice_manager import SliceManager
from backend.app.services.alert_service import AlertService
from ml.inference.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Enums whose values drive the ML integer ↔ SliceType mapping
_INT_TO_SLICE: dict[int, SliceType] = {
    0: SliceType.MMTC,
    1: SliceType.EMBB,
    2: SliceType.URLLC,
}
_SLICE_DOWNGRADE: dict[SliceType, SliceType] = {
    SliceType.URLLC: SliceType.EMBB,
    SliceType.EMBB: SliceType.MMTC,
    SliceType.MMTC: SliceType.MMTC,  # already lowest
}

# Safety event types that trigger alert publishing
_SAFETY_EVENT_TYPES: frozenset[EventType] = frozenset({
    EventType.COLLISION_WARNING,
    EventType.OBSTACLE_ALERT,
    EventType.BRAKE_ALERT,
    EventType.BATTERY_OVERHEAT,
    EventType.SENSOR_DEGRADATION,
})

# Safety EV states that trigger alert publishing
_SAFETY_STATES: frozenset[EVState] = frozenset({
    EVState.CRITICAL,
    EVState.EMERGENCY,
})

# How many messages to process before pruning the priority queue
_PRUNE_INTERVAL: int = 10

# Maximum latency history entries kept in memory
_LATENCY_HISTORY_MAXLEN: int = 10_000

# Typical telemetry payload size in bytes (used for admission control)
_DEFAULT_PAYLOAD_BYTES: int = 512


class SliceOrchestrator:
    """
    Central orchestrator that drives the full VESPER processing pipeline.

    Parameters
    ----------
    queue:
        The async priority queue from which telemetry messages are consumed.
    slice_manager:
        Manages per-slice metrics and records routing decisions.
    alert_service:
        Publishes safety alerts to operators and downstream consumers.
    model_registry:
        Registry of loaded ML models; provides predict / detect helpers.
    settings:
        Application settings object (from ``get_settings()``).
    """

    def __init__(
        self,
        queue: AsyncPriorityQueue,
        slice_manager: SliceManager,
        alert_service: AlertService,
        model_registry: ModelRegistry,
        settings,
    ) -> None:
        self.queue = queue
        self.slice_manager = slice_manager
        self.alert_service = alert_service
        self.model_registry = model_registry
        self.settings = settings

        # Per-EV state machines
        self._state_machines: dict[str, EVStateMachine] = {}

        # Per-EV feature buffers
        self._feature_buffers: dict[str, EVFeatureBuffer] = {}

        # Shared components
        self.urgency_scorer = UrgencyScorer()
        self.rules_engine = RulesEngine()
        self.admission_controller = AdmissionController()
        self.deduplicator = EventDeduplicator()

        # Decision store (ring buffer approximated via list + trim)
        self._decisions: list[SliceDecision] = []
        self._max_decisions: int = 1000

        # Metrics
        self.total_processed: int = 0
        self.total_reassignments: int = 0
        self.safety_event_count: int = 0
        self.decision_latencies: list[float] = []  # milliseconds

        # Prometheus exporter (optional – injected post-construction)
        self.metrics_exporter = None

        # Loop control
        self.running: bool = False
        self._iteration_count: int = 0
        self._loop_task: Optional[asyncio.Task] = None

        logger.info("SliceOrchestrator initialised")

    async def process_message(self, msg: TelemetryMessage) -> Optional[SliceDecision]:
        """
        Run the full processing pipeline for a single TelemetryMessage.

        Steps
        -----
        1.  Deduplication check
        2.  EV state machine update
        3.  Feature extraction
        4.  Urgency scoring
        5.  ML inference (or heuristic fallback)
        6.  Rules engine override
        7.  Admission control
        8.  Build SliceDecision
        9.  Update slice manager
        10. Publish safety alert if needed
        11. Record latency

        Returns
        -------
        SliceDecision or None
            None is returned if the message was identified as a duplicate.
        """
        t_start = time.perf_counter()

        # ── Step 1: Deduplication ─────────────────────────────────────────────
        if await self.deduplicator.is_duplicate(msg):
            logger.debug("Duplicate message dropped: ev=%s seq=%d", msg.ev_id, msg.sequence_num)
            return None

        ev_id = msg.ev_id

        # ── Step 2: EV state machine ──────────────────────────────────────────
        if ev_id not in self._state_machines:
            self._state_machines[ev_id] = EVStateMachine(ev_id=ev_id)
        state_machine = self._state_machines[ev_id]
        ev_state, state_reason = state_machine.transition(msg)

        # ── Step 3: Feature buffer update ─────────────────────────────────────
        if ev_id not in self._feature_buffers:
            self._feature_buffers[ev_id] = EVFeatureBuffer(ev_id=ev_id)
        buffer = self._feature_buffers[ev_id]
        buffer.update(msg)

        # ── Step 4: Urgency scoring ───────────────────────────────────────────
        network_metrics = self.slice_manager.get_all_metrics()
        network_metrics_str = self.slice_manager.get_all_metrics_as_dict()
        features = buffer.extract_features(network_metrics_str)
        urgency_score = self.urgency_scorer.compute(msg, ev_state)
        features["urgency_score"] = urgency_score

        # ── Step 5: ML inference or heuristic fallback ────────────────────────
        ml_slice_int: int
        ml_confidence: float
        shap_values: list[tuple[str, float]] = []
        congestion_pred: float = 0.0

        if self.model_registry.is_ready():
            try:
                t_ml_start = time.perf_counter()
                ml_slice_int, ml_confidence, shap_values = self.model_registry.predict_slice(features)
                congestion_pred = self.model_registry.predict_congestion(features)

                # Anomaly detection on network metrics
                network_flat = self.slice_manager.get_metrics_flat_dict()
                is_anomaly, anomaly_score = self.model_registry.detect_anomaly(network_flat)
                if is_anomaly:
                    logger.warning(
                        "Network anomaly detected (score=%.4f) – bumping message priority for ev=%s",
                        anomaly_score,
                        ev_id,
                    )
                    # Bump message priority one level
                    priority_order = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]
                    current_idx = priority_order.index(msg.priority)
                    msg = msg.model_copy(
                        update={"priority": priority_order[max(0, current_idx - 1)]}
                    )
                    if self.metrics_exporter is not None:
                        self.metrics_exporter.record_anomaly()

                t_ml_end = time.perf_counter()
                ml_latency_ms = (t_ml_end - t_ml_start) * 1000.0
                logger.debug("ML inference in %.3f ms for ev=%s", ml_latency_ms, ev_id)
                if self.metrics_exporter is not None:
                    self.metrics_exporter.record_ml_inference(ml_latency_ms)

            except Exception as exc:
                logger.error("ML inference failed for ev=%s: %s – falling back to heuristic", ev_id, exc)
                ml_slice_int = self.heuristic_fallback(urgency_score, msg.event_type)
                ml_confidence = 0.0
        else:
            ml_slice_int = self.heuristic_fallback(urgency_score, msg.event_type)
            ml_confidence = 0.0

        ml_slice = _INT_TO_SLICE.get(ml_slice_int, SliceType.EMBB)

        # ── Step 6: Rules engine override ─────────────────────────────────────
        slice_metrics_for_rules = self.slice_manager.get_utilization_dict()
        final_slice, final_priority, rule_triggered = self.rules_engine.get_final_decision(
            msg, ev_state, slice_metrics_for_rules
        )

        if rule_triggered is not None:
            # Safety / housekeeping rule fired – use its decision
            logger.debug(
                "Rule %s fired for ev=%s: slice=%s priority=%s",
                rule_triggered,
                ev_id,
                final_slice.value,
                final_priority.name,
            )
            if self.metrics_exporter is not None:
                self.metrics_exporter.record_rule_trigger(rule_triggered)
        else:
            # No rule override – use ML decision; final_priority from rules engine default
            final_slice = ml_slice

        # ── Step 7: Admission control ─────────────────────────────────────────
        payload_size = getattr(msg, "payload_size_bytes", _DEFAULT_PAYLOAD_BYTES)
        self.admission_controller.update_utilization(
            final_slice,
            (network_metrics.get(final_slice) or network_metrics.get(
                SliceType.URLLC  # fallback
            )).utilization_ratio if network_metrics.get(final_slice) else 0.0,
        )
        admission_result = self.admission_controller.admit(
            final_slice, final_priority, payload_size
        )

        if admission_result == "REJECT":
            if final_priority != Priority.CRITICAL:
                # Downgrade slice
                downgraded = _SLICE_DOWNGRADE[final_slice]
                logger.info(
                    "Admission REJECT for ev=%s on %s – downgrading to %s",
                    ev_id,
                    final_slice.value,
                    downgraded.value,
                )
                final_slice = downgraded
                self.total_reassignments += 1
            else:
                # CRITICAL must go through – override rejection
                logger.warning(
                    "Admission REJECT overridden for CRITICAL message ev=%s", ev_id
                )

        elif admission_result == "QUEUE":
            # Secondary buffer – log for now; process in next cycle
            logger.info(
                "Admission QUEUE for ev=%s on %s – will retry next cycle",
                ev_id,
                final_slice.value,
            )

        # ── Step 8: Build SliceDecision ───────────────────────────────────────
        t_end = time.perf_counter()
        decision_latency_ms = (t_end - t_start) * 1000.0

        decision = SliceDecision(
            ev_id=ev_id,
            message_id=f"{ev_id}:{msg.sequence_num}",
            assigned_slice=final_slice,
            confidence_score=ml_confidence,
            urgency_score=urgency_score,
            ev_state=ev_state,
            rule_triggered=rule_triggered,
            model_prediction=ml_slice,
            timestamp=time.time(),
            features_used=features,
            shap_top_features=shap_values,
        )

        # Store decision
        self._decisions.append(decision)
        if len(self._decisions) > self._max_decisions:
            self._decisions = self._decisions[-self._max_decisions:]

        # ── Step 9: Update slice manager ──────────────────────────────────────
        self.slice_manager.record_decision(final_slice)
        self.total_processed += 1

        # ── Step 10: Publish safety alert ─────────────────────────────────────
        is_safety = (
            msg.event_type in _SAFETY_EVENT_TYPES
            or ev_state in _SAFETY_STATES
            or msg.emergency_flag
        )
        if is_safety:
            self.safety_event_count += 1
            await self.alert_service.publish_safety_event(
                ev_id=ev_id,
                event_type=msg.event_type,
                severity=final_priority,
                description=(
                    f"{msg.event_type.value} detected for {ev_id} "
                    f"(state={ev_state.value}, urgency={urgency_score:.3f})"
                ),
                slice_assigned=final_slice,
                response_latency_ms=decision_latency_ms,
            )
            if self.metrics_exporter is not None:
                self.metrics_exporter.record_safety_event(decision_latency_ms)

        # ── Step 11: Record latency ───────────────────────────────────────────
        self.decision_latencies.append(decision_latency_ms)
        if len(self.decision_latencies) > _LATENCY_HISTORY_MAXLEN:
            self.decision_latencies = self.decision_latencies[-_LATENCY_HISTORY_MAXLEN:]

        if self.metrics_exporter is not None:
            self.metrics_exporter.record_message(
                slice=final_slice.value,
                priority=final_priority.name,
                event_type=msg.event_type.value,
                latency_ms=decision_latency_ms,
                delivered=(admission_result != "REJECT"),
            )

        logger.debug(
            "Processed ev=%s seq=%d → slice=%s rule=%s urgency=%.3f latency=%.2f ms",
            ev_id,
            msg.sequence_num,
            final_slice.value,
            rule_triggered or "ML",
            urgency_score,
            decision_latency_ms,
        )

        return decision

    @staticmethod
    def heuristic_fallback(urgency_score: float, event_type: EventType) -> int:
        """
        Simple rule-based slice selection used when ML models are not loaded.

        Returns
        -------
        int
            0 = MMTC, 1 = EMBB, 2 = URLLC
        """
        if urgency_score > 0.75:
            return 2  # URLLC
        if urgency_score > 0.40:
            return 1  # EMBB
        return 0       # MMTC

    async def run_loop(self) -> None:
        """
        Continuously dequeue and process telemetry messages.

        The loop yields control to the event loop via ``asyncio.sleep(0)``
        after each message so that other coroutines (TCP/UDP listeners, API
        handlers) remain responsive.  Every ``_PRUNE_INTERVAL`` iterations the
        priority queue's expired entries are pruned.
        """
        logger.info("SliceOrchestrator run loop started")
        while self.running:
            try:
                msg = await self.queue.get()
                if msg is not None:
                    await self.process_message(msg)
                    self._iteration_count += 1

                    if self._iteration_count % _PRUNE_INTERVAL == 0:
                        pruned = self.queue.prune_expired()
                        if pruned:
                            logger.debug("Pruned %d expired messages from queue", pruned)
            except asyncio.CancelledError:
                logger.info("SliceOrchestrator run loop cancelled")
                break
            except Exception as exc:
                logger.exception("Unhandled error in orchestrator run loop: %s", exc)

            await asyncio.sleep(0)  # yield control

        logger.info("SliceOrchestrator run loop stopped")

    async def start(self) -> None:
        """Start the orchestration loop as a background asyncio task."""
        if self.running:
            logger.warning("SliceOrchestrator.start() called but already running")
            return
        self.running = True
        self._loop_task = asyncio.create_task(self.run_loop(), name="orchestrator_loop")
        logger.info("SliceOrchestrator started")

    async def stop(self) -> None:
        """Gracefully stop the orchestration loop."""
        self.running = False
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        logger.info("SliceOrchestrator stopped")

    def get_stats(self) -> dict:
        """
        Return aggregate orchestrator statistics.

        Includes
        --------
        * total_processed      – messages successfully through the pipeline
        * reassignments        – times a message was downgraded at admission
        * avg_decision_latency_ms
        * p99_decision_latency_ms
        * safety_events        – total safety events dispatched
        * per_ev_state         – current EVState for every tracked vehicle
        """
        latencies = self.decision_latencies
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p99_latency = 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            p99_idx = max(0, int(len(sorted_lat) * 0.99) - 1)
            p99_latency = sorted_lat[p99_idx]

        per_ev = {
            ev_id: sm.get_state().value
            for ev_id, sm in self._state_machines.items()
        }

        return {
            "total_processed": self.total_processed,
            "reassignments": self.total_reassignments,
            "avg_decision_latency_ms": round(avg_latency, 4),
            "p99_decision_latency_ms": round(p99_latency, 4),
            "safety_events": self.safety_event_count,
            "per_ev_state": per_ev,
            "queue_stats": self.queue.stats(),
            "dedup_stats": self.deduplicator.stats(),
            "admission_stats": self.admission_controller.get_stats(),
            "model_registry_status": self.model_registry.status(),
        }

    def get_recent_decisions(self, limit: int = 50) -> list[SliceDecision]:
        """Return the most recent ``limit`` SliceDecisions, newest first."""
        return list(reversed(self._decisions))[:limit]

    def get_decision_by_id(self, decision_id: str) -> Optional[SliceDecision]:
        """Find a decision by its UUID, or return None."""
        for d in self._decisions:
            if d.decision_id == decision_id:
                return d
        return None

    def get_ev_state(self, ev_id: str) -> Optional[EVState]:
        """Return the current EVState for a vehicle, or None if not tracked."""
        sm = self._state_machines.get(ev_id)
        return sm.get_state() if sm else None

    def get_ev_state_history(self, ev_id: str) -> list[tuple[float, str, str]]:
        """
        Return the state history for a vehicle as a list of
        (timestamp, state_name, reason) tuples.
        """
        sm = self._state_machines.get(ev_id)
        if sm is None:
            return []
        return [(ts, state.value, reason) for ts, state, reason in sm.state_history]
