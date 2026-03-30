"""
VESPER – Safety Event Engine
==============================
Generates realistic, temporally correlated safety events using a first-order
discrete-time Markov chain over the ``EventType`` state space.

The engine is intentionally decoupled from the transport layer: callers
receive ``SafetyEvent`` objects and decide how to route or log them.

Architecture
------------
* ``EventSeverity``         – four-level severity enum.
* ``SafetyEvent``           – dataclass representing a single safety incident.
* ``MarkovEventGenerator``  – stateful generator; next state is sampled from
                               a transition matrix that is skewed by live
                               telemetry urgency.
* ``EventFactory``          – static factory with one method per event class.

Markov Chain
------------
States (rows/columns) are ordered as:

    0 NORMAL_TELEMETRY
    1 OBSTACLE_ALERT
    2 COLLISION_WARNING
    3 BRAKE_ALERT
    4 BATTERY_OVERHEAT
    5 SENSOR_DEGRADATION

The base transition matrix encodes domain knowledge:
- Obstacle alerts tend to escalate toward collision warnings.
- Collision warnings either resolve (back to obstacle alert) or persist.
- Battery overheat is a self-sustaining state once entered.
- Sensor degradation usually returns to normal.

When live telemetry urgency > 0.7 the matrix rows are re-weighted to bias
transitions toward OBSTACLE_ALERT and BRAKE_ALERT before sampling.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from backend.app.networking.message_schema import EventType, TelemetryMessage


class EventSeverity(str, Enum):
    """Four-level severity classification for safety events."""

    INFO     = "INFO"
    WARNING  = "WARNING"
    ERROR    = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SafetyEvent:
    """
    A single safety incident record produced by the event engine.

    Fields
    ------
    event_id    : Globally unique identifier (UUID v4 string).
    ev_id       : Vehicle that triggered the event.
    event_type  : Semantic classification (maps to ``EventType`` enum).
    severity    : ``EventSeverity`` level.
    description : Human-readable summary suitable for operator dashboards.
    timestamp   : Unix epoch float (seconds).
    metadata    : Arbitrary key-value pairs for domain-specific context
                  (e.g. obstacle distance, temperature reading).
    """

    event_id:    str            = field(default_factory=lambda: str(uuid.uuid4()))
    ev_id:       str            = ""
    event_type:  EventType      = EventType.NORMAL_TELEMETRY
    severity:    EventSeverity  = EventSeverity.INFO
    description: str            = ""
    timestamp:   float          = field(default_factory=time.time)
    metadata:    Dict[str, Any] = field(default_factory=dict)


# Ordered list of EventType states used as row/column indices.
_MARKOV_STATES: List[EventType] = [
    EventType.NORMAL_TELEMETRY,    # 0
    EventType.OBSTACLE_ALERT,      # 1
    EventType.COLLISION_WARNING,   # 2
    EventType.BRAKE_ALERT,         # 3
    EventType.BATTERY_OVERHEAT,    # 4
    EventType.SENSOR_DEGRADATION,  # 5
]

_STATE_INDEX: Dict[EventType, int] = {s: i for i, s in enumerate(_MARKOV_STATES)}

# Base transition matrix P[i][j] = P(next=j | current=i).
# Rows sum to 1.0.
#
#                  NORM   OBS    COLL   BRAKE  BATT   SENSOR
_BASE_TRANSITION = [
    # 0 NORMAL_TELEMETRY
    [0.70,         0.08,  0.02,  0.06,  0.04,  0.10],
    # 1 OBSTACLE_ALERT
    [0.20,         0.35,  0.30,  0.08,  0.02,  0.05],
    # 2 COLLISION_WARNING
    [0.05,         0.30,  0.45,  0.12,  0.03,  0.05],
    # 3 BRAKE_ALERT
    [0.30,         0.20,  0.10,  0.30,  0.04,  0.06],
    # 4 BATTERY_OVERHEAT
    [0.10,         0.05,  0.05,  0.05,  0.70,  0.05],
    # 5 SENSOR_DEGRADATION
    [0.50,         0.10,  0.05,  0.10,  0.05,  0.20],
]


def _validate_transition_matrix(matrix: List[List[float]]) -> None:
    """Assert that every row of the transition matrix sums to 1 (within float tolerance)."""
    n = len(_MARKOV_STATES)
    assert len(matrix) == n, f"Matrix must have {n} rows, got {len(matrix)}"
    for i, row in enumerate(matrix):
        assert len(row) == n, f"Row {i} must have {n} columns, got {len(row)}"
        total = sum(row)
        assert abs(total - 1.0) < 1e-6, f"Row {i} sums to {total}, expected 1.0"


_validate_transition_matrix(_BASE_TRANSITION)


def _weighted_choice(weights: List[float]) -> int:
    """
    Sample an index from ``weights`` using the alias / cumulative method.

    Returns the index of the chosen element.
    """
    total = sum(weights)
    r = random.random() * total
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r < cumulative:
            return i
    return len(weights) - 1  # rounding guard


def _bias_row_toward_urgency(row: List[float], urgency: float) -> List[float]:
    """
    Return a new probability row that biases toward OBSTACLE_ALERT (idx=1)
    and BRAKE_ALERT (idx=3) when ``urgency`` is high.

    The bias is applied as a soft weighting: when urgency=1.0, up to 40% of
    the probability mass is shifted from NORMAL_TELEMETRY toward the two
    urgent states proportionally.

    The returned row is re-normalised to sum to 1.
    """
    if urgency <= 0.0:
        return list(row)

    result = list(row)
    bias_weight = urgency * 0.40  # maximum 40% mass shifted

    # Take mass from NORMAL_TELEMETRY (idx=0)
    shift_from_normal = min(result[0] * bias_weight, result[0] * 0.8)
    result[0] -= shift_from_normal

    # Distribute to OBSTACLE_ALERT and BRAKE_ALERT equally
    result[1] += shift_from_normal * 0.55   # more toward obstacle
    result[3] += shift_from_normal * 0.45   # and brake

    # Re-normalise
    total = sum(result)
    return [w / total for w in result]


class MarkovEventGenerator:
    """
    First-order discrete-time Markov chain over the 6-state safety event space.

    The chain starts in ``NORMAL_TELEMETRY``.  On each call to ``next_event``
    the current state and live telemetry urgency together determine the
    transition probability vector from which the next state is sampled.

    Parameters
    ----------
    seed : int, optional
        Fixed random seed for reproducibility in tests.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

        self.current_state: EventType = EventType.NORMAL_TELEMETRY

        # Copy the base matrix so instance-level modifications are safe
        self._transition: List[List[float]] = [list(row) for row in _BASE_TRANSITION]

    def next_event(self, ev_id: str, telemetry: TelemetryMessage) -> SafetyEvent:
        """
        Advance the Markov chain and return the resulting ``SafetyEvent``.

        Steps
        -----
        1. Compute the current urgency score from ``telemetry.urgency``.
        2. Retrieve the base transition row for ``current_state``.
        3. If urgency > 0.7, bias the row toward OBSTACLE_ALERT / BRAKE_ALERT.
        4. Sample the next state.
        5. Build and return a ``SafetyEvent`` using ``EventFactory``.
        6. Update ``current_state``.

        The telemetry values are used both for urgency biasing and to
        populate the ``SafetyEvent`` metadata with meaningful context.
        """
        urgency = telemetry.urgency  # uses the @property defined on TelemetryMessage

        current_idx = _STATE_INDEX[self.current_state]
        base_row    = self._transition[current_idx]

        if urgency > 0.7:
            row = _bias_row_toward_urgency(base_row, urgency)
        else:
            row = list(base_row)

        next_idx   = _weighted_choice(row)
        next_state = _MARKOV_STATES[next_idx]

        # Build event using EventFactory
        event = self._build_event(ev_id, next_state, telemetry)

        self.current_state = next_state
        return event

    def _build_event(
        self,
        ev_id: str,
        event_type: EventType,
        telemetry: TelemetryMessage,
    ) -> SafetyEvent:
        """Dispatch to the correct ``EventFactory`` method based on ``event_type``."""
        if event_type == EventType.OBSTACLE_ALERT:
            return EventFactory.create_obstacle_alert(
                ev_id,
                distance_m = telemetry.obstacle_distance_m,
                speed_kmh  = telemetry.speed_kmh,
            )

        if event_type == EventType.COLLISION_WARNING:
            # Compute a rough time-to-impact from distance and speed
            speed_ms = max(0.1, telemetry.speed_kmh / 3.6)
            tti_s    = telemetry.obstacle_distance_m / speed_ms
            return EventFactory.create_collision_warning(ev_id, time_to_impact_s=tti_s)

        if event_type == EventType.BRAKE_ALERT:
            return EventFactory.create_brake_alert(
                ev_id,
                intensity = telemetry.brake_intensity,
                speed_kmh = telemetry.speed_kmh,
            )

        if event_type == EventType.BATTERY_OVERHEAT:
            return EventFactory.create_battery_overheat(
                ev_id, temp_celsius=telemetry.battery_temp_celsius
            )

        if event_type == EventType.SENSOR_DEGRADATION:
            return EventFactory.create_sensor_degradation(
                ev_id, confidence=telemetry.sensor_confidence
            )

        # Default: NORMAL_TELEMETRY
        return SafetyEvent(
            ev_id       = ev_id,
            event_type  = EventType.NORMAL_TELEMETRY,
            severity    = EventSeverity.INFO,
            description = (
                f"Normal telemetry from {ev_id}: "
                f"speed={telemetry.speed_kmh:.1f} km/h, "
                f"SoC={telemetry.state_of_charge_pct:.1f}%, "
                f"temp={telemetry.battery_temp_celsius:.1f}°C."
            ),
            metadata = {
                "speed_kmh":          telemetry.speed_kmh,
                "soc_pct":            telemetry.state_of_charge_pct,
                "battery_temp_c":     telemetry.battery_temp_celsius,
                "sensor_confidence":  telemetry.sensor_confidence,
            },
        )

    def reset(self) -> None:
        """Return the chain to the initial NORMAL_TELEMETRY state."""
        self.current_state = EventType.NORMAL_TELEMETRY


class EventFactory:
    """
    Static factory that constructs ``SafetyEvent`` objects for each
    recognised event class.  All methods follow a consistent contract:

    * The ``event_id`` is auto-generated.
    * The ``timestamp`` is set to ``time.time()`` at construction.
    * ``metadata`` is populated with all domain-specific numeric values so
      downstream consumers do not need to re-parse ``description`` strings.
    """

    @staticmethod
    def create_obstacle_alert(
        ev_id: str,
        distance_m: float,
        speed_kmh: float,
    ) -> SafetyEvent:
        """
        Create an OBSTACLE_ALERT event.

        Severity scales with proximity:
        * < 10 m  → ERROR
        * < 20 m  → WARNING
        * ≥ 20 m  → INFO
        """
        if distance_m < 10.0:
            severity = EventSeverity.ERROR
        elif distance_m < 20.0:
            severity = EventSeverity.WARNING
        else:
            severity = EventSeverity.INFO

        return SafetyEvent(
            ev_id       = ev_id,
            event_type  = EventType.OBSTACLE_ALERT,
            severity    = severity,
            description = (
                f"Obstacle detected {distance_m:.1f} m ahead of {ev_id} "
                f"travelling at {speed_kmh:.1f} km/h. "
                f"Estimated time to reach obstacle: "
                f"{(distance_m / max(0.1, speed_kmh / 3.6)):.1f} s."
            ),
            metadata = {
                "obstacle_distance_m": distance_m,
                "speed_kmh":           speed_kmh,
                "tti_s":               distance_m / max(0.1, speed_kmh / 3.6),
            },
        )

    @staticmethod
    def create_brake_alert(
        ev_id: str,
        intensity: float,
        speed_kmh: float,
    ) -> SafetyEvent:
        """
        Create a BRAKE_ALERT event.

        Severity scales with brake intensity:
        * > 0.9   → CRITICAL
        * > 0.7   → ERROR
        * > 0.5   → WARNING
        * ≤ 0.5   → INFO
        """
        if intensity > 0.9:
            severity = EventSeverity.CRITICAL
        elif intensity > 0.7:
            severity = EventSeverity.ERROR
        elif intensity > 0.5:
            severity = EventSeverity.WARNING
        else:
            severity = EventSeverity.INFO

        braking_force_kn = intensity * 15.0  # rough estimate for a ~1500 kg EV
        decel_ms2 = intensity * 9.8           # maximum ≈ 1 g

        return SafetyEvent(
            ev_id       = ev_id,
            event_type  = EventType.BRAKE_ALERT,
            severity    = severity,
            description = (
                f"Hard braking event on {ev_id}: intensity={intensity:.2f}, "
                f"speed={speed_kmh:.1f} km/h, "
                f"estimated deceleration={decel_ms2:.1f} m/s², "
                f"braking force ≈ {braking_force_kn:.1f} kN."
            ),
            metadata = {
                "brake_intensity":    intensity,
                "speed_kmh":          speed_kmh,
                "deceleration_ms2":   decel_ms2,
                "braking_force_kn":   braking_force_kn,
            },
        )

    @staticmethod
    def create_battery_overheat(
        ev_id: str,
        temp_celsius: float,
    ) -> SafetyEvent:
        """
        Create a BATTERY_OVERHEAT event.

        Severity escalation:
        * > 80 °C  → CRITICAL (thermal runaway territory)
        * > 65 °C  → ERROR    (management system intervention required)
        * > 55 °C  → WARNING  (elevated temperature)
        * ≤ 55 °C  → INFO     (marginal / informational)
        """
        if temp_celsius > 80.0:
            severity = EventSeverity.CRITICAL
            risk_note = "THERMAL RUNAWAY RISK – immediate action required."
        elif temp_celsius > 65.0:
            severity = EventSeverity.ERROR
            risk_note = "Thermal management intervention required."
        elif temp_celsius > 55.0:
            severity = EventSeverity.WARNING
            risk_note = "Monitor closely; reduce motor load."
        else:
            severity = EventSeverity.INFO
            risk_note = "Temperature within elevated but manageable range."

        excess = max(0.0, temp_celsius - 25.0)  # degrees above nominal ambient

        return SafetyEvent(
            ev_id       = ev_id,
            event_type  = EventType.BATTERY_OVERHEAT,
            severity    = severity,
            description = (
                f"Battery thermal event on {ev_id}: "
                f"temp={temp_celsius:.1f}°C (+{excess:.1f}°C above nominal). "
                f"{risk_note}"
            ),
            metadata = {
                "battery_temp_c":     temp_celsius,
                "excess_above_nominal_c": excess,
                "risk_note":          risk_note,
            },
        )

    @staticmethod
    def create_sensor_degradation(
        ev_id: str,
        confidence: float,
    ) -> SafetyEvent:
        """
        Create a SENSOR_DEGRADATION event.

        Severity:
        * < 0.1   → CRITICAL (most sensors non-functional)
        * < 0.3   → ERROR    (significant degradation)
        * < 0.6   → WARNING  (partial degradation)
        * ≥ 0.6   → INFO     (minor degradation)
        """
        if confidence < 0.1:
            severity = EventSeverity.CRITICAL
            guidance = "Vehicle must pull over immediately – autonomous features disabled."
        elif confidence < 0.3:
            severity = EventSeverity.ERROR
            guidance = "Reduce speed and disengage assisted driving features."
        elif confidence < 0.6:
            severity = EventSeverity.WARNING
            guidance = "Increase following distance; monitor sensor status."
        else:
            severity = EventSeverity.INFO
            guidance = "Sensor suite operating below optimal confidence."

        degradation_pct = (1.0 - confidence) * 100.0

        return SafetyEvent(
            ev_id       = ev_id,
            event_type  = EventType.SENSOR_DEGRADATION,
            severity    = severity,
            description = (
                f"Sensor degradation on {ev_id}: "
                f"confidence={confidence:.3f} ({degradation_pct:.1f}% degraded). "
                f"{guidance}"
            ),
            metadata = {
                "sensor_confidence":   confidence,
                "degradation_pct":     degradation_pct,
                "guidance":            guidance,
            },
        )

    @staticmethod
    def create_collision_warning(
        ev_id: str,
        time_to_impact_s: float,
    ) -> SafetyEvent:
        """
        Create a COLLISION_WARNING event.

        Severity based on time-to-impact (TTI):
        * TTI < 1.5 s  → CRITICAL (imminent, autonomous emergency brake)
        * TTI < 3.0 s  → ERROR    (urgent driver action required)
        * TTI < 5.0 s  → WARNING  (hazard ahead, brake recommended)
        * TTI ≥ 5.0 s  → INFO     (early warning)
        """
        tti = max(0.0, time_to_impact_s)

        if tti < 1.5:
            severity = EventSeverity.CRITICAL
            action   = "AUTONOMOUS EMERGENCY BRAKING ACTIVATED."
        elif tti < 3.0:
            severity = EventSeverity.ERROR
            action   = "Immediate driver braking required."
        elif tti < 5.0:
            severity = EventSeverity.WARNING
            action   = "Slow down – obstacle closing rapidly."
        else:
            severity = EventSeverity.INFO
            action   = "Hazard detected ahead; proceed with caution."

        return SafetyEvent(
            ev_id       = ev_id,
            event_type  = EventType.COLLISION_WARNING,
            severity    = severity,
            description = (
                f"Collision warning for {ev_id}: "
                f"time-to-impact={tti:.2f} s. {action}"
            ),
            metadata = {
                "time_to_impact_s": tti,
                "action":           action,
            },
        )
