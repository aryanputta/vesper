"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
state_machine.py – Per-vehicle EV state machine with hysteresis and history.

State diagram
-------------
                          ┌─────────────┐
                          │   NORMAL    │◄──────────────────┐
                          └──────┬──────┘                   │
                    triggers     │                10 all-clear
                          ┌──────▼──────┐                   │
                          │   CAUTION   │◄──────────┐        │
                          └──────┬──────┘           │        │
                    triggers     │          5-clear ─┘        │
                          ┌──────▼──────┐           ▲   5/10-clear
                          │  CRITICAL   │───────────┤        │
                          └──────┬──────┘           │        │
               emergency_flag    │          high     │        │
               latency hint      │          latency  │        │
                          ┌──────▼──────┐           │        │
                          │  EMERGENCY  │───────────┘        │
                          └──────┬──────┘                    │
                    5 all-clear  │                           │
                          ┌──────▼──────┐                    │
                          │  RECOVERY   │────────────────────┘
                          └─────────────┘
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

from ..networking.message_schema import EVState, TelemetryMessage

logger = logging.getLogger(__name__)


# CAUTION triggers
_OBSTACLE_CAUTION_M = 20.0       # metres
_SPEED_CAUTION_KMH = 30.0        # km/h

# CRITICAL triggers
_OBSTACLE_CRITICAL_M = 8.0       # metres
_BRAKE_CRITICAL_THRESHOLD = 0.7  # normalised
_BRAKE_HARD_THRESHOLD = 0.9      # normalised (standalone)
_BATTERY_TEMP_CRITICAL = 65.0    # °C
_SENSOR_CONF_CRITICAL = 0.3      # below this → CRITICAL

# EMERGENCY escalation from CRITICAL
_LATENCY_EMERGENCY_MS = 20.0     # ms hint from caller

# All-clear hysteresis counters
_CLEAR_TO_STEP_DOWN = 5          # consecutive clears to descend one level
_CLEAR_TO_NORMAL = 10            # consecutive clears in RECOVERY → NORMAL

# History ring-buffer capacity
_HISTORY_MAXLEN = 50



class EVStateMachine:
    """
    Finite-state machine tracking the safety status of a single EV.

    Usage
    -----
    .. code-block:: python

        sm = EVStateMachine(ev_id="EV-001")
        new_state, reason = sm.transition(telemetry_msg)
        print(sm.get_state(), sm.time_in_state())
    """

    def __init__(self, ev_id: str) -> None:
        self.ev_id = ev_id
        self._state: EVState = EVState.NORMAL
        self._state_entered_at: float = time.monotonic()
        self._clear_streak: int = 0  # consecutive readings with no triggers

        # Deque of (wall_clock_timestamp: float, state: EVState, reason: str)
        self.state_history: deque[tuple[float, EVState, str]] = deque(maxlen=_HISTORY_MAXLEN)
        self._record_history(EVState.NORMAL, "initial state")

    def transition(
        self,
        msg: TelemetryMessage,
        latency_hint_ms: Optional[float] = None,
    ) -> tuple[EVState, str]:
        """
        Evaluate ``msg`` against transition rules and update internal state.

        Parameters
        ----------
        msg:
            Current telemetry frame from the vehicle.
        latency_hint_ms:
            Optional network latency observed on the URLLC slice at the time
            of processing.  When provided and > 20 ms while already CRITICAL,
            the vehicle escalates to EMERGENCY.

        Returns
        -------
        (new_state, reason)
            ``new_state``  – the EVState after applying rules.
            ``reason``     – human-readable explanation of what changed (or why
                             the state was maintained).
        """
        triggered, reason = self._evaluate_triggers(msg, latency_hint_ms)

        if triggered:
            self._clear_streak = 0
            new_state = self._apply_trigger_rules(msg, latency_hint_ms, reason)
        else:
            self._clear_streak += 1
            new_state, reason = self._apply_clear_rules()

        if new_state != self._state:
            logger.info(
                "EV %s: %s → %s (%s)",
                self.ev_id,
                self._state.name,
                new_state.name,
                reason,
            )
            self._state = new_state
            self._state_entered_at = time.monotonic()
            self._record_history(new_state, reason)
        else:
            logger.debug(
                "EV %s: stays %s – clear_streak=%d (%s)",
                self.ev_id,
                self._state.name,
                self._clear_streak,
                reason,
            )

        return self._state, reason

    def get_state(self) -> EVState:
        """Return the current EVState without modifying it."""
        return self._state

    def time_in_state(self) -> float:
        """Seconds elapsed since entering the current state."""
        return time.monotonic() - self._state_entered_at

    def _evaluate_triggers(
        self,
        msg: TelemetryMessage,
        latency_hint_ms: Optional[float],
    ) -> tuple[bool, str]:
        """
        Check all trigger conditions in priority order.

        Returns (triggered: bool, reason: str).
        """
        # Highest priority: hard emergency flag
        if msg.emergency_flag:
            return True, "emergency_flag asserted by vehicle safety controller"

        # CRITICAL escalation from latency (only meaningful when already CRITICAL)
        if (
            self._state == EVState.CRITICAL
            and latency_hint_ms is not None
            and latency_hint_ms > _LATENCY_EMERGENCY_MS
        ):
            return True, (
                f"URLLC latency {latency_hint_ms:.1f} ms > "
                f"{_LATENCY_EMERGENCY_MS} ms while in CRITICAL"
            )

        # Hard-brake standalone trigger
        if msg.brake_intensity > _BRAKE_HARD_THRESHOLD:
            return True, (
                f"brake_intensity {msg.brake_intensity:.2f} > {_BRAKE_HARD_THRESHOLD}"
            )

        # Close obstacle + heavy braking → CRITICAL
        if (
            msg.obstacle_distance_m < _OBSTACLE_CRITICAL_M
            and msg.brake_intensity > _BRAKE_CRITICAL_THRESHOLD
        ):
            return True, (
                f"obstacle {msg.obstacle_distance_m:.1f} m < {_OBSTACLE_CRITICAL_M} m "
                f"AND brake {msg.brake_intensity:.2f} > {_BRAKE_CRITICAL_THRESHOLD}"
            )

        # Battery overheat
        if msg.battery_temp_celsius > _BATTERY_TEMP_CRITICAL:
            return True, (
                f"battery_temp {msg.battery_temp_celsius:.1f} °C > {_BATTERY_TEMP_CRITICAL} °C"
            )

        # Sensor degradation
        if msg.sensor_confidence < _SENSOR_CONF_CRITICAL:
            return True, (
                f"sensor_confidence {msg.sensor_confidence:.2f} < {_SENSOR_CONF_CRITICAL}"
            )

        # Moderate obstacle + speed → CAUTION
        if (
            msg.obstacle_distance_m < _OBSTACLE_CAUTION_M
            and msg.speed_kmh > _SPEED_CAUTION_KMH
        ):
            return True, (
                f"obstacle {msg.obstacle_distance_m:.1f} m < {_OBSTACLE_CAUTION_M} m "
                f"AND speed {msg.speed_kmh:.1f} km/h > {_SPEED_CAUTION_KMH} km/h"
            )

        return False, "no triggers active"

    def _apply_trigger_rules(
        self,
        msg: TelemetryMessage,
        latency_hint_ms: Optional[float],
        reason: str,
    ) -> EVState:
        """
        Given that at least one trigger fired, determine the target state.
        We select the *most severe* applicable state.
        """
        # EMERGENCY conditions
        if msg.emergency_flag:
            return EVState.EMERGENCY

        if (
            self._state == EVState.CRITICAL
            and latency_hint_ms is not None
            and latency_hint_ms > _LATENCY_EMERGENCY_MS
        ):
            return EVState.EMERGENCY

        # CRITICAL conditions
        if msg.brake_intensity > _BRAKE_HARD_THRESHOLD:
            return EVState.CRITICAL

        if (
            msg.obstacle_distance_m < _OBSTACLE_CRITICAL_M
            and msg.brake_intensity > _BRAKE_CRITICAL_THRESHOLD
        ):
            return EVState.CRITICAL

        if msg.battery_temp_celsius > _BATTERY_TEMP_CRITICAL:
            return EVState.CRITICAL

        if msg.sensor_confidence < _SENSOR_CONF_CRITICAL:
            return EVState.CRITICAL

        # CAUTION condition
        if (
            msg.obstacle_distance_m < _OBSTACLE_CAUTION_M
            and msg.speed_kmh > _SPEED_CAUTION_KMH
        ):
            # Do not downgrade from a more severe state
            if self._state in (EVState.CRITICAL, EVState.EMERGENCY):
                return self._state
            return EVState.CAUTION

        # Fallback: maintain current state if nothing matched (shouldn't happen)
        return self._state

    def _apply_clear_rules(self) -> tuple[EVState, str]:
        """
        When no triggers are active, apply hysteresis step-down logic.

        Returns (new_state, reason).
        """
        current = self._state

        if current == EVState.NORMAL:
            return EVState.NORMAL, "no triggers – steady NORMAL"

        if current == EVState.RECOVERY:
            if self._clear_streak >= _CLEAR_TO_NORMAL:
                return EVState.NORMAL, (
                    f"RECOVERY complete after {self._clear_streak} all-clear readings"
                )
            return EVState.RECOVERY, (
                f"RECOVERY – {self._clear_streak}/{_CLEAR_TO_NORMAL} all-clear readings"
            )

        # For CAUTION, CRITICAL, EMERGENCY: step down after _CLEAR_TO_STEP_DOWN clears
        if self._clear_streak >= _CLEAR_TO_STEP_DOWN:
            step_down_map: dict[EVState, EVState] = {
                EVState.EMERGENCY: EVState.CRITICAL,
                EVState.CRITICAL: EVState.CAUTION,
                EVState.CAUTION: EVState.RECOVERY,
            }
            next_state = step_down_map.get(current, current)
            # Reset streak so we need another burst of clears for the next step
            self._clear_streak = 0
            return next_state, (
                f"stepped down from {current.name} after {_CLEAR_TO_STEP_DOWN} all-clear readings"
            )

        return current, (
            f"no triggers – {self._clear_streak}/{_CLEAR_TO_STEP_DOWN} clears before step-down"
        )

    def _record_history(self, state: EVState, reason: str) -> None:
        """Append an entry to the bounded state history deque."""
        self.state_history.append((time.time(), state, reason))
