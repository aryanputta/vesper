"""
VESPER – EV State Machine Tests
================================
pytest suite for backend.app.core.state_machine.EVStateMachine.

Run from repo root:
    pytest backend/tests/test_state_machine.py -v
"""

from __future__ import annotations

import time

import pytest

from backend.app.core.state_machine import (
    EVStateMachine,
    _CLEAR_TO_NORMAL,
    _CLEAR_TO_STEP_DOWN,
)
from backend.app.networking.message_schema import EVState, EventType, Priority, TelemetryMessage


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_SEQ = 0


def _next_seq() -> int:
    global _SEQ  # noqa: PLW0603
    _SEQ += 1
    return _SEQ


def _msg(
    obstacle_distance_m: float = 100.0,
    brake_intensity: float = 0.0,
    battery_temp: float = 35.0,
    sensor_confidence: float = 0.99,
    speed_kmh: float = 60.0,
    emergency_flag: bool = False,
    ev_id: str = "SM-TEST",
) -> TelemetryMessage:
    return TelemetryMessage(
        ev_id               = ev_id,
        timestamp           = time.time(),
        speed_kmh           = speed_kmh,
        acceleration_ms2    = 0.0,
        brake_intensity     = brake_intensity,
        steering_angle_deg  = 0.0,
        battery_temp_celsius= battery_temp,
        state_of_charge_pct = 80.0,
        motor_load_pct      = 20.0,
        gps_lat             = 37.7749,
        gps_lon             = -122.4194,
        obstacle_distance_m = obstacle_distance_m,
        sensor_confidence   = sensor_confidence,
        emergency_flag      = emergency_flag,
        event_type          = EventType.NORMAL_TELEMETRY,
        priority            = Priority.MEDIUM,
        sequence_num        = _next_seq(),
    )


def _clear_msg(ev_id: str = "SM-TEST") -> TelemetryMessage:
    """Return a message with no triggers active (all-clear)."""
    return _msg(
        obstacle_distance_m = 200.0,
        brake_intensity     = 0.0,
        battery_temp        = 30.0,
        sensor_confidence   = 0.99,
        speed_kmh           = 20.0,  # slow speed: no CAUTION from obstacle+speed
        emergency_flag      = False,
        ev_id               = ev_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_initial_state_is_normal() -> None:
    """Fresh state machine must start in NORMAL state."""
    sm = EVStateMachine(ev_id="SM-001")
    assert sm.get_state() == EVState.NORMAL


def test_obstacle_triggers_caution() -> None:
    """
    Obstacle at 15 m (< 20 m) + speed 40 km/h (> 30 km/h) → CAUTION.
    The state machine must step from NORMAL to CAUTION in a single transition.
    """
    sm  = EVStateMachine(ev_id="SM-002")
    msg = _msg(obstacle_distance_m=15.0, speed_kmh=40.0)

    new_state, reason = sm.transition(msg)

    assert new_state == EVState.CAUTION, (
        f"Expected CAUTION from obstacle+speed, got {new_state}. Reason: {reason}"
    )


def test_close_obstacle_triggers_critical() -> None:
    """
    Obstacle at 5 m (< 8 m) + brake 0.8 (> 0.7) → CRITICAL.
    """
    sm  = EVStateMachine(ev_id="SM-003")
    msg = _msg(obstacle_distance_m=5.0, brake_intensity=0.8)

    new_state, reason = sm.transition(msg)

    assert new_state == EVState.CRITICAL, (
        f"Expected CRITICAL from close obstacle + brake, got {new_state}. Reason: {reason}"
    )


def test_battery_overheat_triggers_critical() -> None:
    """
    battery_temp_celsius = 70 > 65 °C threshold → CRITICAL.
    """
    sm  = EVStateMachine(ev_id="SM-004")
    msg = _msg(battery_temp=70.0)

    new_state, reason = sm.transition(msg)

    assert new_state == EVState.CRITICAL, (
        f"Expected CRITICAL from battery overheat, got {new_state}. Reason: {reason}"
    )


def test_emergency_flag() -> None:
    """
    emergency_flag=True → EMERGENCY regardless of other telemetry fields.
    """
    sm  = EVStateMachine(ev_id="SM-005")
    msg = _msg(emergency_flag=True, obstacle_distance_m=200.0, battery_temp=30.0)

    new_state, reason = sm.transition(msg)

    assert new_state == EVState.EMERGENCY, (
        f"Expected EMERGENCY from emergency_flag, got {new_state}. Reason: {reason}"
    )


def test_recovery_after_clear() -> None:
    """
    Feed the state machine enough consecutive all-clear readings after a
    CRITICAL trigger to drive the state at least one step toward NORMAL.

    Sequence:
    1. Trigger CRITICAL (battery overheat).
    2. Feed _CLEAR_TO_STEP_DOWN all-clear readings → should drop to CAUTION.
    3. Feed _CLEAR_TO_STEP_DOWN more clears → should drop to RECOVERY.
    (We don't require full NORMAL recovery here to keep the test fast.)
    """
    sm = EVStateMachine(ev_id="SM-006")

    # Step 1: trigger CRITICAL
    sm.transition(_msg(battery_temp=70.0))
    assert sm.get_state() == EVState.CRITICAL

    # Step 2: clear until first step-down
    for _ in range(_CLEAR_TO_STEP_DOWN):
        sm.transition(_clear_msg(ev_id="SM-006"))

    assert sm.get_state() == EVState.CAUTION, (
        f"Expected CAUTION after first step-down, got {sm.get_state()}"
    )

    # Step 3: clear until second step-down
    for _ in range(_CLEAR_TO_STEP_DOWN):
        sm.transition(_clear_msg(ev_id="SM-006"))

    assert sm.get_state() == EVState.RECOVERY, (
        f"Expected RECOVERY after second step-down, got {sm.get_state()}"
    )


def test_full_recovery_to_normal() -> None:
    """
    After entering RECOVERY, feeding _CLEAR_TO_NORMAL consecutive all-clear
    readings must transition the machine back to NORMAL.
    """
    sm = EVStateMachine(ev_id="SM-007")

    # Drive to RECOVERY (two levels of step-down from CRITICAL)
    sm.transition(_msg(battery_temp=70.0))
    for _ in range(_CLEAR_TO_STEP_DOWN):
        sm.transition(_clear_msg(ev_id="SM-007"))
    for _ in range(_CLEAR_TO_STEP_DOWN):
        sm.transition(_clear_msg(ev_id="SM-007"))
    assert sm.get_state() == EVState.RECOVERY

    # Now drive to NORMAL
    for _ in range(_CLEAR_TO_NORMAL):
        sm.transition(_clear_msg(ev_id="SM-007"))

    assert sm.get_state() == EVState.NORMAL, (
        f"Expected NORMAL after full recovery, got {sm.get_state()}"
    )


def test_state_history_tracking() -> None:
    """
    After several transitions the state_history deque must be non-empty and
    contain tuples of (float, EVState, str).
    """
    sm = EVStateMachine(ev_id="SM-008")

    sm.transition(_msg(battery_temp=70.0))   # → CRITICAL
    sm.transition(_clear_msg(ev_id="SM-008"))

    assert len(sm.state_history) >= 1, "state_history should be populated after transitions"

    for entry in sm.state_history:
        ts, state, reason = entry
        assert isinstance(ts,     float),    f"Timestamp should be float, got {type(ts)}"
        assert isinstance(state,  EVState),  f"State should be EVState, got {type(state)}"
        assert isinstance(reason, str),      f"Reason should be str, got {type(reason)}"
        assert reason,                       "Reason string should not be empty"


def test_no_downgrade_from_critical_with_caution_trigger() -> None:
    """
    When already in CRITICAL and only a CAUTION-level trigger fires
    (obstacle < 20 m but not < 8 m, no hard brake), the state must NOT
    downgrade to CAUTION.
    """
    sm = EVStateMachine(ev_id="SM-009")

    # Enter CRITICAL via battery overheat
    sm.transition(_msg(battery_temp=70.0))
    assert sm.get_state() == EVState.CRITICAL

    # Send CAUTION-level trigger only (obstacle 15 m, low brake, normal battery)
    caution_msg = _msg(
        obstacle_distance_m = 15.0,
        speed_kmh           = 45.0,
        brake_intensity     = 0.1,
        battery_temp        = 30.0,
    )
    new_state, _ = sm.transition(caution_msg)

    assert new_state == EVState.CRITICAL, (
        "CAUTION-level trigger must not downgrade a vehicle already in CRITICAL state; "
        f"got {new_state}"
    )


def test_latency_hint_escalates_critical_to_emergency() -> None:
    """
    When the vehicle is already CRITICAL and latency_hint_ms > 20 ms is passed,
    the state must escalate to EMERGENCY.
    """
    sm = EVStateMachine(ev_id="SM-010")

    # Get to CRITICAL
    sm.transition(_msg(battery_temp=70.0))
    assert sm.get_state() == EVState.CRITICAL

    # High latency hint while CRITICAL → EMERGENCY
    msg = _clear_msg(ev_id="SM-010")
    new_state, reason = sm.transition(msg, latency_hint_ms=25.0)

    assert new_state == EVState.EMERGENCY, (
        f"High latency while CRITICAL should escalate to EMERGENCY; got {new_state}"
    )


def test_hard_brake_alone_triggers_critical() -> None:
    """
    brake_intensity > 0.9 standalone (no obstacle) → CRITICAL via the hard-brake
    trigger.
    """
    sm  = EVStateMachine(ev_id="SM-011")
    msg = _msg(brake_intensity=0.95, obstacle_distance_m=200.0)

    new_state, reason = sm.transition(msg)

    assert new_state == EVState.CRITICAL, (
        f"Hard brake alone should trigger CRITICAL, got {new_state}. Reason: {reason}"
    )
