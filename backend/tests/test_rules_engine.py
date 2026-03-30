"""
VESPER – Rules Engine Tests
============================
pytest suite for backend.app.policies.rules_engine.RulesEngine.

All tests use real TelemetryMessage objects and the RulesEngine.evaluate_all /
get_final_decision APIs.

Run from repo root:
    pytest backend/tests/test_rules_engine.py -v
"""

from __future__ import annotations

import time

import pytest

from backend.app.networking.message_schema import (
    EVState,
    EventType,
    Priority,
    SliceType,
    TelemetryMessage,
)
from backend.app.policies.rules_engine import RulesEngine


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_SEQ = 0


def _next_seq() -> int:
    global _SEQ  # noqa: PLW0603
    _SEQ += 1
    return _SEQ


def _msg(
    event_type: EventType = EventType.NORMAL_TELEMETRY,
    priority: Priority = Priority.MEDIUM,
    emergency_flag: bool = False,
    battery_temp: float = 35.0,
    brake_intensity: float = 0.1,
    speed_kmh: float = 60.0,
    obstacle_distance_m: float = 100.0,
    sensor_confidence: float = 0.95,
) -> TelemetryMessage:
    return TelemetryMessage(
        ev_id               = "RULE-TEST",
        timestamp           = time.time(),
        speed_kmh           = speed_kmh,
        acceleration_ms2    = 0.5,
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
        event_type          = event_type,
        priority            = priority,
        sequence_num        = _next_seq(),
    )


_NORMAL_METRICS = {
    "urllc_utilization": 0.35,
    "embb_utilization":  0.40,
    "mmtc_utilization":  0.20,
}

_HIGH_URLLC_METRICS = {
    "urllc_utilization": 0.97,  # > 0.95 threshold
    "embb_utilization":  0.50,
    "mmtc_utilization":  0.30,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collision_warning_always_urllc() -> None:
    """
    R01: COLLISION_WARNING or OBSTACLE_ALERT event type → URLLC CRITICAL.
    """
    engine = RulesEngine()

    for et in (EventType.COLLISION_WARNING, EventType.OBSTACLE_ALERT):
        msg = _msg(event_type=et, priority=Priority.HIGH)
        triggered = engine.evaluate_all(msg, EVState.NORMAL, _NORMAL_METRICS)
        rule_ids = {r.rule_id for r in triggered}

        assert "R01" in rule_ids, f"R01 should trigger for {et}"

        final_slice, final_priority, rule_id = engine.get_final_decision(
            msg, EVState.NORMAL, _NORMAL_METRICS
        )
        assert final_slice    == SliceType.URLLC,    f"Expected URLLC for {et}"
        assert final_priority == Priority.CRITICAL,  f"Expected CRITICAL for {et}"
        assert rule_id        == "R01",              f"Winning rule should be R01 for {et}"


def test_emergency_flag_override() -> None:
    """
    R02: emergency_flag=True → URLLC CRITICAL regardless of other fields.
    """
    engine = RulesEngine()
    msg    = _msg(emergency_flag=True, priority=Priority.LOW)

    triggered = engine.evaluate_all(msg, EVState.NORMAL, _NORMAL_METRICS)
    rule_ids  = {r.rule_id for r in triggered}
    assert "R02" in rule_ids, "R02 should trigger when emergency_flag is True"

    final_slice, final_priority, rule_id = engine.get_final_decision(
        msg, EVState.NORMAL, _NORMAL_METRICS
    )
    assert final_slice    == SliceType.URLLC,   "Emergency flag must route to URLLC"
    assert final_priority == Priority.CRITICAL, "Emergency flag must be CRITICAL priority"


def test_battery_overheat_rule() -> None:
    """
    R03: battery_temp_celsius > 65 → URLLC CRITICAL.
    """
    engine = RulesEngine()
    msg    = _msg(battery_temp=70.0)  # 70 > 65 °C

    triggered = engine.evaluate_all(msg, EVState.NORMAL, _NORMAL_METRICS)
    rule_ids  = {r.rule_id for r in triggered}
    assert "R03" in rule_ids, "R03 should trigger when battery_temp > 65 °C"

    final_slice, final_priority, rule_id = engine.get_final_decision(
        msg, EVState.NORMAL, _NORMAL_METRICS
    )
    assert final_slice    == SliceType.URLLC,   "Battery overheat must route to URLLC"
    assert final_priority == Priority.CRITICAL, "Battery overheat must be CRITICAL"
    assert rule_id        == "R03",             "Winning rule should be R03"


def test_battery_below_threshold_no_r03() -> None:
    """
    R03 must NOT trigger when battery_temp_celsius <= 65 °C.
    """
    engine = RulesEngine()
    msg    = _msg(battery_temp=60.0)  # exactly below threshold

    triggered = engine.evaluate_all(msg, EVState.NORMAL, _NORMAL_METRICS)
    rule_ids  = {r.rule_id for r in triggered}
    assert "R03" not in rule_ids, "R03 should NOT trigger when battery_temp <= 65 °C"


def test_log_upload_always_mmtc() -> None:
    """
    R11: LOG_UPLOAD or ANALYTICS_PING → MMTC LOW.
    """
    engine = RulesEngine()

    for et in (EventType.LOG_UPLOAD, EventType.ANALYTICS_PING):
        msg = _msg(event_type=et, priority=Priority.MEDIUM)
        final_slice, final_priority, rule_id = engine.get_final_decision(
            msg, EVState.NORMAL, _NORMAL_METRICS
        )
        assert final_slice    == SliceType.MMTC, f"R11: expected MMTC for {et}"
        assert final_priority == Priority.LOW,   f"R11: expected LOW for {et}"
        assert rule_id        == "R11",          f"R11: winning rule should be R11 for {et}"


def test_diagnostic_always_embb() -> None:
    """
    R12: DIAGNOSTIC → EMBB MEDIUM.
    """
    engine = RulesEngine()
    msg    = _msg(event_type=EventType.DIAGNOSTIC, priority=Priority.LOW)

    final_slice, final_priority, rule_id = engine.get_final_decision(
        msg, EVState.NORMAL, _NORMAL_METRICS
    )
    assert final_slice    == SliceType.EMBB,    "R12: DIAGNOSTIC must route to EMBB"
    assert final_priority == Priority.MEDIUM,   "R12: DIAGNOSTIC must be MEDIUM priority"
    assert rule_id        == "R12",             "Winning rule must be R12"


def test_urllc_shed_low_traffic() -> None:
    """
    R09: When URLLC utilization > 95% and the message is LOW priority, the
    traffic must be shed to MMTC at LOW priority.
    """
    engine = RulesEngine()
    msg    = _msg(priority=Priority.LOW)

    final_slice, final_priority, rule_id = engine.get_final_decision(
        msg, EVState.NORMAL, _HIGH_URLLC_METRICS
    )
    assert final_slice    == SliceType.MMTC, (
        f"R09: LOW traffic should be shed to MMTC under high URLLC load, got {final_slice}"
    )
    assert final_priority == Priority.LOW,   "R09: shedded traffic should remain LOW"
    assert rule_id        == "R09",          f"Winning rule should be R09, got {rule_id}"


def test_safety_rules_beat_shedding() -> None:
    """
    R01 (safety) must win over R09 (shedding) even when URLLC is saturated.
    A COLLISION_WARNING at LOW priority on a saturated URLLC slice must still
    be routed to URLLC CRITICAL.
    """
    engine = RulesEngine()
    msg    = _msg(
        event_type    = EventType.COLLISION_WARNING,
        priority      = Priority.LOW,  # would normally trigger shedding
    )

    final_slice, final_priority, rule_id = engine.get_final_decision(
        msg, EVState.NORMAL, _HIGH_URLLC_METRICS
    )
    assert final_slice    == SliceType.URLLC,   (
        "Safety rule R01 must override shedding and route to URLLC"
    )
    assert final_priority == Priority.CRITICAL, (
        "Safety rule R01 must override priority to CRITICAL"
    )
    assert rule_id        == "R01",             (
        f"Winning rule must be R01 (safety), not shedding; got {rule_id}"
    )


def test_no_rules_fired_returns_default() -> None:
    """
    When no rules trigger, get_final_decision should return a sensible default
    slice based on the message priority (MEDIUM → EMBB, rule_id = None).
    """
    engine  = RulesEngine()
    msg     = _msg(priority=Priority.MEDIUM, event_type=EventType.NORMAL_TELEMETRY)
    metrics = _NORMAL_METRICS

    final_slice, final_priority, rule_id = engine.get_final_decision(
        msg, EVState.NORMAL, metrics
    )
    assert rule_id    is None,            "No rules triggered: rule_id should be None"
    assert final_slice == SliceType.EMBB, f"Default for MEDIUM should be EMBB, got {final_slice}"


def test_r05_hard_brake_at_speed() -> None:
    """
    R05: brake_intensity > 0.9 AND speed_kmh > 50 → URLLC HIGH.
    """
    engine = RulesEngine()
    msg    = _msg(brake_intensity=0.95, speed_kmh=75.0)

    triggered = engine.evaluate_all(msg, EVState.NORMAL, _NORMAL_METRICS)
    rule_ids  = {r.rule_id for r in triggered}
    assert "R05" in rule_ids, "R05 should trigger on hard braking at speed"

    final_slice, final_priority, rule_id = engine.get_final_decision(
        msg, EVState.NORMAL, _NORMAL_METRICS
    )
    assert final_slice    == SliceType.URLLC, "R05 must route to URLLC"
    assert final_priority == Priority.HIGH,   "R05 must assign HIGH priority"
