"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
rules_engine.py – Hard safety rules engine.

Rules are evaluated in order (R01–R12).  Safety rules (R01–R08) take
precedence over load-shedding rules (R09–R10) when determining the final
slice/priority decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from backend.app.networking.message_schema import (
    EVState,
    EventType,
    Priority,
    SliceType,
    TelemetryMessage,
)

logger = logging.getLogger(__name__)



@dataclass
class RuleResult:
    """Outcome of evaluating a single rule against a telemetry message."""

    rule_id: str
    triggered: bool
    override_slice: Optional[SliceType]
    override_priority: Optional[Priority]
    reason: str



class Rule:
    """Encapsulates a single deterministic safety or routing rule."""

    def __init__(
        self,
        rule_id: str,
        description: str,
        condition: Callable[[TelemetryMessage, EVState, dict], bool],
        slice_override: Optional[SliceType],
        priority_override: Optional[Priority],
        reason: str,
    ) -> None:
        self.rule_id = rule_id
        self.description = description
        self.condition = condition
        self.slice_override = slice_override
        self.priority_override = priority_override
        self.reason = reason

    def evaluate(
        self,
        msg: TelemetryMessage,
        ev_state: EVState,
        slice_metrics: dict,
    ) -> RuleResult:
        """
        Evaluate the rule condition against the supplied context.

        Parameters
        ----------
        msg:
            The incoming TelemetryMessage.
        ev_state:
            Current state-machine state for the vehicle.
        slice_metrics:
            Dict with keys ``'urllc_utilization'``, ``'embb_utilization'``,
            ``'mmtc_utilization'`` (float, 0–1).
        """
        try:
            triggered = bool(self.condition(msg, ev_state, slice_metrics))
        except Exception:
            logger.exception("Rule %s condition raised an exception; treating as not triggered.", self.rule_id)
            triggered = False

        return RuleResult(
            rule_id=self.rule_id,
            triggered=triggered,
            override_slice=self.slice_override if triggered else None,
            override_priority=self.priority_override if triggered else None,
            reason=self.reason if triggered else "",
        )



def _escalate_priority(current: Priority) -> Priority:
    """Return the next-higher priority (CRITICAL is already the highest)."""
    order = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]
    idx = order.index(current)
    return order[max(0, idx - 1)]



class RulesEngine:
    """
    Ordered collection of 12 deterministic safety and routing rules.

    Evaluation order
    ----------------
    Rules are evaluated in declaration order (R01 first).  For
    ``get_final_decision`` the first *safety* rule (R01–R08) that triggers
    wins the slice/priority assignment.  Load-shedding rules (R09–R10) are
    only applied when no safety rule fired.  Housekeeping rules (R11–R12)
    override the default assignment regardless of utilisation.
    """

    # IDs for the rule bands
    SAFETY_RULE_IDS = {"R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08"}
    SHED_RULE_IDS = {"R09", "R10"}
    HOUSEKEEPING_RULE_IDS = {"R11", "R12"}
    # Satellite and 6G routing rules evaluated after housekeeping and load-shedding
    SATELLITE_6G_RULE_IDS = {"R13", "R14", "R15", "R16", "R17"}

    def __init__(self) -> None:
        self._rules: list[Rule] = self._build_rules()

    @staticmethod
    def _build_rules() -> list[Rule]:  # noqa: PLR0914
        rules: list[Rule] = []

        # R01 – Collision / obstacle safety
        rules.append(Rule(
            rule_id="R01",
            description="Collision warning or obstacle alert → URLLC CRITICAL",
            condition=lambda msg, _ev, _sm: msg.event_type in (
                EventType.COLLISION_WARNING,
                EventType.OBSTACLE_ALERT,
            ),
            slice_override=SliceType.URLLC,
            priority_override=Priority.CRITICAL,
            reason="Collision or obstacle event detected – maximum urgency required.",
        ))

        # R02 – Hard emergency flag
        rules.append(Rule(
            rule_id="R02",
            description="Emergency flag set → URLLC CRITICAL",
            condition=lambda msg, _ev, _sm: bool(msg.emergency_flag),
            slice_override=SliceType.URLLC,
            priority_override=Priority.CRITICAL,
            reason="Vehicle emergency flag is active.",
        ))

        # R03 – Battery overheat (thermal runaway risk)
        rules.append(Rule(
            rule_id="R03",
            description="Battery temperature > 65 °C → URLLC CRITICAL",
            condition=lambda msg, _ev, _sm: msg.battery_temp_celsius > 65.0,
            slice_override=SliceType.URLLC,
            priority_override=Priority.CRITICAL,
            reason="Battery overheat emergency – temperature exceeds 65 °C.",
        ))

        # R04 – Vehicle in EMERGENCY state
        rules.append(Rule(
            rule_id="R04",
            description="EV state == EMERGENCY → URLLC CRITICAL",
            condition=lambda _msg, ev_state, _sm: ev_state == EVState.EMERGENCY,
            slice_override=SliceType.URLLC,
            priority_override=Priority.CRITICAL,
            reason="Vehicle state machine is in EMERGENCY state.",
        ))

        # R05 – Hard braking at speed
        rules.append(Rule(
            rule_id="R05",
            description="Hard brake (>0.9) at speed (>50 km/h) → URLLC HIGH",
            condition=lambda msg, _ev, _sm: (
                msg.brake_intensity > 0.9 and msg.speed_kmh > 50.0
            ),
            slice_override=SliceType.URLLC,
            priority_override=Priority.HIGH,
            reason="Hard braking at speed detected – collision avoidance manoeuvre possible.",
        ))

        # R06 – Low sensor confidence: escalate priority by one level (no slice override)
        #
        # This rule has no static priority_override because the new priority
        # depends on the *current* message priority.  We encode the special
        # behaviour by using a sentinel value of None for priority_override and
        # handling the escalation inside get_final_decision / evaluate_all.
        # The RuleResult will carry triggered=True; callers check the rule_id
        # to apply one-step escalation.
        rules.append(Rule(
            rule_id="R06",
            description="Sensor confidence < 0.2 → escalate priority by one level",
            condition=lambda msg, _ev, _sm: msg.sensor_confidence < 0.2,
            slice_override=None,
            priority_override=None,  # special: escalate, see get_final_decision
            reason="Sensor confidence critically low – routing priority escalated.",
        ))

        # R07 – Vehicle CRITICAL state with URLLC headroom available
        rules.append(Rule(
            rule_id="R07",
            description="EV state == CRITICAL AND urllc_utilization < 0.9 → URLLC HIGH",
            condition=lambda _msg, ev_state, sm: (
                ev_state == EVState.CRITICAL
                and sm.get("urllc_utilization", 1.0) < 0.9
            ),
            slice_override=SliceType.URLLC,
            priority_override=Priority.HIGH,
            reason="Vehicle in CRITICAL state and URLLC has capacity.",
        ))

        # R08 – Battery overheat event type
        rules.append(Rule(
            rule_id="R08",
            description="EventType == BATTERY_OVERHEAT → URLLC HIGH",
            condition=lambda msg, _ev, _sm: msg.event_type == EventType.BATTERY_OVERHEAT,
            slice_override=SliceType.URLLC,
            priority_override=Priority.HIGH,
            reason="Battery overheat event – high-priority URLLC routing required.",
        ))

        # R09 – Load shedding: low-priority traffic when URLLC saturated
        rules.append(Rule(
            rule_id="R09",
            description="URLLC > 95% util AND priority == LOW → force MMTC LOW",
            condition=lambda msg, _ev, sm: (
                sm.get("urllc_utilization", 0.0) > 0.95
                and msg.priority == Priority.LOW
            ),
            slice_override=SliceType.MMTC,
            priority_override=Priority.LOW,
            reason="URLLC overloaded – shedding LOW-priority traffic to MMTC.",
        ))

        # R10 – Load shedding: medium-priority traffic when URLLC saturated
        rules.append(Rule(
            rule_id="R10",
            description="URLLC > 95% util AND priority == MEDIUM → force EMBB",
            condition=lambda msg, _ev, sm: (
                sm.get("urllc_utilization", 0.0) > 0.95
                and msg.priority == Priority.MEDIUM
            ),
            slice_override=SliceType.EMBB,
            priority_override=Priority.MEDIUM,
            reason="URLLC overloaded – moving MEDIUM-priority traffic to eMBB.",
        ))

        # R11 – Housekeeping: log/analytics traffic always goes to MMTC
        rules.append(Rule(
            rule_id="R11",
            description="LOG_UPLOAD or ANALYTICS_PING → MMTC LOW",
            condition=lambda msg, _ev, _sm: msg.event_type in (
                EventType.LOG_UPLOAD,
                EventType.ANALYTICS_PING,
            ),
            slice_override=SliceType.MMTC,
            priority_override=Priority.LOW,
            reason="Log/analytics traffic assigned to MMTC at LOW priority.",
        ))

        # R12 – Housekeeping: diagnostic traffic goes to EMBB
        rules.append(Rule(
            rule_id="R12",
            description="DIAGNOSTIC → EMBB MEDIUM",
            condition=lambda msg, _ev, _sm: msg.event_type == EventType.DIAGNOSTIC,
            slice_override=SliceType.EMBB,
            priority_override=Priority.MEDIUM,
            reason="Diagnostic traffic assigned to eMBB at MEDIUM priority.",
        ))

        # R13 – NTN offload: satellite link active and URLLC approaching saturation
        # Non-critical traffic is offloaded to the NTN_SAT slice to relieve terrestrial congestion
        rules.append(Rule(
            rule_id="R13",
            description="Satellite link active AND urllc_utilization > 0.90 → NTN_SAT",
            condition=lambda msg, _ev, sm: (
                sm.get("satellite_link_active", False)
                and sm.get("urllc_utilization", 0.0) > 0.90
                and msg.priority in (Priority.LOW, Priority.MEDIUM)
            ),
            slice_override=SliceType.NTN_SAT,
            priority_override=Priority.MEDIUM,
            reason="URLLC near saturation – offloading non-critical traffic to NTN satellite slice.",
        ))

        # R14 – Poor satellite geometry: force off NTN_SAT and fall back to terrestrial
        # A satellite below 10° elevation has a very long slant range and is about to set;
        # continuing to use it would add hundreds of ms of unnecessary latency.
        rules.append(Rule(
            rule_id="R14",
            description="Satellite elevation < 10° → disable NTN_SAT, escalate to terrestrial",
            condition=lambda msg, _ev, sm: (
                sm.get("satellite_elevation_deg", 90.0) < 10.0
                and sm.get("satellite_link_active", False)
            ),
            slice_override=SliceType.EMBB,   # best available terrestrial fallback for most traffic
            priority_override=Priority.HIGH,
            reason="Satellite elevation below 10° – poor link geometry; forced fallback to terrestrial.",
        ))

        # R15 – Emergency with terrestrial saturation: allow NTN_SAT for any traffic
        # Last-resort path when an emergency vehicle cannot get through on any terrestrial slice.
        rules.append(Rule(
            rule_id="R15",
            description="EMERGENCY + all terrestrial > 0.95 util → NTN_SAT",
            condition=lambda msg, ev_state, sm: (
                ev_state == EVState.EMERGENCY
                and sm.get("urllc_utilization", 0.0) > 0.95
                and sm.get("embb_utilization", 0.0) > 0.95
                and sm.get("mmtc_utilization", 0.0) > 0.95
            ),
            slice_override=SliceType.NTN_SAT,
            priority_override=Priority.CRITICAL,
            reason="All terrestrial slices saturated during emergency – using NTN satellite as final path.",
        ))

        # R16 – V2X coordination traffic routes to 6G_URLLC when available
        # V2X messages require sub-millisecond coordination; 6G_URLLC is the preferred path.
        # Falls back to terrestrial URLLC when 6G coverage is not available.
        rules.append(Rule(
            rule_id="R16",
            description="V2X_COORDINATION → 6G_URLLC if available, else URLLC",
            condition=lambda msg, _ev, _sm: msg.event_type == EventType.V2X_COORDINATION,
            slice_override=SliceType.SIX_G_URLLC,
            priority_override=Priority.HIGH,
            reason="V2X coordination routed to 6G_URLLC for sub-millisecond latency.",
        ))

        # R17 – Holographic/XR sync traffic requires high throughput on 6G_EMBB_PLUS
        # Holographic display streams are bandwidth-hungry (Tbps class) but not latency-critical.
        rules.append(Rule(
            rule_id="R17",
            description="HOLOGRAPHIC_SYNC → 6G_eMBB+ or EMBB",
            condition=lambda msg, _ev, _sm: msg.event_type == EventType.HOLOGRAPHIC_SYNC,
            slice_override=SliceType.SIX_G_EMBB_PLUS,
            priority_override=Priority.MEDIUM,
            reason="Holographic sync stream assigned to 6G eMBB+ for Tbps-class throughput.",
        ))

        return rules

    def evaluate_all(
        self,
        msg: TelemetryMessage,
        ev_state: EVState,
        slice_metrics: dict,
    ) -> list[RuleResult]:
        """
        Evaluate every rule and return the list of *all triggered* results.
        """
        results: list[RuleResult] = []
        for rule in self._rules:
            result = rule.evaluate(msg, ev_state, slice_metrics)
            if result.triggered:
                results.append(result)
        return results

    def get_final_decision(
        self,
        msg: TelemetryMessage,
        ev_state: EVState,
        slice_metrics: dict,
    ) -> tuple[SliceType, Priority, Optional[str]]:
        """
        Return ``(slice, priority, rule_id | None)`` for the message.

        Decision precedence
        -------------------
        1. Safety rules R01–R08 (first to trigger wins slice/priority).
           - R06 is special: it has no slice override – it only bumps priority.
        2. Housekeeping rules R11–R12 override the default assignment.
        3. Load-shedding rules R09–R10 (applied only when no safety rule fired).
        4. Fallback: use the slice/priority already on the message.

        R06 interaction
        ---------------
        If R06 fires alongside a safety rule that does carry a slice override,
        the priority from that safety rule is escalated one more step.  If R06
        fires with no other safety rule, the message's *current* priority is
        escalated and the slice falls through to the next applicable rule.
        """
        triggered = self.evaluate_all(msg, ev_state, slice_metrics)

        safety_hits = [r for r in triggered if r.rule_id in self.SAFETY_RULE_IDS]
        shed_hits = [r for r in triggered if r.rule_id in self.SHED_RULE_IDS]
        housekeeping_hits = [r for r in triggered if r.rule_id in self.HOUSEKEEPING_RULE_IDS]
        satellite_6g_hits = [r for r in triggered if r.rule_id in self.SATELLITE_6G_RULE_IDS]

        # Start from the message's current slice assignment
        # (we don't have an assigned slice on the message, so we use a
        # sensible default based on priority).
        current_priority: Priority = msg.priority
        r06_active = any(r.rule_id == "R06" for r in safety_hits)

        # --- Housekeeping rules take absolute precedence for their event types
        if housekeeping_hits:
            # R11 / R12 win unconditionally – these are non-negotiable routing rules
            winning = housekeeping_hits[0]
            final_slice = winning.override_slice
            final_priority = winning.override_priority
            return final_slice, final_priority, winning.rule_id

        # --- Safety rules (R01–R08, excluding R06 pure escalation)
        safety_with_slice = [r for r in safety_hits if r.rule_id != "R06" and r.override_slice is not None]
        if safety_with_slice:
            winning = safety_with_slice[0]
            final_slice = winning.override_slice
            final_priority = winning.override_priority
            # Apply R06 escalation on top if active
            if r06_active and final_priority is not None:
                final_priority = _escalate_priority(final_priority)
            return final_slice, final_priority, winning.rule_id

        # --- R06 alone (no other safety rule with a slice override)
        if r06_active:
            escalated = _escalate_priority(current_priority)
            # No slice override from R06 – fall through to shed/default
            # but carry the escalated priority forward
            current_priority = escalated
            # Apply shed rules with escalated priority
            if shed_hits:
                # Re-check shed conditions against escalated priority
                for shed in shed_hits:
                    if shed.override_slice is not None:
                        return shed.override_slice, current_priority, "R06"
            # No shed applies – keep default slice, return escalated priority
            return _default_slice_for_priority(current_priority), current_priority, "R06"

        # --- Load-shedding rules (only when no safety rule fired)
        if shed_hits:
            winning = shed_hits[0]
            return winning.override_slice, winning.override_priority, winning.rule_id

        # --- Satellite / 6G routing rules (R13–R17): applied after safety and load-shedding
        # These govern specialised slice selection for NTN and 6G slices.
        if satellite_6g_hits:
            winning = satellite_6g_hits[0]
            return winning.override_slice, winning.override_priority, winning.rule_id

        # --- Pure fallback: no rules fired
        return _default_slice_for_priority(current_priority), current_priority, None



def _default_slice_for_priority(priority: Priority) -> SliceType:
    """Map a priority level to a sensible default slice."""
    mapping = {
        Priority.CRITICAL: SliceType.URLLC,
        Priority.HIGH: SliceType.URLLC,
        Priority.MEDIUM: SliceType.EMBB,
        Priority.LOW: SliceType.MMTC,
    }
    return mapping.get(priority, SliceType.EMBB)
