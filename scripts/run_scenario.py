#!/usr/bin/env python3
"""
VESPER – Run a Demo Scenario
============================
Loads a named YAML scenario file, starts the EV fleet and network condition
simulator with the appropriate parameters, applies timed events, and prints a
real-time summary every 5 seconds.

Usage
-----
    python scripts/run_scenario.py --scenario scenario_1_normal
    python scripts/run_scenario.py --scenario scenario_2_obstacle --live-metrics
    python scripts/run_scenario.py --scenario scenario_3_urllc_saturation --headless
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# Make the repo root importable regardless of working directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.app.simulators.ev_simulator import (
    EVDrivingProfile,
    EVFleet,
    ScenarioType,
)

logger = logging.getLogger("vesper.run_scenario")


class NetworkConditionSimulator:
    """
    Tracks simulated per-slice network conditions and applies timed events.

    In a full deployment this would drive a real SDN controller.  Here it
    maintains an in-memory state dict that the scenario runner can query for
    display purposes.
    """

    def __init__(self, conditions: dict[str, dict]) -> None:
        # Mutable state keyed by slice name ("urllc", "embb", "mmtc")
        self.state: dict[str, dict] = {
            name: dict(params) for name, params in conditions.items()
        }
        self._safety_events: int = 0
        self._decision_count: int = 0
        self._decision_start: float = time.monotonic()

    def apply_embb_load_increase(self, target_utilization: float) -> None:
        self.state["embb"]["utilization"] = target_utilization
        logger.info("NetworkSim: eMBB utilization → %.2f", target_utilization)

    def apply_urllc_saturation(self, target_utilization: float) -> None:
        self.state["urllc"]["utilization"] = target_utilization
        logger.info("NetworkSim: URLLC utilization → %.2f", target_utilization)

    def apply_network_degradation(
        self,
        target_slice: str,
        latency_multiplier: float = 1.0,
        loss_rate_multiplier: float = 1.0,
    ) -> None:
        s = self.state.get(target_slice)
        if s:
            s["latency_ms"] = s["latency_ms"] * latency_multiplier
            s["loss_rate"]  = min(1.0, s["loss_rate"] * loss_rate_multiplier)
            logger.info(
                "NetworkSim: %s degraded → latency=%.1fms loss=%.4f",
                target_slice, s["latency_ms"], s["loss_rate"],
            )

    def inject_anomaly(
        self,
        target_slice: str,
        anomaly_type: str,
        latency_ms: float | None = None,
        loss_rate_multiplier: float = 1.0,
    ) -> None:
        slices = list(self.state.keys()) if target_slice == "all" else [target_slice]
        for sl in slices:
            s = self.state.get(sl)
            if not s:
                continue
            if anomaly_type == "latency_spike" and latency_ms is not None:
                s["latency_ms"] = latency_ms
                logger.info("NetworkSim: %s latency spike → %.1fms", sl, latency_ms)
            elif anomaly_type == "packet_storm":
                s["loss_rate"] = min(1.0, s["loss_rate"] * loss_rate_multiplier)
                logger.info("NetworkSim: %s packet storm loss → %.4f", sl, s["loss_rate"])

    def record_safety_event(self) -> None:
        self._safety_events += 1

    def record_decision(self) -> None:
        self._decision_count += 1

    def decisions_per_second(self) -> float:
        elapsed = max(time.monotonic() - self._decision_start, 1e-6)
        return self._decision_count / elapsed

    def summary(self) -> dict:
        return {
            "slices":         self.state,
            "safety_events":  self._safety_events,
            "decisions_ps":   self.decisions_per_second(),
        }


async def _dispatch_events(
    events: list[dict],
    fleet: EVFleet,
    net_sim: NetworkConditionSimulator,
    start_time: float,
) -> None:
    """
    Fire scenario events at the times specified in the YAML file.

    Events are sorted by ``at_seconds`` and fired via asyncio.sleep so that
    the event loop remains responsive throughout.
    """
    sorted_events = sorted(events, key=lambda e: e.get("at_seconds", 0))

    for event in sorted_events:
        at = float(event.get("at_seconds", 0))
        elapsed = time.monotonic() - start_time
        wait = max(0.0, at - elapsed)
        await asyncio.sleep(wait)

        event_type: str = event.get("type", "")
        logger.info("Scenario event at t=%.1fs: %s", at, event_type)

        # Obstacle approach: override a specific EV agent's obstacle distance
        if event_type == "obstacle_approach":
            ev_id = event.get("ev_id")
            start_dist = float(event.get("obstacle_distance_start_m", 50.0))
            close_rate = float(event.get("obstacle_close_rate_m_per_s", 4.0))
            for agent in fleet.agents:
                if ev_id is None or agent.ev_id == ev_id:
                    agent.obstacle_distance = start_dist
                    agent._obstacle_active = True  # noqa: SLF001
                    # Override close rate by patching the scenario tick handler
                    # (simplified: agent will use its internal close logic)
                    logger.info(
                        "Obstacle injected for %s at %.1fm, closing at %.1fm/s",
                        agent.ev_id, start_dist, close_rate,
                    )

        # Network events
        elif event_type == "embb_load_increase":
            net_sim.apply_embb_load_increase(float(event.get("embb_utilization_target", 0.88)))

        elif event_type == "urllc_saturation_increase":
            net_sim.apply_urllc_saturation(float(event.get("target_utilization", 0.97)))

        elif event_type == "network_degradation":
            net_sim.apply_network_degradation(
                target_slice      = event.get("target_slice", "embb"),
                latency_multiplier    = float(event.get("latency_multiplier", 1.0)),
                loss_rate_multiplier  = float(event.get("loss_rate_multiplier", 1.0)),
            )

        elif event_type == "battery_temperature_ramp":
            ev_id     = event.get("ev_id")
            end_temp  = float(event.get("end_temp_celsius", 72.0))
            ramp_s    = float(event.get("ramp_seconds", 30.0))
            start_temp = float(event.get("start_temp_celsius", 55.0))
            for agent in fleet.agents:
                if ev_id is None or agent.ev_id == ev_id:
                    agent.battery_temp = start_temp
                    # Schedule a coroutine to ramp the temperature
                    asyncio.create_task(
                        _ramp_battery(agent, start_temp, end_temp, ramp_s)
                    )

        elif event_type == "inject_network_anomaly":
            net_sim.inject_anomaly(
                target_slice         = event.get("target_slice", "urllc"),
                anomaly_type         = event.get("anomaly_type", "latency_spike"),
                latency_ms           = event.get("latency_ms"),
                loss_rate_multiplier = float(event.get("loss_rate_multiplier", 1.0)),
            )

        else:
            logger.warning("Unknown scenario event type: %s", event_type)


async def _ramp_battery(agent: Any, start: float, end: float, duration_s: float) -> None:
    """Linearly ramp an agent's battery temperature over ``duration_s`` seconds."""
    steps = max(1, int(duration_s * 10))  # 10 Hz resolution
    delta = (end - start) / steps
    for _ in range(steps):
        agent.battery_temp = min(120.0, max(-40.0, agent.battery_temp + delta))
        await asyncio.sleep(0.1)


async def _print_summary_loop(
    fleet: EVFleet,
    net_sim: NetworkConditionSimulator,
    duration_s: float,
    live_metrics: bool,
    headless: bool,
    start_time: float,
) -> None:
    """Print a brief status line every 5 seconds."""
    if headless:
        return

    interval = 5.0
    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= duration_s:
            break
        await asyncio.sleep(interval)

        # Collect EV states
        active_evs = len(fleet.agents)
        summary = net_sim.summary()

        print(
            f"\n[t={elapsed:5.1f}s] Active EVs: {active_evs} | "
            f"Decisions/s: {summary['decisions_ps']:.1f} | "
            f"Safety events: {summary['safety_events']}"
        )

        if live_metrics:
            for sl_name, sl_state in summary["slices"].items():
                print(
                    f"  {sl_name.upper():<6}: "
                    f"util={sl_state.get('utilization', 0):.2f}  "
                    f"latency={sl_state.get('latency_ms', 0):.1f}ms  "
                    f"loss={sl_state.get('loss_rate', 0):.4f}"
                )


def _print_outcome_checklist(expected_outcomes: list[str]) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO OUTCOME VERIFICATION CHECKLIST")
    print("=" * 70)
    for i, outcome in enumerate(expected_outcomes, start=1):
        print(f"  [ ] {i}. {outcome}")
    print("=" * 70)
    print("Review the logs above and check each criterion manually.")
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a VESPER demo scenario.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_scenario.py --scenario scenario_1_normal
  python scripts/run_scenario.py --scenario scenario_2_obstacle --live-metrics
  python scripts/run_scenario.py --scenario scenario_3_urllc_saturation --headless
""",
    )
    parser.add_argument(
        "--scenario",
        default="scenario_1_normal",
        help="Scenario YAML filename (without .yaml extension) inside scenarios/",
    )
    parser.add_argument(
        "--live-metrics",
        action="store_true",
        dest="live_metrics",
        help="Print per-slice metrics in the real-time summary",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Suppress all real-time output (useful for automated runs)",
    )
    return parser


_SCENARIO_TYPE_MAP: dict[str, ScenarioType] = {
    "normal_operations":                  ScenarioType.NORMAL,
    "obstacle_emergency":                 ScenarioType.OBSTACLE_EMERGENCY,
    "urllc_saturation":                   ScenarioType.URLLC_SATURATION,
    "battery_overheat_network_degradation": ScenarioType.BATTERY_OVERHEAT,
    "anomaly_detection":                  ScenarioType.ANOMALY_ATTACK,
}


async def _main(args: argparse.Namespace) -> None:
    scenarios_dir = _REPO_ROOT / "scenarios"
    yaml_name = args.scenario if args.scenario.endswith(".yaml") else f"{args.scenario}.yaml"
    yaml_path = scenarios_dir / yaml_name

    if not yaml_path.exists():
        logger.error("Scenario file not found: %s", yaml_path)
        sys.exit(1)

    with yaml_path.open() as fh:
        scenario_cfg: dict = yaml.safe_load(fh)

    name             = scenario_cfg.get("name", args.scenario)
    description      = scenario_cfg.get("description", "")
    duration_s       = float(scenario_cfg.get("duration_s", 60))
    num_evs          = int(scenario_cfg.get("num_evs", 5))
    net_conditions   = scenario_cfg.get("network_conditions", {})
    ev_profile_cfg   = scenario_cfg.get("ev_profiles", {})
    events           = scenario_cfg.get("events", [])
    expected_outcomes = scenario_cfg.get("expected_outcomes", [])

    if not args.headless:
        print("=" * 70)
        print(f"VESPER Scenario Runner")
        print(f"Scenario : {name}")
        print(f"Desc     : {description}")
        print(f"Duration : {duration_s}s | EVs: {num_evs}")
        print("=" * 70)
        print()

    scenario_type = _SCENARIO_TYPE_MAP.get(name, ScenarioType.NORMAL)

    fleet = EVFleet(
        num_evs  = num_evs,
        scenario = scenario_type,
        tcp_host = "127.0.0.1",
        tcp_port = 9000,
        udp_host = "127.0.0.1",
        udp_port = 9001,
    )

    # Apply ev_profile overrides
    if ev_profile_cfg:
        base_speed  = float(ev_profile_cfg.get("base_speed_kmh", 65))
        speed_var   = float(ev_profile_cfg.get("speed_variance", 10))
        batt_start  = float(ev_profile_cfg.get("battery_start_temp", 42))
        for agent in fleet.agents:
            agent.profile.base_speed        = base_speed
            agent.profile.speed_variance    = speed_var
            agent.profile.battery_start_temp = batt_start
            agent.battery_temp              = batt_start
            agent.speed                     = base_speed

    net_sim = NetworkConditionSimulator(net_conditions)

    start_time = time.monotonic()

    fleet_task   = asyncio.create_task(fleet.run(duration_seconds=duration_s))
    events_task  = asyncio.create_task(
        _dispatch_events(events, fleet, net_sim, start_time)
    )
    summary_task = asyncio.create_task(
        _print_summary_loop(
            fleet, net_sim, duration_s,
            args.live_metrics, args.headless, start_time,
        )
    )

    await fleet_task

    # Clean up helper tasks
    for task in (events_task, summary_task):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    elapsed = time.monotonic() - start_time
    if not args.headless:
        print(f"\nScenario complete in {elapsed:.1f}s.")

    _print_outcome_checklist(expected_outcomes)


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    log_level = logging.WARNING if args.headless else logging.INFO
    logging.basicConfig(
        level  = log_level,
        format = "%(asctime)s %(levelname)-8s %(name)s – %(message)s",
        datefmt= "%H:%M:%S",
    )

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
