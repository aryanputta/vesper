"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
routes_scenarios.py – FastAPI router for simulation scenario control.

Endpoints
---------
GET  /scenarios/list
    List of available scenarios with descriptions.
POST /scenarios/run
    Start a simulation scenario.
POST /scenarios/stop
    Stop the currently running simulation.
GET  /scenarios/status
    Current scenario status: name, elapsed time, events triggered.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Available scenarios catalogue
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, dict[str, Any]] = {
    "normal": {
        "name": "normal",
        "display_name": "Normal Operation",
        "description": (
            "Fleet of EVs operating under typical urban conditions.  "
            "Occasional low-urgency telemetry with no safety events.  "
            "Primarily mMTC and eMBB traffic."
        ),
        "typical_duration_s": 120.0,
        "safety_event_probability": 0.02,
    },
    "emergency": {
        "name": "emergency",
        "display_name": "Emergency Scenario",
        "description": (
            "Multiple simultaneous safety events: hard braking, obstacle alerts, "
            "and battery overheat conditions.  Exercises URLLC slice prioritisation "
            "and rule-engine safety overrides."
        ),
        "typical_duration_s": 60.0,
        "safety_event_probability": 0.50,
    },
    "congestion": {
        "name": "congestion",
        "display_name": "Network Congestion",
        "description": (
            "Simulates progressive URLLC and eMBB slice saturation.  "
            "Tests admission-control downgrading and load-shedding rules."
        ),
        "typical_duration_s": 90.0,
        "safety_event_probability": 0.10,
    },
    "mixed": {
        "name": "mixed",
        "display_name": "Mixed Urban Scenario",
        "description": (
            "Realistic mix of normal telemetry, intermittent safety events, "
            "and varying network conditions.  Good general-purpose benchmark."
        ),
        "typical_duration_s": 120.0,
        "safety_event_probability": 0.15,
    },
    "sensor_degradation": {
        "name": "sensor_degradation",
        "display_name": "Progressive Sensor Degradation",
        "description": (
            "Sensor confidence values gradually decrease across the fleet, "
            "triggering escalating priority assignments and testing the "
            "rules-engine sensor-degradation path."
        ),
        "typical_duration_s": 90.0,
        "safety_event_probability": 0.20,
    },
}


# ---------------------------------------------------------------------------
# In-process simulation state (singleton per worker)
# ---------------------------------------------------------------------------

class _SimulationState:
    """Lightweight simulation lifecycle tracker (per-process singleton)."""

    def __init__(self) -> None:
        self.running: bool = False
        self.scenario_name: Optional[str] = None
        self.started_at: Optional[float] = None
        self.duration_s: float = 0.0
        self.num_evs: int = 0
        self.events_triggered: int = 0
        self._task: Optional[asyncio.Task] = None

    def start(
        self,
        scenario_name: str,
        duration_s: float,
        num_evs: int,
    ) -> None:
        self.running = True
        self.scenario_name = scenario_name
        self.started_at = time.time()
        self.duration_s = duration_s
        self.num_evs = num_evs
        self.events_triggered = 0

    def stop(self) -> None:
        self.running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()

    @property
    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at

    @property
    def remaining_s(self) -> float:
        if not self.running:
            return 0.0
        return max(0.0, self.duration_s - self.elapsed_s)


_sim_state = _SimulationState()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RunScenarioRequest(BaseModel):
    """Request body for POST /scenarios/run."""

    scenario: str = Field(
        ...,
        description="Scenario identifier (see GET /scenarios/list for valid values)",
    )
    duration_s: float = Field(
        default=120.0,
        ge=1.0,
        le=3600.0,
        description="How long (seconds) to run the simulation",
    )
    num_evs: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of simulated EVs",
    )


# ---------------------------------------------------------------------------
# Simulation runner coroutine
# ---------------------------------------------------------------------------

async def _run_simulation(
    scenario_name: str,
    duration_s: float,
    num_evs: int,
    queue,
    settings,
) -> None:
    """
    Background coroutine that drives the EV simulator.

    Attempts to import and use the full EVSimulator from
    ``backend.app.simulators.ev_simulator``.  If that module is not available
    it falls back to a lightweight stub that simply counts elapsed time.
    """
    logger.info(
        "Simulation started: scenario=%s duration=%.1f s num_evs=%d",
        scenario_name,
        duration_s,
        num_evs,
    )

    try:
        from backend.app.simulators.ev_simulator import EVSimulator  # type: ignore[import]
        simulator = EVSimulator(
            num_evs=num_evs,
            scenario=scenario_name,
            duration_s=duration_s,
            telemetry_hz=getattr(settings, "TELEMETRY_HZ", 10.0) if settings else 10.0,
            queue=queue,
        )
        sim_task = asyncio.create_task(simulator.run(), name="ev_simulator")
        _sim_state._task = sim_task
        await asyncio.wait_for(sim_task, timeout=duration_s + 5.0)
    except ImportError:
        # Minimal stub: just wait for the duration
        logger.warning(
            "EVSimulator not importable – running stub simulation for %.1f s", duration_s
        )
        step = 0.5
        elapsed = 0.0
        while elapsed < duration_s and _sim_state.running:
            await asyncio.sleep(step)
            elapsed += step
            _sim_state.events_triggered += 1
    except asyncio.CancelledError:
        logger.info("Simulation cancelled")
    except asyncio.TimeoutError:
        logger.info("Simulation timed out after %.1f s", duration_s)
    except Exception as exc:
        logger.error("Simulation error: %s", exc)
    finally:
        _sim_state.running = False
        logger.info("Simulation finished: scenario=%s", scenario_name)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.get("/list", summary="List available simulation scenarios")
async def list_scenarios() -> dict[str, Any]:
    """
    Return the catalogue of available VESPER simulation scenarios.

    Each entry includes a ``name`` (used as the ``scenario`` field in
    POST /scenarios/run), a human-readable ``display_name``, a ``description``,
    and metadata about typical duration and safety event probability.
    """
    return {
        "scenarios": list(_SCENARIOS.values()),
    }


@router.post("/run", summary="Start a simulation scenario")
async def run_scenario(body: RunScenarioRequest, request: Request) -> dict[str, Any]:
    """
    Start the specified simulation scenario.

    If a simulation is already running it must be stopped first (POST
    /scenarios/stop) before a new one can be started.

    The simulation runs as a background asyncio task and is automatically
    stopped after ``duration_s`` seconds.
    """
    if _sim_state.running:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Scenario '{_sim_state.scenario_name}' is already running. "
                "POST /scenarios/stop first."
            ),
        )

    if body.scenario not in _SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown scenario '{body.scenario}'. "
                f"Valid values: {list(_SCENARIOS.keys())}"
            ),
        )

    settings = getattr(request.app.state, "settings", None)
    queue = getattr(request.app.state, "queue", None)

    _sim_state.start(body.scenario, body.duration_s, body.num_evs)

    sim_task = asyncio.create_task(
        _run_simulation(body.scenario, body.duration_s, body.num_evs, queue, settings),
        name=f"simulation_{body.scenario}",
    )
    _sim_state._task = sim_task

    logger.info(
        "Scenario '%s' started (duration=%.1f s, num_evs=%d)",
        body.scenario,
        body.duration_s,
        body.num_evs,
    )

    return {
        "status": "started",
        "scenario": body.scenario,
        "duration_s": body.duration_s,
        "num_evs": body.num_evs,
        "started_at": _sim_state.started_at,
    }


@router.post("/stop", summary="Stop the currently running simulation")
async def stop_scenario() -> dict[str, Any]:
    """
    Stop the currently running simulation.

    If no simulation is running, returns a 200 with ``status: not_running``.
    """
    if not _sim_state.running:
        return {"status": "not_running", "detail": "No simulation is currently active"}

    scenario_name = _sim_state.scenario_name
    _sim_state.stop()

    logger.info("Scenario '%s' stopped by API request", scenario_name)

    return {
        "status": "stopped",
        "scenario": scenario_name,
        "events_triggered": _sim_state.events_triggered,
        "elapsed_s": round(_sim_state.elapsed_s, 2),
    }


@router.get("/status", summary="Current simulation status")
async def get_scenario_status() -> dict[str, Any]:
    """
    Return the current simulation status.

    Fields
    ------
    * ``running``           – whether a simulation is active
    * ``scenario``          – name of the current (or last) scenario
    * ``elapsed_s``         – seconds since the scenario started
    * ``remaining_s``       – estimated seconds until completion
    * ``events_triggered``  – number of simulated events dispatched so far
    * ``num_evs``           – number of EVs being simulated
    """
    return {
        "running": _sim_state.running,
        "scenario": _sim_state.scenario_name,
        "elapsed_s": round(_sim_state.elapsed_s, 2),
        "remaining_s": round(_sim_state.remaining_s, 2),
        "duration_s": _sim_state.duration_s,
        "events_triggered": _sim_state.events_triggered,
        "num_evs": _sim_state.num_evs,
    }
