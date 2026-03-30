"""
VESPER – Satellite Telemetry API Routes
FastAPI router exposing satellite link status, pass predictions, constellation
coverage, and handoff control for the NTN_SAT slice.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.simulators.satellite_simulator import (
    SatelliteOrchestrator,
    SatelliteChannel,
    SatellitePass,
    CONSTELLATIONS,
)
from backend.app.networking.message_schema import SatelliteLink

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/satellite", tags=["satellite"])

# Module-level orchestrator instance; production code would inject this via FastAPI dependency
_orchestrator: Optional[SatelliteOrchestrator] = None


def get_orchestrator() -> SatelliteOrchestrator:
    """Return the shared SatelliteOrchestrator, initialising it on first call."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SatelliteOrchestrator(num_evs=20)
    return _orchestrator


class HandoffRequest(BaseModel):
    ev_id: str
    target_constellation: str


class FleetCoverageResponse(BaseModel):
    total_evs: int
    evs_with_coverage: int
    coverage_pct: float
    avg_link_quality: float
    active_constellations: list[str]
    timestamp: float


class PassPrediction(BaseModel):
    satellite_id: str
    constellation: str
    aos_time: float
    los_time: float
    max_elevation_deg: float
    duration_minutes: float


class ConstellationCoverageResponse(BaseModel):
    constellation: str
    evs_covered: list[str]
    total_covered: int


class HandoffResponse(BaseModel):
    ev_id: str
    previous_constellation: str
    target_constellation: str
    success: bool
    message: str
    timestamp: float


@router.get("/status", response_model=FleetCoverageResponse)
async def get_fleet_status() -> FleetCoverageResponse:
    """
    Return fleet-wide satellite coverage summary.

    Includes total EV count, number currently under satellite coverage,
    coverage percentage, average link quality (dB), and active constellation names.
    """
    orch = get_orchestrator()
    coverage = orch.get_fleet_coverage()
    return FleetCoverageResponse(
        total_evs=coverage["total_evs"],
        evs_with_coverage=coverage["evs_with_coverage"],
        coverage_pct=coverage["coverage_pct"],
        avg_link_quality=coverage["avg_link_quality"],
        active_constellations=coverage["active_constellations"],
        timestamp=time.time(),
    )


@router.get("/{ev_id}/link", response_model=Optional[SatelliteLink])
async def get_ev_link(ev_id: str) -> Optional[SatelliteLink]:
    """
    Return the current satellite link details for a specific EV.

    Returns null (HTTP 200 with null body) when the EV has no active satellite
    coverage at the moment of the request.
    """
    orch = get_orchestrator()
    if ev_id not in orch.channels:
        raise HTTPException(status_code=404, detail=f"EV '{ev_id}' not found in fleet.")
    link = orch.get_best_link_for_ev(ev_id)
    return link


@router.get("/{ev_id}/passes", response_model=list[PassPrediction])
async def get_ev_passes(ev_id: str) -> list[PassPrediction]:
    """
    Return predictions for the next 3 satellite passes for the specified EV.

    Each pass includes AOS/LOS times (Unix epoch seconds), peak elevation,
    and total pass duration in minutes.  Pass predictions are generated from
    the orbital mechanics of the EV's assigned constellation.
    """
    orch = get_orchestrator()
    if ev_id not in orch.channels:
        raise HTTPException(status_code=404, detail=f"EV '{ev_id}' not found in fleet.")

    channel = orch.channels[ev_id]
    passes: list[PassPrediction] = []
    now = time.time()
    offset = 0.0

    # Generate 3 simulated future passes using the channel's scheduling logic
    for _ in range(3):
        # Temporarily save and restore the real pass state so we don't disturb the live sim
        original_pass = channel.current_pass
        channel.current_pass = None   # force scheduler to produce a new pass
        predicted = channel._schedule_next_pass()
        channel.current_pass = original_pass

        # Shift the predicted AOS forward by the cumulative offset from previous passes
        predicted.aos_time = now + offset + (predicted.aos_time - now)
        predicted.los_time = predicted.aos_time + (predicted.los_time - predicted.aos_time)
        offset += (predicted.los_time - predicted.aos_time) + 1800.0   # ~30 min gap between passes

        passes.append(PassPrediction(
            satellite_id=predicted.satellite_id,
            constellation=predicted.constellation,
            aos_time=round(predicted.aos_time, 1),
            los_time=round(predicted.los_time, 1),
            max_elevation_deg=round(predicted.max_elevation_deg, 1),
            duration_minutes=round((predicted.los_time - predicted.aos_time) / 60.0, 1),
        ))

    return passes


@router.get("/constellation/{name}/coverage", response_model=ConstellationCoverageResponse)
async def get_constellation_coverage(name: str) -> ConstellationCoverageResponse:
    """
    List which EVs currently have active coverage from the named constellation.

    Valid constellation names: Starlink, OneWeb, Kuiper.
    """
    if name not in CONSTELLATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown constellation '{name}'. Known: {list(CONSTELLATIONS.keys())}",
        )
    orch = get_orchestrator()
    covered_evs = [
        ev_id
        for ev_id, link in orch.active_links.items()
        if link.constellation == name
    ]
    return ConstellationCoverageResponse(
        constellation=name,
        evs_covered=covered_evs,
        total_covered=len(covered_evs),
    )


@router.post("/handoff", response_model=HandoffResponse)
async def trigger_satellite_handoff(request: HandoffRequest) -> HandoffResponse:
    """
    Trigger a satellite constellation handoff for the specified EV.

    The EV's SatelliteChannel is reconfigured to prefer the target constellation
    on its next pass scheduling cycle.  The current pass is terminated immediately
    so the channel will acquire the new constellation at the next AOS.
    """
    orch = get_orchestrator()
    if request.ev_id not in orch.channels:
        raise HTTPException(status_code=404, detail=f"EV '{request.ev_id}' not found in fleet.")
    if request.target_constellation not in CONSTELLATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown target constellation '{request.target_constellation}'.",
        )

    channel = orch.channels[request.ev_id]
    previous = channel.constellation

    if previous == request.target_constellation:
        return HandoffResponse(
            ev_id=request.ev_id,
            previous_constellation=previous,
            target_constellation=request.target_constellation,
            success=False,
            message="EV is already on the requested constellation.",
            timestamp=time.time(),
        )

    # Execute handoff: switch constellation and force pass reschedule
    channel.constellation = request.target_constellation
    channel.orbit = CONSTELLATIONS[request.target_constellation]
    channel.current_pass = None   # next update() call will schedule a fresh pass
    channel.link_active = False
    channel.link_quality = 0.0

    # Remove the stale active link so the API reflects the handoff immediately
    orch.active_links.pop(request.ev_id, None)

    logger.info(
        "Satellite handoff: %s switched from %s to %s",
        request.ev_id, previous, request.target_constellation,
    )
    return HandoffResponse(
        ev_id=request.ev_id,
        previous_constellation=previous,
        target_constellation=request.target_constellation,
        success=True,
        message=f"Handoff initiated. EV will acquire {request.target_constellation} on next pass.",
        timestamp=time.time(),
    )
