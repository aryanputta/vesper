"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
routes_telemetry.py – FastAPI router for EV telemetry and state queries.

Endpoints
---------
GET /telemetry/{ev_id}/latest
    Most recent TelemetryMessage for the given EV (from Redis).
GET /telemetry/{ev_id}/state
    Current EVState plus full state history for the given EV.
GET /telemetry/fleet/summary
    Fleet-wide summary: count per EVState, list of active alerts.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from backend.app.networking.message_schema import EVState, TelemetryMessage

router = APIRouter()

# Redis key pattern used by the simulator / TCP listener to cache latest telemetry
_LATEST_KEY_PATTERN = "vesper:telemetry:latest:{ev_id}"


def _get_orchestrator(request: Request):
    orch = getattr(request.app.state, "orchestrator", None)
    if orch is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialised")
    return orch


def _get_redis(request: Request) -> Optional[Any]:
    return getattr(request.app.state, "redis", None)


@router.get(
    "/fleet/summary",
    summary="Fleet-wide EV state summary",
    # Note: this must appear BEFORE /{ev_id}/... routes to avoid path collisions
)
async def get_fleet_summary(request: Request) -> dict[str, Any]:
    """
    Return an aggregate summary of the entire EV fleet.

    Includes
    --------
    * ``total_evs``         – number of EVs currently tracked by the orchestrator
    * ``state_counts``      – count of EVs in each EVState
    * ``active_alerts``     – alerts raised in the last 60 seconds
    * ``safety_event_count``– total safety events since startup
    """
    orch = _get_orchestrator(request)
    stats = orch.get_stats()
    per_ev: dict[str, str] = stats.get("per_ev_state", {})

    state_counts: dict[str, int] = {s.value: 0 for s in EVState}
    for state_str in per_ev.values():
        if state_str in state_counts:
            state_counts[state_str] += 1

    alert_service = getattr(request.app.state, "alert_service", None)
    active_alerts: list[dict] = []
    if alert_service is not None:
        try:
            raw_alerts = alert_service.get_recent_alerts(limit=20)
            cutoff = time.time() - 60.0
            for a in raw_alerts:
                if a.timestamp >= cutoff:
                    active_alerts.append({
                        "alert_id": a.alert_id,
                        "ev_id": a.ev_id,
                        "event_type": a.event_type.value,
                        "severity": a.severity.name,
                        "timestamp": a.timestamp,
                        "response_latency_ms": a.response_latency_ms,
                    })
        except Exception:
            pass

    return {
        "timestamp": time.time(),
        "total_evs": len(per_ev),
        "state_counts": state_counts,
        "active_alerts": active_alerts,
        "safety_event_count": stats.get("safety_events", 0),
    }


@router.get(
    "/{ev_id}/latest",
    summary="Latest TelemetryMessage for an EV",
)
async def get_latest_telemetry(ev_id: str, request: Request) -> dict[str, Any]:
    """
    Return the most recent telemetry frame for ``ev_id``.

    The frame is read from Redis (key ``vesper:telemetry:latest:{ev_id}``).
    If Redis is unavailable or the EV has not yet sent any data, a 404 is
    returned.
    """
    redis = _get_redis(request)
    if redis is None:
        raise HTTPException(
            status_code=503,
            detail="Redis not available – cannot retrieve cached telemetry",
        )

    key = _LATEST_KEY_PATTERN.format(ev_id=ev_id)
    try:
        raw = await redis.get(key)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Redis error: {exc}") from exc

    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry data found for EV '{ev_id}'",
        )

    try:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        msg = TelemetryMessage.model_validate_json(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to deserialise telemetry for EV '{ev_id}': {exc}",
        ) from exc

    return {
        "ev_id": ev_id,
        "latest": msg.model_dump(mode="json"),
    }


@router.get(
    "/{ev_id}/state",
    summary="Current EVState and state history for an EV",
)
async def get_ev_state(ev_id: str, request: Request) -> dict[str, Any]:
    """
    Return the current FSM state and the full state transition history for
    ``ev_id``.

    If the orchestrator has not yet processed any message from this vehicle
    a 404 is returned.
    """
    orch = _get_orchestrator(request)
    current_state = orch.get_ev_state(ev_id)

    if current_state is None:
        raise HTTPException(
            status_code=404,
            detail=f"EV '{ev_id}' is not currently tracked by the orchestrator",
        )

    history = orch.get_ev_state_history(ev_id)

    return {
        "ev_id": ev_id,
        "current_state": current_state.value,
        "history": [
            {"timestamp": ts, "state": state, "reason": reason}
            for ts, state, reason in history
        ],
    }
