"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
routes_slices.py – FastAPI router for 5G network slice status and metrics.

Endpoints
---------
GET /slices/status
    Returns current metrics snapshot for all three slices.
GET /slices/{slice_name}/metrics
    Returns the last 100 metric measurements for a specific slice.
GET /slices/{slice_name}/history
    Returns the last 1000 metric measurements for a specific slice.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from backend.app.networking.message_schema import NetworkMetrics, SliceType

router = APIRouter()


def _get_slice_manager(request: Request):
    """Extract SliceManager from app.state; raise 503 if not initialised."""
    sm = getattr(request.app.state, "slice_manager", None)
    if sm is None:
        raise HTTPException(status_code=503, detail="SliceManager not initialised")
    return sm


def _resolve_slice_name(slice_name: str) -> SliceType:
    """Convert a URL path segment to a SliceType enum value."""
    normalised = slice_name.upper()
    try:
        return SliceType(normalised)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown slice '{slice_name}'. "
                f"Valid values: {[s.value for s in SliceType]}"
            ),
        )


def _serialise_metrics(m: NetworkMetrics) -> dict[str, Any]:
    """Convert a NetworkMetrics instance to a JSON-safe dict."""
    return {
        "slice_type": m.slice_type.value,
        "latency_ms": m.latency_ms,
        "jitter_ms": m.jitter_ms,
        "packet_loss_rate": m.packet_loss_rate,
        "throughput_mbps": m.throughput_mbps,
        "utilization_ratio": m.utilization_ratio,
        "queue_depth": m.queue_depth,
        "timestamp": m.timestamp,
    }


@router.get("/status", summary="Current metrics for all slices")
async def get_all_slice_status(request: Request) -> dict[str, Any]:
    """
    Return a snapshot of the current KPIs for all three 5G slices:
    URLLC, eMBB, and mMTC.

    The response includes latency, packet loss, utilisation, queue depth,
    and throughput for each slice, plus a top-level ``timestamp`` indicating
    when the response was composed.
    """
    sm = _get_slice_manager(request)
    all_metrics = sm.get_all_metrics()

    slices: dict[str, Any] = {}
    for slice_type in SliceType:
        m = all_metrics.get(slice_type)
        if m is not None:
            slices[slice_type.value] = _serialise_metrics(m)
        else:
            # Slice not yet populated – return sentinel values
            slices[slice_type.value] = {
                "slice_type": slice_type.value,
                "latency_ms": None,
                "jitter_ms": None,
                "packet_loss_rate": None,
                "throughput_mbps": None,
                "utilization_ratio": None,
                "queue_depth": None,
                "timestamp": None,
            }

    return {
        "timestamp": time.time(),
        "slices": slices,
    }


@router.get(
    "/{slice_name}/metrics",
    summary="Time-series metrics for a specific slice (last 100 points)",
)
async def get_slice_metrics(slice_name: str, request: Request) -> dict[str, Any]:
    """
    Return the last 100 NetworkMetrics measurements for the requested slice.

    Path parameter ``slice_name`` is case-insensitive.
    Valid values: ``URLLC``, ``EMBB``, ``MMTC``.
    """
    sm = _get_slice_manager(request)
    slice_type = _resolve_slice_name(slice_name)

    series = sm.get_slice_metrics_series(slice_type, limit=100)

    return {
        "slice": slice_type.value,
        "count": len(series),
        "metrics": [_serialise_metrics(m) for m in series],
    }


@router.get(
    "/{slice_name}/history",
    summary="Full metrics history for a specific slice (last 1000 measurements)",
)
async def get_slice_history(slice_name: str, request: Request) -> dict[str, Any]:
    """
    Return the last 1 000 NetworkMetrics measurements for the requested slice.

    This endpoint is suitable for chart rendering in the dashboard.  For
    real-time streaming, connect directly via WebSocket (not implemented here).
    """
    sm = _get_slice_manager(request)
    slice_type = _resolve_slice_name(slice_name)

    history = sm.get_slice_history(slice_type, limit=1000)

    return {
        "slice": slice_type.value,
        "count": len(history),
        "history": [_serialise_metrics(m) for m in history],
    }
