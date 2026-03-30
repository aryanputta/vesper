"""
VESPER – 6G Metrics API Routes
FastAPI router exposing 6G node metrics, ISAC sensing events, beam assignments,
digital twin sync status, and network energy efficiency.
"""

from __future__ import annotations

import time
import logging
import random
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.app.simulators.sixg_simulator import (
    SixGNode,
    SixGRICController,
    SixGBand,
    ISACMeasurement,
)
from backend.app.networking.message_schema import SixGMetrics

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sixg", tags=["6G"])

# Module-level controller instance; production code would inject this via FastAPI dependency
_ric: Optional[SixGRICController] = None


def get_ric() -> SixGRICController:
    """Return the shared SixGRICController, creating default nodes on first call."""
    global _ric
    if _ric is None:
        nodes = [
            SixGNode("node-alpha", (0.0, 0.0), SixGBand.SUB_THZ),
            SixGNode("node-beta", (200.0, 0.0), SixGBand.SUB_THZ),
            SixGNode("node-gamma", (100.0, 173.0), SixGBand.FR3),   # FR3 for wider coverage
            SixGNode("node-delta", (300.0, 173.0), SixGBand.THZ),   # experimental THz node
        ]
        _ric = SixGRICController(nodes)
        # Pre-populate with some simulated EV connections
        for i in range(1, 11):
            ev_id = f"EV-{i:03d}"
            node = list(_ric.nodes.values())[i % len(_ric.nodes)]
            node.connected_evs.add(ev_id)
            # Seed the digital twin with a placeholder entry
            node._digital_twin[ev_id] = {
                "state": {"speed_kmh": random.uniform(0, 80), "timestamp": time.time()},
                "ingested_at": time.time(),
            }
    return _ric


class NodeStatusResponse(BaseModel):
    node_id: str
    band: str
    position_x: float
    position_y: float
    connected_evs: list[str]
    isac_enabled: bool
    total_sensing_events: int


class BeamAssignmentResponse(BaseModel):
    ev_id: str
    node_id: str
    timestamp: float


class DigitalTwinEntry(BaseModel):
    ev_id: str
    node_id: str
    sync_latency_ms: float
    last_updated: float


class EnergyMetricsResponse(BaseModel):
    active_nodes: int
    connected_evs: int
    avg_throughput_gbps: float
    total_power_watts: float
    energy_efficiency_mbps_per_watt: float
    timestamp: float


@router.get("/nodes", response_model=list[NodeStatusResponse])
async def list_nodes() -> list[NodeStatusResponse]:
    """
    Return status for all active 6G nodes managed by the RIC.

    Each entry includes node ID, frequency band, position (metres, local frame),
    currently connected EVs, ISAC capability flag, and cumulative sensing event count.
    """
    ric = get_ric()
    result = []
    for node_id, node in ric.nodes.items():
        result.append(NodeStatusResponse(
            node_id=node_id,
            band=node.band.value,
            position_x=node.position[0],
            position_y=node.position[1],
            connected_evs=list(node.connected_evs),
            isac_enabled=node.isac_enabled,
            total_sensing_events=len(node._isac_measurements),
        ))
    return result


@router.get("/nodes/{node_id}/metrics", response_model=SixGMetrics)
async def get_node_metrics(node_id: str) -> SixGMetrics:
    """
    Return the current SixGMetrics snapshot for a specific 6G node.

    Includes RIC node ID, THz band status, subcarrier spacing, AI beamforming gain,
    semantic compression ratio, digital twin sync latency, and energy efficiency.
    """
    ric = get_ric()
    if node_id not in ric.nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found.")
    return ric.nodes[node_id].get_metrics()


@router.get("/isac/detections", response_model=list[ISACMeasurement])
async def get_isac_detections() -> list[ISACMeasurement]:
    """
    Return the most recent 50 ISAC sensing events across all 6G nodes.

    ISAC (Integrated Sensing and Communications) uses the same waveform for
    both data transmission and radar sensing.  Each event captures target
    distance, radial velocity, radar cross section, and sensing confidence.
    """
    ric = get_ric()
    all_measurements: list[ISACMeasurement] = []
    for node in ric.nodes.values():
        all_measurements.extend(node._isac_measurements)

    # Sort by timestamp descending and return the latest 50
    all_measurements.sort(key=lambda m: m.timestamp, reverse=True)
    return all_measurements[:50]


@router.get("/beams", response_model=list[BeamAssignmentResponse])
async def get_beam_assignments() -> list[BeamAssignmentResponse]:
    """
    Return current beam-to-EV assignments across all 6G nodes.

    The RIC controller assigns each EV to a specific node based on proximity
    and load balancing.  This endpoint returns the current assignment table
    useful for visualising beam topology in the dashboard.
    """
    ric = get_ric()
    now = time.time()
    # Re-derive assignments from connected_evs in each node (matches internal state)
    result = []
    for node_id, node in ric.nodes.items():
        for ev_id in node.connected_evs:
            result.append(BeamAssignmentResponse(
                ev_id=ev_id,
                node_id=node_id,
                timestamp=now,
            ))
    return result


@router.get("/digital-twin", response_model=list[DigitalTwinEntry])
async def get_digital_twin_status() -> list[DigitalTwinEntry]:
    """
    Return digital twin sync status for all tracked EVs.

    The digital twin is the 6G node's in-memory representation of each
    connected UE's physical state.  Sync latency measures how stale that
    representation is — low values indicate near-real-time fidelity.
    """
    ric = get_ric()
    now = time.time()
    result: list[DigitalTwinEntry] = []
    for node_id, node in ric.nodes.items():
        for ev_id, entry in node._digital_twin.items():
            ingested_at = entry.get("ingested_at", now)
            sync_latency_ms = (now - ingested_at) * 1000.0
            result.append(DigitalTwinEntry(
                ev_id=ev_id,
                node_id=node_id,
                sync_latency_ms=round(max(0.0, sync_latency_ms), 3),
                last_updated=round(ingested_at, 3),
            ))
    return result


@router.get("/energy", response_model=EnergyMetricsResponse)
async def get_energy_metrics() -> EnergyMetricsResponse:
    """
    Return network energy efficiency metrics for the 6G node cluster.

    6G targets dramatically better energy efficiency than 5G (bit/joule).
    This endpoint reports total power consumption, aggregate throughput,
    and the derived Mbps/W efficiency figure across all active nodes.
    """
    ric = get_ric()
    summary = ric.get_fleet_summary()
    total_power = summary["active_nodes"] * SixGNode.NODE_POWER_W
    return EnergyMetricsResponse(
        active_nodes=summary["active_nodes"],
        connected_evs=summary["connected_evs"],
        avg_throughput_gbps=summary["avg_throughput_gbps"],
        total_power_watts=total_power,
        energy_efficiency_mbps_per_watt=summary["energy_efficiency_mbps_per_watt"],
        timestamp=time.time(),
    )
