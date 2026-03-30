"""
VESPER – 6G Network Feature Simulator
Simulates sub-THz beamforming, AI-native RAN control, semantic communications,
digital twin integration, ISAC sensing, and energy efficiency for 6G slices.
"""

import asyncio
import math
import random
import time
import logging
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.app.networking.message_schema import SixGMetrics

logger = logging.getLogger(__name__)


class SixGBand(Enum):
    FR3 = "FR3_7-24GHz"          # Frequency Range 3: upper mid-band, good coverage/capacity balance
    SUB_THZ = "SubTHz_92-300GHz" # Sub-terahertz: very high throughput, short range (~100-200m)
    THZ = "THz_300GHz+"          # True THz: experimental, sub-10m range, molecular absorption peaks


@dataclass
class ISACMeasurement:
    """
    Output of an Integrated Sensing and Communications (ISAC) radar scan.
    The same waveform used for data transmission doubles as a radar pulse,
    allowing the 6G node to sense vehicle positions and velocities.
    """
    target_distance_m: float
    target_velocity_ms: float
    radar_cross_section_dbsm: float  # radar cross section in dBsm
    sensing_confidence: float         # 0–1 probability this measurement is valid
    timestamp: float


class SemanticCompressionEngine:
    """
    Simulates 6G semantic communications compression.

    In semantic comms the transmitter extracts and sends only the *meaning*
    of a message rather than raw bits.  When the receiver already holds
    similar context (high context_similarity), dramatic compression is possible
    because only the delta needs to be conveyed.
    """

    def __init__(self) -> None:
        self._prev_context_hash: Optional[int] = None

    def compress(self, payload_bytes: int, context_similarity: float) -> tuple[int, float]:
        """
        Compute compressed payload size and the achieved compression ratio.

        Parameters
        ----------
        payload_bytes:
            Raw message size before semantic encoding.
        context_similarity:
            Float in [0, 1].  1.0 = identical to previously seen context
            (maximum compressibility); 0.0 = completely novel content.

        Returns
        -------
        (compressed_bytes, compression_ratio)
            compressed_bytes is always at least 50 bytes (header overhead).
            compression_ratio is compressed / original.
        """
        # Up to 80% savings when context is maximally similar
        base_ratio = 1.0 - (context_similarity * 0.8)
        # Add realistic encoder variance
        ratio = base_ratio + random.gauss(0.0, 0.05)
        ratio = max(0.05, min(1.0, ratio))   # physically plausible range
        compressed_bytes = max(50, int(payload_bytes * ratio))
        actual_ratio = compressed_bytes / max(1, payload_bytes)
        return compressed_bytes, actual_ratio


class SixGNode:
    """
    Simulates a single 6G base station node.

    Capabilities:
    - AI-assisted massive MIMO beamforming (1024-element arrays)
    - Sub-THz path loss with molecular absorption
    - ISAC dual-function sensing radar
    - Digital-twin state tracking per connected UE
    - Semantic compression for all outgoing data flows
    """

    SPEED_OF_LIGHT = 3e8         # m/s
    MAX_RANGE_M = 200.0          # effective THz cell radius in metres
    BANDWIDTH_GHZ = 10.0         # 10 GHz of spectrum available at sub-THz
    THERMAL_NOISE_DBM = -174.0   # thermal noise floor dBm/Hz at room temperature
    NODE_POWER_W = 500.0         # nominal node power consumption in watts

    def __init__(
        self,
        node_id: str,
        position: tuple[float, float],
        band: SixGBand = SixGBand.SUB_THZ,
    ) -> None:
        self.node_id = node_id
        self.position = position      # (x_m, y_m) in a local coordinate frame
        self.band = band
        self.isac_enabled = True
        self.semantic_engine = SemanticCompressionEngine()
        self.connected_evs: set[str] = set()

        # Digital twin: maps ev_id → last telemetry dict with ingestion timestamp
        self._digital_twin: dict[str, dict] = {}

        # ISAC history for fleet-level aggregation
        self._isac_measurements: list[ISACMeasurement] = []

        # Rolling throughput tracker for energy efficiency
        self._total_bits_served: float = 0.0
        self._session_start = time.time()

    def compute_beamforming_gain(
        self,
        ev_position: tuple[float, float],
        num_antenna_elements: int = 1024,
    ) -> float:
        """
        Estimate AI-assisted massive MIMO beamforming gain in dB.

        Array gain is the ideal coherent combining gain = 10 log10(N).
        A geometric efficiency factor accounts for the beam being off boresight.
        """
        array_gain_db = 10.0 * math.log10(num_antenna_elements)   # ~30 dB for 1024 elements

        # Compute angle from boresight (node points upward, boresight = directly above)
        dx = ev_position[0] - self.position[0]
        dy = ev_position[1] - self.position[1]
        distance = math.sqrt(dx ** 2 + dy ** 2) + 1e-6   # avoid div-by-zero
        # Boresight is taken as the forward axis (x-direction)
        cos_angle = abs(dx) / distance
        # Geometric efficiency mapped to additional gain variation (±3 dB)
        geometric_efficiency_db = 10.0 * math.log10(max(0.1, cos_angle))
        gain_db = array_gain_db + geometric_efficiency_db
        return round(gain_db, 2)

    def compute_thz_path_loss(self, distance_m: float, freq_ghz: float = 140.0) -> float:
        """
        Compute total THz path loss combining FSPL and molecular absorption.

        Molecular absorption at 140 GHz is approximately 0.4 dB/m in humid air,
        making THz communications highly range-limited (effective reach ~100-200 m).
        """
        freq_hz = freq_ghz * 1e9
        # Free-space path loss (Friis formula)
        fspl_db = (
            20.0 * math.log10(max(1.0, distance_m))
            + 20.0 * math.log10(freq_hz)
            + 20.0 * math.log10(4.0 * math.pi / self.SPEED_OF_LIGHT)
        )
        # Molecular absorption loss: ~0.4 dB/m at 140 GHz in standard atmosphere
        alpha_db_per_m = 0.4
        absorption_db = alpha_db_per_m * distance_m
        return round(fspl_db + absorption_db, 2)

    def compute_effective_throughput(
        self, distance_m: float, ev_speed_kmh: float
    ) -> float:
        """
        Estimate effective user throughput in Gbps.

        Uses Shannon capacity approximation.  SNR degrades with THz path loss
        and worsens for fast-moving vehicles due to beam tracking latency.
        Returns 0.0 when beyond the maximum cell range.
        """
        if distance_m > self.MAX_RANGE_M:
            return 0.0

        bandwidth_hz = self.BANDWIDTH_GHZ * 1e9
        path_loss_db = self.compute_thz_path_loss(distance_m)
        beamforming_gain = self.compute_beamforming_gain(
            (self.position[0] + distance_m, self.position[1])
        )

        # TX power assumed 23 dBm (200 mW) for a node-side downlink
        tx_power_dbm = 23.0
        noise_power_dbm = self.THERMAL_NOISE_DBM + 10.0 * math.log10(bandwidth_hz)

        snr_db = tx_power_dbm + beamforming_gain - path_loss_db - noise_power_dbm
        snr_linear = 10.0 ** (snr_db / 10.0)

        # Shannon capacity (theoretical upper bound)
        capacity_bps = bandwidth_hz * math.log2(1.0 + max(0.0, snr_linear))

        # Mobility penalty: fast vehicles cause beam misalignment and Doppler spread
        speed_ms = ev_speed_kmh / 3.6
        mobility_penalty = max(0.1, 1.0 - (speed_ms / 100.0) * 0.5)   # up to 50% at 100 m/s

        throughput_gbps = (capacity_bps * mobility_penalty) / 1e9
        return round(max(0.0, throughput_gbps), 3)

    def run_isac_sensing(
        self, ev_id: str, distance_m: float, velocity_ms: float
    ) -> ISACMeasurement:
        """
        Simulate an ISAC radar measurement of a nearby EV.

        The 6G waveform doubles as a radar probe, returning range and radial
        velocity with cm-level and cm/s-level accuracy respectively.
        """
        # Add realistic measurement noise
        measured_distance = distance_m + random.gauss(0.0, 0.1)    # ~10 cm range accuracy
        measured_velocity = velocity_ms + random.gauss(0.0, 0.05)  # ~5 cm/s velocity accuracy

        # Radar cross section of a vehicle is typically 10–20 dBsm
        rcs_dbsm = random.gauss(15.0, 3.0)

        # Sensing confidence degrades with range and at very high velocities
        range_factor = max(0.0, 1.0 - (distance_m / self.MAX_RANGE_M))
        velocity_factor = max(0.3, 1.0 - abs(velocity_ms) / 50.0)
        confidence = min(1.0, range_factor * velocity_factor + random.gauss(0.0, 0.05))

        measurement = ISACMeasurement(
            target_distance_m=round(measured_distance, 3),
            target_velocity_ms=round(measured_velocity, 3),
            radar_cross_section_dbsm=round(rcs_dbsm, 2),
            sensing_confidence=round(max(0.0, confidence), 3),
            timestamp=time.time(),
        )
        self._isac_measurements.append(measurement)
        # Keep only the last 500 measurements in memory
        if len(self._isac_measurements) > 500:
            self._isac_measurements = self._isac_measurements[-500:]
        return measurement

    def update_digital_twin(self, ev_id: str, telemetry: dict) -> float:
        """
        Update the digital twin state for an EV and return the sync latency.

        Sync latency is the age of the twin relative to when the telemetry was
        generated — a proxy for how accurately the twin represents reality.
        """
        now = time.time()
        telemetry_time = telemetry.get("timestamp", now)
        self._digital_twin[ev_id] = {"state": telemetry, "ingested_at": now}
        sync_latency_ms = (now - telemetry_time) * 1000.0
        return round(max(0.0, sync_latency_ms), 3)

    def get_metrics(self) -> SixGMetrics:
        """Return a current SixGMetrics snapshot for this node."""
        thz_active = self.band in (SixGBand.SUB_THZ, SixGBand.THZ)

        # 6G numerology: 480 kHz SCS for sub-THz, 960 kHz for true THz
        if self.band == SixGBand.THZ:
            scs_khz = 960.0
        elif self.band == SixGBand.SUB_THZ:
            scs_khz = 480.0
        else:
            scs_khz = 120.0   # FR3 uses 5G NR-like numerology extended

        # Average beamforming gain across connected EVs (use representative value if none)
        if self.connected_evs:
            gain_db = 28.0 + random.gauss(0.0, 1.5)   # typical 1024-element gain with noise
        else:
            gain_db = 0.0

        # Semantic compression: estimate ratio from recent activity
        compression_ratio = random.uniform(0.15, 0.6)   # 40–85% compression typical in practice

        # Digital twin sync latency: worst-case across connected EVs
        twin_lats = []
        for ev_id in self.connected_evs:
            entry = self._digital_twin.get(ev_id)
            if entry:
                age_ms = (time.time() - entry["ingested_at"]) * 1000.0
                twin_lats.append(age_ms)
        sync_latency_ms = max(twin_lats) if twin_lats else 0.0

        # Energy efficiency: total served Gbps / node power
        elapsed = max(1.0, time.time() - self._session_start)
        avg_throughput_gbps = self._total_bits_served / elapsed / 1e9
        efficiency = (avg_throughput_gbps * 1000.0) / self.NODE_POWER_W   # Mbps/W

        return SixGMetrics(
            ric_node_id=self.node_id,
            thz_band_active=thz_active,
            subcarrier_spacing_khz=scs_khz,
            ai_beamforming_gain_db=round(gain_db, 2),
            semantic_compression_ratio=round(compression_ratio, 4),
            digital_twin_sync_latency_ms=round(sync_latency_ms, 3),
            network_energy_efficiency_mbps_per_watt=round(efficiency, 4),
            timestamp=time.time(),
        )


class SixGRICController:
    """
    AI-native RAN Intelligent Controller (RIC), O-RAN xApp inspired.

    Manages multiple 6G nodes: assigns beams, drives handoff decisions,
    and reports fleet-level energy efficiency.
    """

    def __init__(self, nodes: list[SixGNode]) -> None:
        self.nodes = {n.node_id: n for n in nodes}
        # Current beam assignment: ev_id → node_id
        self._beam_assignments: dict[str, str] = {}
        # Per-EV metrics history for trend-based handoff decisions
        self._metrics_history: dict[str, list[float]] = {}
        self.energy_optimisation_active = True

    def optimize_beam_assignments(
        self, ev_positions: dict[str, tuple[float, float]]
    ) -> dict[str, str]:
        """
        Greedy beam assignment: assign each EV to the nearest 6G node.

        In a real RIC this would use reinforcement learning policies (xApps);
        here we use Euclidean distance as a simple proxy.
        """
        assignments: dict[str, str] = {}
        for ev_id, pos in ev_positions.items():
            best_node = None
            best_dist = float("inf")
            for node_id, node in self.nodes.items():
                dx = pos[0] - node.position[0]
                dy = pos[1] - node.position[1]
                dist = math.sqrt(dx ** 2 + dy ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_node = node_id
            if best_node is not None:
                assignments[ev_id] = best_node
                self.nodes[best_node].connected_evs.add(ev_id)

        self._beam_assignments = assignments
        return assignments

    def handoff_decision(
        self,
        ev_id: str,
        current_node: str,
        metrics_history: list[float],
    ) -> Optional[str]:
        """
        Recommend a handoff if the recent throughput trend is degrading.

        Examines the last 3 measurements; if all three are monotonically
        decreasing (signal weakening trend), recommends the next-best node.
        Returns a new node_id or None if no handoff is warranted.
        """
        if len(metrics_history) < 3:
            return None

        # Check for strict downward trend in the most recent 3 samples
        recent = metrics_history[-3:]
        is_degrading = all(recent[i] > recent[i + 1] for i in range(len(recent) - 1))
        if not is_degrading:
            return None

        # Find alternative node (pick the node with most headroom, i.e. fewest connected EVs)
        candidates = [
            (nid, len(n.connected_evs))
            for nid, n in self.nodes.items()
            if nid != current_node
        ]
        if not candidates:
            return None

        target_node = min(candidates, key=lambda x: x[1])[0]
        logger.info("[RIC] Recommending handoff for %s: %s → %s", ev_id, current_node, target_node)
        return target_node

    def compute_network_energy_efficiency(
        self,
        total_throughput_gbps: float,
        total_power_watts: float = 500.0,
    ) -> float:
        """Return network energy efficiency in Mbps/W."""
        if total_power_watts <= 0:
            return 0.0
        efficiency = (total_throughput_gbps * 1000.0) / total_power_watts
        return round(efficiency, 4)

    def get_fleet_summary(self) -> dict:
        """Aggregate metrics across all managed 6G nodes."""
        active_nodes = len(self.nodes)
        all_connected_evs: set[str] = set()
        total_sensing_events = 0
        throughput_samples: list[float] = []

        for node in self.nodes.values():
            all_connected_evs.update(node.connected_evs)
            total_sensing_events += len(node._isac_measurements)
            # Sample a representative throughput per connected EV
            for ev_id in node.connected_evs:
                throughput_samples.append(random.uniform(0.5, 8.0))   # Gbps range

        avg_throughput = (
            sum(throughput_samples) / len(throughput_samples) if throughput_samples else 0.0
        )
        total_power = active_nodes * SixGNode.NODE_POWER_W
        efficiency = self.compute_network_energy_efficiency(avg_throughput * active_nodes, total_power)

        return {
            "active_nodes": active_nodes,
            "connected_evs": len(all_connected_evs),
            "avg_throughput_gbps": round(avg_throughput, 3),
            "total_sensing_events": total_sensing_events,
            "energy_efficiency_mbps_per_watt": efficiency,
        }
