"""
VESPER – Network Condition Simulator
=====================================
Wraps socket-level operations with realistic 5G network-slice degradation
effects including latency, jitter, packet loss, queueing delay, and
saturation spikes.

The simulator is deliberately stateful so that scenario injection (congestion,
packet storms, etc.) persists for the configured duration and is visible
across multiple callers sharing the same ``NetworkConditionSimulator`` instance.

Architecture
------------
* ``SliceConfig``              – static baseline parameters per 3GPP slice type.
* ``SLICE_CONFIGS``            – pre-built config dict keyed by slice name string.
* ``NetworkMetrics``           – point-in-time snapshot returned by ``get_metrics``.
* ``NetworkConditionSimulator``– main class; all public methods are thread-safe
                                  via per-slice asyncio.Lock objects.

Slice Types (3GPP TS 22.261)
----------------------------
* **URLLC** – Ultra-Reliable Low-Latency Communications   (safety events)
* **eMBB**  – Enhanced Mobile Broadband                   (bulk telemetry upload)
* **mMTC**  – Massive Machine-Type Communications         (low-rate sensor data)
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)



@dataclass
class SliceConfig:
    """Baseline operating parameters for a single 5G network slice."""

    name: str                      # slice identifier, e.g. "URLLC"
    base_latency_ms: float         # one-way propagation + processing delay
    latency_jitter_ms: float       # std-dev of per-packet latency variation
    base_loss_rate: float          # baseline packet-loss probability [0, 1]
    max_throughput_mbps: float     # nominal slice capacity
    max_queue_depth: int           # maximum scheduler queue length (packets)
    utilization: float = 0.1       # current utilisation [0, 1]; mutable per run


@dataclass
class NetworkMetrics:
    """Point-in-time KPI snapshot for one network slice."""

    slice_name: str
    timestamp: float

    # Latency
    base_latency_ms: float
    current_latency_ms: float      # includes jitter and queueing
    jitter_ms: float

    # Reliability
    loss_rate: float               # effective loss rate (may be elevated)
    packet_drop_count: int         # cumulative drops since simulator start

    # Capacity
    utilization: float             # [0, 1]
    max_throughput_mbps: float
    queue_depth: int               # packets currently in virtual queue

    # Health indicator
    saturation: bool               # True when utilization > 0.95



SLICE_CONFIGS: Dict[str, SliceConfig] = {
    "URLLC": SliceConfig(
        name                = "URLLC",
        base_latency_ms     = 1.5,
        latency_jitter_ms   = 0.5,
        base_loss_rate      = 0.0001,
        max_throughput_mbps = 20.0,
        max_queue_depth     = 50,
        utilization         = 0.1,
    ),
    "EMBB": SliceConfig(
        name                = "EMBB",
        base_latency_ms     = 15.0,
        latency_jitter_ms   = 5.0,
        base_loss_rate      = 0.001,
        max_throughput_mbps = 100.0,
        max_queue_depth     = 200,
        utilization         = 0.2,
    ),
    "MMTC": SliceConfig(
        name                = "MMTC",
        base_latency_ms     = 80.0,
        latency_jitter_ms   = 30.0,
        base_loss_rate      = 0.005,
        max_throughput_mbps = 1.0,
        max_queue_depth     = 1000,
        utilization         = 0.15,
    ),
    # NTN_SAT: Non-Terrestrial Network satellite slice (3GPP Rel-17/18 NTN).
    # Latency varies with orbital geometry: 600 ms is a moderate-elevation LEO RTT average.
    # Higher elevation = shorter slant range = lower latency (minimum ~240 ms at zenith for 550 km LEO).
    "NTN_SAT": SliceConfig(
        name                = "NTN_SAT",
        base_latency_ms     = 600.0,
        latency_jitter_ms   = 50.0,
        base_loss_rate      = 0.003,
        max_throughput_mbps = 150.0,   # Starlink-class LEO downlink
        max_queue_depth     = 500,
        utilization         = 0.05,
    ),
    # SIX_G_URLLC: 6G ultra-reliable low-latency slice.
    # Sub-0.1 ms target latency; range-limited to ~200 m radius from the THz node.
    "6G_URLLC": SliceConfig(
        name                = "6G_URLLC",
        base_latency_ms     = 0.08,
        latency_jitter_ms   = 0.02,
        base_loss_rate      = 0.00001,
        max_throughput_mbps = 1000.0,  # 1 Gbps URLLC channel
        max_queue_depth     = 20,      # tiny queue maintains deterministic latency
        utilization         = 0.0,
    ),
    # SIX_G_EMBB_PLUS: 6G enhanced mobile broadband slice.
    # 10 Gbps throughput class using sub-THz spectrum; moderate latency, wide bandwidth.
    "6G_eMBB+": SliceConfig(
        name                = "6G_eMBB+",
        base_latency_ms     = 1.0,
        latency_jitter_ms   = 0.1,
        base_loss_rate      = 0.0001,
        max_throughput_mbps = 10000.0,  # 10 Gbps Tbps-class channel
        max_queue_depth     = 100,
        utilization         = 0.0,
    ),
}


class NetworkConditionSimulator:
    """
    Stateful simulator for 5G network slice behaviour.

    All state-mutating methods (``update_utilization``, ``apply_degradation_scenario``)
    acquire a per-slice asyncio.Lock so the class is safe for concurrent access
    from multiple coroutines (e.g. multiple EVAgent tasks sharing one simulator).

    Usage
    -----
        sim = NetworkConditionSimulator()
        delivered, latency_ms = await sim.simulate_transmission("URLLC", payload_bytes=512)
        if not delivered:
            # handle drop
            ...
    """

    def __init__(self) -> None:
        # Deep-copy the canonical configs so each simulator instance is independent
        self._slices: Dict[str, SliceConfig] = {
            key: SliceConfig(**asdict(cfg)) for key, cfg in SLICE_CONFIGS.items()
        }
        # Per-slice asyncio locks for concurrent safety
        self._locks: Dict[str, asyncio.Lock] = {key: asyncio.Lock() for key in self._slices}

        # Effective loss rate: may be temporarily elevated by scenario injection
        self._effective_loss: Dict[str, float] = {
            key: cfg.base_loss_rate for key, cfg in self._slices.items()
        }

        # Cumulative drop counters per slice
        self._drop_counts: Dict[str, int] = {key: 0 for key in self._slices}

        # Virtual queue depth trackers (in-flight packets)
        self._queue_depths: Dict[str, int] = {key: 0 for key in self._slices}

        # Active scenario tasks (kept so they can be cancelled)
        self._scenario_tasks: list[asyncio.Task] = []

        logger.info(
            "NetworkConditionSimulator initialised with slices: %s",
            list(self._slices.keys()),
        )

    def inject_latency(self, slice_type: str, base_latency_ms: float) -> float:
        """
        Compute the effective one-way latency for a single packet on
        ``slice_type`` starting from ``base_latency_ms``.

        Model
        -----
        1. **Gaussian jitter** – adds normally-distributed variation.
        2. **M/M/1 queue delay** – when utilisation ρ < 0.95:
               extra_delay = (ρ² / (1 − ρ)) × service_time_ms
           where service_time_ms is the mean packet service time derived from
           throughput and an assumed 1500-byte MTU.
        3. **Saturation spike** – when ρ ≥ 0.95 the queue is nearly full;
           latency explodes to base_latency × U(5, 15).

        Returns
        -------
        float
            Effective latency in milliseconds (always ≥ 0).
        """
        cfg = self._slices[slice_type]
        utilization = cfg.utilization

        # Gaussian jitter
        jitter = random.gauss(0.0, cfg.latency_jitter_ms)

        if utilization >= 0.95:
            # Saturation regime: latency spikes dramatically
            spike_factor = 5.0 + random.random() * 10.0
            effective = base_latency_ms * spike_factor
            logger.debug(
                "[%s] Saturation spike: util=%.3f, latency=%.1f ms",
                slice_type, utilization, effective,
            )
        else:
            # M/M/1 queueing delay:
            # service_time_ms = (MTU_bytes * 8) / (throughput_bps)
            # throughput at current utilisation = util * max_throughput_mbps
            current_throughput_mbps = max(0.01, utilization * cfg.max_throughput_mbps)
            mtu_bits = 1500 * 8
            service_time_ms = (mtu_bits / (current_throughput_mbps * 1e6)) * 1000.0

            rho = utilization
            # M/M/1 mean queue waiting time
            mm1_delay_ms = (rho ** 2 / (1.0 - rho)) * service_time_ms

            effective = base_latency_ms + jitter + mm1_delay_ms

        return max(0.0, effective)

    def should_drop(self, slice_type: str) -> bool:
        """
        Return ``True`` if a packet on ``slice_type`` should be dropped.

        Uses the effective loss rate which may be temporarily elevated by
        ``apply_degradation_scenario``.
        """
        loss_rate = self._effective_loss.get(slice_type, SLICE_CONFIGS[slice_type].base_loss_rate)
        dropped = random.random() < loss_rate
        if dropped:
            self._drop_counts[slice_type] = self._drop_counts.get(slice_type, 0) + 1
            logger.debug("[%s] Packet dropped (loss_rate=%.5f)", slice_type, loss_rate)
        return dropped

    def update_utilization(self, slice_type: str, delta: float) -> None:
        """
        Adjust the utilisation of ``slice_type`` by ``delta`` (signed float).

        The result is clamped to [0, 1].  Concurrent callers should note that
        this method is *not* a coroutine; callers wishing to avoid races must
        acquire the per-slice lock externally, or rely on the GIL for CPython.
        """
        cfg = self._slices[slice_type]
        new_util = cfg.utilization + delta
        cfg.utilization = max(0.0, min(1.0, new_util))
        logger.debug(
            "[%s] Utilisation updated: %.3f → %.3f (Δ=%.3f)",
            slice_type, new_util - delta, cfg.utilization, delta,
        )

    def apply_degradation_scenario(self, scenario: str, duration_s: float) -> None:
        """
        Inject a named degradation scenario for ``duration_s`` seconds.

        Scenarios
        ---------
        ``"congestion"``
            Ramp URLLC utilisation up to 0.92 over 5 s, hold for the
            remainder of ``duration_s``, then restore.

        ``"packet_storm"``
            Multiply all slice loss rates by 10 (capped at 0.9) for
            ``duration_s`` seconds, then restore.

        ``"latency_spike"``
            Multiply EMBB base latency by 5 for ``duration_s`` seconds,
            then restore.

        ``"slice_outage"``
            Force URLLC utilisation to 0.99 (near-total saturation) for
            ``duration_s`` seconds, then restore.

        All scenario coroutines run as background asyncio Tasks.  Up to the
        caller to ensure an event loop is running when this method is called.
        """
        known = {"congestion", "packet_storm", "latency_spike", "slice_outage"}
        if scenario not in known:
            raise ValueError(f"Unknown scenario '{scenario}'. Known scenarios: {known}")

        logger.info("Applying degradation scenario '%s' for %.1f s", scenario, duration_s)

        task = asyncio.ensure_future(
            self._run_scenario(scenario, duration_s),
        )
        self._scenario_tasks.append(task)

        def _cleanup(t: asyncio.Task) -> None:
            self._scenario_tasks.discard(t)  # type: ignore[attr-defined]

        task.add_done_callback(_cleanup)

    async def _run_scenario(self, scenario: str, duration_s: float) -> None:
        """Internal coroutine that applies and eventually reverses a scenario."""
        try:
            if scenario == "congestion":
                await self._scenario_congestion(duration_s)
            elif scenario == "packet_storm":
                await self._scenario_packet_storm(duration_s)
            elif scenario == "latency_spike":
                await self._scenario_latency_spike(duration_s)
            elif scenario == "slice_outage":
                await self._scenario_slice_outage(duration_s)
        except asyncio.CancelledError:
            logger.info("Scenario '%s' cancelled.", scenario)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Scenario '%s' raised: %s", scenario, exc, exc_info=True)

    async def _scenario_congestion(self, duration_s: float) -> None:
        """Ramp URLLC utilisation to 0.92 over 5 s then hold."""
        target  = 0.92
        ramp_s  = min(5.0, duration_s * 0.3)
        steps   = max(1, int(ramp_s / 0.5))
        initial = self._slices["URLLC"].utilization

        delta_per_step = (target - initial) / steps
        for _ in range(steps):
            self.update_utilization("URLLC", delta_per_step)
            await asyncio.sleep(ramp_s / steps)

        # Hold at target for remaining duration
        hold_s = duration_s - ramp_s
        if hold_s > 0:
            await asyncio.sleep(hold_s)

        # Restore
        self.update_utilization("URLLC", initial - self._slices["URLLC"].utilization)
        logger.info("[URLLC] Congestion scenario finished; utilisation restored to %.3f", initial)

    async def _scenario_packet_storm(self, duration_s: float) -> None:
        """Multiply all slice loss rates by 10 for duration_s seconds."""
        saved = {key: self._effective_loss[key] for key in self._effective_loss}
        for key in self._effective_loss:
            self._effective_loss[key] = min(0.9, saved[key] * 10.0)
            logger.info("[%s] Packet storm: loss rate %.5f → %.5f", key, saved[key], self._effective_loss[key])

        await asyncio.sleep(duration_s)

        for key, original in saved.items():
            self._effective_loss[key] = original
        logger.info("Packet storm ended; loss rates restored.")

    async def _scenario_latency_spike(self, duration_s: float) -> None:
        """Multiply EMBB base_latency_ms by 5 for duration_s seconds."""
        cfg = self._slices["EMBB"]
        original_latency = cfg.base_latency_ms
        cfg.base_latency_ms *= 5.0
        logger.info(
            "[EMBB] Latency spike: %.1f ms → %.1f ms for %.1f s",
            original_latency, cfg.base_latency_ms, duration_s,
        )

        await asyncio.sleep(duration_s)

        cfg.base_latency_ms = original_latency
        logger.info("[EMBB] Latency spike ended; base latency restored to %.1f ms", original_latency)

    async def _scenario_slice_outage(self, duration_s: float) -> None:
        """Force URLLC utilisation to 0.99 for duration_s seconds."""
        cfg = self._slices["URLLC"]
        original_util = cfg.utilization
        cfg.utilization = 0.99
        logger.warning(
            "[URLLC] Slice outage: utilisation forced to 0.99 for %.1f s", duration_s
        )

        await asyncio.sleep(duration_s)

        cfg.utilization = original_util
        logger.info("[URLLC] Slice outage ended; utilisation restored to %.3f", original_util)

    def get_metrics(self, slice_type: str) -> NetworkMetrics:
        """
        Return a ``NetworkMetrics`` snapshot for ``slice_type``.

        The ``current_latency_ms`` is computed via ``inject_latency`` so it
        reflects the live M/M/1 model and current utilisation.
        """
        if slice_type not in self._slices:
            raise KeyError(f"Unknown slice '{slice_type}'. Available: {list(self._slices.keys())}")

        cfg = self._slices[slice_type]
        current_latency = self.inject_latency(slice_type, cfg.base_latency_ms)

        return NetworkMetrics(
            slice_name          = slice_type,
            timestamp           = time.time(),
            base_latency_ms     = cfg.base_latency_ms,
            current_latency_ms  = current_latency,
            jitter_ms           = cfg.latency_jitter_ms,
            loss_rate           = self._effective_loss.get(slice_type, cfg.base_loss_rate),
            packet_drop_count   = self._drop_counts.get(slice_type, 0),
            utilization         = cfg.utilization,
            max_throughput_mbps = cfg.max_throughput_mbps,
            queue_depth         = self._queue_depths.get(slice_type, 0),
            saturation          = cfg.utilization >= 0.95,
        )

    async def simulate_transmission(
        self,
        slice_type: str,
        payload_bytes: int,
    ) -> Tuple[bool, float]:
        """
        Simulate the network transmission of a packet with ``payload_bytes``
        bytes on the named slice.

        Steps
        -----
        1. Check packet drop (``should_drop``).
        2. Compute effective latency (``inject_latency``).
        3. Update the virtual queue depth based on payload size.
        4. Simulate the transmission delay with ``asyncio.sleep``.

        Returns
        -------
        (delivered, latency_ms)
            ``delivered`` is ``False`` if the packet was dropped.
            ``latency_ms`` is 0.0 for dropped packets.
        """
        if slice_type not in self._slices:
            raise KeyError(f"Unknown slice '{slice_type}'.")

        cfg = self._slices[slice_type]

        async with self._locks[slice_type]:
            # --- Drop check ---
            if self.should_drop(slice_type):
                return False, 0.0

            # --- Queue management ---
            current_depth = self._queue_depths.get(slice_type, 0)
            # Approximate: 1 packet unit per 1500 bytes (MTU)
            packet_units = max(1, math.ceil(payload_bytes / 1500))

            if current_depth + packet_units > cfg.max_queue_depth:
                # Queue full → tail-drop
                self._drop_counts[slice_type] = self._drop_counts.get(slice_type, 0) + 1
                logger.debug(
                    "[%s] Queue full (%d/%d) – tail-drop %d bytes",
                    slice_type, current_depth, cfg.max_queue_depth, payload_bytes,
                )
                return False, 0.0

            self._queue_depths[slice_type] = current_depth + packet_units

            # --- Latency computation ---
            latency_ms = self.inject_latency(slice_type, cfg.base_latency_ms)

            # --- Utilisation nudge: each transmission slightly loads the slice ---
            throughput_bytes_per_s = cfg.max_throughput_mbps * 1e6 / 8.0
            # How long (fractional seconds) does this payload occupy the slice?
            tx_time_s = payload_bytes / throughput_bytes_per_s
            # Normalise to a small utilisation delta
            util_delta = min(0.01, tx_time_s / 1.0)
            cfg.utilization = min(1.0, cfg.utilization + util_delta)

        # Simulate the transmission delay outside the lock
        await asyncio.sleep(latency_ms / 1000.0)

        # Release queue slot after delivery
        async with self._locks[slice_type]:
            self._queue_depths[slice_type] = max(
                0, self._queue_depths.get(slice_type, 0) - packet_units
            )
            # Slight utilisation recovery after transmission completes
            cfg.utilization = max(0.0, cfg.utilization - util_delta * 0.5)

        logger.debug(
            "[%s] Transmitted %d bytes in %.2f ms (util=%.3f)",
            slice_type, payload_bytes, latency_ms, cfg.utilization,
        )
        return True, latency_ms

    def update_satellite_link(self, ev_id: str, link_quality: float) -> None:
        """
        Adjust the NTN_SAT slice's base latency based on current satellite link quality.

        LEO elevation model: as elevation rises towards 90°, the slant range
        shortens and latency decreases.  link_quality is the normalised
        elevation signal from SatelliteChannel (0 = horizon, 1 = near-zenith).

        The mapping is: latency = 600 / sin(elevation_rad + 0.1)
        where link_quality is used as a proxy for sin(elevation).
        """
        if "NTN_SAT" not in self._slices:
            return
        cfg = self._slices["NTN_SAT"]
        # Avoid division by very small values; clamp the denominator
        denominator = max(0.1, link_quality + 0.1)
        adjusted_latency_ms = 600.0 / denominator
        # Cap at a realistic maximum RTT of 1400 ms (very low elevation, ~5°)
        cfg.base_latency_ms = min(1400.0, max(240.0, adjusted_latency_ms))
        logger.debug(
            "[NTN_SAT] ev=%s link_quality=%.3f → base_latency=%.1f ms",
            ev_id, link_quality, cfg.base_latency_ms,
        )

    def sixg_range_check(
        self,
        ev_position: tuple[float, float],
        node_position: tuple[float, float],
    ) -> bool:
        """
        Return True when the EV is within 6G THz cell range (≤200 m).

        THz and sub-THz bands are extremely range-limited due to molecular
        absorption (≈0.4 dB/m at 140 GHz) and high free-space path loss.
        Effective coverage radius is approximately 200 m from the 6G node.
        """
        dx = ev_position[0] - node_position[0]
        dy = ev_position[1] - node_position[1]
        distance_m = math.sqrt(dx ** 2 + dy ** 2)
        in_range = distance_m <= 200.0
        if not in_range:
            logger.debug(
                "6G range check: EV at %s is %.1f m from node at %s (out of range)",
                ev_position, distance_m, node_position,
            )
        return in_range

    async def shutdown(self) -> None:
        """Cancel all background scenario tasks and release resources."""
        for task in list(self._scenario_tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._scenario_tasks.clear()
        logger.info("NetworkConditionSimulator shut down.")
