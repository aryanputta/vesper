"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
slice_forwarder.py – Routes slice-decided messages through the network
                     simulator and tracks per-slice delivery statistics.

The ``NetworkConditionSimulator`` is defined here as a base class so that
callers can import and subclass it without a circular dependency.  The VESPER
simulator module will provide a concrete implementation; the forwarder accepts
any object implementing the same ``simulate_transmission`` signature.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Protocol, runtime_checkable

from backend.app.networking.message_schema import SliceDecision, SliceType, TelemetryMessage

logger = logging.getLogger(__name__)



@runtime_checkable
class NetworkConditionSimulator(Protocol):
    """
    Protocol (structural type) for any network condition simulator accepted by
    ``SliceForwarder``.

    Implementors must provide exactly one coroutine:

    ``simulate_transmission(slice_type, payload_bytes) -> (delivered, latency_ms)``
    """

    async def simulate_transmission(
        self,
        slice_type: SliceType,
        payload_bytes: int,
    ) -> tuple[bool, float]:
        """
        Simulate transmitting ``payload_bytes`` bytes over ``slice_type``.

        Returns
        -------
        delivered: bool
            ``True`` if the packet was delivered, ``False`` if dropped.
        latency_ms: float
            Simulated one-way latency in milliseconds.
        """
        ...  # pragma: no cover


class DefaultNetworkSimulator:
    """
    Simple built-in simulator with realistic but fixed slice characteristics.

    Intended as a drop-in when no external simulator is configured.

    Slice parameters
    ----------------
    URLLC : 1 ms base latency, 0.1% loss
    eMBB  : 10 ms base latency, 0.5% loss
    MMTC  : 50 ms base latency, 1% loss
    """

    import random as _random

    _PROFILES: dict[SliceType, dict] = {
        SliceType.URLLC: {"base_latency_ms": 1.0,  "jitter_ms": 0.5,  "loss_rate": 0.001},
        SliceType.EMBB:  {"base_latency_ms": 10.0, "jitter_ms": 3.0,  "loss_rate": 0.005},
        SliceType.MMTC:  {"base_latency_ms": 50.0, "jitter_ms": 10.0, "loss_rate": 0.010},
    }

    async def simulate_transmission(
        self,
        slice_type: SliceType,
        payload_bytes: int,
    ) -> tuple[bool, float]:
        import random
        profile = self._PROFILES[slice_type]
        jitter = random.gauss(0, profile["jitter_ms"])
        latency_ms = max(0.1, profile["base_latency_ms"] + jitter)

        # Simulate propagation delay
        await asyncio.sleep(latency_ms / 1000.0)

        delivered = random.random() >= profile["loss_rate"]
        return delivered, latency_ms



class _SliceStats:
    """Internal per-slice delivery statistics."""

    def __init__(self) -> None:
        self.delivered_count: int = 0
        self.dropped_count: int = 0
        self.total_latency_sum: float = 0.0

    @property
    def total_count(self) -> int:
        return self.delivered_count + self.dropped_count

    @property
    def delivery_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return self.delivered_count / self.total_count

    @property
    def avg_latency_ms(self) -> float:
        if self.delivered_count == 0:
            return 0.0
        return self.total_latency_sum / self.delivered_count

    def record(self, delivered: bool, latency_ms: float) -> None:
        if delivered:
            self.delivered_count += 1
            self.total_latency_sum += latency_ms
        else:
            self.dropped_count += 1

    def to_dict(self) -> dict:
        return {
            "delivered_count": self.delivered_count,
            "dropped_count": self.dropped_count,
            "total_count": self.total_count,
            "delivery_rate": round(self.delivery_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
        }



class SliceForwarder:
    """
    Routes a ``TelemetryMessage`` + ``SliceDecision`` pair to the assigned
    network slice via the supplied ``NetworkConditionSimulator``.

    Parameters
    ----------
    network_simulator:
        Any object implementing ``NetworkConditionSimulator``.  If ``None``,
        the built-in ``DefaultNetworkSimulator`` is used.
    """

    def __init__(
        self,
        network_simulator: Optional[NetworkConditionSimulator] = None,
    ) -> None:
        self._simulator: NetworkConditionSimulator = (
            network_simulator if network_simulator is not None
            else DefaultNetworkSimulator()
        )
        self._stats: dict[SliceType, _SliceStats] = {
            st: _SliceStats() for st in SliceType
        }
        self._total_forwarded: int = 0

    async def forward(
        self,
        msg: TelemetryMessage,
        decision: SliceDecision,
    ) -> tuple[bool, float]:
        """
        Forward a message to its assigned slice.

        Parameters
        ----------
        msg:
            The original telemetry message.
        decision:
            The routing decision; ``decision.assigned_slice`` determines the
            target slice.

        Returns
        -------
        delivered: bool
            Whether the simulated transmission was successful.
        latency_ms: float
            Simulated end-to-end latency in milliseconds.
        """
        slice_type = decision.assigned_slice
        payload_size = self._estimate_payload_size(msg)

        logger.debug(
            "Forwarding %s seq=%d to %s (payload=%d bytes)",
            msg.ev_id,
            msg.sequence_num,
            slice_type.value,
            payload_size,
        )

        try:
            delivered, latency_ms = await self._simulator.simulate_transmission(
                slice_type, payload_size
            )
        except Exception:
            logger.exception(
                "simulate_transmission raised for %s on slice %s; treating as dropped.",
                msg.ev_id,
                slice_type.value,
            )
            delivered, latency_ms = False, 0.0

        self._stats[slice_type].record(delivered, latency_ms)
        self._total_forwarded += 1

        if delivered:
            logger.debug(
                "Delivered %s seq=%d via %s in %.2f ms",
                msg.ev_id,
                msg.sequence_num,
                slice_type.value,
                latency_ms,
            )
        else:
            logger.warning(
                "DROPPED %s seq=%d on slice %s",
                msg.ev_id,
                msg.sequence_num,
                slice_type.value,
            )

        return delivered, latency_ms

    @staticmethod
    def _estimate_payload_size(msg: TelemetryMessage) -> int:
        """
        Estimate the on-wire payload size.

        We use a fixed overhead + per-field heuristic because serialising
        the full Pydantic model on the hot path is expensive.
        A typical TelemetryMessage JSON is ~350–450 bytes; we use 400 bytes.
        """
        return 400  # bytes (constant approximation)

    def get_stats(self) -> dict:
        """Return per-slice delivery rates, counts, and average latencies."""
        return {
            "total_forwarded": self._total_forwarded,
            "per_slice": {
                st.value: self._stats[st].to_dict() for st in SliceType
            },
        }
