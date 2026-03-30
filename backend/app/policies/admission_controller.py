"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
admission_controller.py – Token-bucket admission controller per network slice.

Each network slice has its own TokenBucket.  The AdmissionController
orchestrates admission decisions using both utilisation thresholds and
per-slice token availability.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Literal

from backend.app.networking.message_schema import Priority, SliceType

logger = logging.getLogger(__name__)



class TokenBucket:
    """
    Classic token-bucket rate limiter.

    Parameters
    ----------
    rate_mbps:
        Sustained token replenishment rate in Mbit/s.
    burst_mb:
        Maximum burst capacity in Megabytes (MiB for implementation simplicity;
        the spec says MB so we treat 1 MB = 1 000 000 bytes for network context).
    """

    _BITS_PER_BYTE = 8
    _BYTES_PER_MBIT = 125_000  # 1 Mbit = 125 000 bytes

    def __init__(self, rate_mbps: float, burst_mb: float) -> None:
        if rate_mbps <= 0:
            raise ValueError("rate_mbps must be positive")
        if burst_mb <= 0:
            raise ValueError("burst_mb must be positive")

        self._rate_bps: float = rate_mbps * self._BYTES_PER_MBIT  # bytes per second
        self._capacity: float = burst_mb * 1_000_000              # bytes
        self._tokens: float = self._capacity                      # start full
        self._last_refill: float = time.monotonic()
        self._lock: threading.Lock = threading.Lock()

        # Statistics
        self._total_admitted_bytes: int = 0
        self._total_rejected_bytes: int = 0
        self._admit_count: int = 0
        self._reject_count: int = 0

    def consume(self, bytes_requested: int) -> bool:
        """
        Attempt to consume ``bytes_requested`` tokens.

        Returns ``True`` if the bucket had sufficient tokens (admitted),
        ``False`` otherwise (rejected).  A rejection does *not* consume any
        tokens.
        """
        with self._lock:
            self._do_refill()
            if self._tokens >= bytes_requested:
                self._tokens -= bytes_requested
                self._total_admitted_bytes += bytes_requested
                self._admit_count += 1
                return True
            else:
                self._total_rejected_bytes += bytes_requested
                self._reject_count += 1
                return False

    def refill(self) -> None:
        """
        Manually trigger a refill.  Call this once per second from a
        scheduler; the bucket also refills lazily inside ``consume``.
        """
        with self._lock:
            self._do_refill()

    def available_bytes(self) -> int:
        """Return the current number of available tokens (bytes)."""
        with self._lock:
            self._do_refill()
            return int(self._tokens)

    def _do_refill(self) -> None:
        """Accumulate tokens since last call.  Must be called while holding the lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_bps)
            self._last_refill = now

    def stats(self) -> dict:
        return {
            "available_bytes": self.available_bytes(),
            "capacity_bytes": int(self._capacity),
            "rate_bps": self._rate_bps,
            "admit_count": self._admit_count,
            "reject_count": self._reject_count,
            "total_admitted_bytes": self._total_admitted_bytes,
            "total_rejected_bytes": self._total_rejected_bytes,
        }



class AdmissionController:
    """
    Per-slice admission controller combining token buckets with utilisation
    thresholds.

    Slice capacities
    ----------------
    * URLLC  – 20 Mbit/s sustained, 2 MB burst
    * eMBB   – 100 Mbit/s sustained, 10 MB burst
    * MMTC   – 1 Mbit/s sustained, 0.5 MB burst

    Decision logic (in order)
    -------------------------
    1. CRITICAL priority → always ADMIT (bypass token bucket, log if overloaded).
    2. If slice utilisation > 0.95 and priority == LOW → REJECT.
    3. If slice utilisation > 0.90 and priority == MEDIUM → QUEUE.
    4. Otherwise → check token bucket:
       - Token bucket says yes → ADMIT.
       - Token bucket says no  → QUEUE.
    """

    _BUCKET_CONFIG: dict[SliceType, tuple[float, float]] = {
        SliceType.URLLC: (20.0, 2.0),
        SliceType.EMBB: (100.0, 10.0),
        SliceType.MMTC: (1.0, 0.5),
    }

    def __init__(self) -> None:
        self._buckets: dict[SliceType, TokenBucket] = {
            st: TokenBucket(rate_mbps, burst_mb)
            for st, (rate_mbps, burst_mb) in self._BUCKET_CONFIG.items()
        }
        self._utilization: dict[SliceType, float] = {
            SliceType.URLLC: 0.0,
            SliceType.EMBB: 0.0,
            SliceType.MMTC: 0.0,
        }
        # Per-slice decision counters
        self._counters: dict[SliceType, dict[str, int]] = {
            st: {"ADMIT": 0, "QUEUE": 0, "REJECT": 0} for st in SliceType
        }

    def admit(
        self,
        slice_type: SliceType,
        priority: Priority,
        payload_size_bytes: int,
    ) -> Literal["ADMIT", "QUEUE", "REJECT"]:
        """
        Decide whether to admit, queue, or reject a message.

        Parameters
        ----------
        slice_type:
            Target network slice.
        priority:
            Message priority level.
        payload_size_bytes:
            Payload size in bytes; used for token-bucket consumption.
        """
        util = self._utilization.get(slice_type, 0.0)
        bucket = self._buckets[slice_type]

        # 1. CRITICAL always admitted
        if priority == Priority.CRITICAL:
            if util > 0.95:
                logger.warning(
                    "CRITICAL message admitted to overloaded slice %s (util=%.2f)",
                    slice_type.value,
                    util,
                )
            # Consume tokens for accounting but do not reject
            bucket.consume(payload_size_bytes)
            decision: Literal["ADMIT", "QUEUE", "REJECT"] = "ADMIT"

        # 2. Reject low-priority traffic on heavily-loaded slices
        elif util > 0.95 and priority == Priority.LOW:
            decision = "REJECT"

        # 3. Queue medium-priority traffic on loaded slices
        elif util > 0.90 and priority == Priority.MEDIUM:
            decision = "QUEUE"

        # 4. Token-bucket check
        else:
            admitted = bucket.consume(payload_size_bytes)
            decision = "ADMIT" if admitted else "QUEUE"

        self._counters[slice_type][decision] += 1
        logger.debug(
            "admit slice=%s priority=%s size=%d util=%.2f → %s",
            slice_type.value,
            priority.name,
            payload_size_bytes,
            util,
            decision,
        )
        return decision

    def update_utilization(self, slice_type: SliceType, util: float) -> None:
        """Update the cached utilisation ratio for a slice (0.0–1.0)."""
        if not (0.0 <= util <= 1.0):
            raise ValueError(f"Utilisation must be in [0, 1], got {util}")
        self._utilization[slice_type] = util

    def tick(self) -> None:
        """Trigger a manual refill on all token buckets.  Call at 1 Hz."""
        for bucket in self._buckets.values():
            bucket.refill()

    def get_stats(self) -> dict:
        """Return a comprehensive snapshot of admission controller state."""
        return {
            "utilization": {st.value: util for st, util in self._utilization.items()},
            "decisions": {
                st.value: dict(counts) for st, counts in self._counters.items()
            },
            "token_buckets": {
                st.value: self._buckets[st].stats() for st in SliceType
            },
        }
