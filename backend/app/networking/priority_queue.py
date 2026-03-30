"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
priority_queue.py – Thread-safe asyncio priority queue with TTL expiry.

Messages are ordered by (Priority.value, timestamp, sequence_num) so that
lower numeric priority values (i.e. CRITICAL=0) are dequeued first, and ties
within a priority level are broken by arrival time (FIFO).
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .message_schema import Priority, TelemetryMessage

logger = logging.getLogger(__name__)

# TTL configuration (seconds) per priority level
# None means "never expire"
_TTL_SECONDS: dict[Priority, Optional[float]] = {
    Priority.CRITICAL: None,    # never expires
    Priority.HIGH: 10.0,
    Priority.MEDIUM: 2.0,
    Priority.LOW: 0.5,
}



@dataclass(order=False)
class PriorityMessage:
    """
    Wrapper that makes TelemetryMessage comparable for heapq.

    Ordering: (priority.value ASC, timestamp ASC, sequence_num ASC)
    This ensures CRITICAL (value=0) messages surface before HIGH (value=1),
    and within the same priority, older messages surface first.
    """

    priority: Priority
    timestamp: float           # wall-clock insertion time (not message timestamp)
    sequence: int              # message sequence_num used as tie-breaker
    message: TelemetryMessage
    inserted_at: float = field(default_factory=time.monotonic, compare=False)

    def __lt__(self, other: "PriorityMessage") -> bool:
        return (self.priority.value, self.timestamp, self.sequence) < (
            other.priority.value,
            other.timestamp,
            other.sequence,
        )

    def __le__(self, other: "PriorityMessage") -> bool:
        return self == other or self < other

    def __gt__(self, other: "PriorityMessage") -> bool:
        return not self <= other

    def __ge__(self, other: "PriorityMessage") -> bool:
        return not self < other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PriorityMessage):
            return NotImplemented
        return (self.priority.value, self.timestamp, self.sequence) == (
            other.priority.value,
            other.timestamp,
            other.sequence,
        )

    def is_expired(self) -> bool:
        """Return True if this message has exceeded its TTL."""
        ttl = _TTL_SECONDS.get(self.priority)
        if ttl is None:
            return False
        age = time.monotonic() - self.inserted_at
        return age > ttl



class AsyncPriorityQueue:
    """
    Asyncio-compatible, thread-safe min-heap priority queue for
    TelemetryMessage objects.

    Features
    --------
    * CRITICAL messages are never dropped due to TTL.
    * Expired LOW / MEDIUM / HIGH messages are pruned lazily on every ``get``
      and can be eagerly pruned by calling ``prune_expired()``.
    * ``put`` and ``get`` are coroutines and acquire an asyncio.Lock so that
      concurrent producers / consumers remain consistent without GIL tricks.
    * ``get_nowait`` is a non-blocking variant that returns ``None`` immediately
      if the queue is empty or the top item is expired.
    """

    def __init__(self) -> None:
        self._heap: list[PriorityMessage] = []
        self._lock: asyncio.Lock = asyncio.Lock()
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._total_expired: int = 0

    async def put(self, message: TelemetryMessage) -> None:
        """
        Enqueue a TelemetryMessage.

        The insertion timestamp is taken as the message's own ``timestamp``
        field so that upstream ordering is preserved within a priority band.
        """
        item = PriorityMessage(
            priority=message.priority,
            timestamp=message.timestamp,
            sequence=message.sequence_num,
            message=message,
        )
        async with self._lock:
            heapq.heappush(self._heap, item)
            self._total_enqueued += 1
            logger.debug(
                "Enqueued %s seq=%d priority=%s heap_size=%d",
                message.ev_id,
                message.sequence_num,
                message.priority.name,
                len(self._heap),
            )

    async def get(self) -> Optional[TelemetryMessage]:
        """
        Dequeue the highest-priority, non-expired message.

        Expired messages encountered at the head of the heap are silently
        discarded (counted in stats) and the next candidate is examined.
        Returns ``None`` if the queue is empty after pruning.
        """
        async with self._lock:
            return self._pop_valid()

    async def get_nowait(self) -> Optional[TelemetryMessage]:
        """
        Non-blocking variant of ``get``.  Returns ``None`` immediately if the
        lock cannot be acquired or the queue is empty / all-expired.
        """
        if self._lock.locked():
            return None
        async with self._lock:
            return self._pop_valid()

    def _pop_valid(self) -> Optional[TelemetryMessage]:
        """
        Internal: pop and return the first non-expired item, discarding any
        expired items that sit at the heap root.  Caller must hold ``_lock``.
        """
        while self._heap:
            item = heapq.heappop(self._heap)
            if item.is_expired():
                self._total_expired += 1
                logger.debug(
                    "Expired %s seq=%d priority=%s",
                    item.message.ev_id,
                    item.message.sequence_num,
                    item.priority.name,
                )
                continue
            self._total_dequeued += 1
            return item.message
        return None

    def prune_expired(self) -> int:
        """
        Eagerly remove all expired messages from the heap.

        This is an O(n) operation and rebuilds the heap.  It is safe to call
        from a synchronous context (e.g. a background maintenance task) as
        long as no concurrent coroutine is inside ``put`` / ``get``.

        Returns the number of items pruned.
        """
        before = len(self._heap)
        valid: list[PriorityMessage] = [item for item in self._heap if not item.is_expired()]
        pruned = before - len(valid)
        if pruned:
            heapq.heapify(valid)
            self._heap = valid
            self._total_expired += pruned
            logger.info("Pruned %d expired messages from priority queue", pruned)
        return pruned

    def size(self) -> int:
        """Current number of items in the heap (including not-yet-pruned expired ones)."""
        return len(self._heap)

    def stats(self) -> dict:
        """
        Return a snapshot of queue statistics.

        Counts per priority level reflect the current heap contents (may include
        expired items that have not yet been pruned).
        """
        counts: dict[str, int] = {p.name: 0 for p in Priority}
        for item in self._heap:
            counts[item.priority.name] += 1

        return {
            "heap_size": len(self._heap),
            "counts_by_priority": counts,
            "total_enqueued": self._total_enqueued,
            "total_dequeued": self._total_dequeued,
            "total_expired": self._total_expired,
            "ttl_config_seconds": {
                p.name: _TTL_SECONDS[p] for p in Priority
            },
        }
