"""
VESPER – Priority Queue Tests
==============================
pytest suite for backend.app.networking.priority_queue.AsyncPriorityQueue.

Run from repo root:
    pytest backend/tests/test_priority_queue.py -v
"""

from __future__ import annotations

import asyncio
import time
from typing import List

import pytest

from backend.app.networking.message_schema import EventType, Priority, TelemetryMessage
from backend.app.networking.priority_queue import AsyncPriorityQueue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEQ_COUNTER = 0


def _next_seq() -> int:
    global _SEQ_COUNTER  # noqa: PLW0603
    _SEQ_COUNTER += 1
    return _SEQ_COUNTER


def _make_msg(
    priority: Priority = Priority.MEDIUM,
    ev_id: str = "TEST-001",
    seq: int | None = None,
    event_type: EventType = EventType.NORMAL_TELEMETRY,
) -> TelemetryMessage:
    return TelemetryMessage(
        ev_id               = ev_id,
        timestamp           = time.time(),
        speed_kmh           = 60.0,
        acceleration_ms2    = 0.0,
        brake_intensity     = 0.0,
        steering_angle_deg  = 0.0,
        battery_temp_celsius= 30.0,
        state_of_charge_pct = 80.0,
        motor_load_pct      = 20.0,
        gps_lat             = 37.7749,
        gps_lon             = -122.4194,
        obstacle_distance_m = 100.0,
        sensor_confidence   = 0.99,
        emergency_flag      = False,
        event_type          = event_type,
        priority            = priority,
        sequence_num        = seq if seq is not None else _next_seq(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critical_before_low() -> None:
    """
    Insert a LOW-priority message first, then a CRITICAL message.
    Verify that the CRITICAL message is dequeued first regardless of insertion
    order.
    """
    q = AsyncPriorityQueue()

    low_msg      = _make_msg(priority=Priority.LOW,      seq=1)
    critical_msg = _make_msg(priority=Priority.CRITICAL, seq=2)

    await q.put(low_msg)
    await q.put(critical_msg)

    first  = await q.get()
    second = await q.get()

    assert first  is not None
    assert second is not None
    assert first.priority  == Priority.CRITICAL, (
        f"Expected CRITICAL first, got {first.priority}"
    )
    assert second.priority == Priority.LOW, (
        f"Expected LOW second, got {second.priority}"
    )


@pytest.mark.asyncio
async def test_ttl_expiry() -> None:
    """
    Insert a LOW-priority message (TTL = 0.5 s), wait longer than the TTL,
    then call prune_expired() and verify the queue is empty.
    """
    q = AsyncPriorityQueue()

    low_msg = _make_msg(priority=Priority.LOW, seq=_next_seq())
    await q.put(low_msg)

    assert q.size() == 1, "Queue should have one item before expiry"

    # Wait for the LOW TTL (0.5 s) to pass
    await asyncio.sleep(0.6)

    pruned = q.prune_expired()
    assert pruned == 1, f"Expected 1 item pruned, got {pruned}"
    assert q.size() == 0, "Queue should be empty after prune"


@pytest.mark.asyncio
async def test_concurrent_puts() -> None:
    """
    Launch 100 concurrent ``put`` coroutines and verify all 100 messages are
    retrievable from the queue.
    """
    q = AsyncPriorityQueue()
    n = 100

    msgs = [_make_msg(priority=Priority.MEDIUM, seq=i) for i in range(n)]

    # Fire all puts concurrently
    await asyncio.gather(*[q.put(msg) for msg in msgs])

    assert q.size() == n, f"Expected {n} items in queue, found {q.size()}"

    # Drain and collect
    received: List[TelemetryMessage] = []
    for _ in range(n):
        msg = await q.get()
        if msg is not None:
            received.append(msg)

    assert len(received) == n, (
        f"Expected to receive {n} messages, got {len(received)}"
    )


@pytest.mark.asyncio
async def test_empty_get() -> None:
    """
    Calling ``get`` on an empty queue should return ``None`` immediately rather
    than raising an exception or blocking indefinitely.
    """
    q = AsyncPriorityQueue()

    result = await q.get()
    assert result is None, f"Expected None from empty queue, got {result}"


@pytest.mark.asyncio
async def test_stats() -> None:
    """
    Verify that queue statistics correctly reflect the number of enqueued,
    dequeued, and per-priority-class items.
    """
    q = AsyncPriorityQueue()

    # Insert 2 CRITICAL, 3 MEDIUM
    crit_msgs   = [_make_msg(priority=Priority.CRITICAL, seq=i)     for i in range(2)]
    medium_msgs = [_make_msg(priority=Priority.MEDIUM,   seq=i + 10) for i in range(3)]

    for msg in crit_msgs + medium_msgs:
        await q.put(msg)

    s = q.stats()
    assert s["total_enqueued"] == 5,    f"total_enqueued wrong: {s}"
    assert s["total_dequeued"] == 0,    f"total_dequeued should be 0 before any get: {s}"
    assert s["counts_by_priority"]["CRITICAL"] == 2, f"CRITICAL count wrong: {s}"
    assert s["counts_by_priority"]["MEDIUM"]   == 3, f"MEDIUM count wrong: {s}"

    # Dequeue two items
    await q.get()
    await q.get()

    s2 = q.stats()
    assert s2["total_dequeued"] == 2, f"total_dequeued should be 2 after two gets: {s2}"
    assert s2["heap_size"] == 3,      f"heap_size should be 3 after two pops: {s2}"


@pytest.mark.asyncio
async def test_priority_ordering_all_levels() -> None:
    """
    Insert one message at each priority level in reverse order (LOW first),
    then dequeue and verify strictly ascending urgency order (CRITICAL → LOW).
    """
    q = AsyncPriorityQueue()

    priorities = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
    for p in priorities:
        await q.put(_make_msg(priority=p, seq=_next_seq()))

    order: List[Priority] = []
    for _ in range(4):
        msg = await q.get()
        if msg is not None:
            order.append(msg.priority)

    expected = [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]
    assert order == expected, f"Dequeue order wrong: {order}"


@pytest.mark.asyncio
async def test_critical_never_expires() -> None:
    """
    CRITICAL messages have no TTL.  After sleeping beyond the LOW TTL, a
    CRITICAL message should NOT be pruned by prune_expired().
    """
    q = AsyncPriorityQueue()

    crit_msg = _make_msg(priority=Priority.CRITICAL, seq=_next_seq())
    await q.put(crit_msg)

    await asyncio.sleep(0.7)  # beyond LOW and MEDIUM TTLs

    pruned = q.prune_expired()
    assert pruned == 0,   "CRITICAL message should never be pruned"
    assert q.size() == 1, "CRITICAL message should still be in queue"
