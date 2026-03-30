"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
event_deduplicator.py – Redis-backed (with in-memory fallback) event deduplication.

Duplicate detection is key to preventing the network slice manager from
re-processing the same safety event multiple times when a vehicle retransmits
or when multiple ingress nodes receive the same UDP telemetry frame.

A message is considered a duplicate if another message with the same
(ev_id, event_type, truncated-timestamp) key was seen within the configured
TTL window.  The key is a truncated SHA-256 digest so Redis memory footprint
stays bounded.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from typing import Any, Optional

from ..networking.message_schema import TelemetryMessage

logger = logging.getLogger(__name__)

# We use ``Any`` here so that callers can pass either ``redis.asyncio.Redis``
# or any duck-typed compatible client without requiring redis to be installed
# in environments that use only the in-memory fallback.
_RedisClient = Any


class EventDeduplicator:
    """
    Deduplicates TelemetryMessage events using Redis as the shared store.

    Deduplication key
    -----------------
    The key is derived from the first 16 hex characters of the SHA-256 digest
    of the string ``"{ev_id}:{event_type}:{truncated_timestamp}"``, where the
    timestamp is truncated to one decimal place (100 ms resolution).  This
    means two frames from the same vehicle, of the same event type, arriving
    within 100 ms of each other will share a key and the second will be
    considered a duplicate.

    Fallback
    --------
    If ``redis_client`` is ``None`` or if any Redis operation raises an
    exception, the deduplicator falls back to an asyncio-guarded in-memory
    dictionary.  The in-memory store is process-local and not shared across
    workers, so it provides best-effort deduplication only.

    Parameters
    ----------
    redis_client:
        An initialised ``redis.asyncio.Redis`` client (or compatible).
        Pass ``None`` to force in-memory mode.
    ttl_ms:
        Key time-to-live in milliseconds.  Defaults to 500 ms.
    """

    def __init__(
        self,
        redis_client: Optional[_RedisClient] = None,
        ttl_ms: int = 500,
    ) -> None:
        self._redis: Optional[_RedisClient] = redis_client
        self._ttl_ms: int = ttl_ms

        # In-memory fallback: {key: expiry_monotonic_seconds}
        self._mem_store: dict[str, float] = {}
        self._mem_lock: asyncio.Lock = asyncio.Lock()

        # Diagnostics
        self._redis_errors: int = 0
        self._duplicates_caught: int = 0
        self._total_checked: int = 0

        if redis_client is None:
            logger.info(
                "EventDeduplicator: no Redis client provided – using in-memory fallback"
            )

    async def is_duplicate(self, msg: TelemetryMessage) -> bool:
        """
        Check whether *msg* is a duplicate of a recently seen event.

        Parameters
        ----------
        msg:
            The telemetry message to check.

        Returns
        -------
        bool
            ``True`` if the event has already been seen within the TTL window
            (i.e. this message is a duplicate and should be discarded).
            ``False`` if the event is novel (the key has been registered so
            subsequent identical messages will return ``True``).
        """
        self._total_checked += 1
        key = self._make_key(msg)
        is_dup = await self._check_and_set(key)
        if is_dup:
            self._duplicates_caught += 1
            logger.debug(
                "Duplicate detected: ev=%s event=%s key=%s",
                msg.ev_id,
                msg.event_type,
                key,
            )
        return is_dup

    @staticmethod
    def _make_key(msg: TelemetryMessage) -> str:
        """
        Derive the 16-character hex deduplication key for *msg*.

        The timestamp is truncated to one decimal place so that messages
        within the same 100 ms slot share a key regardless of sub-10 ms
        jitter.
        """
        truncated_ts = int(msg.timestamp * 10) / 10
        raw = f"{msg.ev_id}:{msg.event_type}:{truncated_ts}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return digest[:16]

    async def _check_and_set(self, key: str) -> bool:
        """
        Attempt to SET *key* in Redis (NX + PEXPIRE).

        Returns ``True`` if the key already existed (duplicate), ``False`` if
        it was freshly set (novel event).  Falls back to in-memory on error.
        """
        if self._redis is not None:
            try:
                return await self._redis_check_and_set(key)
            except Exception as exc:  # noqa: BLE001
                self._redis_errors += 1
                if self._redis_errors == 1 or self._redis_errors % 100 == 0:
                    logger.warning(
                        "Redis error in EventDeduplicator (count=%d): %s – using in-memory fallback",
                        self._redis_errors,
                        exc,
                    )
        return await self._mem_check_and_set(key)

    async def _redis_check_and_set(self, key: str) -> bool:
        """
        Use Redis SET with NX (set-if-not-exists) and PEXPIRE.

        The atomic SET … NX … PX … command returns ``True`` when the key was
        newly set and ``None`` / ``False`` when it already existed.

        Note: ``redis.asyncio`` returns the result of ``SET key value NX PX ms``
        as ``True`` on success or ``None`` on key-exists.
        """
        # redis-py / redis.asyncio API:
        #   set(name, value, nx=True, px=ttl_ms)
        # Returns: True (set) | None (already existed)
        result = await self._redis.set(
            f"vesper:dedup:{key}",
            "1",
            nx=True,
            px=self._ttl_ms,
        )
        # result is True → newly set → NOT a duplicate
        # result is None → key already existed → IS a duplicate
        return result is None

    async def _mem_check_and_set(self, key: str) -> bool:
        """
        In-memory fallback using an asyncio.Lock-guarded dict.

        Expired entries are pruned on every call (amortised O(n) per call, but
        the dict remains small because entries expire quickly).
        """
        now = time.monotonic()
        ttl_seconds = self._ttl_ms / 1000.0
        expiry = now + ttl_seconds

        async with self._mem_lock:
            # Prune expired entries (lazy GC)
            expired_keys = [k for k, exp in self._mem_store.items() if exp <= now]
            for k in expired_keys:
                del self._mem_store[k]

            if key in self._mem_store:
                # Key still live → duplicate
                return True

            # Novel event: register it
            self._mem_store[key] = expiry
            return False

    def stats(self) -> dict[str, int]:
        """Return operational statistics for monitoring."""
        return {
            "total_checked": self._total_checked,
            "duplicates_caught": self._duplicates_caught,
            "redis_errors": self._redis_errors,
            "mem_store_size": len(self._mem_store),
            "ttl_ms": self._ttl_ms,
            "using_redis": self._redis is not None,
        }

    async def flush(self) -> None:
        """
        Clear all tracked keys.

        Useful in tests or when the deduplication window needs to be reset
        (e.g. after a controlled vehicle restart).
        """
        async with self._mem_lock:
            self._mem_store.clear()

        if self._redis is not None:
            try:
                # Scan for and delete all VESPER dedup keys
                cursor = 0
                pattern = "vesper:dedup:*"
                while True:
                    cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                    if keys:
                        await self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to flush Redis dedup keys: %s", exc)
