"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
alert_service.py – Redis pub/sub alert broadcaster and persistent store.

``AlertService`` publishes alerts and decisions to Redis channels and stores
alerts in a Redis sorted set keyed by timestamp.  All Redis operations are
handled gracefully: if Redis is unavailable the service degrades to in-memory
storage and logs a warning.

Channel layout
--------------
``vesper:alerts``
    AlertEvent JSON objects published on every safety alert.
``vesper:decisions``
    SliceDecision JSON objects published on every routing decision.
``vesper:alerts:store``
    Redis sorted set (score = timestamp) for persistent alert history.
    Capped at 1 000 members.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, Callable, Optional

from backend.app.networking.message_schema import AlertEvent, EventType, Priority, SliceDecision, SliceType

logger = logging.getLogger(__name__)

_ALERT_CHANNEL = "vesper:alerts"
_DECISION_CHANNEL = "vesper:decisions"
_ALERT_STORE_KEY = "vesper:alerts:store"
_MAX_STORED_ALERTS = 1000
_IN_MEMORY_BUFFER_MAXLEN = 1000  # fallback when Redis is unavailable


class AlertService:
    """
    Redis pub/sub alert broadcaster and persistent alert store.

    Parameters
    ----------
    redis_client:
        An async Redis client (e.g. ``redis.asyncio.Redis``).  May be ``None``
        for in-memory-only operation (useful in unit tests or development).

    Usage
    -----
    Publish an alert::

        await alert_service.publish_alert(alert)

    Subscribe to alerts::

        async def on_alert(alert: AlertEvent) -> None:
            print(alert)
        await alert_service.subscribe_alerts(on_alert)

    Query recent alerts::

        recent = alert_service.get_recent_alerts(n=10)
    """

    def __init__(self, redis_client: Any = None) -> None:
        self._redis = redis_client
        self._callbacks: list[Callable] = []
        self._total_alerts_published: int = 0
        self._total_decisions_published: int = 0

        # In-memory fallback buffer (also used for get_recent_alerts when Redis
        # is not available)
        self._in_memory_alerts: deque[AlertEvent] = deque(maxlen=_IN_MEMORY_BUFFER_MAXLEN)

        # Background subscription task (started by subscribe_alerts)
        self._subscription_task: Optional[asyncio.Task] = None

    async def publish_alert(self, alert: AlertEvent) -> None:
        """
        Publish an ``AlertEvent`` to the ``vesper:alerts`` channel and store it.

        Parameters
        ----------
        alert:
            The alert to broadcast and persist.
        """
        self._in_memory_alerts.append(alert)
        self._total_alerts_published += 1

        payload = alert.model_dump_json()

        if self._redis is not None:
            try:
                await self._redis.publish(_ALERT_CHANNEL, payload)
            except Exception as exc:
                logger.warning("Failed to publish alert to Redis channel: %s", exc)
        else:
            logger.debug("Redis unavailable; alert stored in-memory only: %s", alert.alert_id)

        # Also persist to sorted set
        self.store_alert(alert)

        logger.warning(
            "ALERT published [%s] ev=%s event=%s severity=%s",
            alert.alert_id,
            alert.ev_id,
            alert.event_type.value,
            alert.severity.name,
        )

        # Notify local subscribers (used by WebSocket broadcaster etc.)
        for cb in self._callbacks:
            try:
                result = cb(alert)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Alert callback raised an exception")

    async def publish_decision(self, decision: SliceDecision) -> None:
        """
        Publish a ``SliceDecision`` to the ``vesper:decisions`` channel.

        Parameters
        ----------
        decision:
            The routing decision to broadcast.
        """
        self._total_decisions_published += 1
        payload = decision.model_dump_json()

        if self._redis is not None:
            try:
                await self._redis.publish(_DECISION_CHANNEL, payload)
            except Exception as exc:
                logger.warning("Failed to publish decision to Redis channel: %s", exc)
        else:
            logger.debug(
                "Redis unavailable; decision %s not broadcast.", decision.decision_id
            )

    async def subscribe_alerts(self, callback: Callable) -> None:
        """
        Subscribe to the ``vesper:alerts`` Redis channel.

        ``callback`` is called for every message received.  It may be async
        or sync.  Each message is decoded from JSON and validated as an
        ``AlertEvent`` before being passed to the callback.

        If Redis is unavailable the callback is registered as a local
        in-process subscriber instead (receives alerts published via
        ``publish_alert`` in the same process).

        Parameters
        ----------
        callback:
            Callable with signature ``(alert: AlertEvent) -> None | Awaitable``.
        """
        self._callbacks.append(callback)

        if self._redis is None:
            logger.info(
                "subscribe_alerts: Redis unavailable – registered as local callback only."
            )
            return

        # Start background task to drive the Redis subscription
        if self._subscription_task is None or self._subscription_task.done():
            self._subscription_task = asyncio.create_task(
                self._redis_subscription_loop(), name="vesper-alert-subscriber"
            )
            logger.info("subscribe_alerts: Redis subscription task started.")

    async def _redis_subscription_loop(self) -> None:
        """
        Background task: subscribe to ``vesper:alerts`` and dispatch messages.

        Runs until cancelled.
        """
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(_ALERT_CHANNEL)
            logger.info("Redis subscription active on channel '%s'", _ALERT_CHANNEL)

            async for raw_message in pubsub.listen():
                if raw_message is None:
                    continue
                if raw_message.get("type") != "message":
                    continue

                data = raw_message.get("data")
                if not data:
                    continue

                try:
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    alert = AlertEvent.model_validate_json(data)
                except Exception as exc:
                    logger.warning("Failed to decode Redis alert message: %s", exc)
                    continue

                for cb in self._callbacks:
                    try:
                        result = cb(alert)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception:
                        logger.exception("Alert callback raised during Redis dispatch")

        except asyncio.CancelledError:
            logger.info("Redis subscription loop cancelled.")
            raise
        except Exception:
            logger.exception("Unexpected error in Redis subscription loop.")

    def store_alert(self, alert: AlertEvent) -> None:
        """
        Persist an alert to Redis sorted set (score = timestamp).

        Keeps only the most recent ``_MAX_STORED_ALERTS`` entries.
        Falls back to in-memory storage if Redis is unavailable.

        Note: This is a *synchronous* method.  The Redis sorted-set write is
        scheduled as a fire-and-forget task when called from an async context.
        """
        # Always keep in-memory copy
        # (the deque was already appended in publish_alert; avoid double-append)

        if self._redis is None:
            return  # already stored in self._in_memory_alerts

        # Schedule the async Redis zset write as a task if an event loop is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._redis_store_alert(alert))
        except RuntimeError:
            # No running loop – caller is synchronous
            pass

    async def _redis_store_alert(self, alert: AlertEvent) -> None:
        """Async helper to write the alert to the Redis sorted set."""
        try:
            payload = alert.model_dump_json()
            # Use alert_id as the member string (unique) with timestamp as score
            await self._redis.zadd(
                _ALERT_STORE_KEY,
                {payload: alert.timestamp},
            )
            # Trim to last _MAX_STORED_ALERTS entries (keep highest scores = newest)
            count = await self._redis.zcard(_ALERT_STORE_KEY)
            if count > _MAX_STORED_ALERTS:
                excess = count - _MAX_STORED_ALERTS
                await self._redis.zpopmin(_ALERT_STORE_KEY, excess)
        except Exception as exc:
            logger.warning("Failed to persist alert to Redis sorted set: %s", exc)

    def get_recent_alerts(self, n: int = 50) -> list[AlertEvent]:
        """
        Return the ``n`` most recent alerts (newest first).

        Reads from in-memory buffer.  For Redis-backed retrieval use
        ``get_recent_alerts_async`` instead.

        Parameters
        ----------
        n:
            Maximum number of alerts to return.
        """
        alerts = list(self._in_memory_alerts)
        # Reverse to get newest first
        return list(reversed(alerts))[:n]

    async def get_recent_alerts_async(self, n: int = 50) -> list[AlertEvent]:
        """
        Return the ``n`` most recent alerts from Redis (newest first).

        Falls back to the in-memory buffer if Redis is unavailable.
        """
        if self._redis is None:
            return self.get_recent_alerts(n)

        try:
            # Sorted set: highest score (timestamp) = newest
            raw_entries = await self._redis.zrevrange(
                _ALERT_STORE_KEY, 0, n - 1
            )
            alerts: list[AlertEvent] = []
            for raw in raw_entries:
                try:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    alert = AlertEvent.model_validate_json(raw)
                    alerts.append(alert)
                except Exception as exc:
                    logger.warning("Failed to decode stored alert: %s", exc)
            return alerts
        except Exception as exc:
            logger.warning("Redis zrevrange failed, falling back to in-memory: %s", exc)
            return self.get_recent_alerts(n)

    async def publish_safety_event(
        self,
        ev_id: str,
        event_type: EventType,
        severity: Priority,
        description: str,
        slice_assigned: SliceType,
        response_latency_ms: float,
    ) -> AlertEvent:
        """
        Create an ``AlertEvent`` and publish it.

        Provided for backward compatibility with the existing orchestrator.
        """
        alert = AlertEvent(
            ev_id=ev_id,
            event_type=event_type,
            severity=severity,
            description=description,
            timestamp=time.time(),
            slice_assigned=slice_assigned,
            response_latency_ms=response_latency_ms,
        )
        await self.publish_alert(alert)
        return alert

    def register_callback(self, coro_fn: Callable) -> None:
        """Register an in-process callback (sync or async) for new alerts."""
        self._callbacks.append(coro_fn)

    def unregister_callback(self, coro_fn: Callable) -> None:
        """Remove a previously registered callback."""
        try:
            self._callbacks.remove(coro_fn)
        except ValueError:
            pass

    def get_alerts_for_ev(self, ev_id: str, limit: int = 20) -> list[AlertEvent]:
        """Return the most recent alerts for a specific vehicle (in-memory)."""
        matching = [a for a in self._in_memory_alerts if a.ev_id == ev_id]
        return list(reversed(matching))[:limit]

    def get_active_alerts(self, max_age_s: float = 60.0) -> list[AlertEvent]:
        """Return alerts younger than ``max_age_s`` seconds (in-memory)."""
        cutoff = time.time() - max_age_s
        return [a for a in self._in_memory_alerts if a.timestamp >= cutoff]

    def get_stats(self) -> dict:
        """Return a summary of alert service activity."""
        return {
            "total_alerts_published": self._total_alerts_published,
            "total_decisions_published": self._total_decisions_published,
            "in_memory_buffer_size": len(self._in_memory_alerts),
            "in_memory_buffer_capacity": _IN_MEMORY_BUFFER_MAXLEN,
            "active_alerts_60s": len(self.get_active_alerts(60.0)),
            "redis_available": self._redis is not None,
            "subscription_task_running": (
                self._subscription_task is not None
                and not self._subscription_task.done()
            ),
        }
