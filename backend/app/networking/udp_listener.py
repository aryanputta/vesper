"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
udp_listener.py – Async UDP server for safety events (low-latency path).

UDP is used for safety-critical telemetry where the extra TCP handshake
overhead is unacceptable.  Every received datagram is immediately placed onto
the priority queue at CRITICAL or HIGH priority depending on the message's own
priority field.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

from pydantic import ValidationError

from backend.app.networking.message_schema import Priority, TelemetryMessage
from backend.app.networking.priority_queue import AsyncPriorityQueue

logger = logging.getLogger(__name__)


class UDPListener(asyncio.DatagramProtocol):
    """
    Async UDP server for high-priority EV telemetry.

    Implements ``asyncio.DatagramProtocol`` so that the event loop calls
    ``datagram_received`` directly – no ``await`` overhead on the receive path.

    Because ``datagram_received`` is synchronous, the message is put into the
    priority queue via ``asyncio.ensure_future`` / ``loop.create_task`` so the
    coroutine runs on the running event loop without blocking.

    Parameters
    ----------
    host:
        Interface to bind (e.g. ``"0.0.0.0"``).
    port:
        UDP port to listen on.
    message_callback:
        Callable invoked for every successfully parsed message.
        May be async or sync.
    queue:
        Shared ``AsyncPriorityQueue``; messages are enqueued here.
    """

    def __init__(
        self,
        host: str,
        port: int,
        message_callback: Callable,
        queue: AsyncPriorityQueue,
    ) -> None:
        self._host = host
        self._port = port
        self._message_callback = message_callback
        self._queue = queue

        self._transport: Optional[asyncio.DatagramTransport] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Counters
        self._packets_received: int = 0
        self._parse_errors: int = 0
        self._high_priority_count: int = 0   # messages with CRITICAL or HIGH priority

    async def start(self) -> None:
        """Create the UDP endpoint and start listening."""
        self._loop = asyncio.get_running_loop()
        transport, _ = await self._loop.create_datagram_endpoint(
            lambda: self,
            local_addr=(self._host, self._port),
        )
        self._transport = transport
        logger.info(
            "UDPListener started on %s:%s",
            self._host,
            self._port,
        )

    async def stop(self) -> None:
        """Close the UDP transport."""
        if self._transport is not None:
            self._transport.close()
            self._transport = None
            logger.info("UDPListener stopped.")

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        """
        Called by the event loop for each received datagram.

        Parses the datagram as a JSON ``TelemetryMessage``.  On success, the
        message is enqueued and the user callback is scheduled.  On failure,
        the error counter is incremented.
        """
        self._packets_received += 1

        msg = self._parse_datagram(data, addr)
        if msg is None:
            return

        # Track high-priority messages
        if msg.priority in (Priority.CRITICAL, Priority.HIGH):
            self._high_priority_count += 1

        # Schedule async operations on the running event loop
        loop = self._loop or asyncio.get_event_loop()
        loop.create_task(self._enqueue_and_callback(msg))

    def error_received(self, exc: Exception) -> None:
        """Called when a send or receive operation raises an OS-level error."""
        logger.error("UDPListener error_received: %s", exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        """Called when the connection is closed."""
        if exc:
            logger.error("UDPListener connection_lost with error: %s", exc)
        else:
            logger.info("UDPListener connection_lost (clean shutdown).")

    async def _enqueue_and_callback(self, msg: TelemetryMessage) -> None:
        """Put the message on the priority queue and invoke the user callback."""
        try:
            await self._queue.put(msg)
        except Exception:
            logger.exception("Failed to enqueue UDP message from %s", msg.ev_id)

        try:
            result = self._message_callback(msg)
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("message_callback raised for UDP message %s", msg.ev_id)

    def _parse_datagram(
        self,
        data: bytes,
        addr: tuple,
    ) -> Optional[TelemetryMessage]:
        """
        Decode a raw UDP datagram into a ``TelemetryMessage``.

        Returns ``None`` and increments the parse-error counter on failure.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
            return TelemetryMessage.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._parse_errors += 1
            logger.warning(
                "UDP JSON decode error from %s:%s – %s",
                addr[0], addr[1], exc,
            )
        except ValidationError as exc:
            self._parse_errors += 1
            logger.warning(
                "UDP validation error from %s:%s – %s",
                addr[0], addr[1], exc,
            )
        except Exception:
            self._parse_errors += 1
            logger.exception("Unexpected error parsing UDP datagram from %s:%s", addr[0], addr[1])
        return None

    def stats(self) -> dict:
        """Return a snapshot of listener statistics."""
        return {
            "host": self._host,
            "port": self._port,
            "transport_open": self._transport is not None,
            "packets_received": self._packets_received,
            "parse_errors": self._parse_errors,
            "high_priority_count": self._high_priority_count,
        }
