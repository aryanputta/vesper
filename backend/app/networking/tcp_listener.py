"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
tcp_listener.py – Async TCP server that receives EV telemetry frames.

Frames are newline-delimited JSON objects, each parseable as a
TelemetryMessage.  Malformed frames are logged and discarded; the connection
remains open.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

from pydantic import ValidationError

from backend.app.networking.message_schema import TelemetryMessage
from backend.app.networking.priority_queue import AsyncPriorityQueue

logger = logging.getLogger(__name__)


class TCPListener:
    """
    Async TCP server for receiving EV telemetry.

    Each connected EV client sends newline-delimited JSON frames.  The
    listener parses them into ``TelemetryMessage`` objects, invokes a
    caller-supplied callback, and places the message into an
    ``AsyncPriorityQueue`` for downstream processing.

    Parameters
    ----------
    host:
        Interface to bind (e.g. ``"0.0.0.0"``).
    port:
        TCP port to listen on.
    message_callback:
        Async or sync callable invoked for every successfully parsed message.
        Signature: ``callback(msg: TelemetryMessage) -> None | Awaitable``.
    queue:
        The shared priority queue; parsed messages are placed here via
        ``queue.put(msg)``.
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

        self._server: Optional[asyncio.AbstractServer] = None

        # Counters
        self._connected_clients: int = 0
        self._messages_received: int = 0
        self._parse_errors: int = 0

    async def start(self) -> None:
        """Start the TCP server and begin accepting connections."""
        self._server = await asyncio.start_server(
            self.handle_client,
            host=self._host,
            port=self._port,
        )
        addr = self._server.sockets[0].getsockname() if self._server.sockets else (self._host, self._port)
        logger.info("TCPListener started on %s:%s", addr[0], addr[1])
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            logger.info("TCPListener stopped.")
            self._server = None

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handle a single TCP client connection.

        Reads newline-delimited JSON frames until the connection is closed or
        an unrecoverable I/O error occurs.  Malformed or invalid frames
        increment the parse-error counter and are skipped.
        """
        peer = writer.get_extra_info("peername", default=("?", "?"))
        logger.info("Client connected: %s:%s", peer[0], peer[1])
        self._connected_clients += 1

        try:
            while True:
                try:
                    line = await reader.readline()
                except (ConnectionResetError, asyncio.IncompleteReadError, OSError) as exc:
                    logger.debug("Connection from %s:%s closed during read: %s", peer[0], peer[1], exc)
                    break

                if not line:
                    # EOF – client disconnected
                    logger.info("Client disconnected: %s:%s", peer[0], peer[1])
                    break

                raw = line.strip()
                if not raw:
                    continue  # blank line, skip silently

                msg = self._parse_frame(raw, peer)
                if msg is None:
                    continue

                self._messages_received += 1

                # Invoke user callback (supports both sync and async callables)
                try:
                    result = self._message_callback(msg)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.exception("message_callback raised an exception for %s", msg.ev_id)

                # Enqueue for downstream processing
                try:
                    await self._queue.put(msg)
                except Exception:
                    logger.exception("Failed to enqueue message from %s", msg.ev_id)

        finally:
            self._connected_clients -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def _parse_frame(
        self,
        raw: bytes,
        peer: tuple,
    ) -> Optional[TelemetryMessage]:
        """
        Attempt to decode a raw bytes frame into a ``TelemetryMessage``.

        Returns ``None`` and increments the error counter on failure.
        """
        try:
            payload = json.loads(raw.decode("utf-8"))
            return TelemetryMessage.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._parse_errors += 1
            logger.warning(
                "JSON decode error from %s:%s – %s (raw=%r)",
                peer[0], peer[1], exc, raw[:120],
            )
        except ValidationError as exc:
            self._parse_errors += 1
            logger.warning(
                "Validation error from %s:%s – %s",
                peer[0], peer[1], exc,
            )
        except Exception:
            self._parse_errors += 1
            logger.exception("Unexpected error parsing frame from %s:%s", peer[0], peer[1])
        return None

    def stats(self) -> dict:
        """Return a snapshot of listener statistics."""
        return {
            "host": self._host,
            "port": self._port,
            "connected_clients": self._connected_clients,
            "messages_received": self._messages_received,
            "parse_errors": self._parse_errors,
            "server_running": self._server is not None,
        }
