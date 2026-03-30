"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
main.py – FastAPI application entry point.

Startup sequence (via lifespan context manager)
------------------------------------------------
1. Connect to Redis
2. Initialise SliceManager, AlertService, ModelRegistry
3. Create AsyncPriorityQueue and SliceOrchestrator
4. Start the orchestrator loop
5. Start TCP telemetry listener (port 9001)
6. Start UDP safety listener (port 9002)

Shutdown sequence
-----------------
1. Stop TCP/UDP listeners
2. Stop orchestrator
3. Close Redis connection
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.networking.message_schema import TelemetryMessage
from backend.app.networking.priority_queue import AsyncPriorityQueue
from backend.app.services.alert_service import AlertService
from backend.app.services.orchestrator import SliceOrchestrator
from backend.app.services.slice_manager import SliceManager
from ml.inference.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TCP telemetry listener
# ---------------------------------------------------------------------------


async def _handle_tcp_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    queue: AsyncPriorityQueue,
) -> None:
    """
    Handle a single TCP telemetry connection.

    Protocol: 4-byte big-endian length prefix followed by UTF-8 JSON payload.
    """
    peer = writer.get_extra_info("peername")
    logger.debug("TCP connection from %s", peer)
    try:
        while True:
            # Read 4-byte length prefix
            header = await asyncio.wait_for(reader.readexactly(4), timeout=30.0)
            if not header:
                break
            (length,) = struct.unpack(">I", header)
            if length == 0 or length > 65_536:
                logger.warning("TCP: invalid message length %d from %s", length, peer)
                break
            payload = await asyncio.wait_for(reader.readexactly(length), timeout=10.0)
            try:
                msg = TelemetryMessage.model_validate_json(payload)
                await queue.put(msg)
            except Exception as exc:
                logger.debug("TCP: JSON parse error from %s: %s", peer, exc)
    except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionResetError):
        pass
    except Exception as exc:
        logger.debug("TCP client handler error for %s: %s", peer, exc)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.debug("TCP connection closed: %s", peer)


async def _start_tcp_listener(
    host: str,
    port: int,
    queue: AsyncPriorityQueue,
) -> asyncio.Server:
    """Start the asyncio TCP telemetry server."""
    server = await asyncio.start_server(
        lambda r, w: _handle_tcp_client(r, w, queue),
        host=host,
        port=port,
    )
    logger.info("TCP telemetry listener started on %s:%d", host, port)
    return server


# ---------------------------------------------------------------------------
# UDP safety listener
# ---------------------------------------------------------------------------


class _UDPProtocol(asyncio.DatagramProtocol):
    """asyncio UDP protocol that enqueues received telemetry datagrams."""

    def __init__(self, queue: AsyncPriorityQueue, loop: asyncio.AbstractEventLoop) -> None:
        self._queue = queue
        self._loop = loop

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        try:
            msg = TelemetryMessage.model_validate_json(data)
            asyncio.ensure_future(self._queue.put(msg), loop=self._loop)
        except Exception as exc:
            logger.debug("UDP: parse error from %s: %s", addr, exc)

    def error_received(self, exc: Exception) -> None:
        logger.warning("UDP listener error: %s", exc)


async def _start_udp_listener(
    host: str,
    port: int,
    queue: AsyncPriorityQueue,
) -> asyncio.BaseTransport:
    """Bind the UDP safety listener and return the transport."""
    loop = asyncio.get_event_loop()
    transport, _ = await loop.create_datagram_endpoint(
        lambda: _UDPProtocol(queue, loop),
        local_addr=(host, port),
    )
    logger.info("UDP safety listener started on %s:%d", host, port)
    return transport


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager managing application startup and shutdown.

    Everything placed before ``yield`` runs on startup; everything after
    ``yield`` runs on shutdown.
    """
    settings = get_settings()

    # ── Configure logging ─────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ── Redis connection ──────────────────────────────────────────────────
    redis_client: Any = None
    try:
        redis_client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,
            socket_connect_timeout=3,
        )
        await redis_client.ping()
        logger.info("Redis connected: %s", settings.REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable (%s) – running without Redis", exc)
        redis_client = None

    # ── Core components ───────────────────────────────────────────────────
    slice_manager = SliceManager(settings=settings)
    alert_service = AlertService(redis_client=redis_client)

    model_registry = ModelRegistry(models_path=settings.ML_MODELS_PATH)
    if settings.ML_INFERENCE_ENABLED:
        loaded = model_registry.load_models()
        if loaded:
            logger.info("ML models loaded successfully")
        else:
            logger.info("No ML models found – running in heuristic mode")

    queue: AsyncPriorityQueue = AsyncPriorityQueue()

    orchestrator = SliceOrchestrator(
        queue=queue,
        slice_manager=slice_manager,
        alert_service=alert_service,
        model_registry=model_registry,
        settings=settings,
    )

    # ── Store in app.state for router access ──────────────────────────────
    app.state.settings = settings
    app.state.redis = redis_client
    app.state.slice_manager = slice_manager
    app.state.alert_service = alert_service
    app.state.model_registry = model_registry
    app.state.queue = queue
    app.state.orchestrator = orchestrator

    # ── Start orchestrator ────────────────────────────────────────────────
    await orchestrator.start()
    logger.info("SliceOrchestrator started")

    # ── Start TCP listener ────────────────────────────────────────────────
    tcp_server: Any = None
    try:
        tcp_server = await _start_tcp_listener(settings.API_HOST, settings.TCP_TELEMETRY_PORT, queue)
        app.state.tcp_server = tcp_server
    except OSError as exc:
        logger.warning("Could not start TCP listener on port %d: %s", settings.TCP_TELEMETRY_PORT, exc)

    # ── Start UDP listener ────────────────────────────────────────────────
    udp_transport: Any = None
    try:
        udp_transport = await _start_udp_listener(settings.API_HOST, settings.UDP_SAFETY_PORT, queue)
        app.state.udp_transport = udp_transport
    except OSError as exc:
        logger.warning("Could not start UDP listener on port %d: %s", settings.UDP_SAFETY_PORT, exc)

    logger.info("VESPER startup complete")

    # ── Yield control to FastAPI ──────────────────────────────────────────
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    logger.info("VESPER shutting down …")

    if udp_transport is not None:
        try:
            udp_transport.close()
        except Exception:
            pass

    if tcp_server is not None:
        try:
            tcp_server.close()
            await tcp_server.wait_closed()
        except Exception:
            pass

    await orchestrator.stop()

    if redis_client is not None:
        try:
            await redis_client.aclose()
        except Exception:
            pass

    logger.info("VESPER shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="VESPER",
    description="5G Slice-Aware EV Safety Orchestrator",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include API routers ───────────────────────────────────────────────────
from backend.app.api.routes_slices import router as slices_router
from backend.app.api.routes_telemetry import router as telemetry_router
from backend.app.api.routes_decisions import router as decisions_router
from backend.app.api.routes_scenarios import router as scenarios_router

app.include_router(slices_router, prefix="/slices", tags=["slices"])
app.include_router(telemetry_router, prefix="/telemetry", tags=["telemetry"])
app.include_router(decisions_router, prefix="/decisions", tags=["decisions"])
app.include_router(scenarios_router, prefix="/scenarios", tags=["scenarios"])


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """
    Lightweight health check endpoint.

    Returns a JSON document with ``status`` and per-component health
    information.  Suitable for load-balancer probes.
    """
    components: dict[str, Any] = {}

    # Redis
    redis_client = getattr(app.state, "redis", None)
    if redis_client is not None:
        try:
            await redis_client.ping()
            components["redis"] = "ok"
        except Exception:
            components["redis"] = "degraded"
    else:
        components["redis"] = "unavailable"

    # Orchestrator
    orchestrator = getattr(app.state, "orchestrator", None)
    if orchestrator is not None:
        stats = orchestrator.get_stats()
        components["orchestrator"] = {
            "status": "running" if orchestrator.running else "stopped",
            "total_processed": stats["total_processed"],
            "avg_latency_ms": stats["avg_decision_latency_ms"],
        }
    else:
        components["orchestrator"] = "not initialised"

    # Slice manager
    slice_manager = getattr(app.state, "slice_manager", None)
    if slice_manager is not None:
        sm_stats = slice_manager.get_stats()
        components["slice_manager"] = {
            "status": "ok",
            "total_decisions": sm_stats.get("total_decisions", 0),
        }
    else:
        components["slice_manager"] = "not initialised"

    # ML model registry
    model_registry = getattr(app.state, "model_registry", None)
    if model_registry is not None:
        components["model_registry"] = model_registry.status()
    else:
        components["model_registry"] = "not initialised"

    # Queue
    queue = getattr(app.state, "queue", None)
    if queue is not None:
        q_stats = queue.stats()
        components["priority_queue"] = {
            "heap_size": q_stats["heap_size"],
            "total_enqueued": q_stats["total_enqueued"],
        }

    overall_status = "ok"
    if components.get("orchestrator") == "not initialised":
        overall_status = "degraded"

    return {
        "status": overall_status,
        "timestamp": time.time(),
        "version": "1.0.0",
        "components": components,
    }
