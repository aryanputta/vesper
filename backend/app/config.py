"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
config.py – Centralised application settings using pydantic-settings.

All values can be overridden via environment variables or a .env file placed
in the working directory.  Field names are used as-is for env-var lookup
(case-insensitive by default in pydantic-settings).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    VESPER application settings.

    Each field maps 1-to-1 to an environment variable of the same name.
    Defaults are suitable for local development; override in .env or the
    host environment for staging / production.
    """

    # ------------------------------------------------------------------
    # Server / network
    # ------------------------------------------------------------------
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    TCP_TELEMETRY_PORT: int = 9001
    UDP_SAFETY_PORT: int = 9002

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379"

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------
    DATABASE_URL: str = "postgresql+asyncpg://vesper:vesper@localhost:5432/vesper"

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    NUM_EVS: int = 10
    SCENARIO: str = "normal"
    SIMULATION_DURATION_S: float = 120.0
    TELEMETRY_HZ: float = 10.0

    # ------------------------------------------------------------------
    # Network slice SLA targets
    # ------------------------------------------------------------------

    # URLLC – Ultra-Reliable Low-Latency Communications
    URLLC_LATENCY_MS: float = 1.5
    URLLC_JITTER_MS: float = 0.5
    URLLC_LOSS_RATE: float = 0.0001

    # eMBB – Enhanced Mobile Broadband
    EMBB_LATENCY_MS: float = 15.0
    EMBB_JITTER_MS: float = 5.0
    EMBB_LOSS_RATE: float = 0.001

    # mMTC – Massive Machine-Type Communications
    MMTC_LATENCY_MS: float = 80.0
    MMTC_JITTER_MS: float = 30.0
    MMTC_LOSS_RATE: float = 0.005

    # ------------------------------------------------------------------
    # Machine learning
    # ------------------------------------------------------------------
    ML_MODELS_PATH: str = "./ml/models"
    ML_INFERENCE_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------
    # Pydantic model config
    # ------------------------------------------------------------------
    model_config = ConfigDict(env_file=".env")


@lru_cache()
def get_settings() -> Settings:
    """
    Return a cached Settings singleton.

    The ``@lru_cache`` decorator ensures the .env file is parsed only once
    per process lifetime.  Call ``get_settings.cache_clear()`` in tests that
    need to inject different environment variables.
    """
    return Settings()
