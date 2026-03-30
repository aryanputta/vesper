#!/usr/bin/env python3
"""
VESPER – Component Latency Benchmark
=====================================
Measures and prints throughput / latency statistics for the core VESPER
processing pipeline.

Components benchmarked
----------------------
1. AsyncPriorityQueue – insert + pop throughput and p99 latency.
2. UrgencyScorer (Python path) – 100 k single-sample calls.
3. UrgencyScorer (C extension) – 100 k calls + speedup ratio vs. Python.
4. XGBoost slice classifier – single-sample inference, 10 k calls.
5. RulesEngine.evaluate_all – 100 k calls.
6. EVFeatureBuffer.extract_features – 100 k calls.

Usage
-----
    python scripts/benchmark.py
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from backend.app.networking.message_schema import (
    EVState,
    EventType,
    NetworkMetrics,
    Priority,
    SliceType,
    TelemetryMessage,
)
from backend.app.networking.priority_queue import AsyncPriorityQueue
from backend.app.core.urgency_scorer import UrgencyScorer
from backend.app.core.feature_extractor import EVFeatureBuffer
from backend.app.policies.rules_engine import RulesEngine


def _make_telemetry(
    ev_id: str = "BM-001",
    seq: int = 0,
    priority: Priority = Priority.MEDIUM,
    event_type: EventType = EventType.NORMAL_TELEMETRY,
    obstacle_distance_m: float = 80.0,
    brake_intensity: float = 0.1,
    battery_temp: float = 38.0,
    sensor_confidence: float = 0.95,
    speed_kmh: float = 65.0,
    emergency_flag: bool = False,
) -> TelemetryMessage:
    return TelemetryMessage(
        ev_id               = ev_id,
        timestamp           = time.time(),
        speed_kmh           = speed_kmh,
        acceleration_ms2    = 0.5,
        brake_intensity     = brake_intensity,
        steering_angle_deg  = 2.0,
        battery_temp_celsius= battery_temp,
        state_of_charge_pct = 80.0,
        motor_load_pct      = 35.0,
        gps_lat             = 37.7749,
        gps_lon             = -122.4194,
        obstacle_distance_m = obstacle_distance_m,
        sensor_confidence   = sensor_confidence,
        emergency_flag      = emergency_flag,
        event_type          = event_type,
        priority            = priority,
        sequence_num        = seq,
    )


def _make_network_metrics(
    slice_type: SliceType = SliceType.URLLC,
    latency_ms: float = 1.5,
    utilization: float = 0.35,
    loss_rate: float = 0.0001,
) -> NetworkMetrics:
    return NetworkMetrics(
        slice_type        = slice_type,
        latency_ms        = latency_ms,
        jitter_ms         = 0.3,
        packet_loss_rate  = loss_rate,
        throughput_mbps   = 100.0,
        utilization_ratio = utilization,
        queue_depth       = 5,
        timestamp         = time.time(),
    )


def _percentile(data: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile (0–100) of ``data`` (sorted ascending)."""
    if not data:
        return 0.0
    idx = max(0, int((pct / 100.0) * len(data)) - 1)
    return data[idx]


def _benchmark_sync(fn: Callable, n: int) -> dict[str, float]:
    """Time ``n`` synchronous calls to ``fn``; return stats dict."""
    latencies_us: list[float] = []
    t_wall_start = time.perf_counter()

    for i in range(n):
        t0 = time.perf_counter()
        fn(i)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1_000_000)

    t_wall_total = time.perf_counter() - t_wall_start
    latencies_us.sort()

    return {
        "n":            n,
        "total_s":      t_wall_total,
        "throughput":   n / t_wall_total,
        "mean_us":      statistics.mean(latencies_us),
        "p50_us":       _percentile(latencies_us, 50),
        "p95_us":       _percentile(latencies_us, 95),
        "p99_us":       _percentile(latencies_us, 99),
    }


async def _bench_priority_queue(n: int = 100_000) -> dict[str, Any]:
    """Benchmark insert + pop of ``n`` messages through AsyncPriorityQueue."""
    q = AsyncPriorityQueue()

    # Pre-build messages (construction cost excluded from timing)
    msgs_critical = [
        _make_telemetry(seq=i, priority=Priority.CRITICAL) for i in range(n // 2)
    ]
    msgs_low = [
        _make_telemetry(seq=i + n // 2, priority=Priority.LOW) for i in range(n // 2)
    ]

    insert_start = time.perf_counter()
    for msg in msgs_critical + msgs_low:
        await q.put(msg)
    insert_total = time.perf_counter() - insert_start

    pop_latencies_us: list[float] = []
    pop_start = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        result = await q.get()
        t1 = time.perf_counter()
        if result is not None:
            pop_latencies_us.append((t1 - t0) * 1_000_000)
    pop_total = time.perf_counter() - pop_start

    pop_latencies_us.sort()
    return {
        "insert_throughput":   n / max(insert_total, 1e-9),
        "pop_throughput":      len(pop_latencies_us) / max(pop_total, 1e-9),
        "pop_p99_us":          _percentile(pop_latencies_us, 99),
        "pop_p50_us":          _percentile(pop_latencies_us, 50),
    }


def _bench_urgency_scorer(n: int = 100_000) -> dict[str, Any]:
    scorer = UrgencyScorer()
    msg    = _make_telemetry()
    ev_st  = EVState.CAUTION

    # Force Python path
    import backend.app.core.urgency_scorer as _us_mod
    original_c = _us_mod._c_compute  # noqa: SLF001

    # Python path
    _us_mod._c_compute = None  # noqa: SLF001
    py_stats = _benchmark_sync(lambda _i: scorer.compute(msg, ev_st), n)

    # C path (if available)
    _us_mod._c_compute = original_c  # restore
    if original_c is not None:
        c_stats = _benchmark_sync(lambda _i: scorer.compute(msg, ev_st), n)
        speedup = c_stats["throughput"] / py_stats["throughput"]
    else:
        c_stats = None
        speedup = None

    return {
        "python": py_stats,
        "c":      c_stats,
        "speedup_ratio": speedup,
    }


def _bench_xgboost(n: int = 10_000) -> dict[str, Any] | None:
    """
    Benchmark the XGBoost slice classifier if a trained model is present.
    Returns None and logs a warning if the model cannot be loaded.
    """
    try:
        import numpy as np
        import xgboost as xgb  # type: ignore[import]

        model_path = _REPO_ROOT / "ml" / "models" / "slice_classifier.json"
        if not model_path.exists():
            return None  # model not yet trained

        booster = xgb.Booster()
        booster.load_model(str(model_path))

        # Build a dummy feature matrix matching the 22-feature schema
        rng     = np.random.default_rng(42)
        X_dummy = rng.random((1, 22)).astype(np.float32)
        dmatrix = xgb.DMatrix(X_dummy)

        def _infer(_i: int) -> None:
            booster.predict(dmatrix)

        return _benchmark_sync(_infer, n)

    except ImportError:
        return None


def _bench_rules_engine(n: int = 100_000) -> dict[str, Any]:
    engine = RulesEngine()
    msg    = _make_telemetry()
    ev_st  = EVState.NORMAL
    metrics = {
        "urllc_utilization": 0.35,
        "embb_utilization":  0.40,
        "mmtc_utilization":  0.20,
    }

    return _benchmark_sync(
        lambda _i: engine.evaluate_all(msg, ev_st, metrics),
        n,
    )


def _bench_feature_extractor(n: int = 100_000) -> dict[str, Any]:
    buf = EVFeatureBuffer(ev_id="BM-001")

    # Warm the buffer with 50 frames
    for i in range(50):
        buf.update(_make_telemetry(seq=i))

    net = {
        SliceType.URLLC: _make_network_metrics(SliceType.URLLC, 1.5, 0.35, 0.0001),
        SliceType.EMBB:  _make_network_metrics(SliceType.EMBB, 15.0, 0.40, 0.001),
        SliceType.MMTC:  _make_network_metrics(SliceType.MMTC, 80.0, 0.20, 0.005),
    }

    return _benchmark_sync(
        lambda _i: buf.extract_features(net, urgency_score=0.25),
        n,
    )


def _fmt(val: float | None, unit: str = "") -> str:
    if val is None:
        return "N/A"
    if unit:
        return f"{val:>12,.1f} {unit}"
    return f"{val:>12,.1f}"


def _print_table(results: dict[str, Any]) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print("VESPER Component Benchmark Results")
    print(sep)

    pq = results.get("priority_queue", {})
    print(f"\n{'─'*78}")
    print(f"{'Component':<35} {'Throughput':>14} {'p50_us':>10} {'p99_us':>10}")
    print(f"{'─'*78}")

    print(
        f"{'1. PriorityQueue (insert)':<35}"
        f"{_fmt(pq.get('insert_throughput'), 'calls/s'):>25}"
        f"{'—':>11}"
        f"{'—':>11}"
    )
    print(
        f"{'   PriorityQueue (pop)':<35}"
        f"{_fmt(pq.get('pop_throughput'), 'calls/s'):>25}"
        f"{_fmt(pq.get('pop_p50_us'), 'µs'):>21}"
        f"{_fmt(pq.get('pop_p99_us'), 'µs'):>11}"
    )

    us = results.get("urgency_scorer", {})
    py = us.get("python", {}) or {}
    c  = us.get("c", {}) or {}
    print(
        f"{'2. UrgencyScorer (Python)':<35}"
        f"{_fmt(py.get('throughput'), 'calls/s'):>25}"
        f"{_fmt(py.get('p50_us'), 'µs'):>21}"
        f"{_fmt(py.get('p99_us'), 'µs'):>11}"
    )
    if c:
        print(
            f"{'3. UrgencyScorer (C extension)':<35}"
            f"{_fmt(c.get('throughput'), 'calls/s'):>25}"
            f"{_fmt(c.get('p50_us'), 'µs'):>21}"
            f"{_fmt(c.get('p99_us'), 'µs'):>11}"
        )
        sr = us.get("speedup_ratio")
        if sr is not None:
            print(f"   C speedup ratio: {sr:.1f}x")
    else:
        print(f"{'3. UrgencyScorer (C extension)':<35}  (not available – .so not built)")

    xg = results.get("xgboost")
    if xg:
        print(
            f"{'4. XGBoost inference':<35}"
            f"{_fmt(xg.get('throughput'), 'calls/s'):>25}"
            f"{_fmt(xg.get('p50_us'), 'µs'):>21}"
            f"{_fmt(xg.get('p99_us'), 'µs'):>11}"
        )
    else:
        print(f"{'4. XGBoost inference':<35}  (model not found or xgboost not installed)")

    re = results.get("rules_engine", {})
    print(
        f"{'5. RulesEngine.evaluate_all':<35}"
        f"{_fmt(re.get('throughput'), 'calls/s'):>25}"
        f"{_fmt(re.get('p50_us'), 'µs'):>21}"
        f"{_fmt(re.get('p99_us'), 'µs'):>11}"
    )

    fe = results.get("feature_extractor", {})
    print(
        f"{'6. EVFeatureBuffer.extract':<35}"
        f"{_fmt(fe.get('throughput'), 'calls/s'):>25}"
        f"{_fmt(fe.get('p50_us'), 'µs'):>21}"
        f"{_fmt(fe.get('p99_us'), 'µs'):>11}"
    )

    print(f"{'─'*78}")
    print()


async def _async_main() -> None:
    print("VESPER Benchmark – warming up…")

    results: dict[str, Any] = {}

    print("[1/6] Benchmarking AsyncPriorityQueue (100k messages)…")
    results["priority_queue"] = await _bench_priority_queue(n=100_000)

    print("[2/6] Benchmarking UrgencyScorer (100k calls)…")
    results["urgency_scorer"] = _bench_urgency_scorer(n=100_000)

    print("[4/6] Benchmarking XGBoost classifier (10k calls)…")
    results["xgboost"] = _bench_xgboost(n=10_000)

    print("[5/6] Benchmarking RulesEngine (100k calls)…")
    results["rules_engine"] = _bench_rules_engine(n=100_000)

    print("[6/6] Benchmarking FeatureExtractor (100k calls)…")
    results["feature_extractor"] = _bench_feature_extractor(n=100_000)

    _print_table(results)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
