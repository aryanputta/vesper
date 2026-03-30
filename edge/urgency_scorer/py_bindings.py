"""
VESPER – Edge Urgency Scorer Python Bindings
============================================
Ctypes wrapper around the compiled C urgency scorer (urgency_scorer.so).
Falls back to a pure-Python implementation when the shared library is not
available so that the module is importable on systems without a C compiler.

Usage
-----
    from edge.urgency_scorer.py_bindings import compute_urgency_c, benchmark

    result = compute_urgency_c({
        "safety_flag": 0.0,
        "obstacle_distance_m": 25.0,
        "brake_intensity": 0.3,
        "battery_temp_celsius": 45.0,
        "sensor_confidence": 0.9,
        "speed_kmh": 65.0,
        "ev_state": 0,
    })
    print(result)
"""

from __future__ import annotations

import ctypes
import logging
import os
import statistics
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class _EVTelemetry(ctypes.Structure):
    """Maps to the C ``EVTelemetry`` struct in urgency.h."""

    _fields_ = [
        ("safety_flag",           ctypes.c_float),
        ("obstacle_distance_m",   ctypes.c_float),
        ("brake_intensity",       ctypes.c_float),
        ("battery_temp_celsius",  ctypes.c_float),
        ("sensor_confidence",     ctypes.c_float),
        ("speed_kmh",             ctypes.c_float),
        ("ev_state",              ctypes.c_int),
    ]


class _UrgencyResult(ctypes.Structure):
    """Maps to the C ``UrgencyResult`` struct in urgency.h."""

    _fields_ = [
        ("urgency_score",    ctypes.c_float),
        ("priority_class",   ctypes.c_int),
        ("component_scores", ctypes.c_float * 5),
    ]


_lib: Optional[ctypes.CDLL] = None
_c_fn: Optional[ctypes.CFUNCTYPE] = None  # type: ignore[type-arg]

_PRIORITY_NAMES = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}


def _load_library() -> bool:
    """
    Try to load ``urgency_scorer.so`` from the same directory as this file.
    Returns True on success, False on failure.
    """
    global _lib, _c_fn  # noqa: PLW0603

    so_path = Path(__file__).parent / "urgency_scorer.so"
    if not so_path.exists():
        logger.debug("urgency_scorer.so not found at %s; using pure-Python fallback.", so_path)
        return False

    try:
        lib = ctypes.CDLL(str(so_path))

        fn = lib.compute_urgency
        fn.restype = _UrgencyResult
        fn.argtypes = [ctypes.POINTER(_EVTelemetry)]

        # Smoke-test
        t = _EVTelemetry(
            safety_flag=0.0,
            obstacle_distance_m=999.0,
            brake_intensity=0.0,
            battery_temp_celsius=20.0,
            sensor_confidence=1.0,
            speed_kmh=0.0,
            ev_state=0,
        )
        r = fn(ctypes.byref(t))
        if not (0.0 <= r.urgency_score <= 1.0):
            raise ValueError(f"Smoke-test produced invalid urgency_score: {r.urgency_score}")

        _lib = lib
        _c_fn = fn
        logger.info("Loaded C urgency scorer from %s", so_path)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to load urgency_scorer.so: %s; using pure-Python fallback.", exc)
        _lib = None
        _c_fn = None
        return False


_c_available = _load_library()


_W_SAFETY   = 0.30
_W_OBSTACLE = 0.25
_W_BRAKE    = 0.20
_W_BATTERY  = 0.15
_W_SENSOR   = 0.10

_STATE_MULTIPLIERS = {0: 1.0, 1: 1.1, 2: 1.3, 3: 1.5, 4: 1.0}


def _clampf(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _python_compute_urgency(t: dict) -> dict:
    """
    Pure-Python equivalent of the C ``compute_urgency`` function.

    Parameters
    ----------
    t : dict
        Must contain the keys matching ``EVTelemetry`` fields.

    Returns
    -------
    dict with keys: urgency_score, priority_class, priority_name, component_scores
    """
    safety_flag          = float(t.get("safety_flag", 0.0))
    obstacle_distance_m  = float(t.get("obstacle_distance_m", 999.0))
    brake_intensity      = float(t.get("brake_intensity", 0.0))
    battery_temp_celsius = float(t.get("battery_temp_celsius", 20.0))
    sensor_confidence    = float(t.get("sensor_confidence", 1.0))
    ev_state             = int(t.get("ev_state", 0))

    safety_score   = _clampf(safety_flag, 0.0, 1.0)
    obstacle_score = (
        _clampf((50.0 - obstacle_distance_m) / 50.0, 0.0, 1.0)
        if obstacle_distance_m < 50.0 else 0.0
    )
    brake_score    = _clampf(brake_intensity, 0.0, 1.0)
    battery_score  = _clampf((battery_temp_celsius - 40.0) / 40.0, 0.0, 1.0)
    sensor_score   = _clampf(1.0 - sensor_confidence, 0.0, 1.0)

    weighted_sum = (
        _W_SAFETY   * safety_score
        + _W_OBSTACLE * obstacle_score
        + _W_BRAKE    * brake_score
        + _W_BATTERY  * battery_score
        + _W_SENSOR   * sensor_score
    )

    multiplier   = _STATE_MULTIPLIERS.get(ev_state, 1.0)
    final_score  = _clampf(weighted_sum * multiplier, 0.0, 1.0)

    if   final_score >= 0.75: priority_class = 3
    elif final_score >= 0.50: priority_class = 2
    elif final_score >= 0.25: priority_class = 1
    else:                     priority_class = 0

    return {
        "urgency_score":    final_score,
        "priority_class":   priority_class,
        "priority_name":    _PRIORITY_NAMES[priority_class],
        "component_scores": [
            safety_score,
            obstacle_score,
            brake_score,
            battery_score,
            sensor_score,
        ],
    }


def compute_urgency_c(telemetry_dict: dict) -> dict:
    """
    Compute urgency score for an EV telemetry snapshot.

    Uses the compiled C extension when available; automatically falls back to
    the pure-Python implementation otherwise.

    Parameters
    ----------
    telemetry_dict : dict
        Keys: safety_flag, obstacle_distance_m, brake_intensity,
              battery_temp_celsius, sensor_confidence, speed_kmh, ev_state.

    Returns
    -------
    dict
        urgency_score   : float [0.0, 1.0]
        priority_class  : int   {0=LOW, 1=MEDIUM, 2=HIGH, 3=CRITICAL}
        priority_name   : str
        component_scores: list[float]  [safety, obstacle, brake, battery, sensor]
        backend         : str  "c" | "python"
    """
    if _c_fn is not None:
        t = _EVTelemetry(
            safety_flag          = float(telemetry_dict.get("safety_flag", 0.0)),
            obstacle_distance_m  = float(telemetry_dict.get("obstacle_distance_m", 999.0)),
            brake_intensity      = float(telemetry_dict.get("brake_intensity", 0.0)),
            battery_temp_celsius = float(telemetry_dict.get("battery_temp_celsius", 20.0)),
            sensor_confidence    = float(telemetry_dict.get("sensor_confidence", 1.0)),
            speed_kmh            = float(telemetry_dict.get("speed_kmh", 0.0)),
            ev_state             = int(telemetry_dict.get("ev_state", 0)),
        )
        r = _c_fn(ctypes.byref(t))
        return {
            "urgency_score":    float(r.urgency_score),
            "priority_class":   int(r.priority_class),
            "priority_name":    _PRIORITY_NAMES.get(int(r.priority_class), "UNKNOWN"),
            "component_scores": [float(r.component_scores[i]) for i in range(5)],
            "backend":          "c",
        }

    result = _python_compute_urgency(telemetry_dict)
    result["backend"] = "python"
    return result


def benchmark(n: int = 10_000) -> dict:
    """
    Benchmark the urgency scorer for ``n`` calls.

    Runs both the C extension (if available) and pure-Python implementation.

    Parameters
    ----------
    n : int
        Number of invocations per backend.

    Returns
    -------
    dict
        timing_stats sub-dicts for each backend:
            min_us, max_us, mean_us, p50_us, p99_us, throughput_calls_per_sec
        speedup_ratio : float (C throughput / Python throughput), or None if C unavailable.
    """
    sample = {
        "safety_flag":          0.0,
        "obstacle_distance_m":  25.0,
        "brake_intensity":      0.4,
        "battery_temp_celsius": 52.0,
        "sensor_confidence":    0.85,
        "speed_kmh":            72.0,
        "ev_state":             1,  # CAUTION
    }

    def _time_backend(fn, iterations: int) -> dict:
        latencies_us: list[float] = []
        start_wall = time.perf_counter()
        for _ in range(iterations):
            t0 = time.perf_counter()
            fn(sample)
            t1 = time.perf_counter()
            latencies_us.append((t1 - t0) * 1_000_000)
        total_s = time.perf_counter() - start_wall

        latencies_us.sort()
        p99_idx = max(0, int(0.99 * iterations) - 1)
        p50_idx = max(0, int(0.50 * iterations) - 1)
        return {
            "min_us":                  latencies_us[0],
            "max_us":                  latencies_us[-1],
            "mean_us":                 statistics.mean(latencies_us),
            "p50_us":                  latencies_us[p50_idx],
            "p99_us":                  latencies_us[p99_idx],
            "throughput_calls_per_sec": iterations / total_s,
        }

    results: dict = {}

    # Python backend
    results["python"] = _time_backend(_python_compute_urgency, n)

    # C backend (if available)
    if _c_available and _c_fn is not None:
        def _c_wrapper(d: dict) -> dict:
            return compute_urgency_c(d)

        results["c"] = _time_backend(_c_wrapper, n)
        py_tput = results["python"]["throughput_calls_per_sec"]
        c_tput  = results["c"]["throughput_calls_per_sec"]
        results["speedup_ratio"] = c_tput / py_tput if py_tput > 0 else None
    else:
        results["c"] = None
        results["speedup_ratio"] = None

    results["n_iterations"] = n
    return results


if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    print("=" * 70)
    print("VESPER Urgency Scorer – Python Bindings Benchmark")
    print("=" * 70)

    n = 10_000
    print(f"\nRunning {n:,} iterations per backend...\n")

    stats = benchmark(n=n)

    backends = ["python"]
    if stats["c"] is not None:
        backends.append("c")

    header = f"{'Backend':<10} {'mean_us':>10} {'p50_us':>10} {'p99_us':>10} {'throughput':>16}"
    print(header)
    print("-" * len(header))

    for backend in backends:
        s = stats[backend]
        print(
            f"{backend:<10} "
            f"{s['mean_us']:>10.2f} "
            f"{s['p50_us']:>10.2f} "
            f"{s['p99_us']:>10.2f} "
            f"{s['throughput_calls_per_sec']:>14,.0f}/s"
        )

    if stats["speedup_ratio"] is not None:
        print(f"\nC speedup ratio: {stats['speedup_ratio']:.1f}x faster than Python")
    else:
        print("\nC extension not available – only Python backend benchmarked.")

    print("\nSingle-call example:")
    example = compute_urgency_c({
        "safety_flag":          0.0,
        "obstacle_distance_m":  8.0,
        "brake_intensity":      0.85,
        "battery_temp_celsius": 62.0,
        "sensor_confidence":    0.75,
        "speed_kmh":            55.0,
        "ev_state":             2,  # CRITICAL
    })
    print(json.dumps(example, indent=2))
