#!/usr/bin/env python3
"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
generate_dataset.py – Generate synthetic training dataset for VESPER ML models.

Produces a CSV with realistic, physics-informed EV telemetry + 5G slice
metrics across five driving/network scenarios.  Run from the project root:

    python scripts/generate_dataset.py --samples 100000 --output data/raw/training_data.csv
"""

import argparse
import csv
import math
import os
import random
import sys
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SCENARIOS = ["normal", "obstacle_emergency", "urllc_saturation", "battery_overheat", "anomaly_attack"]

# Scenario proportions (must sum to 1.0)
SCENARIO_WEIGHTS = {
    "normal":              0.40,
    "obstacle_emergency":  0.20,
    "urllc_saturation":    0.15,
    "battery_overheat":    0.15,
    "anomaly_attack":      0.10,
}

# Slice label integers: 0=MMTC, 1=EMBB, 2=URLLC
MMTC  = 0
EMBB  = 1
URLLC = 2

# EV state integers: 0=NORMAL, 1=CAUTION, 2=CRITICAL, 3=EMERGENCY, 4=RECOVERY
EV_NORMAL    = 0
EV_CAUTION   = 1
EV_CRITICAL  = 2
EV_EMERGENCY = 3
EV_RECOVERY  = 4

# Urgency score component weights (must match UrgencyScorer in backend)
W_SAFETY = 0.30
W_OBS    = 0.25
W_BRAKE  = 0.20
W_BATT   = 0.15
W_SENS   = 0.10

BATT_TEMP_BASE  = 40.0
BATT_TEMP_RANGE = 40.0
OBSTACLE_CAP    = 50.0


def _compute_urgency(
    emergency_flag: bool,
    obstacle_m: float,
    brake: float,
    battery_temp: float,
    sensor_conf: float,
) -> float:
    """Reproduce UrgencyScorer logic for ground-truth label generation."""
    sf_score  = 1.0 if emergency_flag else 0.0
    obs_score = max(0.0, 1.0 - obstacle_m / OBSTACLE_CAP) if obstacle_m < OBSTACLE_CAP else 0.0
    brk_score = float(brake)
    bat_raw   = (battery_temp - BATT_TEMP_BASE) / BATT_TEMP_RANGE
    bat_score = max(0.0, min(1.0, bat_raw))
    sen_score = 1.0 - float(sensor_conf)

    raw = (
        W_SAFETY * sf_score
        + W_OBS   * obs_score
        + W_BRAKE * brk_score
        + W_BATT  * bat_score
        + W_SENS  * sen_score
    )
    return min(1.0, max(0.0, raw))


def _assign_slice_label(urgency: float, scenario: str, urllc_util: float) -> int:
    """
    Rule-based ground-truth slice assignment.

    URLLC:  urgency >= 0.50  OR  urllc_util >= 0.95 (saturation forces escalation)
    EMBB:   urgency >= 0.20  OR  (normal-ish driving with decent throughput demand)
    MMTC:   everything else  (low-rate sensor pings, bulk log uploads)
    """
    if urgency >= 0.50 or (scenario == "urllc_saturation" and urllc_util >= 0.95):
        return URLLC
    if urgency >= 0.20 or scenario in ("obstacle_emergency", "battery_overheat"):
        return EMBB
    return MMTC


def _assign_ev_state(urgency: float, emergency_flag: bool) -> int:
    if emergency_flag or urgency >= 0.75:
        return EV_EMERGENCY
    if urgency >= 0.50:
        return EV_CRITICAL
    if urgency >= 0.25:
        return EV_CAUTION
    return EV_NORMAL


def _noise(sigma: float) -> float:
    """Return Gaussian noise with zero mean and given std-dev."""
    return random.gauss(0.0, sigma)


def _gen_normal(t: float) -> dict:
    """Stable urban/suburban driving.  Low urgency → mostly EMBB/MMTC."""
    speed       = random.uniform(40.0, 80.0) + _noise(3.0)
    accel       = random.gauss(0.0, 0.5)
    brake       = max(0.0, min(1.0, random.gauss(0.05, 0.03)))
    batt_temp   = random.uniform(35.0, 50.0) + _noise(1.0)
    soc         = random.uniform(40.0, 90.0)
    obstacle    = random.uniform(25.0, 120.0) + _noise(2.0)
    sensor_conf = min(1.0, max(0.0, random.gauss(0.92, 0.03)))
    motor_load  = random.uniform(20.0, 60.0)
    emergency   = False

    urllc_lat   = random.gauss(4.5,  0.5)
    urllc_util  = random.gauss(0.35, 0.08)
    urllc_loss  = max(0.0, random.gauss(0.002, 0.001))
    urllc_tput  = random.gauss(18.0, 2.0)
    embb_lat    = random.gauss(18.0, 3.0)
    embb_util   = random.gauss(0.55, 0.10)
    mmtc_util   = random.gauss(0.30, 0.07)

    return dict(
        speed_kmh=max(0.0, speed), acceleration_ms2=accel,
        brake_intensity=brake, battery_temp_celsius=batt_temp,
        state_of_charge_pct=max(0.0, min(100.0, soc)),
        obstacle_distance_m=max(0.0, obstacle),
        sensor_confidence=sensor_conf, motor_load_pct=max(0.0, min(100.0, motor_load)),
        emergency_flag=emergency,
        urllc_latency_ms=max(0.1, urllc_lat),
        urllc_utilization=max(0.0, min(1.0, urllc_util)),
        urllc_loss_rate=max(0.0, min(1.0, urllc_loss)),
        urllc_throughput_mbps=max(0.0, urllc_tput),
        embb_latency_ms=max(0.1, embb_lat),
        embb_utilization=max(0.0, min(1.0, embb_util)),
        mmtc_utilization=max(0.0, min(1.0, mmtc_util)),
    )


def _gen_obstacle_emergency(t: float, seq: int) -> dict:
    """
    Obstacle approaching at speed – distance decreases from 80 m to 2 m
    over ~200 samples, braking increases correspondingly.
    """
    # Phase: 0=approaching (100 samples), 1=emergency braking (100+)
    phase_t = (seq % 300) / 300.0       # normalised position within cycle

    if phase_t < 0.4:
        # Closing phase: obstacle distance drops 80 → 5 m
        obstacle = 80.0 - phase_t / 0.4 * 75.0 + _noise(1.5)
        brake    = min(1.0, max(0.0, phase_t / 0.4 * 0.6 + _noise(0.05)))
        speed    = max(0.0, 70.0 - phase_t / 0.4 * 50.0 + _noise(2.0))
        emergency = False
    elif phase_t < 0.6:
        # Hard braking: obstacle very close
        obstacle = max(1.0, 5.0 - (phase_t - 0.4) / 0.2 * 3.0 + _noise(0.5))
        brake    = min(1.0, max(0.0, 0.85 + _noise(0.05)))
        speed    = max(0.0, 20.0 - (phase_t - 0.4) / 0.2 * 20.0 + _noise(1.0))
        emergency = True
    else:
        # Recovery: pulling away
        obstacle = min(80.0, (phase_t - 0.6) / 0.4 * 60.0 + 5.0 + _noise(2.0))
        brake    = max(0.0, 0.3 - (phase_t - 0.6) / 0.4 * 0.3 + _noise(0.03))
        speed    = min(60.0, (phase_t - 0.6) / 0.4 * 40.0 + _noise(2.0))
        emergency = False

    batt_temp   = random.uniform(38.0, 52.0) + _noise(1.0)
    soc         = random.uniform(30.0, 80.0)
    sensor_conf = min(1.0, max(0.0, random.gauss(0.88, 0.04)))
    motor_load  = max(0.0, min(100.0, abs(brake * 80.0) + _noise(5.0)))
    accel       = -brake * 8.0 + _noise(0.3)

    urllc_lat  = max(0.1, random.gauss(5.5, 1.0) + (1.0 if emergency else 0.0))
    urllc_util = min(1.0, max(0.0, random.gauss(0.65 if emergency else 0.45, 0.08)))
    urllc_loss = max(0.0, random.gauss(0.003, 0.001))
    urllc_tput = max(0.0, random.gauss(15.0, 2.0))
    embb_lat   = max(0.1, random.gauss(20.0, 3.0))
    embb_util  = max(0.0, min(1.0, random.gauss(0.50, 0.10)))
    mmtc_util  = max(0.0, min(1.0, random.gauss(0.28, 0.07)))

    return dict(
        speed_kmh=speed, acceleration_ms2=accel,
        brake_intensity=min(1.0, max(0.0, brake)),
        battery_temp_celsius=batt_temp,
        state_of_charge_pct=max(0.0, min(100.0, soc)),
        obstacle_distance_m=max(0.0, obstacle),
        sensor_confidence=sensor_conf, motor_load_pct=motor_load,
        emergency_flag=emergency,
        urllc_latency_ms=urllc_lat,
        urllc_utilization=urllc_util, urllc_loss_rate=urllc_loss,
        urllc_throughput_mbps=urllc_tput,
        embb_latency_ms=embb_lat, embb_utilization=embb_util,
        mmtc_utilization=mmtc_util,
    )


def _gen_urllc_saturation(t: float, seq: int) -> dict:
    """
    URLLC slice near/at capacity – forces EMBB fallback for non-critical traffic.
    Critical events still need URLLC, creating label diversity.
    """
    urllc_util = min(1.0, max(0.0, random.gauss(0.93, 0.03)))
    urllc_loss = max(0.0, random.gauss(0.015, 0.005))   # higher loss under saturation
    urllc_lat  = max(0.1, random.gauss(12.0, 3.0))       # latency spikes
    urllc_tput = max(0.0, random.gauss(8.0,  1.5))       # throughput degraded
    embb_util  = min(1.0, max(0.0, random.gauss(0.75, 0.08)))
    mmtc_util  = min(1.0, max(0.0, random.gauss(0.60, 0.10)))
    embb_lat   = max(0.1, random.gauss(25.0, 5.0))

    speed       = random.uniform(30.0, 70.0) + _noise(3.0)
    accel       = random.gauss(0.0, 0.8)
    brake       = max(0.0, min(1.0, random.gauss(0.10, 0.05)))
    batt_temp   = random.uniform(40.0, 58.0) + _noise(1.5)
    soc         = random.uniform(25.0, 75.0)
    obstacle    = random.uniform(15.0, 80.0) + _noise(2.0)
    sensor_conf = min(1.0, max(0.0, random.gauss(0.85, 0.05)))
    motor_load  = random.uniform(30.0, 70.0)
    emergency   = urllc_util > 0.97 and random.random() < 0.1  # occasional critical event

    return dict(
        speed_kmh=max(0.0, speed), acceleration_ms2=accel,
        brake_intensity=brake, battery_temp_celsius=batt_temp,
        state_of_charge_pct=max(0.0, min(100.0, soc)),
        obstacle_distance_m=max(0.0, obstacle),
        sensor_confidence=sensor_conf, motor_load_pct=max(0.0, min(100.0, motor_load)),
        emergency_flag=emergency,
        urllc_latency_ms=urllc_lat, urllc_utilization=urllc_util,
        urllc_loss_rate=urllc_loss, urllc_throughput_mbps=urllc_tput,
        embb_latency_ms=embb_lat, embb_utilization=embb_util,
        mmtc_utilization=mmtc_util,
    )


def _gen_battery_overheat(t: float, seq: int) -> dict:
    """
    Battery temperature rises from 60 °C to 75 °C over the scenario window.
    Triggers URLLC escalation as BMS sends priority alerts.
    """
    # Temperature rises linearly with each sample, cycling over 400 samples
    cycle_pos = (seq % 400) / 400.0
    batt_temp = 60.0 + cycle_pos * 15.0 + _noise(0.5)   # 60→75 °C

    emergency = batt_temp > 70.0 and random.random() < 0.4
    brake     = max(0.0, min(1.0, random.gauss(0.05, 0.03)))
    speed     = max(0.0, random.uniform(20.0, 60.0) + _noise(2.0))
    accel     = random.gauss(0.0, 0.5)
    soc       = max(0.0, min(100.0, random.uniform(15.0, 60.0)))
    obstacle  = random.uniform(20.0, 100.0) + _noise(3.0)
    sensor_conf = min(1.0, max(0.0, random.gauss(0.90, 0.03)))
    motor_load  = max(0.0, min(100.0, random.uniform(40.0, 80.0) + _noise(5.0)))

    # Network remains mostly stable; BMS alert traffic spikes URLLC utilisation
    urllc_util = min(1.0, max(0.0, 0.50 + (batt_temp - 60.0) / 15.0 * 0.35 + _noise(0.05)))
    urllc_lat  = max(0.1, random.gauss(6.0, 1.0))
    urllc_loss = max(0.0, random.gauss(0.005, 0.002))
    urllc_tput = max(0.0, random.gauss(14.0, 2.0))
    embb_lat   = max(0.1, random.gauss(20.0, 3.0))
    embb_util  = max(0.0, min(1.0, random.gauss(0.50, 0.08)))
    mmtc_util  = max(0.0, min(1.0, random.gauss(0.35, 0.07)))

    return dict(
        speed_kmh=speed, acceleration_ms2=accel, brake_intensity=brake,
        battery_temp_celsius=batt_temp,
        state_of_charge_pct=max(0.0, min(100.0, soc)),
        obstacle_distance_m=max(0.0, obstacle),
        sensor_confidence=sensor_conf, motor_load_pct=motor_load,
        emergency_flag=emergency,
        urllc_latency_ms=urllc_lat, urllc_utilization=urllc_util,
        urllc_loss_rate=urllc_loss, urllc_throughput_mbps=urllc_tput,
        embb_latency_ms=embb_lat, embb_utilization=embb_util,
        mmtc_utilization=mmtc_util,
    )


def _gen_anomaly_attack(t: float, seq: int) -> dict:
    """
    Erratic sensor readings consistent with a sensor spoofing / cyber-attack.
    Sensor confidence degrades, readings become inconsistent, URLLC escalation.
    """
    # Sensor confidence degrades sporadically
    sensor_conf = max(0.0, min(1.0, random.gauss(0.25, 0.15)))

    # Speed is erratic (spoofed GPS/wheel sensor)
    speed = max(0.0, random.gauss(55.0, 20.0) + random.choice([-1, 1]) * random.gauss(15.0, 5.0))
    accel = random.gauss(0.0, 3.0)   # large variance = erratic
    brake = max(0.0, min(1.0, random.gauss(0.20, 0.15)))

    batt_temp  = random.uniform(38.0, 65.0) + _noise(3.0)   # noisy temp
    soc        = max(0.0, min(100.0, random.uniform(10.0, 90.0)))
    obstacle   = max(0.0, random.gauss(30.0, 25.0))          # erratic distance
    motor_load = max(0.0, min(100.0, random.gauss(50.0, 20.0)))

    # Emergency flag may be spuriously asserted
    emergency = sensor_conf < 0.20 and random.random() < 0.35

    # Network metrics also show anomalies (possible DDoS against slice mgr)
    urllc_lat  = max(0.1, random.gauss(8.0, 5.0) + random.choice([0.0, 30.0]) * random.random())
    urllc_util = min(1.0, max(0.0, random.gauss(0.70, 0.15)))
    urllc_loss = max(0.0, min(0.3, random.gauss(0.02, 0.01)))
    urllc_tput = max(0.0, random.gauss(10.0, 5.0))
    embb_lat   = max(0.1, random.gauss(25.0, 8.0))
    embb_util  = max(0.0, min(1.0, random.gauss(0.60, 0.15)))
    mmtc_util  = max(0.0, min(1.0, random.gauss(0.45, 0.12)))

    return dict(
        speed_kmh=speed, acceleration_ms2=accel, brake_intensity=brake,
        battery_temp_celsius=batt_temp,
        state_of_charge_pct=max(0.0, min(100.0, soc)),
        obstacle_distance_m=max(0.0, obstacle),
        sensor_confidence=sensor_conf, motor_load_pct=motor_load,
        emergency_flag=emergency,
        urllc_latency_ms=urllc_lat, urllc_utilization=urllc_util,
        urllc_loss_rate=urllc_loss, urllc_throughput_mbps=urllc_tput,
        embb_latency_ms=embb_lat, embb_utilization=embb_util,
        mmtc_utilization=mmtc_util,
    )


def generate_sample(scenario: str, t: float, ev_id: str, seq: int) -> dict:
    """
    Generate a single synthetic telemetry + network + label row.

    Parameters
    ----------
    scenario:
        One of the five VESPER scenarios.
    t:
        Simulation time (Unix-epoch seconds).
    ev_id:
        Vehicle identifier string.
    seq:
        Monotonically-increasing per-vehicle sequence counter.

    Returns
    -------
    dict with all CSV column fields.
    """
    if scenario == "normal":
        fields = _gen_normal(t)
    elif scenario == "obstacle_emergency":
        fields = _gen_obstacle_emergency(t, seq)
    elif scenario == "urllc_saturation":
        fields = _gen_urllc_saturation(t, seq)
    elif scenario == "battery_overheat":
        fields = _gen_battery_overheat(t, seq)
    elif scenario == "anomaly_attack":
        fields = _gen_anomaly_attack(t, seq)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    # Compute urgency score (same logic as UrgencyScorer in backend)
    urgency = _compute_urgency(
        emergency_flag=fields["emergency_flag"],
        obstacle_m=fields["obstacle_distance_m"],
        brake=fields["brake_intensity"],
        battery_temp=fields["battery_temp_celsius"],
        sensor_conf=fields["sensor_confidence"],
    )
    fields["urgency_score"] = round(urgency, 6)

    # Ground-truth labels
    slice_label    = _assign_slice_label(urgency, scenario, fields["urllc_utilization"])
    ev_state_label = _assign_ev_state(urgency, fields["emergency_flag"])

    return {
        "ev_id":                  ev_id,
        "timestamp":              round(t, 4),
        "scenario":               scenario,
        "speed_kmh":              round(fields["speed_kmh"],              4),
        "acceleration_ms2":       round(fields["acceleration_ms2"],       4),
        "brake_intensity":        round(fields["brake_intensity"],         6),
        "battery_temp_celsius":   round(fields["battery_temp_celsius"],   4),
        "state_of_charge_pct":    round(fields["state_of_charge_pct"],    4),
        "obstacle_distance_m":    round(fields["obstacle_distance_m"],    4),
        "sensor_confidence":      round(fields["sensor_confidence"],       6),
        "motor_load_pct":         round(fields["motor_load_pct"],          4),
        "emergency_flag":         int(fields["emergency_flag"]),
        "urllc_latency_ms":       round(fields["urllc_latency_ms"],       4),
        "urllc_utilization":      round(fields["urllc_utilization"],      6),
        "urllc_loss_rate":        round(fields["urllc_loss_rate"],         8),
        "urllc_throughput_mbps":  round(fields["urllc_throughput_mbps"],  4),
        "embb_latency_ms":        round(fields["embb_latency_ms"],        4),
        "embb_utilization":       round(fields["embb_utilization"],        6),
        "mmtc_utilization":       round(fields["mmtc_utilization"],        6),
        "urgency_score":          round(urgency,                           6),
        "slice_label":            slice_label,
        "ev_state_label":         ev_state_label,
    }


FIELDNAMES = [
    "ev_id", "timestamp", "scenario",
    "speed_kmh", "acceleration_ms2", "brake_intensity",
    "battery_temp_celsius", "state_of_charge_pct",
    "obstacle_distance_m", "sensor_confidence", "motor_load_pct",
    "emergency_flag",
    "urllc_latency_ms", "urllc_utilization", "urllc_loss_rate",
    "urllc_throughput_mbps",
    "embb_latency_ms", "embb_utilization",
    "mmtc_utilization",
    "urgency_score",
    "slice_label",    # 0=MMTC, 1=EMBB, 2=URLLC
    "ev_state_label", # 0=NORMAL … 4=RECOVERY
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic VESPER training dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Total number of rows to generate")
    parser.add_argument("--output",  default="data/raw/training_data.csv",
                        help="Output CSV path (relative to project root)")
    parser.add_argument("--seed",    type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n-evs",   type=int, default=10,
                        help="Number of distinct EV IDs to distribute samples across")
    args = parser.parse_args()

    random.seed(args.seed)

    # Resolve output path relative to project root (parent of scripts/)
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    output_path  = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Build EV IDs
    ev_ids = [f"EV-{i:03d}" for i in range(1, args.n_evs + 1)]

    # Pre-compute scenario counts proportionally
    scenario_counts: dict[str, int] = {}
    allocated = 0
    scenarios_list = list(SCENARIO_WEIGHTS.keys())
    for i, sc in enumerate(scenarios_list):
        if i == len(scenarios_list) - 1:
            scenario_counts[sc] = args.samples - allocated
        else:
            scenario_counts[sc] = int(args.samples * SCENARIO_WEIGHTS[sc])
            allocated += scenario_counts[sc]

    # Sequence counters per EV
    ev_seq: dict[str, int] = {eid: 0 for eid in ev_ids}
    # Base timestamps per EV
    base_ts = time.time() - args.samples * 0.1  # 10 Hz → 0.1 s per sample
    ev_ts: dict[str, float] = {eid: base_ts for eid in ev_ids}

    print(f"Generating {args.samples:,} samples → {output_path}")
    print(f"Scenario distribution: {scenario_counts}")

    t_start = time.perf_counter()

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()

        label_counts = {MMTC: 0, EMBB: 0, URLLC: 0}

        for sc in scenarios_list:
            n = scenario_counts[sc]
            for i in range(n):
                ev_id = random.choice(ev_ids)
                t     = ev_ts[ev_id]
                seq   = ev_seq[ev_id]

                row = generate_sample(sc, t, ev_id, seq)
                writer.writerow(row)

                label_counts[row["slice_label"]] += 1
                ev_seq[ev_id] += 1
                ev_ts[ev_id]  += 0.1   # 10 Hz

    elapsed = time.perf_counter() - t_start

    total = args.samples
    print(f"\nDone in {elapsed:.2f}s  ({total / elapsed:,.0f} rows/s)")
    print(f"\nSlice label distribution:")
    labels = ["MMTC", "EMBB", "URLLC"]
    for idx, name in enumerate(labels):
        cnt = label_counts[idx]
        print(f"  {name:6s} ({idx}): {cnt:8,d}  ({100.0 * cnt / total:.1f}%)")
    print(f"\nOutput: {os.path.abspath(output_path)}")
    print(f"Rows: {total:,}  |  EVs: {args.n_evs}")


if __name__ == "__main__":
    main()
