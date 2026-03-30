# VESPER System Architecture

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Component Interaction Diagram](#2-component-interaction-diagram)
3. [End-to-End Data Flow](#3-end-to-end-data-flow)
4. [Network Simulation Model](#4-network-simulation-model)
5. [ML Pipeline](#5-ml-pipeline)
6. [Safety Guarantee](#6-safety-guarantee)
7. [Observability Strategy](#7-observability-strategy)

---

## 1. System Overview

VESPER (Vehicle Edge Slice Priority and Emergency Router) is a distributed real-time system that bridges two domains: 5G network slice management and EV fleet safety orchestration. At its core, VESPER continuously evaluates incoming vehicle telemetry, assigns each packet class to the appropriate 5G network slice (URLLC, eMBB, or mMTC), and provides a real-time dashboard for operators monitoring a fleet of up to 20 simulated EVs.

The system is designed around three hard constraints:

1. **Latency constraint**: Safety-critical events (collision alerts, emergency braking, V2X proximity warnings) must reach the slice scheduler within 3ms of ingestion. This drives the choice of UDP transport, the C urgency scorer, and the in-process rules engine.

2. **Safety constraint**: The ML model may not route a safety-critical event to a non-URLLC slice regardless of its confidence. The deterministic rules engine is the final arbiter.

3. **Explainability constraint**: Every slice routing decision must be explainable in near-real-time. SHAP values are computed synchronously during inference and stored alongside each decision record.

### Deployment Topology

In production-intent mode (Docker Compose), VESPER runs as five services:

- **backend** (FastAPI + asyncio, port 8000): Core orchestration engine
- **dashboard** (React + Nginx, port 3000): Real-time operator interface
- **db** (TimescaleDB/PostgreSQL, port 5432): Time-series telemetry storage
- **redis** (port 6379): Ephemeral state store for in-flight EV state and queue depths
- **prometheus** (port 9090): Metrics scraping and alerting

An optional **ml_trainer** service (profile: training) runs the XGBoost training pipeline on demand.

---

## 2. Component Interaction Diagram

```
 EV Fleet (simulated, UDP @ 10-100 Hz per vehicle)
 ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐
 │ EV-1 │ │ EV-2 │ │ EV-3 │ ... │EV-20 │
 └──┬───┘ └──┬───┘ └──┬───┘     └──┬───┘
    └─────────┴─────────┴─────────── ┘
                    │ UDP :9000
                    ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  INGESTION LAYER  (backend/ingestion/)                       │
 │                                                              │
 │  ┌─────────────────┐    ┌─────────────────┐                  │
 │  │  UDP Socket Srv │───▶│ TelemetryParser │                  │
 │  │  (asyncio)      │    │ (msgpack/JSON)  │                  │
 │  └─────────────────┘    └────────┬────────┘                  │
 │                                  │ TelemetryFrame             │
 │                                  ▼                           │
 │                         ┌────────────────┐                   │
 │                         │ EV State Machine│◀─── Redis state  │
 │                         │ NORMAL→CAUTION  │                   │
 │                         │ →CRITICAL→EMERG.│                   │
 │                         └────────┬────────┘                  │
 └──────────────────────────────────┼───────────────────────────┘
                                    │ EVStateEvent
                    ┌───────────────┼──────────────────┐
                    │               │                   │
                    ▼               ▼                   ▼
 ┌─────────────┐  ┌────────────┐  ┌──────────────────────────┐
 │  C URGENCY  │  │  ML ENGINE │  │  SAFETY RULES ENGINE     │
 │  SCORER     │  │  (XGBoost) │  │                          │
 │  .so / FFI  │  │            │  │  Rule evaluation ≤0.1ms  │
 │             │  │  Features  │  │                          │
 │  speed,     │──▶  extracted │  │  IF state == EMERGENCY   │
 │  decel,     │  │  from state│  │    → FORCE URLLC         │
 │  proximity, │  │            │  │  IF proximity < 5m       │
 │  temp...    │  │  SHAP vals │  │    AND speed > 30        │
 │             │  │  computed  │  │    → FORCE URLLC         │
 │  urgency    │  │  inline    │  │  IF battery_temp > 65°C  │
 │  score [0,1]│  │            │  │    → FORCE URLLC         │
 └──────┬──────┘  └─────┬──────┘  └──────────┬───────────────┘
        │               │                     │
        │  urgency      │  ml_slice,          │ forced_slice
        └───────────────┴──────────┬──────────┘
                                   │ SliceDecision
                                   ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  SLICE SCHEDULER  (backend/services/slice_scheduler.py)     │
 │                                                             │
 │  1. Rules engine result takes precedence                    │
 │  2. If no override: use ML prediction                       │
 │  3. Log decision + SHAP values to DB                        │
 │  4. Enqueue to appropriate slice queue                      │
 │                                                             │
 │  ┌──────────────┐ ┌──────────────┐ ┌────────────────────┐  │
 │  │ URLLC Queue  │ │ eMBB Queue   │ │ mMTC Queue         │  │
 │  │  heapq       │ │  WFQ         │ │  FIFO deque        │  │
 │  │  priority    │ │  weights     │ │                    │  │
 │  └──────┬───────┘ └──────┬───────┘ └─────────┬──────────┘  │
 └─────────┼────────────────┼───────────────────┼─────────────┘
           └────────────────┴───────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  NETWORK SLICE SIMULATOR  (backend/network/)                │
 │                                                             │
 │  M/M/1 queue per slice                                      │
 │  λ (arrival rate) updated from queue depth                  │
 │  μ (service rate) fixed per slice tier                      │
 │  W = 1/(μ - λ) + noise                                      │
 │                                                             │
 │  Outputs: simulated latency_ms, packet_loss_pct             │
 └──────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
 ┌─────────────────────┐        ┌───────────────────────────┐
 │  TimescaleDB        │        │  Prometheus (12 metrics)  │
 │  Hypertables:       │        │                           │
 │  - telemetry_frames │        │  vesper_slice_latency_ms  │
 │  - slice_decisions  │        │  vesper_ev_state          │
 │  - safety_events    │        │  vesper_ml_inference_ms   │
 │  - slice_metrics    │        │  ...                      │
 └─────────────────────┘        └──────────┬────────────────┘
                                           │
                                           ▼
 ┌──────────────────────────────────────────────────────────┐
 │  FASTAPI REST LAYER  (:8000)                             │
 │                                                          │
 │  GET /slices/status           → slice utilization        │
 │  GET /slices/:name/metrics    → per-slice latency        │
 │  GET /telemetry/fleet/summary → all EV states            │
 │  GET /decisions/recent        → last 20 decisions        │
 │  GET /alerts/history          → safety event log         │
 │  GET /ml/stats                → inference metrics        │
 │  GET /health                  → system health check      │
 └──────────────────────┬───────────────────────────────────┘
                        │ HTTP polling (500ms – 2s)
                        ▼
 ┌──────────────────────────────────────────────────────────┐
 │  REACT DASHBOARD  (:3000)                                │
 │                                                          │
 │  Panel 1: Slice utilization gauges (URLLC/eMBB/mMTC)    │
 │  Panel 2: EV fleet grid (up to 20 vehicles)              │
 │  Panel 3: Real-time latency line chart (Recharts)        │
 │  Panel 4: Live decision feed (scrolling)                 │
 │  Panel 5: Safety event timeline (30 min horizon)         │
 │  Panel 6: Confidence distribution + anomaly counter      │
 └──────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Data Flow

### Happy Path: Normal Telemetry Frame

```
T=0ms    EV-07 sends UDP frame { speed: 45, decel: 0.1, proximity: 82m, ... }
T=0.1ms  UDP socket receives datagram, asyncio callback fires
T=0.2ms  TelemetryParser decodes msgpack/JSON, validates schema
T=0.3ms  EVStateMachine updates EV-07 state (remains NORMAL)
T=0.4ms  C urgency_scorer computes score = 0.18 (low)
T=0.5ms  Feature extractor builds 12-feature vector
T=0.9ms  XGBoost predict_proba → [URLLC: 0.04, eMBB: 0.21, mMTC: 0.75]
T=1.0ms  SHAP values computed (TreeExplainer, exact)
T=1.1ms  Rules engine evaluates → no override conditions met
T=1.2ms  SliceScheduler assigns mMTC (probability 0.75)
T=1.3ms  Decision logged to TimescaleDB (async, non-blocking)
T=1.4ms  Prometheus counter incremented
T=1.4ms  Frame enqueued to mMTC FIFO queue
T=~2ms   M/M/1 simulator processes frame, emits latency_ms = 47ms
```

### Emergency Path: Collision Alert

```
T=0ms    EV-07 emergency brakes: { speed: 62, decel: 0.91, proximity: 3.2m, state: EMERGENCY }
T=0.1ms  UDP socket receives datagram (priority: high in asyncio queue)
T=0.2ms  TelemetryParser decodes, validates
T=0.3ms  EVStateMachine transitions NORMAL → EMERGENCY, emits alert
T=0.4ms  C urgency_scorer: score = 0.97 (saturation curve hits ceiling)
T=0.5ms  Rules engine evaluates BEFORE ML:
           Rule: state == EMERGENCY → FORCE URLLC (fires immediately)
T=0.5ms  [ML inference skipped — rules engine already decided]
T=0.6ms  SliceScheduler assigns URLLC (rule override, confidence=1.0, rule_id="EMERGENCY_OVERRIDE")
T=0.7ms  Decision logged to DB: { assigned_slice: URLLC, confidence: 1.0, rule_triggered: "EMERGENCY_OVERRIDE" }
T=0.7ms  Safety event logged to alerts table
T=0.8ms  Prometheus: vesper_rules_overrides_total{rule_id="EMERGENCY_OVERRIDE"} += 1
T=0.8ms  Frame pushed to URLLC priority queue (highest priority item)
T=~1.5ms M/M/1 simulator dequeues from URLLC → latency_ms = 2.1ms
```

Note: On the emergency path, ML inference is deliberately skipped after a rule fires to save ~0.4ms. The rules engine is evaluated first for exactly this reason.

---

## 4. Network Simulation Model

### M/M/1 Queue Approximation

Each 5G slice is modeled as an independent M/M/1 queue, a classical queueing theory model with:
- Markovian (Poisson) inter-arrival times with rate λ (packets/second)
- Markovian (exponential) service times with rate μ (packets/second)
- Single server (representing the radio access network scheduler)
- Infinite queue capacity (conservative — real 5G queues are bounded)

The mean waiting time (Little's Law):
```
W = 1 / (μ - λ)   for ρ = λ/μ < 1

Where:
  ρ  = utilization (0 → 1 as queue saturates)
  W  = mean waiting time
  Wq = mean queue wait = ρ / (μ - λ)
  Lq = mean queue length = ρ² / (1 - ρ)
```

### Slice Parameters

| Slice | μ (service rate) | Target ρ | Simulated latency at ρ=0.5 | Simulated latency at ρ=0.9 |
|---|---|---|---|---|
| URLLC | 1000 pkt/s | <0.3 | ~1.0ms | ~9.0ms (degraded) |
| eMBB | 200 pkt/s | <0.7 | ~16ms | ~45ms |
| mMTC | 50 pkt/s | <0.8 | ~50ms | ~180ms |

A small Gaussian noise term `N(0, σ²)` with σ proportional to ρ is added to each simulated latency sample to produce realistic trace variation.

### Scenario Load Injection

The `run_scenario.py` script can inject synthetic background traffic to drive ρ toward a target value, enabling stress scenarios (Scenario 3) that push slices toward saturation without requiring physical radio hardware.

### Packet Loss Model

Packet loss is modeled as a function of queue utilization:
```python
loss_pct = max(0, (utilization_pct - 85) * 0.5)  # Linear above 85% util
```
This is conservative relative to real 5G behavior but adequate for demonstrating graceful degradation in the dashboard.

---

## 5. ML Pipeline

### Feature Engineering

Each telemetry frame is transformed into a 12-dimensional feature vector before XGBoost inference:

| Feature | Type | Source | Rationale |
|---|---|---|---|
| `urgency_score` | float [0,1] | C scorer | Composite urgency — most predictive feature |
| `speed_kmh` | float | EV telemetry | Speed correlates with URLLC need |
| `deceleration_g` | float | EV telemetry | Hard braking indicator |
| `proximity_m` | float | EV LIDAR/radar | Proximity to obstacles/other vehicles |
| `battery_soc` | float [0,1] | EV BMS | Low SoC → eMBB for diagnostics |
| `battery_temp_c` | float | EV BMS | High temp → thermal event risk |
| `slice_util_urllc` | float [0,1] | Network sim | Slice availability context |
| `slice_util_embb` | float [0,1] | Network sim | Slice availability context |
| `slice_util_mmtc` | float [0,1] | Network sim | Slice availability context |
| `event_type_encoded` | int | State machine | One-hot reduced: normal/caution/critical |
| `time_since_last_critical` | float | State history | Recency of safety events |
| `hour_of_day` | int [0,23] | System time | Traffic pattern context |

### Training Data Generation

The synthetic dataset (`scripts/generate_dataset.py`) creates labeled samples by:
1. Sampling EV state distributions from realistic urban driving profiles
2. Running the deterministic rules engine to assign ground-truth labels for safety-critical cases
3. Using domain heuristics (speed × proximity × urgency thresholds) to label the boundary region
4. Adding ~10% label noise to prevent overconfidence

The result is a class-balanced dataset (approx. 15% URLLC, 35% eMBB, 50% mMTC) that reflects realistic urban fleet traffic patterns.

### XGBoost Training Configuration

```python
XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist',    # Fast histogram-based training
    n_jobs=-1,
)
```

### Evaluation Methodology

Train/validation/test split: 70/15/15 with stratified sampling to maintain class balance across splits. Evaluation metrics:
- Per-class F1 (safety emphasis: URLLC false negatives cost more than false positives)
- Confusion matrix with cost-weighted accuracy (URLLC→mMTC misclassification cost = 10×)
- Calibration curve (well-calibrated probabilities are essential for the confidence threshold in the dashboard)
- Inference latency: measured with `timeit` over 10,000 single-sample predictions

### SHAP Integration

After each XGBoost prediction, a SHAP `TreeExplainer` computes exact Shapley values for the predicted class. These are stored as a JSON array (top 5 features by absolute SHAP value) in the `slice_decisions` table and surfaced in the dashboard's safety event popup. This design choice — computing SHAP synchronously during inference — costs ~0.3ms per call but is critical for the explainability requirement. Async SHAP computation was considered and rejected because it would break the causal link between decision and explanation.

---

## 6. Safety Guarantee

### The Problem with ML-Only Systems

A probabilistic classifier assigns a probability to each class at inference time. Even a 94% accurate model will misclassify 6% of inputs. In a fleet of 20 vehicles at 100 Hz, that is 120 misclassifications per second. For most event types (diagnostic uploads, OTA metadata), this is inconsequential. But for a vehicle at 4m from a pedestrian traveling at 60 km/h, a single misclassification that routes a V2X collision alert through mMTC instead of URLLC can mean a 400ms delay — the difference between a near-miss and an impact.

This is the fundamental reason why **the rules engine sits above the ML model in the decision stack**, not below it.

### Rules Engine Design

The safety rules are defined in `backend/rules/safety_rules.yaml` as a human-readable, version-controlled, auditable list of conditions and forced slice assignments. The rules engine evaluates all applicable rules in O(R) time (R = number of rules, currently ~12) before ML inference begins on the emergency code path.

Rules are evaluated in priority order. The first matching rule fires and short-circuits further evaluation:

```yaml
# safety_rules.yaml (excerpt)
rules:
  - id: EMERGENCY_STATE
    priority: 1
    condition: "ev.state == 'EMERGENCY'"
    action: force_slice(URLLC)
    description: "Any vehicle in EMERGENCY state always gets URLLC"

  - id: COLLISION_IMMINENT
    priority: 2
    condition: "ev.proximity_m < 5.0 AND ev.speed_kmh > 20"
    action: force_slice(URLLC)
    description: "Imminent collision detection — V2X alert must be URLLC"

  - id: THERMAL_RUNAWAY
    priority: 3
    condition: "ev.battery_temp_c > 65.0"
    action: force_slice(URLLC)
    description: "Thermal runaway risk — diagnostic stream must be URLLC"

  - id: CRITICAL_STATE_RECENT
    priority: 4
    condition: "ev.state == 'CRITICAL' AND ev.time_since_last_critical < 30"
    action: force_slice(URLLC)
    description: "Recent CRITICAL event — maintain URLLC for 30 seconds"
```

### Why This Architecture is Correct for Safety-Critical Systems

This architecture is analogous to the avionics pattern where automated flight envelope protection (deterministic) is layered above the autopilot's trajectory optimizer (learned/heuristic). The envelope protection cannot be overridden by the optimizer regardless of its confidence. VESPER applies the same pattern to 5G slice routing.

From a formal verification standpoint: the rules engine is a finite state machine with a small, bounded rule set that can be exhaustively tested. The XGBoost model cannot be formally verified — its behavior on out-of-distribution inputs is undefined. The safety guarantee derives entirely from the rules engine, not the model.

The rules engine also provides the override audit trail (logged as `rule_triggered` in every decision record) that enables post-incident root cause analysis without needing to reconstruct ML model state.

---

## 7. Observability Strategy

VESPER's observability is built around three pillars: metrics, traces, and decision logs.

### Metrics (Prometheus)

All 12 custom Prometheus metrics are registered at application startup via the `metrics_collector.py` service. The FastAPI app exposes `/metrics` in Prometheus text format. Key design choices:
- Latency metrics are **histograms** (not gauges) to expose p50/p95/p99 percentiles via Prometheus `histogram_quantile()`
- Utilization is a **gauge** (instantaneous value, not a rate)
- Decision counts and rule overrides are **counters** (monotonically increasing, rate-queryable)

### Decision Logs (TimescaleDB)

Every slice routing decision is written to a `slice_decisions` hypertable partitioned by `timestamp`. The schema:
```sql
CREATE TABLE slice_decisions (
    timestamp        TIMESTAMPTZ NOT NULL,
    ev_id            TEXT NOT NULL,
    event_type       TEXT,
    urgency_score    FLOAT,
    assigned_slice   TEXT,
    confidence       FLOAT,
    rule_triggered   TEXT,          -- NULL if ML decision
    shap_top5        JSONB,         -- [{"feature": "...", "value": 0.xx}, ...]
    latency_ms       FLOAT          -- end-to-end pipeline latency
);
SELECT create_hypertable('slice_decisions', 'timestamp');
```

Hypertable chunking at 1-hour intervals allows fast range queries for the decision feed and safety timeline without table scans.

### Dashboard Polling Strategy

The React dashboard uses staggered polling intervals to avoid simultaneous API burst:
- Slice status + fleet state: 1000ms (real-time operational awareness)
- Latency chart: 1000ms (one new data point per second per line)
- Decision feed: 500ms (fastest-moving panel — new decisions arrive frequently)
- Alert history + model stats: 2000ms (slower-changing data)

All API calls use a 800ms timeout with graceful fallback to last-known-good data. This prevents the dashboard from going blank during transient backend restarts.

### Log Correlation

Every telemetry frame is assigned a `frame_id` (UUID) at ingestion time. This ID propagates through the urgency scorer, rules engine, ML inference, and decision log, enabling end-to-end trace reconstruction for any given safety event.
