# VESPER Demo Scenarios Guide

This guide walks through all five demonstration scenarios, covering setup, what to watch in the dashboard, key metrics to highlight, and talking points for technical interviews or presentations.

---

## Scenario 1 — Baseline Normal Operation

### Description
20 EVs in normal urban driving conditions: speeds 20–60 km/h, no proximity events, battery parameters nominal. This scenario establishes the baseline system behavior and demonstrates the intelligent routing split between slices under normal load.

### Setup Steps
```bash
# Ensure system is running
docker-compose -f infra/compose/docker-compose.yml up -d

# Verify all services healthy
curl http://localhost:8000/health

# Start scenario (runs for 5 minutes by default)
python scripts/run_scenario.py \
  --scenario scenarios/scenario_1_baseline.yaml \
  --duration 300
```

### What to Watch in Dashboard

**Panel 1 — Slice Utilization**
- mMTC: ~55–65% utilization (steady, dominant slice for periodic telemetry)
- eMBB: ~25–35% (sensor uploads, diagnostics)
- URLLC: ~5–10% (near-zero; only occasional low-urgency assignments)
- All latency indicators green

**Panel 2 — EV Fleet Grid**
- All 20 cards showing NORMAL (green) state badge
- Urgency scores uniformly low (0.05–0.25 range)
- Even distribution across eMBB and mMTC slice assignments

**Panel 3 — Latency Chart**
- mMTC line: smooth 50–100ms with small natural variance
- eMBB line: 15–35ms
- URLLC line: sporadic samples near 1–3ms (rare assignments)
- All lines well below their respective SLA thresholds

**Panel 4 — Decision Feed**
- Steady stream of mMTC and eMBB assignments
- Confidence scores predominantly 0.75–0.92 range
- No rule_triggered entries
- Rows show no green highlights (below 0.85 confidence threshold is normal at baseline)

**Panel 5 — Safety Timeline**
- Sparse dots, mostly blue (info) and green (low severity)
- No red or orange dots
- Demonstrates clean baseline before introducing stress

**Panel 6 — Model Confidence Panel**
- Confidence distribution weighted toward 0.7–0.85 and 0.85–1.0 buckets
- Anomaly counter at 0 or 1
- ML inference p99 displayed — should be <1ms

### Key Metrics to Highlight
- Total slice assignments: ~200/s at 10 Hz per vehicle × 20 vehicles
- ML inference p99 < 1ms (visible in Panel 6)
- Zero rule overrides (confirm via `vesper_rules_overrides_total` in Prometheus)
- mMTC queue depth stable (not growing)

### Interview Talking Points

**"Why does normal driving use mMTC and not URLLC?"**
Periodic telemetry pings (GPS position, speed, battery SoC) don't have sub-5ms latency requirements — they're informational, not safety-critical. Routing them through URLLC wastes reserved radio resources and raises per-bit cost. VESPER's ML model learned this distribution from the training data: the urgency score for periodic telemetry is consistently low, and the slice utilization features steer the model toward mMTC. This is the fundamental value proposition — intelligent resource sharing rather than blanket URLLC assignment.

**"How does the system handle 20 vehicles at 10–100 Hz?"**
The asyncio UDP server processes frames in a non-blocking event loop. The C urgency scorer eliminates the hot-loop Python overhead. TimescaleDB writes are async (fire-and-forget to a background task queue), so they don't block the main inference pipeline. In profiling, the bottleneck at baseline load is the XGBoost inference itself at ~0.4ms mean — well within budget.

---

## Scenario 2 — Sudden Obstacle (Emergency Brake Cascade)

### Description
EV-07 detects a pedestrian at 3.2m while traveling at 62 km/h and triggers an emergency brake event. The V2X system broadcasts a collision warning to all nearby vehicles (EV-03, EV-08, EV-11, EV-15). This tests the emergency path latency, rules engine correctness, and cascade behavior across the fleet.

### Setup Steps
```bash
python scripts/run_scenario.py \
  --scenario scenarios/scenario_2_obstacle.yaml \
  --verbose

# The scenario runs the following event sequence:
# T=0s:   All EVs in NORMAL state (5s lead-in)
# T=5s:   EV-07 proximity drops to 3.2m, speed 62 km/h → EMERGENCY
# T=5.1s: V2X alert broadcast to EV-03, EV-08, EV-11, EV-15 → CRITICAL
# T=7s:   EV-07 decelerates, proximity increases → RECOVERY
# T=15s:  All EVs return to CAUTION, then NORMAL
```

### What to Watch in Dashboard

**Panel 1 — Slice Utilization (critical moment: T=5s)**
- URLLC spikes from ~8% to 85–90% within 1–2 polling intervals
- Watch for the color shift: green → yellow → red on the URLLC gauge
- eMBB drops as cascade vehicles redirect to URLLC
- The spike is narrow (seconds) — demonstrates fast response and fast recovery

**Panel 2 — EV Fleet Grid**
- EV-07: badge transitions NORMAL → EMERGENCY (red) instantly
- EV-03, 08, 11, 15: transition to CRITICAL (orange) within one polling cycle
- Urgency scores for affected vehicles spike to 0.90–0.97
- Remaining EVs stay green NORMAL

**Panel 3 — Latency Chart**
- URLLC latency may briefly spike as queue loads (watch M/M/1 model respond)
- Even under spike, URLLC stays well below eMBB baseline latency
- Recovery visible as latency returns to baseline within ~10s

**Panel 4 — Decision Feed (most dramatic panel for this scenario)**
- Row flood: 5 rapid URLLC assignments for EV-07 and cascade vehicles
- All rows highlighted green (rule override forces confidence=1.0)
- `rule_triggered` column shows "EMERGENCY_OVERRIDE" or "COLLISION_IMMINENT"
- Normal mMTC/eMBB flow resumes after ~7s
- Scroll through to show the before/after contrast

**Panel 5 — Safety Timeline**
- Red critical dot appears at T=5s (EV-07 EMERGENCY)
- Orange high-severity dots appear for cascade vehicles
- Click EV-07's red dot → SHAP popup shows:
  - `proximity_m`: large negative SHAP (very close = high urgency)
  - `speed_kmh`: large positive contribution
  - `urgency_score`: dominant feature at 0.97
  - `state_encoded`: EMERGENCY flag contribution

**Panel 6 — Model Confidence + Anomaly**
- Anomaly counter ticks up (emergency events qualify as anomalies)
- Confidence distribution briefly shifts left (rules override = confidence=1.0, but reported as rule-driven in separate counter)

### Key Metrics to Highlight
- Time from EV-07 EMERGENCY to URLLC assignment: <1ms (rules engine path)
- URLLC spike duration: ~7s (tied to scenario timeline)
- Rules override counter increments: 5 (one per affected vehicle)
- No eMBB or mMTC assignments for affected vehicles during emergency window

### Interview Talking Points

**"Why is the rules engine evaluated before ML inference on the emergency path?"**
Every millisecond counts at 62 km/h: the vehicle travels 1.7cm per millisecond. Skipping ML inference on confirmed emergency events saves ~0.4–0.8ms and eliminates the risk of the model making a wrong call on an edge case. The rules engine evaluates in ~0.08ms with zero model uncertainty. For the 5% of events that are genuine safety-critical emergencies, we want determinism, not probability.

**"What if URLLC is at 95% utilization when the emergency hits?"**
The rules engine assigns URLLC regardless. The queue priority ensures the emergency frame is at the head of the URLLC queue. In a real 5G network, the Radio Access Network scheduler would honor QoS Class Identifiers (QCI 1 for URLLC) even during congestion — our simulation mirrors this by using a heapq priority queue where emergency frames get the minimum priority key (highest priority). Latency may increase under saturation (see Scenario 3), but the frame will always get there first.

**"What does SHAP tell us here?"**
The SHAP values for the collision event show `proximity_m` and `speed_kmh` as the dominant features — which is exactly what you'd expect physically. This validates both the model's behavior and the feature engineering. For an auditor or safety engineer, being able to point to a specific decision and show exactly which sensor readings drove it is critically important for ISO 26262 compliance documentation.

---

## Scenario 3 — Slice Saturation + Graceful Degradation

### Description
The URLLC slice is artificially loaded to ρ=0.92 utilization via synthetic background traffic injection. New emergency events must still be routed correctly while the system demonstrates graceful degradation: eMBB traffic reroutes to mMTC, latency increases are visible in the chart, but safety-critical traffic is never dropped.

### Setup Steps
```bash
# Start the scenario with background traffic injection
python scripts/run_scenario.py \
  --scenario scenarios/scenario_3_saturation.yaml \
  --background-traffic-rate 950 \
  --verbose

# After 30s of saturation load, an emergency event fires:
# T=30s: EV-12 triggers EMERGENCY state under saturated URLLC
```

### What to Watch in Dashboard

**Panel 1 — Slice Utilization**
- Watch URLLC climb from baseline to ~90% over the first 20s as background traffic injects
- Gauge color: green → yellow → red
- Packet loss percentage for URLLC starts climbing above 85% utilization
- eMBB and mMTC utilization rise as some traffic attempts to reroute

**Panel 3 — Latency Chart**
- URLLC latency visibly increases as ρ → 0.9: M/M/1 mean latency W = 1/(μ-λ) diverges
- At ρ=0.92 with μ=1000: W = 1/(1000-920) = 12.5ms (well above 5ms SLA)
- Red dashed line would cross URLLC SLA threshold (if configured)
- eMBB and mMTC latency also increase as they absorb rerouted traffic

**Panel 4 — Decision Feed**
- Under saturation, ML model starts factoring in `slice_util_urllc` feature
- Some non-critical traffic shifts from eMBB to mMTC (model adapts to congestion context)
- When EV-12 emergency fires at T=30s: URLLC assignment appears DESPITE high utilization
- `rule_triggered: EMERGENCY_OVERRIDE` confirms the rules engine forced it through

**Panel 5 — Safety Timeline**
- Dense cluster of medium-severity events during saturation window (queue warnings)
- EV-12 critical dot at T=30s despite the red background

### Key Metrics to Highlight
- URLLC latency at ρ=0.92: ~12ms (M/M/1 model response)
- Rules engine still forces URLLC for EV-12 at T=30s: correctness preserved under stress
- Packet loss climbs to ~3.5% for URLLC (above 85% utilization threshold)
- `vesper_rules_overrides_total` shows override fired despite saturation

### Interview Talking Points

**"How does the ML model respond to slice saturation?"**
The slice utilization features (`slice_util_urllc/embb/mmtc`) are real-time inputs to the model — they're polled at each inference call, not computed at training time. When URLLC hits 90%+ utilization, the model has learned (from training scenarios) that this context reduces the practical benefit of URLLC routing for non-critical traffic and shifts borderline decisions toward eMBB. This is an emergent behavior from feature engineering, not a hardcoded rule — the model genuinely adapts.

**"What happens to overall system performance under saturation?"**
Non-safety-critical latency degrades gracefully. The M/M/1 model's divergence curve is non-linear — small increases in ρ beyond 0.9 cause large latency increases. This is intentional: it demonstrates the real cost of slice misuse. If everything were routed through URLLC indiscriminately, the slice would saturate and everyone's latency would spike to 12ms+. VESPER's intelligent routing keeps mMTC at 50–100ms for bulk traffic and preserves URLLC's 1–5ms for the events that actually need it.

---

## Scenario 4 — Battery Thermal Event (Progressive Urgency Escalation)

### Description
EV-12's battery temperature rises from 32°C to 78°C over 90 seconds, simulating a lithium-ion thermal runaway precursor. The urgency scorer and ML model respond continuously as temperature rises, transitioning the vehicle's diagnostic stream from mMTC through eMBB to URLLC. This scenario demonstrates the system's sensitivity to continuous signals, not just discrete state transitions.

### Setup Steps
```bash
python scripts/run_scenario.py \
  --scenario scenarios/scenario_4_battery.yaml \
  --ev EV-12 \
  --duration 120

# Temperature ramp profile:
# T=0s:   battery_temp = 32°C  → mMTC routing, urgency ~0.15
# T=30s:  battery_temp = 48°C  → eMBB routing, urgency ~0.42
# T=60s:  battery_temp = 62°C  → eMBB/URLLC boundary, urgency ~0.71
# T=75s:  battery_temp = 68°C  → Rules engine fires THERMAL_RUNAWAY → URLLC forced
# T=90s:  battery_temp = 78°C  → URLLC, state CRITICAL
```

### What to Watch in Dashboard

**Panel 2 — EV Fleet Grid (EV-12 card)**
- State badge transitions: NORMAL → CAUTION (yellow) at ~T=35s
- Urgency score bar visibly filling: watch the bar grow in real time
- State transitions to CRITICAL at T=75s (red/orange badge)
- Active slice assignment shown changing beneath the EV ID

**Panel 4 — Decision Feed**
- Filter mentally on EV-12 rows
- Early rows: mMTC, confidence ~0.82
- Mid rows: eMBB, confidence ~0.77 (lower — model is less certain at boundary)
- Late rows: URLLC, `rule_triggered: THERMAL_RUNAWAY`
- Confidence histogram shape changes across the three phases

**Panel 6 — SHAP Popup**
Click any EV-12 safety event in Panel 5. The SHAP breakdown will show `battery_temp_c` as the dominant feature contribution, followed by `urgency_score` and `battery_soc`. This is the most compelling demonstration of SHAP explainability — a single feature's rising value is directly traceable through the model's decision.

### Key Metrics to Highlight
- Smooth urgency score ramp (not discrete jumps) — demonstrates C scorer's continuous computation
- ML model adapts routing before the rules engine fires (at 62°C, model is already routing eMBB)
- Rules engine fires at exactly 65°C (THERMAL_RUNAWAY threshold) — deterministic
- `battery_temp_c` SHAP value is the dominant contributor at all three decision points

### Interview Talking Points

**"How does the urgency scorer handle a continuous variable like temperature?"**
The C urgency scorer applies a sigmoid saturation curve to battery temperature with a threshold around 50°C. Below threshold, temperature contributes minimally to urgency. Above it, the contribution increases non-linearly — a 65°C reading is not just slightly more urgent than 55°C, it is exponentially more so. This mirrors the actual thermal runaway risk profile of lithium-ion cells. The exact curve parameters are calibrated against BMS vendor specifications for the EV model in the scenario.

**"Why does the ML model shift routing before the rules engine fires?"**
This is actually desirable behavior and validates the feature engineering. The ML model learned from training data that high battery temperatures correlate with URLLC-worthy events. It starts shifting EV-12's diagnostic stream toward URLLC even before the hard threshold trips. The rules engine provides the safety floor (the hard guarantee), but the ML model provides the early warning. They're complementary, not redundant.

---

## Scenario 5 — Multi-EV Emergency + Simultaneous Rule Override

### Description
Three vehicles (EV-03, EV-09, EV-14) simultaneously enter EMERGENCY state due to a simulated multi-vehicle incident on a highway section. All three trigger rule overrides concurrently. This tests system correctness under concurrent load, demonstrates the anomaly detection counter, and shows the rules engine's behavior when multiple high-priority overrides compete for URLLC bandwidth.

### Setup Steps
```bash
python scripts/run_scenario.py \
  --scenario scenarios/scenario_5_multi_emergency.yaml \
  --verbose

# Event sequence:
# T=0s:   Baseline — all 20 EVs normal
# T=5s:   Multi-vehicle incident event injected
#          EV-03, EV-09, EV-14 → EMERGENCY simultaneously
#          EV-01, EV-07, EV-11 → CRITICAL (nearby vehicles)
# T=5-15s: Recovery phase — vehicles decelerate, RECOVERY state
# T=25s:  All vehicles return to CAUTION, then NORMAL
```

### What to Watch in Dashboard

**Panel 2 — EV Fleet Grid**
- Three red EMERGENCY badges appear simultaneously at T=5s — the visual impact is dramatic
- Three more orange CRITICAL badges for nearby vehicles
- Urgency bars for all six affected vehicles near maximum
- Remaining 14 vehicles stay NORMAL — system correctly isolates the incident

**Panel 1 — Slice Utilization**
- URLLC jumps to 70–85% instantly (6 vehicles × high-frequency URLLC frames)
- Watch the real-time utilization bar — it should peak within 1–2 dashboard polling cycles
- Latency chart (Panel 3) shows URLLC latency spike under the concurrent load

**Panel 4 — Decision Feed**
- 6 rapid URLLC rows appear with `rule_triggered` populated
- Multiple rule IDs visible: EMERGENCY_OVERRIDE and COLLISION_IMMINENT firing for different vehicles
- Confidence=1.0 for all rule-driven decisions
- Normal background traffic still flowing (mMTC/eMBB rows interspersed)
- After T=15s: RECOVERY state vehicles get URLLC for 30s post-event (CRITICAL_STATE_RECENT rule)

**Panel 6 — Model Confidence + Anomaly**
- Anomaly counter: ticks from baseline to 3+ immediately at T=5s
- Confidence distribution: watch the 0.85–1.0 bucket grow as rule-driven decisions (confidence=1.0) dominate
- ML inference p99 may briefly increase as system is under higher load
- After incident: distribution normalizes back toward baseline within ~20s

**Panel 5 — Safety Timeline**
- Cluster of red and orange dots at the T=5s mark
- Visual density of dots tells the severity story at a glance
- Clicking any dot → SHAP popup with per-vehicle breakdown

### Key Metrics to Highlight
- All 3 EMERGENCY vehicles get URLLC within one rules engine evaluation cycle (~0.1ms)
- Rules overrides counter: +3 for EMERGENCY_OVERRIDE, +3 for nearby CRITICAL vehicles
- Anomaly counter: +3–6 events depending on what qualifies as anomaly
- URLLC latency stays below 5ms SLA despite 70–85% utilization spike (heapq priority preserves ordering)
- System returns to baseline within 20s of incident resolution

### Interview Talking Points

**"What happens when 3 vehicles need URLLC simultaneously?"**
The slice scheduler doesn't need to arbitrate between them at the rules layer — all three get URLLC, because the rules engine assigns independently per vehicle and doesn't have a global "one winner" constraint. The arbitration happens at the queue level: all three frames go into the URLLC heapq priority queue, and the queue's service rate (μ = 1000 pkt/s) processes them in urgency order. At 6 concurrent high-urgency vehicles, we're adding maybe 6 frames per 10ms window to a queue that can handle 1000 — it barely registers as load.

**"How would this scale to 100 or 1000 vehicles?"**
The current architecture bottlenecks at UDP socket throughput and XGBoost inference. At 100 vehicles × 100 Hz = 10,000 frames/second, the single asyncio event loop would saturate. The natural scaling path is: (1) partition by EV ID range across multiple ingestion workers, (2) move ML inference to a process pool (CPU-bound), (3) Redis pub/sub for cross-worker state sharing. The rules engine and C scorer scale linearly since they're stateless per-frame. TimescaleDB handles 100k+ inserts/second with connection pooling.

**"What does the anomaly counter represent technically?"**
The anomaly detector runs a simple statistical threshold: if an EV's urgency score exceeds mean + 3σ of its recent history, it's flagged as anomalous. For EMERGENCY state vehicles, the urgency score always trips this (which is correct — emergency events are anomalous by definition). The counter is useful as a leading indicator in the dashboard: a spike in anomalies before any state transitions often means the urgency scorer is detecting degraded vehicle behavior that hasn't yet crossed a discrete state threshold.

**"How do you demonstrate this is production-ready vs. a demo?"**
Several design choices distinguish this from a prototype: (1) every decision is logged with full provenance (frame_id, SHAP values, rule_triggered), not just aggregated stats; (2) the rules engine is defined in an auditable YAML config, not hardcoded conditionals; (3) API errors in the dashboard fall back to last-known-good data rather than blank panels; (4) the M/M/1 queue model is parameterized against real 3GPP 5G NR slice specifications, not arbitrary numbers; (5) the C extension is benchmarked and the benchmark is reproducible. That said, it is a simulation — real deployment would need integration with an actual 5G core network's slice management API (3GPP TS 28.531).
