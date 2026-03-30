# VESPER 6G and Satellite Integration Architecture

## Why 6G Matters for V2X

Vehicle-to-everything (V2X) communication imposes requirements that 5G can approach but not fully satisfy. Cooperative driving at highway speeds requires message latency in the sub-1 ms range for collision avoidance consensus. 5G URLLC achieves 1–2 ms in lab conditions; real-world deployments typically land at 5–10 ms due to backhaul and scheduling overhead.

6G targets sub-0.1 ms over-the-air latency by moving scheduling intelligence into the radio node itself (AI-native RAN), eliminating centralized grant cycles. For VESPER this means:

- Brake alerts propagate to surrounding vehicles before the driver's foot fully depresses the pedal.
- Intersection coordination messages synchronize stop/go decisions across 20+ vehicles in a 50 m radius at 960 kHz subcarrier spacing (8x denser than 5G NR's maximum 120 kHz numerology), giving deterministic slot boundaries.
- ISAC (Integrated Sensing and Communications) dual-use waveforms let each 6G node simultaneously serve as a radar, detecting vehicles, cyclists, and pedestrians without dedicated sensor hardware.

The AI-native RAN principle means the base station runs inference continuously. Beam prediction models track a vehicle's trajectory 300 ms ahead, pre-positioning antenna weights so the beam is already optimized when the vehicle arrives at a new angle. This eliminates the 10–30 ms beam recovery penalty that degrades 5G mmWave in fast-moving scenarios.

## NTN in 3GPP Release 17 and 18

3GPP introduced Non-Terrestrial Networks (NTN) formally in Release 17 (frozen March 2022). The key changes that make satellite viable as a 5G/6G access path:

**HARQ timing adaptation.** Standard 5G HARQ retransmission assumes round-trip times under 4 ms. LEO satellites have RTTs of 240–1400 ms depending on elevation angle. Release 17 extends HARQ process counts and timers to accommodate these delays without triggering spurious retransmissions.

**Doppler pre-compensation.** A LEO satellite at 550 km altitude moves at ~7.5 km/s relative to the ground, causing Doppler shifts up to ±150 kHz at Ka-band (28 GHz). Release 17 mandates UE-side pre-compensation, shifting the transmitted carrier frequency to cancel orbital motion so the satellite receives a nominally stationary signal.

**Timing advance.** NTN requires timing advance values up to 67,000 µs (vs. 5G NR's maximum of 2 ms), derived from the UE's GPS location and the satellite's ephemeris broadcast.

Release 18 adds service continuity between terrestrial and non-terrestrial networks — the mechanisms VESPER uses for seamless handoff between a congested urban URLLC slice and the NTN_SAT backup path.

## LEO Satellite Mechanics

**Orbital altitude.** Starlink operates at 550 km, OneWeb at 1200 km, Kuiper at 630 km. Lower altitude reduces latency but requires more satellites for global coverage (the shell is smaller in area).

**Orbital period.** At 550 km the period is ~95.5 minutes. The ISS at 408 km completes an orbit every 92.7 minutes. Period scales with altitude as T ∝ (R_Earth + h)^(3/2) via Kepler's third law.

**Pass duration.** From any ground point, a LEO satellite is above the 5° elevation mask for 8–15 minutes per pass. A vehicle driving at 100 km/h moves ~10 km during a typical pass, which is negligible relative to the 2000+ km satellite footprint diameter, so pass duration is effectively the same whether stationary or moving.

**Doppler profile.** Maximum Doppler shift occurs at AOS (satellite just rising, moving mostly toward the observer) and LOS (satellite setting, moving mostly away). Doppler is zero at the moment of peak elevation (closest approach). For a 28 GHz Ka-band link, maximum shift ≈ ±700 kHz, well within the receiver's AFC range.

**Coverage gap.** Because LEO orbits are inclined, there are periodic gaps in coverage at mid-latitudes. The VESPER orchestrator tracks AOS/LOS for all constellation members visible to the fleet, scheduling NTN_SAT slice usage only when elevation ≥ 5° and proactively switching to another constellation (e.g. OneWeb's polar orbit when Starlink coverage lapses).

## Ka-Band Link Budget

A link budget determines whether received signal power is sufficient for a target data rate. For the VESPER satellite channel at Ka-band:

```
Transmit power (EV terminal):   +33 dBm   (2 W phased array)
Antenna gain (terminal):        +35 dBi   (flat phased array)
Free-space path loss (FSPL):   -XXX dBm  (depends on slant range)
Molecular absorption (rain):     -3 dB    (clear sky; add 10 dB for heavy rain)
Noise figure (receiver):         -3 dB
```

FSPL at 550 km slant range (45° elevation) for 28 GHz:

```
distance = 550 / sin(45°) = 778 km = 778,000 m
FSPL = 20 log10(778000) + 20 log10(28e9) + 20 log10(4π / 3e8)
     = 117.8 + 189.0 − 21.5 = 185.3 dB
```

Net link budget: 33 + 35 − 185.3 − 3 = −120.3 dBm received signal, well above the −130 dBm sensitivity of a Ka-band LNB, giving ~10 dB margin for rain fade. The link budget tightens to zero margin at ~5° elevation (slant range >6300 km), which is why VESPER sets the AOS/LOS mask at exactly 5°.

## Satellite-Terrestrial Handoff Strategy

VESPER's handoff policy is governed by rules R13–R15 in the rules engine:

**R13 (NTN offload):** When URLLC utilization exceeds 90% and a satellite link is active, non-critical traffic is moved to NTN_SAT. This keeps the terrestrial URLLC slice available for safety events while using satellite bandwidth for bulk telemetry.

**R14 (geometry guard):** When satellite elevation drops below 10°, NTN_SAT is disabled regardless of terrestrial load. Below 10° the RTT exceeds 800 ms and link margin is marginal; the rule forces an immediate switch to EMBB with elevated priority to compensate.

**R15 (emergency last-resort):** If all three terrestrial slices simultaneously exceed 95% utilization during an EMERGENCY vehicle state, NTN_SAT is pressed into service as a critical path. This scenario is rare (total network failure) but ensures liveness for safety-critical messages.

The orchestrator also runs constellation diversity tracking. If the fleet has only Starlink active, a degrading link prompts a proactive handoff recommendation to OneWeb or Kuiper via the `get_handoff_recommendation()` method. The `/satellite/handoff` API endpoint executes this switch by reconfiguring the channel's constellation and clearing the current pass, allowing the next update cycle to acquire the new constellation.

## THz Frequency Characteristics

Sub-THz (92–300 GHz) and true THz (>300 GHz) bands offer multi-GHz contiguous bandwidth — enabling the 10 Gbps+ throughput class in the 6G_eMBB+ slice — but suffer from fundamental physical constraints:

**Molecular absorption.** Water vapour and oxygen molecules absorb electromagnetic energy at specific THz frequencies. The 60 GHz band has ~15 dB/km absorption (used in 5G backhaul for its security benefit of limited range). At 140 GHz the absorption is ~0.4 dB/m in humid air, limiting effective range to 100–200 m for viable link budgets.

**Free-space path loss.** FSPL scales as frequency squared. At 140 GHz the path loss at 100 m is ~116 dB vs. ~82 dB at 28 GHz. Massive MIMO beamforming with 1024 antenna elements compensates ~30 dB of this difference, but range remains fundamentally constrained.

**Rain fade.** Precipitation introduces 1–40 dB/km additional loss at 140 GHz depending on rainfall rate. This is the dominant link availability impairment for outdoor deployments, which is why VESPER's 6G nodes are modeled as urban hot-spots (covered parking areas, intersections) rather than macro cells.

**Use case fit for VESPER.** The 200 m constraint makes THz ideal for dense intersection scenarios: a single 6G node at a controlled intersection can serve all approaching vehicles within range with sub-0.1 ms V2X coordination messages and simultaneously act as an ISAC sensor, detecting pedestrians and cyclists without additional hardware.

## Semantic Communications

Traditional communications compress bits. Semantic communications compress *meaning*. In a 6G semantic framework, the encoder extracts the semantic intent of a message (e.g. "vehicle approaching intersection from north at 40 km/h") and transmits a compact symbolic representation. The receiver, holding a shared context model, reconstructs the full message from the semantic token plus its prior knowledge.

The `SemanticCompressionEngine` in VESPER simulates this with a `context_similarity` parameter:

- `context_similarity = 0.9`: The new message is 90% similar to the last one (e.g. another telemetry frame 100 ms later from the same vehicle). The engine achieves ~72% compression, sending only the delta.
- `context_similarity = 0.1`: Completely novel content (e.g. a sudden emergency alert after normal telemetry). Compression is minimal (~8% savings).

In practice, 6G standardization bodies (ITU-T, 3GPP Study Items post-Release 18) are exploring semantic layer protocols that would integrate into the 5G service data adaptation protocol (SDAP) stack. The VESPER model provides a realistic throughput-accuracy tradeoff interface that can be plugged into a real semantic codec.

## ISAC for Obstacle Detection

ISAC (Integrated Sensing and Communications) is a 6G design philosophy where the same hardware and waveform serve both the communications and radar functions simultaneously. Benefits for VESPER:

**Infrastructure sensing without extra hardware.** Each 6G intersection node automatically becomes a traffic surveillance sensor, tracking vehicle positions, speeds, and sizes (via radar cross section) without camera or LiDAR installation.

**Coherent sensing.** Because the 6G node controls the transmit waveform precisely, it can apply sophisticated OFDM-based sensing: each subcarrier carries both data and contributes to a range-Doppler measurement. Range resolution ≈ c / (2 × bandwidth) = 3×10⁸ / (2 × 10⁹) = 15 cm at 1 GHz bandwidth, improving to 1.5 cm at 10 GHz bandwidth.

**Privacy-preserving.** Radar returns yield position and velocity but not identity, unlike camera-based sensing. The `ISACMeasurement` model in VESPER captures `target_distance_m`, `target_velocity_ms`, and `radar_cross_section_dbsm` without any personally identifiable information.

VESPER's `isac_obstacle_confidence` feature feeds directly into the slice classifier: when ISAC detects an obstacle with high confidence, the urgency score rises and the model is biased toward URLLC assignment for the nearest vehicles, pre-emptively routing safety alerts before the on-vehicle sensors even trigger.

## How VESPER Uses Satellite as a Backup Path

The satellite integration follows a tiered fallback architecture:

```
Primary:   6G_URLLC (< 200 m from node, V2X events)
          or 6G_eMBB+ (< 200 m from node, holographic/XR)
          or URLLC (terrestrial 5G, safety events)
          or EMBB (terrestrial 5G, bulk telemetry)
          or MMTC (terrestrial 5G, low-rate IoT sensor data)

Backup:    NTN_SAT (satellite, elevation > 5°, for offload or emergency)
```

The `SatelliteOrchestrator.run_update_loop()` runs as an asyncio background task, continuously updating `active_links` and emitting handoff events. The network slice manager reads `satellite_link_active` and `satellite_elevation_deg` from the fleet state and passes them into the rules engine's `slice_metrics` dict, enabling rules R13–R15 to fire.

Latency is adjusted dynamically via `update_satellite_link(ev_id, link_quality)` in the `NetworkConditionSimulator`: as the satellite rises toward zenith the NTN_SAT slice's `base_latency_ms` drops from 600 ms toward 240 ms, and the M/M/1 queueing model reflects this improved geometry in real time.

## Digital Twin in 6G

A digital twin is a live virtual replica of a physical entity updated in near-real-time. In 6G, the RAN node maintains a digital twin of every connected UE, enabling:

**Predictive beam management.** By extrapolating the twin's trajectory model, the AI RAN controller (RIC) positions the next beam 200–500 ms ahead. This eliminates the "beam failure" events that are the primary cause of throughput drops for fast-moving 5G mmWave users.

**Network-level simulation.** The orchestrator runs a fleet digital twin, simulating future slice load to pre-allocate capacity. VESPER's `update_digital_twin(ev_id, telemetry)` method ingests each telemetry frame and returns the sync latency — the time gap between when the telemetry was generated and when it reached the node. A sync latency under 5 ms indicates the twin is tracking reality faithfully.

**Fault prediction.** The twin captures battery temperature trends, sensor confidence degradation, and drive cycle patterns. Anomaly detection on twin state can predict battery thermal runaway events 2–5 minutes before on-vehicle sensors trigger an alert, allowing preemptive slice reservation.

## Energy Efficiency in 6G

5G base stations consume 100–2000 W each. The global mobile network accounts for ~2% of world electricity consumption. 6G targets a 10–100x improvement in energy efficiency (bits per joule) through:

**AI-driven sleep modes.** When no UEs are within range, a 6G THz node powers down its RF chains completely. The 200 m range limit means most nodes idle most of the time in typical deployments — a feature that becomes a power-saving advantage.

**Semantic compression.** Transmitting 72% fewer bits for a given application throughput directly reduces transmit power and air time, improving the bits/joule metric by the same factor.

**Massive MIMO coherent gain.** Beamforming concentrates energy where the UE is, avoiding isotropic radiation. The 30 dB array gain means 1000x less power needed to achieve the same received SNR compared to an omnidirectional antenna — or equivalently, the same power covers 30x the distance.

VESPER's `compute_network_energy_efficiency()` returns a `Mbps/W` figure. The `/sixg/energy` API endpoint aggregates this across all nodes, giving an operator view of the network's carbon footprint during a simulation run.

## Interview Talking Points

**What is NTN and why does it matter for EVs?**
NTN stands for Non-Terrestrial Network — 3GPP's framework (Release 17+) for integrating satellite links into the 5G core. For EVs, it provides a resilient backup path when terrestrial base stations are congested or unavailable (e.g. highway dead zones, disaster scenarios). VESPER uses it as a last-resort path for emergency vehicles when all three terrestrial slices exceed 95% utilization.

**How does VESPER decide when to use the satellite slice?**
Three deterministic rules (R13–R15) in the rules engine govern this. R13 offloads non-critical traffic when URLLC utilization exceeds 90% and a satellite link is active. R14 disables the satellite slice when elevation drops below 10° (poor geometry, high latency, low margin). R15 is the emergency override — it forces satellite use even for critical traffic when the terrestrial network has fully collapsed.

**What makes 6G different from 5G for latency?**
5G URLLC achieves ~1 ms one-way latency. 6G targets sub-0.1 ms through AI-native scheduling that eliminates the central grant cycle, plus sub-THz spectrum with subcarrier spacings of 480–960 kHz (vs. 120 kHz in 5G NR), giving slot durations of ~1 µs instead of ~8 µs. The massive MIMO array also reduces retransmission probability, so fewer retransmissions means lower effective latency.

**What is ISAC and how does VESPER use it?**
ISAC uses the same 6G waveform for both data communication and radar sensing simultaneously. In VESPER, each 6G intersection node is also a radar, detecting vehicle positions, speeds, and radar cross sections. The `isac_obstacle_confidence` feature is fed into the ML classifier — when the infrastructure radar detects a high-confidence obstacle near an EV, VESPER pre-routes the next telemetry frame to URLLC before the vehicle's own sensors have even triggered.

**How does semantic compression work and why does it matter for V2X?**
In 6G semantic communications, the transmitter encodes the *meaning* of a message rather than raw bits. Two consecutive telemetry frames that differ only slightly (context_similarity ~0.9) can be compressed by up to 80% — instead of sending 512 bytes, only ~100 bytes are transmitted. This reduces air-time, power consumption, and queue occupancy. For VESPER's V2X coordination messages, it means more vehicles can share the same 6G_URLLC slice without congestion.

**What is a digital twin in the context of 6G RAN?**
A digital twin is a live in-memory replica of a UE's state maintained at the base station. The 6G RIC uses the twin to predict beam directions, pre-allocate resources, and detect anomalies (e.g. sudden battery temperature spike). VESPER's `update_digital_twin()` tracks how fresh each vehicle's twin is — sync latency under 5 ms means the network is effectively operating on real-time state, enabling predictive rather than reactive resource management.
