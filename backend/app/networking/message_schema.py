"""
VESPER – Intelligent 5G Slice-Aware EV Safety and Telemetry Control System
message_schema.py – Pydantic v2 models for all inter-component message types.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator



class SliceType(str, Enum):
    """3GPP 5G network slice categories used by VESPER."""

    URLLC = "URLLC"                   # Ultra-Reliable Low-Latency Communications
    EMBB = "EMBB"                     # Enhanced Mobile Broadband
    MMTC = "MMTC"                     # Massive Machine-Type Communications
    NTN_SAT = "NTN_SAT"               # Non-Terrestrial Network – Satellite (3GPP Rel-17/18)
    SIX_G_URLLC = "6G_URLLC"         # 6G ultra-reliable, sub-0.1 ms latency target
    SIX_G_EMBB_PLUS = "6G_eMBB+"     # 6G enhanced mobile broadband, Tbps class


class Priority(int, Enum):
    """Message priority levels.  Lower numeric value = higher urgency."""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class EVState(str, Enum):
    """Finite states of a single electric vehicle in the VESPER state machine."""

    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"
    RECOVERY = "RECOVERY"


class EventType(str, Enum):
    """Classification of the telemetry event being reported."""

    OBSTACLE_ALERT = "OBSTACLE_ALERT"
    BRAKE_ALERT = "BRAKE_ALERT"
    BATTERY_OVERHEAT = "BATTERY_OVERHEAT"
    SENSOR_DEGRADATION = "SENSOR_DEGRADATION"
    COLLISION_WARNING = "COLLISION_WARNING"
    NORMAL_TELEMETRY = "NORMAL_TELEMETRY"
    DIAGNOSTIC = "DIAGNOSTIC"
    LOG_UPLOAD = "LOG_UPLOAD"
    ANALYTICS_PING = "ANALYTICS_PING"
    SATELLITE_HANDOFF = "SATELLITE_HANDOFF"       # LEO/MEO satellite link handoff event
    V2X_COORDINATION = "V2X_COORDINATION"         # Vehicle-to-everything coordination message
    SPECTRUM_SENSING = "SPECTRUM_SENSING"         # ISAC spectrum sensing report
    HOLOGRAPHIC_SYNC = "HOLOGRAPHIC_SYNC"         # 6G holographic/XR synchronisation frame
    FEDERATED_UPDATE = "FEDERATED_UPDATE"         # Federated ML model weight update



class TelemetryMessage(BaseModel):
    """
    Primary telemetry frame emitted by every EV at up to 10 Hz.

    All physical quantities use SI units unless noted in the field description.
    """

    ev_id: str = Field(..., description="Unique vehicle identifier (e.g. 'EV-001')")
    timestamp: float = Field(
        ..., description="Unix epoch timestamp in seconds (float, sub-second precision)"
    )
    speed_kmh: float = Field(..., ge=0.0, description="Vehicle speed in km/h")
    acceleration_ms2: float = Field(
        ..., description="Longitudinal acceleration in m/s². Negative = deceleration."
    )
    brake_intensity: float = Field(
        ..., ge=0.0, le=1.0, description="Normalised brake pedal depression [0, 1]"
    )
    steering_angle_deg: float = Field(
        ..., ge=-540.0, le=540.0, description="Steering wheel angle in degrees"
    )
    battery_temp_celsius: float = Field(
        ..., description="High-voltage battery pack temperature in °C"
    )
    state_of_charge_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Battery state of charge [0, 100] %"
    )
    motor_load_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Motor load as percentage of rated power"
    )
    gps_lat: float = Field(..., ge=-90.0, le=90.0, description="WGS-84 latitude")
    gps_lon: float = Field(..., ge=-180.0, le=180.0, description="WGS-84 longitude")
    obstacle_distance_m: float = Field(
        ..., ge=0.0, description="Distance to nearest detected obstacle in metres"
    )
    sensor_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Aggregate sensor suite confidence [0, 1]"
    )
    emergency_flag: bool = Field(
        default=False,
        description="Hard emergency flag set by on-vehicle safety controller",
    )
    event_type: EventType = Field(
        default=EventType.NORMAL_TELEMETRY, description="Semantic classification of this frame"
    )
    priority: Priority = Field(
        default=Priority.MEDIUM, description="Routing priority for the 5G slice manager"
    )
    sequence_num: int = Field(
        ..., ge=0, description="Monotonically-increasing per-vehicle sequence counter"
    )

    model_config = {"use_enum_values": False}

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timestamp must be a positive Unix epoch value")
        return v

    @field_validator("battery_temp_celsius")
    @classmethod
    def battery_temp_sanity(cls, v: float) -> float:
        # Physical sanity check: batteries don't operate outside –40 °C to 120 °C
        if not (-40.0 <= v <= 120.0):
            raise ValueError(f"battery_temp_celsius {v} is outside physical range [-40, 120]")
        return v

    @property
    def urgency(self) -> float:
        """
        Scalar urgency score in [0, 1] synthesised from safety-critical fields.

        Computation priority (first max wins for critical thresholds):
        * Imminent collision (obstacle < 8 m or emergency_flag)  → 1.0
        * Obstacle proximity gradient                            → 0.55–0.85
        * Hard braking (brake_intensity)                         → up to 0.9
        * Battery overtemperature gradient                       → up to 1.0
        * Sensor degradation (1 – confidence)                    → up to 0.7
        """
        score = 0.0

        # Emergency flag is an absolute override
        if self.emergency_flag:
            return 1.0

        # Obstacle proximity (highest weight)
        if self.obstacle_distance_m < 8.0:
            score = max(score, 1.0)
        elif self.obstacle_distance_m < 20.0:
            score = max(score, 0.85)
        elif self.obstacle_distance_m < 40.0:
            score = max(score, 0.55)

        # Hard braking
        score = max(score, self.brake_intensity * 0.9)

        # Battery thermal runaway
        if self.battery_temp_celsius > 80.0:
            score = max(score, 1.0)
        elif self.battery_temp_celsius > 65.0:
            score = max(score, 0.85)
        elif self.battery_temp_celsius > 55.0:
            score = max(score, 0.65)

        # Sensor degradation
        score = max(score, (1.0 - self.sensor_confidence) * 0.7)

        return min(score, 1.0)



class NetworkMetrics(BaseModel):
    """
    Point-in-time snapshot of a single 5G network slice's KPIs, collected
    by the VESPER network monitor at ~1 Hz per slice.
    """

    slice_type: SliceType = Field(..., description="Which 5G slice these metrics describe")
    latency_ms: float = Field(..., ge=0.0, description="One-way latency in milliseconds")
    jitter_ms: float = Field(..., ge=0.0, description="Latency jitter (std-dev) in milliseconds")
    packet_loss_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of packets lost [0, 1]"
    )
    throughput_mbps: float = Field(..., ge=0.0, description="Observed throughput in Mbit/s")
    utilization_ratio: float = Field(
        ..., ge=0.0, le=1.0, description="Slice capacity utilisation [0, 1]"
    )
    queue_depth: int = Field(..., ge=0, description="Current number of packets in the slice queue")
    timestamp: float = Field(..., description="Unix epoch timestamp when metrics were sampled")

    model_config = {"use_enum_values": False}

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timestamp must be a positive Unix epoch value")
        return v



class SliceDecision(BaseModel):
    """
    Output record produced by the VESPER slice manager for every telemetry
    frame processed.  Consumed by the network controller and stored for audit.
    """

    decision_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique decision identifier (UUID v4)",
    )
    ev_id: str = Field(..., description="Vehicle this decision applies to")
    message_id: str = Field(
        ..., description="Identifier of the TelemetryMessage that triggered this decision"
    )
    assigned_slice: SliceType = Field(..., description="5G slice assigned by the manager")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Model or rule confidence in [0, 1]"
    )
    urgency_score: float = Field(
        ..., ge=0.0, le=1.0, description="Computed urgency in [0, 1] from UrgencyScorer"
    )
    ev_state: EVState = Field(..., description="EV state at the time of decision")
    rule_triggered: Optional[str] = Field(
        default=None,
        description="Human-readable label of the deterministic rule that fired (if any)",
    )
    model_prediction: SliceType = Field(
        ..., description="Raw slice predicted by the ML model (may differ from assigned_slice)"
    )
    timestamp: float = Field(..., description="Unix epoch timestamp of this decision")
    features_used: dict[str, Any] = Field(
        default_factory=dict,
        description="Feature vector snapshot used as model input",
    )
    shap_top_features: list[tuple[str, float]] = Field(
        default_factory=list,
        description="Top SHAP (feature_name, shap_value) pairs, sorted by |value| desc",
    )

    model_config = {"use_enum_values": False}

    @field_validator("shap_top_features", mode="before")
    @classmethod
    def coerce_shap_features(cls, v: Any) -> list[tuple[str, float]]:
        """Accept list-of-lists as well as list-of-tuples (e.g. from JSON)."""
        if isinstance(v, list):
            return [tuple(item) if isinstance(item, list) else item for item in v]
        return v



class AlertEvent(BaseModel):
    """
    Raised by the VESPER alert subsystem when a safety threshold is breached.
    Forwarded to operators and logged to the time-series store.
    """

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique alert identifier (UUID v4)",
    )
    ev_id: str = Field(..., description="Vehicle that triggered the alert")
    event_type: EventType = Field(..., description="Kind of event that raised this alert")
    severity: Priority = Field(..., description="Severity level, reusing Priority scale")
    description: str = Field(..., description="Human-readable explanation of the alert condition")
    timestamp: float = Field(..., description="Unix epoch timestamp when the alert was generated")
    slice_assigned: SliceType = Field(
        ..., description="5G slice used to transmit this alert"
    )
    response_latency_ms: float = Field(
        ..., ge=0.0, description="End-to-end latency from event detection to alert dispatch in ms"
    )

    model_config = {"use_enum_values": False}

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("timestamp must be a positive Unix epoch value")
        return v

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Alert description must not be empty")
        return v.strip()


class SatelliteLink(BaseModel):
    """
    Point-in-time descriptor of an active LEO/MEO/GEO satellite link for one EV.

    Populated by SatelliteChannel.update() and forwarded to the NTN slice manager
    to adjust NTN_SAT routing decisions based on real-time link quality.
    """

    link_id: str = Field(..., description="Unique link identifier, typically '<ev_id>-<satellite_id>'")
    constellation: str = Field(..., description="Operator constellation name, e.g. 'Starlink', 'OneWeb', 'Kuiper'")
    satellite_id: str = Field(..., description="Individual satellite identifier within the constellation")
    elevation_angle_deg: float = Field(..., description="Current satellite elevation above horizon in degrees")
    signal_strength_dbm: float = Field(..., description="Received signal power in dBm (typically –80 to –110 dBm)")
    doppler_shift_hz: float = Field(..., description="Carrier Doppler frequency shift in Hz caused by orbital motion")
    round_trip_latency_ms: float = Field(..., ge=0.0, description="Full round-trip propagation latency in ms")
    link_budget_db: float = Field(..., description="Net link budget (TX power + antenna gain – path loss – noise) in dB")
    beam_id: str = Field(..., description="Satellite spot-beam identifier currently serving this EV")
    handoff_pending: bool = Field(default=False, description="True when the orchestrator has scheduled a beam/satellite handoff")
    ground_station_id: str = Field(..., description="Gateway ground station routing this satellite link")
    timestamp: float = Field(..., description="Unix epoch timestamp when this link snapshot was captured")

    model_config = {"use_enum_values": False}


class SixGMetrics(BaseModel):
    """
    Key performance indicators for a single 6G RAN node as reported by the
    RAN Intelligent Controller (RIC).  Consumed by the VESPER slice manager
    to decide between SIX_G_URLLC and SIX_G_EMBB_PLUS routing.
    """

    ric_node_id: str = Field(..., description="RAN Intelligent Controller node identifier (xApp domain)")
    thz_band_active: bool = Field(..., description="True when the sub-THz (92–300 GHz) band carrier is up")
    subcarrier_spacing_khz: float = Field(
        ..., description="Numerology subcarrier spacing in kHz. 6G uses 480/960 kHz vs 5G NR's 120 kHz max."
    )
    ai_beamforming_gain_db: float = Field(
        ..., description="Achieved beamforming gain in dB from the AI-assisted massive MIMO controller"
    )
    semantic_compression_ratio: float = Field(
        ..., ge=0.0, le=1.0,
        description="Ratio of compressed-to-original payload size via semantic comms (0 = fully compressed)"
    )
    digital_twin_sync_latency_ms: float = Field(
        ..., ge=0.0,
        description="Age of the most recent digital-twin state sync in milliseconds"
    )
    network_energy_efficiency_mbps_per_watt: float = Field(
        ..., ge=0.0,
        description="Network energy efficiency: total throughput (Mbps) divided by total node power (W)"
    )
    timestamp: float = Field(..., description="Unix epoch timestamp when these metrics were sampled")

    model_config = {"use_enum_values": False}
