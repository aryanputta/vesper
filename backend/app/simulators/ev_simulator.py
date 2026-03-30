"""
VESPER – EV Telemetry Simulator
================================
Async, multi-agent simulator that drives one or more virtual electric vehicles
and streams their telemetry to the VESPER backend over TCP (normal frames) and
UDP (safety-critical events).

Architecture
------------
* ``EVDrivingProfile``  – static vehicle behavioural parameters.
* ``ScenarioType``      – enum of injectable fault / stress scenarios.
* ``EVAgent``           – single-vehicle coroutine; runs at 10 Hz.
* ``EVFleet``           – launches N EVAgent tasks concurrently.

Usage (standalone)
------------------
    import asyncio
    from backend.app.simulators.ev_simulator import EVFleet, ScenarioType

    async def main():
        fleet = EVFleet(
            num_evs=4,
            scenario=ScenarioType.OBSTACLE_EMERGENCY,
            tcp_host="127.0.0.1", tcp_port=9000,
            udp_host="127.0.0.1", udp_port=9001,
        )
        await fleet.run(duration_seconds=60.0)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import socket
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from backend.app.networking.message_schema import (
    EventType,
    EVState,
    Priority,
    TelemetryMessage,
)

logger = logging.getLogger(__name__)



@dataclass
class EVDrivingProfile:
    """Parameterises the stochastic driving behaviour of a single EV agent."""

    # Speed envelope
    base_speed: float = 60.0          # km/h – mean cruise speed
    speed_variance: float = 8.0       # km/h – std-dev of speed random walk

    # Acceleration
    base_accel: float = 0.5           # m/s² – nominal longitudinal acceleration

    # Thermal model
    battery_start_temp: float = 25.0  # °C – initial pack temperature
    battery_heat_rate: float = 0.04   # °C per tick per unit motor load

    # Event injection probabilities (per tick)
    obstacle_probability: float = 0.0   # probability per tick of spawning obstacle
    event_probability: float = 0.02     # probability per tick of a random safety event



class ScenarioType(Enum):
    """High-level scenario injected into the simulation run."""

    NORMAL             = "NORMAL"
    OBSTACLE_EMERGENCY = "OBSTACLE_EMERGENCY"
    URLLC_SATURATION   = "URLLC_SATURATION"
    BATTERY_OVERHEAT   = "BATTERY_OVERHEAT"
    ANOMALY_ATTACK     = "ANOMALY_ATTACK"



_TICK_RATE_HZ    = 10                 # simulation ticks per second
_TICK_PERIOD_S   = 1.0 / _TICK_RATE_HZ

# GPS drift per tick (≈ 1–2 metres at 60 km/h)
_GPS_LAT_DRIFT   = 0.000008           # degrees latitude per tick
_GPS_LON_DRIFT   = 0.000010           # degrees longitude per tick

# Starting GPS position (San Francisco, CA – representative urban grid)
_BASE_LAT        = 37.7749
_BASE_LON        = -122.4194

# Scenario timing (seconds from agent start)
_OBSTACLE_START_T     = 10.0          # when obstacle appears in OBSTACLE_EMERGENCY
_OBSTACLE_INIT_DIST_M = 50.0          # initial obstacle distance in metres
_OBSTACLE_CLOSE_RATE  = 2.0           # metres closed per tick
_OVERHEAT_START_T     = 15.0          # when rapid heating starts in BATTERY_OVERHEAT
_OVERHEAT_RATE_MULT   = 6.0           # multiplier on battery_heat_rate



class EVAgent:
    """
    Single electric-vehicle telemetry agent.

    The agent maintains a continuous physical state (speed, temperature, GPS,
    etc.) that is evolved each tick via ``update_state()``, then serialised
    into a ``TelemetryMessage`` by ``generate_message()``, and dispatched by
    ``send_message()``.

    Transport strategy
    ------------------
    * **CRITICAL / HIGH** events  → UDP (low-latency fire-and-forget).
    * **MEDIUM / LOW** telemetry  → TCP (reliable, ordered stream).
    """

    def __init__(
        self,
        ev_id: str,
        tcp_host: str,
        tcp_port: int,
        udp_host: str,
        udp_port: int,
        profile: EVDrivingProfile,
        scenario: ScenarioType,
    ) -> None:
        self.ev_id     = ev_id
        self.tcp_host  = tcp_host
        self.tcp_port  = tcp_port
        self.udp_host  = udp_host
        self.udp_port  = udp_port
        self.profile   = profile
        self.scenario  = scenario

        self.speed: float            = profile.base_speed          # km/h
        self.acceleration: float     = 0.0                         # m/s²
        self.brake_intensity: float  = 0.0                         # [0, 1]
        self.steering: float         = 0.0                         # degrees
        self.battery_temp: float     = profile.battery_start_temp  # °C
        self.soc: float              = 100.0                       # %
        self.motor_load: float       = 0.3                         # [0, 1]
        self.lat: float              = _BASE_LAT + random.uniform(-0.002, 0.002)
        self.lon: float              = _BASE_LON + random.uniform(-0.002, 0.002)
        self.obstacle_distance: float = 999.0                      # metres
        self.sensor_confidence: float = 1.0                        # [0, 1]
        self.emergency_flag: bool     = False
        self.sequence_num: int        = 0

        self._elapsed: float         = 0.0    # seconds since run() started
        self._tcp_writer: Optional[asyncio.StreamWriter] = None
        self._udp_socket: Optional[socket.socket]        = None

        # Obstacle scenario: track whether obstacle is currently active
        self._obstacle_active: bool  = False

        # Anomaly scenario: counter for burst injection
        self._anomaly_counter: int   = 0

    async def connect(self) -> None:
        """
        Establish the TCP stream connection and prepare the UDP socket.

        TCP carries ordered, reliable telemetry.
        UDP is connectionless; we simply bind a local socket and remember the
        destination address – no handshake required.
        """
        logger.info("[%s] Connecting TCP → %s:%d", self.ev_id, self.tcp_host, self.tcp_port)
        try:
            _reader, self._tcp_writer = await asyncio.open_connection(
                self.tcp_host, self.tcp_port
            )
            logger.info("[%s] TCP connected.", self.ev_id)
        except (ConnectionRefusedError, OSError) as exc:
            logger.warning(
                "[%s] TCP connection failed (%s) – telemetry will be logged only.", self.ev_id, exc
            )
            self._tcp_writer = None

        # UDP – connectionless; just create the socket
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.setblocking(False)
        logger.info("[%s] UDP socket ready → %s:%d", self.ev_id, self.udp_host, self.udp_port)

    async def disconnect(self) -> None:
        """Cleanly close the TCP writer and UDP socket."""
        if self._tcp_writer is not None:
            try:
                self._tcp_writer.close()
                await self._tcp_writer.wait_closed()
            except Exception as exc:  # noqa: BLE001
                logger.debug("[%s] TCP close error: %s", self.ev_id, exc)
            self._tcp_writer = None
            logger.info("[%s] TCP disconnected.", self.ev_id)

        if self._udp_socket is not None:
            try:
                self._udp_socket.close()
            except OSError:
                pass
            self._udp_socket = None
            logger.info("[%s] UDP socket closed.", self.ev_id)

    async def run(self, duration_seconds: float = 60.0) -> None:
        """
        Drive the simulation at ``_TICK_RATE_HZ`` for ``duration_seconds``.

        Each tick:
        1. ``update_state()``     – advance the physical model.
        2. ``generate_message()`` – produce a TelemetryMessage snapshot.
        3. ``send_message()``     – dispatch via TCP or UDP.
        """
        await self.connect()
        logger.info("[%s] Starting run for %.1fs @ %dHz.", self.ev_id, duration_seconds, _TICK_RATE_HZ)

        tick_start = time.monotonic()

        try:
            while self._elapsed < duration_seconds:
                loop_t0 = time.monotonic()

                self.update_state()
                msg = self.generate_message()
                await self.send_message(msg)

                self._elapsed += _TICK_PERIOD_S
                self.sequence_num += 1

                # Maintain real-time pacing: sleep the remainder of the tick period
                elapsed_tick = time.monotonic() - loop_t0
                sleep_for = max(0.0, _TICK_PERIOD_S - elapsed_tick)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            logger.info("[%s] Run cancelled after %.1fs.", self.ev_id, self._elapsed)
        finally:
            await self.disconnect()
            total = time.monotonic() - tick_start
            logger.info(
                "[%s] Run complete. %d frames in %.2fs (%.1f fps).",
                self.ev_id, self.sequence_num, total,
                self.sequence_num / max(total, 1e-6),
            )

    def update_state(self) -> None:
        """
        Advance all physical-state variables by one tick (0.1 s).

        The update order matters: kinematics → thermal → GPS → sensors →
        scenario injection.
        """
        t = self._elapsed  # convenient alias

        speed_noise = random.gauss(0.0, self.profile.speed_variance * 0.1)
        target_speed = self.profile.base_speed + speed_noise
        # First-order lag (τ ≈ 2 s) so speed changes feel physical
        alpha = _TICK_PERIOD_S / 2.0
        self.speed = max(0.0, self.speed + alpha * (target_speed - self.speed))

        # Derive acceleration as the discrete derivative of speed (km/h → m/s)
        prev_speed_ms = (self.speed / 3.6)
        speed_delta_ms = (target_speed - self.speed) / 3.6
        self.acceleration = float(speed_delta_ms / _TICK_PERIOD_S)
        self.acceleration = max(-8.0, min(4.0, self.acceleration))  # physical limits

        # Brake intensity: positive when decelerating sharply
        if self.acceleration < -1.0:
            self.brake_intensity = min(1.0, abs(self.acceleration) / 8.0)
        else:
            # Gradual release
            self.brake_intensity = max(0.0, self.brake_intensity - 0.05)

        # Steering: gentle sinusoidal lane-keeping oscillation
        self.steering = 15.0 * math.sin(2 * math.pi * t / 8.0) + random.gauss(0.0, 1.5)
        self.steering = max(-45.0, min(45.0, self.steering))

        # Motor load proportional to speed and acceleration demand
        base_load = (self.speed / 120.0) * 0.6
        accel_load = max(0.0, self.acceleration / 4.0) * 0.4
        self.motor_load = min(1.0, base_load + accel_load + random.gauss(0.0, 0.02))

        heat_rate = self.profile.battery_heat_rate

        if self.scenario == ScenarioType.BATTERY_OVERHEAT and t >= _OVERHEAT_START_T:
            heat_rate *= _OVERHEAT_RATE_MULT

        self.battery_temp += heat_rate * self.motor_load
        # Passive cooling effect at idle
        ambient = 22.0
        self.battery_temp += (ambient - self.battery_temp) * 0.001

        # Approximate: 1% SoC per ~3 km travelled (60 km/h → 0.6 km per 36 ticks)
        drain = (self.speed / 3.6) * _TICK_PERIOD_S / 3000.0 * 100.0
        self.soc = max(0.0, self.soc - drain)

        heading_rad = math.radians(self.steering + 90.0)  # rough heading from steering
        self.lat += _GPS_LAT_DRIFT * math.sin(heading_rad)
        self.lon += _GPS_LON_DRIFT * math.cos(heading_rad)
        # Keep within a reasonable bounding box around start
        self.lat = max(_BASE_LAT - 0.05, min(_BASE_LAT + 0.05, self.lat))
        self.lon = max(_BASE_LON - 0.05, min(_BASE_LON + 0.05, self.lon))

        if random.random() < 0.01:
            self.sensor_confidence -= random.uniform(0.05, 0.3)
        else:
            self.sensor_confidence += 0.01  # gradual recovery
        self.sensor_confidence = max(0.0, min(1.0, self.sensor_confidence))

        if self.scenario == ScenarioType.OBSTACLE_EMERGENCY:
            if t >= _OBSTACLE_START_T:
                if not self._obstacle_active:
                    self._obstacle_active = True
                    self.obstacle_distance = _OBSTACLE_INIT_DIST_M
                    logger.info("[%s] Obstacle appeared at %.1fm", self.ev_id, self.obstacle_distance)

                # Close at _OBSTACLE_CLOSE_RATE m/tick, but not below 0
                self.obstacle_distance = max(0.0, self.obstacle_distance - _OBSTACLE_CLOSE_RATE)

                # Emergency braking response when obstacle is dangerously close
                if self.obstacle_distance < 15.0:
                    self.brake_intensity = min(1.0, self.brake_intensity + 0.15)
                    self.speed = max(0.0, self.speed - 5.0)
                    self.emergency_flag = self.obstacle_distance < 5.0
            else:
                self.obstacle_distance = 999.0

        if self.scenario == ScenarioType.ANOMALY_ATTACK:
            # Burst every ~5 s (50 ticks), lasting 3 ticks
            self._anomaly_counter += 1
            if self._anomaly_counter % 50 < 3:
                # Impossible speed / temp combo
                self.speed = random.choice([
                    -10.0,             # negative speed
                    350.0,             # impossibly fast
                    self.speed,        # or occasionally normal to fool detectors
                ])
                self.battery_temp = random.choice([
                    -60.0,             # sub-physical
                    200.0,             # catastrophic
                    self.battery_temp,
                ])
                self.sensor_confidence = random.uniform(0.0, 0.15)
                logger.debug(
                    "[%s] ANOMALY injected: speed=%.1f temp=%.1f",
                    self.ev_id, self.speed, self.battery_temp,
                )

        if (
            self.scenario != ScenarioType.OBSTACLE_EMERGENCY
            and self.obstacle_distance < 5.0
        ):
            self.emergency_flag = True
        elif self.obstacle_distance >= 10.0 and not self._obstacle_active:
            self.emergency_flag = False

    def generate_message(self) -> TelemetryMessage:
        """
        Construct a ``TelemetryMessage`` from the current physical state.

        Field-name mapping follows the existing Pydantic schema in
        ``backend.app.networking.message_schema``.
        """
        event_type, priority = self.classify_event_type()
        ev_state = self._derive_ev_state()

        msg = TelemetryMessage(
            ev_id             = self.ev_id,
            timestamp         = time.time(),
            speed_kmh         = round(max(0.0, self.speed), 2),
            acceleration_ms2  = round(self.acceleration, 3),
            brake_intensity   = round(max(0.0, min(1.0, self.brake_intensity)), 3),
            steering_angle_deg= round(self.steering, 2),
            battery_temp_celsius = round(
                # Clamp to physical range [-40, 120] so Pydantic validator passes
                max(-40.0, min(120.0, self.battery_temp)), 2
            ),
            state_of_charge_pct = round(max(0.0, min(100.0, self.soc)), 2),
            motor_load_pct    = round(max(0.0, min(100.0, self.motor_load * 100.0)), 2),
            gps_lat           = round(self.lat, 7),
            gps_lon           = round(self.lon, 7),
            obstacle_distance_m = round(max(0.0, self.obstacle_distance), 2),
            sensor_confidence = round(max(0.0, min(1.0, self.sensor_confidence)), 4),
            emergency_flag    = self.emergency_flag,
            event_type        = event_type,
            priority          = priority,
            sequence_num      = self.sequence_num,
        )
        return msg

    def _derive_ev_state(self) -> EVState:
        """Map physical state to the EVState finite state machine."""
        if self.emergency_flag or self.obstacle_distance < 5.0:
            return EVState.EMERGENCY
        if self.obstacle_distance < 20.0 or self.battery_temp > 65.0:
            return EVState.CRITICAL
        if (
            self.brake_intensity > 0.5
            or self.obstacle_distance < 40.0
            or self.sensor_confidence < 0.5
        ):
            return EVState.CAUTION
        return EVState.NORMAL

    def classify_event_type(self) -> Tuple[EventType, Priority]:
        """
        Classify the current physical state into an ``(EventType, Priority)``
        pair that determines routing and slice selection downstream.

        Decision hierarchy (first match wins):
        1. obstacle_distance < 8 m  → COLLISION_WARNING / CRITICAL
        2. obstacle_distance < 20 m → OBSTACLE_ALERT    / HIGH
        3. brake_intensity > 0.8    → BRAKE_ALERT       / HIGH
        4. battery_temp > 65 °C     → BATTERY_OVERHEAT  / CRITICAL
        5. sensor_confidence < 0.3  → SENSOR_DEGRADATION / MEDIUM
        6. otherwise                → NORMAL_TELEMETRY  / LOW or MEDIUM
        """
        if self.obstacle_distance < 8.0:
            return EventType.COLLISION_WARNING, Priority.CRITICAL

        if self.obstacle_distance < 20.0:
            return EventType.OBSTACLE_ALERT, Priority.HIGH

        if self.brake_intensity > 0.8:
            return EventType.BRAKE_ALERT, Priority.HIGH

        if self.battery_temp > 65.0:
            return EventType.BATTERY_OVERHEAT, Priority.CRITICAL

        if self.sensor_confidence < 0.3:
            return EventType.SENSOR_DEGRADATION, Priority.MEDIUM

        # Normal telemetry – use MEDIUM when speed is high, LOW otherwise
        if self.speed > 80.0:
            return EventType.NORMAL_TELEMETRY, Priority.MEDIUM
        return EventType.NORMAL_TELEMETRY, Priority.LOW

    async def send_message(self, msg: TelemetryMessage) -> None:
        """
        Serialise ``msg`` to JSON (newline-delimited) and dispatch it.

        * CRITICAL / HIGH priority → UDP fire-and-forget (latency first).
        * MEDIUM  / LOW  priority → TCP reliable stream.

        If the relevant transport is unavailable the message is logged at
        WARNING level and silently dropped (best-effort semantics match
        real-world 5G slice behaviour under saturation).
        """
        payload = (json.dumps(msg.model_dump(mode="json")) + "\n").encode("utf-8")

        if msg.priority in (Priority.CRITICAL, Priority.HIGH):
            await self._send_udp(payload, msg)
        else:
            await self._send_tcp(payload, msg)

    async def _send_udp(self, payload: bytes, msg: TelemetryMessage) -> None:
        """Send ``payload`` over UDP to the VESPER edge gateway."""
        if self._udp_socket is None:
            logger.warning("[%s] UDP socket not available; dropping %s seq=%d",
                           self.ev_id, msg.event_type.name, msg.sequence_num)
            return

        loop = asyncio.get_running_loop()
        try:
            # sendto is non-blocking after setblocking(False); wrap in executor
            # to avoid blocking the event loop on a rare OS send-buffer full.
            await loop.run_in_executor(
                None,
                lambda: self._udp_socket.sendto(payload, (self.udp_host, self.udp_port)),  # type: ignore[union-attr]
            )
            logger.debug(
                "[%s] UDP → %s:%d | %s seq=%d %d bytes",
                self.ev_id, self.udp_host, self.udp_port,
                msg.event_type.name, msg.sequence_num, len(payload),
            )
        except OSError as exc:
            logger.warning("[%s] UDP send error: %s", self.ev_id, exc)

    async def _send_tcp(self, payload: bytes, msg: TelemetryMessage) -> None:
        """Write ``payload`` to the persistent TCP stream."""
        if self._tcp_writer is None:
            logger.debug(
                "[%s] TCP unavailable; dropping %s seq=%d",
                self.ev_id, msg.event_type.name, msg.sequence_num,
            )
            return

        try:
            self._tcp_writer.write(payload)
            await self._tcp_writer.drain()
            logger.debug(
                "[%s] TCP → %s:%d | %s seq=%d %d bytes",
                self.ev_id, self.tcp_host, self.tcp_port,
                msg.event_type.name, msg.sequence_num, len(payload),
            )
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            logger.warning("[%s] TCP send error (%s); will retry on next tick.", self.ev_id, exc)
            # Attempt reconnect for next tick
            self._tcp_writer = None
            asyncio.ensure_future(self._reconnect_tcp())

    async def _reconnect_tcp(self) -> None:
        """Attempt to re-establish the TCP connection with a short back-off."""
        for attempt in range(1, 4):
            await asyncio.sleep(attempt * 0.5)
            logger.info("[%s] TCP reconnect attempt %d…", self.ev_id, attempt)
            try:
                _reader, self._tcp_writer = await asyncio.open_connection(
                    self.tcp_host, self.tcp_port
                )
                logger.info("[%s] TCP reconnected.", self.ev_id)
                return
            except (ConnectionRefusedError, OSError) as exc:
                logger.warning("[%s] TCP reconnect %d failed: %s", self.ev_id, attempt, exc)
        logger.error("[%s] TCP reconnect exhausted; continuing UDP-only.", self.ev_id)



class EVFleet:
    """
    Manages a fleet of ``EVAgent`` instances and runs them concurrently.

    Each vehicle is assigned a unique EV-ID of the form ``EV-{index:03d}``
    and its own ``EVDrivingProfile`` with slight randomisation so the fleet
    does not behave identically.
    """

    # Scenario → per-vehicle profile tweaks
    _SCENARIO_PROFILE_OVERRIDES: dict[ScenarioType, dict] = {
        ScenarioType.OBSTACLE_EMERGENCY: {
            "obstacle_probability": 0.0,  # managed by agent internally
        },
        ScenarioType.BATTERY_OVERHEAT: {
            "battery_heat_rate": 0.12,
        },
        ScenarioType.ANOMALY_ATTACK: {
            "event_probability": 0.05,
        },
        ScenarioType.URLLC_SATURATION: {
            "event_probability": 0.08,
        },
        ScenarioType.NORMAL: {},
    }

    def __init__(
        self,
        num_evs: int,
        scenario: ScenarioType,
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 9000,
        udp_host: str = "127.0.0.1",
        udp_port: int = 9001,
    ) -> None:
        if num_evs < 1:
            raise ValueError(f"num_evs must be >= 1, got {num_evs}")

        self.num_evs  = num_evs
        self.scenario = scenario
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.udp_host = udp_host
        self.udp_port = udp_port

        overrides = self._SCENARIO_PROFILE_OVERRIDES.get(scenario, {})

        self.agents: list[EVAgent] = []
        for i in range(num_evs):
            # Small per-vehicle randomisation so fleet is heterogeneous
            profile = EVDrivingProfile(
                base_speed         = random.uniform(50.0, 90.0),
                speed_variance     = random.uniform(5.0, 12.0),
                base_accel         = random.uniform(0.3, 0.8),
                battery_start_temp = random.uniform(22.0, 30.0),
                battery_heat_rate  = overrides.get("battery_heat_rate", random.uniform(0.03, 0.06)),
                obstacle_probability = overrides.get("obstacle_probability", 0.0),
                event_probability  = overrides.get("event_probability", 0.02),
            )
            agent = EVAgent(
                ev_id    = f"EV-{i + 1:03d}",
                tcp_host = tcp_host,
                tcp_port = tcp_port,
                udp_host = udp_host,
                udp_port = udp_port,
                profile  = profile,
                scenario = scenario,
            )
            self.agents.append(agent)

        logger.info(
            "EVFleet initialised: %d vehicles, scenario=%s, TCP=%s:%d, UDP=%s:%d",
            num_evs, scenario.name, tcp_host, tcp_port, udp_host, udp_port,
        )

    async def run(self, duration_seconds: float = 60.0) -> None:
        """
        Run all ``EVAgent`` coroutines concurrently for ``duration_seconds``.

        Uses ``asyncio.gather`` so that individual agent failures are caught
        and logged without aborting the whole fleet.
        """
        logger.info(
            "EVFleet starting %d agents for %.1fs.", self.num_evs, duration_seconds
        )

        tasks = [
            asyncio.create_task(
                agent.run(duration_seconds),
                name=f"agent-{agent.ev_id}",
            )
            for agent in self.agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                logger.error(
                    "Agent %s raised an exception: %s", agent.ev_id, result, exc_info=result
                )

        logger.info("EVFleet run complete.")
