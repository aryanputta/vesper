"""
VESPER – Satellite Telemetry Simulator
Simulates LEO/MEO/GEO satellite pass geometry, link budgets, Doppler shifts,
and round-trip latency for NTN (Non-Terrestrial Network) slice management.
"""

import asyncio
import math
import random
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from backend.app.networking.message_schema import SatelliteLink

logger = logging.getLogger(__name__)


class ConstellationType(Enum):
    LEO = "LEO"   # Low Earth Orbit, ~550–1200 km
    MEO = "MEO"   # Medium Earth Orbit, ~8000–20000 km
    GEO = "GEO"   # Geostationary, ~35786 km


@dataclass
class SatelliteOrbit:
    """Static orbital parameters for one constellation tier."""
    orbit_altitude_km: float
    inclination_deg: float
    period_minutes: float
    constellation: str


CONSTELLATIONS: dict[str, SatelliteOrbit] = {
    "Starlink": SatelliteOrbit(550, 53.0, 95.5, "Starlink"),
    "OneWeb": SatelliteOrbit(1200, 87.9, 109.0, "OneWeb"),
    "Kuiper": SatelliteOrbit(630, 51.9, 97.0, "Kuiper"),
}


@dataclass
class SatellitePass:
    """Describes a single overhead pass of a satellite as seen from a ground point."""
    satellite_id: str
    constellation: str
    aos_time: float        # acquisition of signal – Unix epoch seconds
    los_time: float        # loss of signal – Unix epoch seconds
    max_elevation_deg: float
    current_elevation_deg: float = 0.0


class SatelliteChannel:
    """
    Simulates the full satellite communication link for one EV.

    Models pass geometry, free-space path loss, Ka-band link budget,
    Doppler shift from orbital velocity, and round-trip latency.
    """

    KA_FREQ_GHZ = 28.0          # Ka-band downlink centre frequency
    KA_FREQ_HZ = 28.0e9
    SPEED_OF_LIGHT_MS = 3e8     # m/s
    TX_POWER_DBM = 33.0         # typical LEO terminal transmit power
    ANTENNA_GAIN_DB = 35.0      # phased-array antenna gain
    NOISE_FIGURE_DB = 3.0       # receiver noise figure
    PROCESSING_DELAY_MS = 5.0   # satellite on-board processing delay
    MIN_ELEVATION_DEG = 5.0     # AOS/LOS threshold – link unusable below 5°

    def __init__(
        self,
        ev_id: str,
        ground_lat: float,
        ground_lon: float,
        constellation: str = "Starlink",
    ) -> None:
        self.ev_id = ev_id
        self.ground_lat = ground_lat
        self.ground_lon = ground_lon
        self.constellation = constellation
        self.orbit = CONSTELLATIONS[constellation]

        self.current_pass: Optional[SatellitePass] = None
        self.link_active: bool = False
        self.link_quality: float = 0.0   # normalised 0–1

        # Schedule the first pass slightly in the future so startup isn't instant
        self.current_pass = self._schedule_next_pass()

    def compute_elevation(self, satellite_id: str, t: float) -> float:
        """
        Return instantaneous elevation angle in degrees during an active pass.

        Uses a sinusoidal arc model: elevation = max_el × sin(π × progress),
        which approximates the rise-peak-set shape of a LEO pass trajectory.
        Returns 0.0 if t is outside the AOS–LOS window.
        """
        if self.current_pass is None or self.current_pass.satellite_id != satellite_id:
            return 0.0
        p = self.current_pass
        if t < p.aos_time or t > p.los_time:
            return 0.0
        progress = (t - p.aos_time) / (p.los_time - p.aos_time)
        return p.max_elevation_deg * math.sin(math.pi * progress)

    def compute_link_budget(self, elevation_deg: float, altitude_km: float) -> float:
        """
        Compute the net Ka-band link budget in dB.

        Free-space path loss (FSPL) is calculated for the slant-range distance
        from ground terminal to satellite.  Returns a large negative value when
        the link is below the minimum elevation threshold.
        """
        if elevation_deg < self.MIN_ELEVATION_DEG:
            return -999.0

        elevation_rad = math.radians(elevation_deg)
        slant_range_km = altitude_km / math.sin(elevation_rad)
        slant_range_m = slant_range_km * 1000.0

        # Friis free-space path loss: 20log10(d) + 20log10(f) + 20log10(4π/c)
        fspl_db = (
            20 * math.log10(slant_range_m)
            + 20 * math.log10(self.KA_FREQ_HZ)
            + 20 * math.log10((4 * math.pi) / self.SPEED_OF_LIGHT_MS)
        )

        link_budget = (
            self.TX_POWER_DBM
            + self.ANTENNA_GAIN_DB
            - fspl_db
            - self.NOISE_FIGURE_DB
        )
        return link_budget

    def compute_doppler(self, elevation_deg: float, orbit_velocity_km_s: float = 7.5) -> float:
        """
        Estimate Doppler frequency shift in Hz.

        The component of orbital velocity along the line of sight is
        v_los = v_orbit × cos(elevation) — maximum at low elevation (satellite
        approaching horizon) and zero at zenith.
        """
        elevation_rad = math.radians(elevation_deg)
        velocity_ms = orbit_velocity_km_s * 1000.0
        doppler_hz = (velocity_ms * math.cos(elevation_rad) / self.SPEED_OF_LIGHT_MS) * self.KA_FREQ_HZ
        return doppler_hz

    def compute_latency(self, altitude_km: float, elevation_deg: float) -> float:
        """
        Compute round-trip latency in milliseconds.

        Slant range is computed from altitude and elevation angle (with a
        floor at 5° to avoid near-zero division).  RTT = 2 × one-way + processing.
        """
        min_elev_rad = math.radians(max(elevation_deg, self.MIN_ELEVATION_DEG))
        slant_range_km = altitude_km / math.sin(min_elev_rad)
        one_way_ms = (slant_range_km / 300.0) * 1000.0   # 300 km/ms ≈ speed of light
        rtt_ms = 2.0 * one_way_ms + self.PROCESSING_DELAY_MS
        return rtt_ms

    def _schedule_next_pass(self) -> SatellitePass:
        """
        Randomly schedule the next satellite pass over this EV ground point.

        AOS is 30–300 s from now; pass duration is 8–15 minutes (realistic LEO
        visibility window); max elevation is drawn uniformly from 15°–85°.
        """
        now = time.time()
        aos = now + random.uniform(30.0, 300.0)
        duration_s = random.uniform(8.0 * 60.0, 15.0 * 60.0)
        los = aos + duration_s
        max_el = random.uniform(15.0, 85.0)
        sat_num = random.randint(1, 4000)
        satellite_id = f"{self.constellation}-{sat_num:04d}"
        return SatellitePass(
            satellite_id=satellite_id,
            constellation=self.constellation,
            aos_time=aos,
            los_time=los,
            max_elevation_deg=max_el,
            current_elevation_deg=0.0,
        )

    async def update(self, t: float) -> Optional[SatelliteLink]:
        """
        Advance simulation state to time t and return a SatelliteLink if the
        link is currently active (elevation ≥ MIN_ELEVATION_DEG).

        Schedules the next pass automatically when the current one has ended.
        """
        if self.current_pass is None or t > self.current_pass.los_time:
            self.current_pass = self._schedule_next_pass()
            self.link_active = False
            self.link_quality = 0.0
            return None

        elevation = self.compute_elevation(self.current_pass.satellite_id, t)
        self.current_pass.current_elevation_deg = elevation

        if elevation < self.MIN_ELEVATION_DEG:
            self.link_active = False
            self.link_quality = 0.0
            return None

        # Link is above threshold — compute all metrics
        alt = self.orbit.orbit_altitude_km
        link_budget = self.compute_link_budget(elevation, alt)
        doppler = self.compute_doppler(elevation)
        rtt_ms = self.compute_latency(alt, elevation)

        # Normalise link quality: map link_budget from ~-140 dB (terrible) to ~-80 dB (excellent)
        # Clamp to [0,1] so the policy layer can use it directly
        quality = max(0.0, min(1.0, (link_budget + 140.0) / 60.0))
        self.link_quality = quality
        self.link_active = True

        # Approximate received signal power from the link budget (in dBm)
        signal_dbm = self.TX_POWER_DBM + self.ANTENNA_GAIN_DB - abs(link_budget)

        beam_id = f"beam-{hash(self.current_pass.satellite_id) % 512:03d}"
        ground_station_id = f"gs-{int(abs(self.ground_lat)) % 10:02d}"

        return SatelliteLink(
            link_id=f"{self.ev_id}-{self.current_pass.satellite_id}",
            constellation=self.constellation,
            satellite_id=self.current_pass.satellite_id,
            elevation_angle_deg=round(elevation, 2),
            signal_strength_dbm=round(signal_dbm, 2),
            doppler_shift_hz=round(doppler, 1),
            round_trip_latency_ms=round(rtt_ms, 2),
            link_budget_db=round(link_budget, 2),
            beam_id=beam_id,
            handoff_pending=self.get_handoff_recommendation() is not None,
            ground_station_id=ground_station_id,
            timestamp=t,
        )

    def get_handoff_recommendation(self) -> Optional[str]:
        """
        Suggest a constellation handoff when current link quality is poor.

        Returns the name of a better constellation, or None if the current link
        is acceptable or there is no active pass.
        """
        if not self.link_active or self.link_quality >= 0.3:
            return None

        # Recommend a different constellation as backup
        alternatives = [c for c in CONSTELLATIONS if c != self.constellation]
        if alternatives:
            # Prefer the constellation at highest orbit altitude for better geometry
            best = max(alternatives, key=lambda c: CONSTELLATIONS[c].orbit_altitude_km)
            return best
        return None


class SatelliteOrchestrator:
    """
    Fleet-level satellite link manager.

    Maintains one SatelliteChannel per EV and aggregates fleet-wide
    coverage statistics, handoff events, and constellation diversity.
    """

    def __init__(self, num_evs: int) -> None:
        self.num_evs = num_evs
        # Spread vehicles slightly across a ~10 km radius area around a nominal position
        self.channels: dict[str, SatelliteChannel] = {}
        for i in range(num_evs):
            ev_id = f"EV-{i + 1:03d}"
            # Small lat/lon offsets to simulate vehicles spread across a region
            lat = 37.7749 + (i % 10) * 0.001
            lon = -122.4194 + (i // 10) * 0.001
            constellation = list(CONSTELLATIONS.keys())[i % len(CONSTELLATIONS)]
            self.channels[ev_id] = SatelliteChannel(ev_id, lat, lon, constellation)

        self.active_links: dict[str, SatelliteLink] = {}
        self.handoff_events: list[dict] = []

    async def run_update_loop(self, interval_s: float = 1.0) -> None:
        """
        Continuously update all satellite channels and emit handoff events
        when link status changes.  Runs indefinitely until cancelled.
        """
        while True:
            t = time.time()
            for ev_id, channel in self.channels.items():
                prev_active = ev_id in self.active_links
                link = await channel.update(t)

                if link is not None:
                    self.active_links[ev_id] = link
                    if not prev_active:
                        # AOS event – new link established
                        logger.info(
                            "[%s] Satellite AOS: %s el=%.1f°",
                            ev_id, link.satellite_id, link.elevation_angle_deg,
                        )
                else:
                    if prev_active:
                        # LOS event – link dropped
                        old_link = self.active_links.pop(ev_id)
                        event = {
                            "ev_id": ev_id,
                            "type": "LOS",
                            "constellation": old_link.constellation,
                            "timestamp": t,
                        }
                        self.handoff_events.append(event)
                        logger.info("[%s] Satellite LOS: %s", ev_id, old_link.satellite_id)
                    elif ev_id in self.active_links:
                        del self.active_links[ev_id]

            # Log constellation diversity every 30 iterations
            constellations_in_use = {lnk.constellation for lnk in self.active_links.values()}
            logger.debug(
                "Fleet satellite coverage: %d/%d EVs, constellations: %s",
                len(self.active_links),
                self.num_evs,
                constellations_in_use,
            )

            await asyncio.sleep(interval_s)

    def get_fleet_coverage(self) -> dict:
        """Return fleet-wide satellite coverage summary statistics."""
        evs_with_coverage = len(self.active_links)
        coverage_pct = (evs_with_coverage / self.num_evs * 100.0) if self.num_evs > 0 else 0.0

        qualities = [lnk.link_budget_db for lnk in self.active_links.values()]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0.0

        active_constellations = list({lnk.constellation for lnk in self.active_links.values()})

        return {
            "total_evs": self.num_evs,
            "evs_with_coverage": evs_with_coverage,
            "coverage_pct": round(coverage_pct, 1),
            "avg_link_quality": round(avg_quality, 2),
            "active_constellations": active_constellations,
        }

    def get_best_link_for_ev(self, ev_id: str) -> Optional[SatelliteLink]:
        """Return the current satellite link for the specified EV, if active."""
        return self.active_links.get(ev_id)
