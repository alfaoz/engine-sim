"""Engine configurations. All dimensions in SI (metres, etc.).

Firing/timing angles are given in cycle coordinates where 0 = firing TDC:
  4-stroke cycle spans 0..720 deg  ->  power 0..180, exhaust 180..360,
                                       intake 360..540, compression 540..720
  2-stroke cycle spans 0..360 deg  ->  power/exhaust/scavenge in one rev
"""
from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class VehicleConfig:
    name: str
    mass: float            # kg (incl. occupants)
    cd: float              # drag coefficient
    frontal_area: float    # m^2
    crr: float             # rolling resistance coefficient
    wheel_radius: float    # m
    final_drive: float
    gear_ratios: List[float] = field(default_factory=list)

    def n_gears(self):
        return len(self.gear_ratios)


@dataclass
class EngineConfig:
    name: str
    bore: float                 # m
    stroke: float               # m
    conrod: float               # m
    compression_ratio: float
    n_cylinders: int
    stroke_cycle: int = 4       # 2 or 4
    diesel: bool = False
    turbo_boost: float = 0.0    # gauge boost pressure (Pa), 0 = NA

    # combustion / valve timing in cycle-coordinate degrees
    ignition_btdc: float = 22.0     # deg before firing TDC
    burn_duration: float = 55.0     # deg
    evo: float = 145.0              # exhaust valve opens (deg, from firing TDC)
    evc: float = 375.0
    ivo: float = 355.0
    ivc: float = 585.0

    redline_rpm: float = 7000.0
    idle_rpm: float = 850.0
    inertia: float = 0.18           # kg m^2 rotating inertia (flywheel-ish)
    power_tune: float = 1.0         # combustion calibration multiplier

    # exhaust geometry (single collector+tailpipe lumped model)
    pipe_length: float = 1.4        # m, header+collector+tailpipe
    pipe_diameter: float = 0.050    # m
    runner_spread: float = 0.18     # fraction of pipe over which cyls inject

    firing_order: List[int] = field(default_factory=list)

    def displacement_cc(self) -> float:
        v = math.pi / 4 * self.bore ** 2 * self.stroke * self.n_cylinders
        return v * 1e6

    def cylinder_cc(self) -> float:
        return math.pi / 4 * self.bore ** 2 * self.stroke * 1e6


# ---------------------------------------------------------------------------
# Presets — geometry/CR taken from real engines (approximate but representative)
# ---------------------------------------------------------------------------

def _moped_50cc():
    return EngineConfig(
        name="50cc 2-stroke single (moped)",
        bore=0.0405, stroke=0.0395, conrod=0.080,
        compression_ratio=7.5, n_cylinders=1, stroke_cycle=2,
        # turbo_boost here stands in for crankcase scavenge delivery
        diesel=False, turbo_boost=28000.0,
        ignition_btdc=18.0, burn_duration=42.0,
        # 2-stroke ports (cycle 0..360): exhaust around BDC(180)
        evo=98.0, evc=262.0, ivo=118.0, ivc=242.0,
        redline_rpm=9500.0, idle_rpm=1600.0, inertia=0.012,
        pipe_length=0.85, pipe_diameter=0.028, runner_spread=0.0,
        firing_order=[1],
    )


def _two_stroke_250():
    return EngineConfig(
        name="250cc 2-stroke single (dirt bike)",
        bore=0.0665, stroke=0.072, conrod=0.125,
        compression_ratio=9.0, n_cylinders=1, stroke_cycle=2,
        diesel=False, turbo_boost=35000.0,
        ignition_btdc=20.0, burn_duration=42.0,
        evo=92.0, evc=268.0, ivo=115.0, ivc=245.0,
        redline_rpm=10500.0, idle_rpm=1700.0, inertia=0.02,
        pipe_length=1.1, pipe_diameter=0.034, runner_spread=0.0,
        firing_order=[1],
    )


def _moto_125():
    return EngineConfig(
        name="125cc 4-stroke single (bike)",
        bore=0.058, stroke=0.0478, conrod=0.095,
        compression_ratio=11.0, n_cylinders=1, stroke_cycle=4,
        ignition_btdc=24.0, burn_duration=52.0,
        redline_rpm=10500.0, idle_rpm=1500.0, inertia=0.02,
        pipe_length=1.0, pipe_diameter=0.032, runner_spread=0.0,
        firing_order=[1],
    )


def _sportbike_600():
    return EngineConfig(
        name="600cc inline-4 sportbike",
        bore=0.0670, stroke=0.0426, conrod=0.090,
        compression_ratio=12.5, n_cylinders=4, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=48.0,
        redline_rpm=15000.0, idle_rpm=1300.0, inertia=0.05,
        pipe_length=1.3, pipe_diameter=0.045, runner_spread=0.24,
        firing_order=[1, 2, 4, 3],
    )


def _thumper_single():
    return EngineConfig(
        name="650cc 4-stroke single (thumper)",
        bore=0.100, stroke=0.0827, conrod=0.150,
        compression_ratio=10.0, n_cylinders=1, stroke_cycle=4,
        ignition_btdc=26.0, burn_duration=55.0,
        redline_rpm=7000.0, idle_rpm=1200.0, inertia=0.09,
        pipe_length=1.5, pipe_diameter=0.045, runner_spread=0.0,
        firing_order=[1],
    )


def _triple_1000():
    return EngineConfig(
        name="1.0L inline-3 (modern NA)",
        bore=0.0715, stroke=0.0843, conrod=0.138,
        compression_ratio=11.5, n_cylinders=3, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=52.0,
        redline_rpm=6500.0, idle_rpm=800.0, inertia=0.14,
        pipe_length=1.6, pipe_diameter=0.045, runner_spread=0.22,
        firing_order=[1, 2, 3],
    )


def _tdi_1400():
    return EngineConfig(
        name="1.4 TDI inline-4 (turbo diesel)",
        bore=0.0795, stroke=0.0955, conrod=0.144,
        compression_ratio=19.5, n_cylinders=4, stroke_cycle=4,
        diesel=True, turbo_boost=80000.0,
        ignition_btdc=6.0, burn_duration=65.0,
        redline_rpm=5000.0, idle_rpm=850.0, inertia=0.22,
        pipe_length=1.8, pipe_diameter=0.050, runner_spread=0.20,
        firing_order=[1, 3, 4, 2],
    )


def _gti_2000():
    return EngineConfig(
        name="2.0L inline-4 (turbo petrol)",
        bore=0.0825, stroke=0.0925, conrod=0.144,
        compression_ratio=9.6, n_cylinders=4, stroke_cycle=4,
        diesel=False, turbo_boost=60000.0,
        ignition_btdc=18.0, burn_duration=50.0,
        redline_rpm=6800.0, idle_rpm=820.0, inertia=0.20,
        pipe_length=1.7, pipe_diameter=0.057, runner_spread=0.22,
        firing_order=[1, 3, 4, 2],
    )


def _v8_5000():
    return EngineConfig(
        name="5.0L V8 (NA petrol)",
        bore=0.0930, stroke=0.0927, conrod=0.155,
        compression_ratio=11.0, n_cylinders=8, stroke_cycle=4,
        ignition_btdc=24.0, burn_duration=52.0,
        redline_rpm=7200.0, idle_rpm=750.0, inertia=0.30,
        pipe_length=2.0, pipe_diameter=0.063, runner_spread=0.30,
        firing_order=[1, 5, 4, 8, 6, 3, 7, 2],  # cross-plane Coyote-ish
    )


def _na_1600():
    return EngineConfig(
        name="1.6L inline-4 (NA economy)",
        bore=0.0790, stroke=0.0815, conrod=0.139,
        compression_ratio=10.8, n_cylinders=4, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=52.0,
        redline_rpm=6500.0, idle_rpm=780.0, inertia=0.18,
        pipe_length=1.7, pipe_diameter=0.048, runner_spread=0.22,
        firing_order=[1, 3, 4, 2],
    )


def _inline5_2500():
    return EngineConfig(
        name="2.5L inline-5 (warble)",
        bore=0.0825, stroke=0.0928, conrod=0.144,
        compression_ratio=10.0, n_cylinders=5, stroke_cycle=4,
        diesel=False, turbo_boost=50000.0,
        ignition_btdc=18.0, burn_duration=50.0,
        redline_rpm=7000.0, idle_rpm=800.0, inertia=0.24,
        pipe_length=1.8, pipe_diameter=0.057, runner_spread=0.26,
        firing_order=[1, 2, 4, 5, 3],
    )


def _inline6_3000():
    return EngineConfig(
        name="3.0L inline-6 (smooth)",
        bore=0.0840, stroke=0.0900, conrod=0.145,
        compression_ratio=11.0, n_cylinders=6, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=50.0,
        redline_rpm=7000.0, idle_rpm=720.0, inertia=0.28,
        pipe_length=1.9, pipe_diameter=0.060, runner_spread=0.28,
        firing_order=[1, 5, 3, 6, 2, 4],
    )


def _v8_muscle_6200():
    return EngineConfig(
        name="6.2L V8 OHV (muscle)",
        bore=0.1034, stroke=0.0921, conrod=0.154,
        compression_ratio=10.7, n_cylinders=8, stroke_cycle=4,
        ignition_btdc=26.0, burn_duration=54.0,
        redline_rpm=6200.0, idle_rpm=700.0, inertia=0.34,
        pipe_length=2.1, pipe_diameter=0.066, runner_spread=0.30,
        firing_order=[1, 8, 7, 2, 6, 5, 4, 3],
    )


def _tdi_v6_3000():
    return EngineConfig(
        name="3.0 V6 TDI (turbo diesel)",
        bore=0.083, stroke=0.0914, conrod=0.168,
        compression_ratio=16.8, n_cylinders=6, stroke_cycle=4,
        diesel=True, turbo_boost=110000.0,
        ignition_btdc=6.0, burn_duration=62.0,
        redline_rpm=4800.0, idle_rpm=720.0, inertia=0.55,
        pipe_length=2.0, pipe_diameter=0.060, runner_spread=0.26,
        firing_order=[1, 4, 3, 6, 2, 5], power_tune=1.15,
    )


def _diesel_v8_66():
    return EngineConfig(
        name="6.6 V8 turbo diesel (pickup)",
        bore=0.103, stroke=0.099, conrod=0.160,
        compression_ratio=16.0, n_cylinders=8, stroke_cycle=4,
        diesel=True, turbo_boost=140000.0,
        ignition_btdc=5.0, burn_duration=64.0,
        redline_rpm=3600.0, idle_rpm=680.0, inertia=2.5,
        pipe_length=2.3, pipe_diameter=0.075, runner_spread=0.30,
        firing_order=[1, 8, 7, 2, 6, 5, 4, 3], power_tune=1.0,
    )


def _diesel_i6_67():
    return EngineConfig(
        name="6.7 i6 turbo diesel (truck)",
        bore=0.107, stroke=0.124, conrod=0.192,
        compression_ratio=17.3, n_cylinders=6, stroke_cycle=4,
        diesel=True, turbo_boost=160000.0,
        ignition_btdc=4.0, burn_duration=68.0,
        redline_rpm=3200.0, idle_rpm=700.0, inertia=3.5,
        pipe_length=2.5, pipe_diameter=0.085, runner_spread=0.28,
        firing_order=[1, 5, 3, 6, 2, 4], power_tune=1.0,
    )


def _diesel_i6_126():
    return EngineConfig(
        name="12.6 i6 diesel (bus/lorry)",
        bore=0.130, stroke=0.158, conrod=0.255,
        compression_ratio=17.0, n_cylinders=6, stroke_cycle=4,
        diesel=True, turbo_boost=190000.0,
        ignition_btdc=3.0, burn_duration=74.0,
        redline_rpm=2300.0, idle_rpm=550.0, inertia=9.0,
        pipe_length=3.0, pipe_diameter=0.110, runner_spread=0.28,
        firing_order=[1, 5, 3, 6, 2, 4], power_tune=1.0,
    )


PRESETS = {
    "50cc 2-stroke single": _moped_50cc,
    "250cc 2-stroke single": _two_stroke_250,
    "125cc 4-stroke single": _moto_125,
    "650cc thumper single": _thumper_single,
    "600cc inline-4 sportbike": _sportbike_600,
    "1.0L inline-3": _triple_1000,
    "1.6L inline-4 NA": _na_1600,
    "2.0L turbo-4": _gti_2000,
    "2.5L inline-5": _inline5_2500,
    "3.0L inline-6": _inline6_3000,
    "1.4 TDI diesel-4": _tdi_1400,
    "3.0 V6 TDI diesel": _tdi_v6_3000,
    "6.6 V8 diesel (pickup)": _diesel_v8_66,
    "6.7 i6 diesel (truck)": _diesel_i6_67,
    "12.6 i6 diesel (bus)": _diesel_i6_126,
    "5.0L V8": _v8_5000,
    "6.2L V8 muscle": _v8_muscle_6200,
}


def get_preset(name: str) -> EngineConfig:
    return PRESETS[name]()


# ---------------------------------------------------------------------------
# Vehicle presets — set mass / aero / gearing => road load on the engine
# ---------------------------------------------------------------------------

VEHICLES = {
    "Bench / neutral (free rev)": VehicleConfig(
        name="Bench", mass=1.0, cd=0.0, frontal_area=0.0, crr=0.0,
        wheel_radius=0.3, final_drive=1.0, gear_ratios=[]),
    "Scooter (CVT)": VehicleConfig(
        name="Scooter", mass=130.0, cd=0.7, frontal_area=0.55, crr=0.020,
        wheel_radius=0.18, final_drive=4.2, gear_ratios=[2.2]),
    "Moped": VehicleConfig(
        name="Moped", mass=95.0, cd=0.9, frontal_area=0.6, crr=0.020,
        wheel_radius=0.20, final_drive=5.2, gear_ratios=[2.8]),
    "Motorcycle (sport)": VehicleConfig(
        name="Sport bike", mass=240.0, cd=0.58, frontal_area=0.6, crr=0.018,
        # final_drive folds in the primary reduction (~1.9) + chain (~2.9)
        wheel_radius=0.30, final_drive=5.6,
        gear_ratios=[2.62, 1.95, 1.58, 1.36, 1.21, 1.10]),
    "City car": VehicleConfig(
        name="City car", mass=960.0, cd=0.32, frontal_area=2.0, crr=0.012,
        wheel_radius=0.28, final_drive=4.1,
        gear_ratios=[3.55, 1.91, 1.28, 0.95, 0.76]),
    "Hatchback": VehicleConfig(
        name="Hatchback", mass=1150.0, cd=0.31, frontal_area=2.1, crr=0.012,
        wheel_radius=0.30, final_drive=3.9,
        gear_ratios=[3.45, 1.95, 1.28, 0.97, 0.79]),
    "Hot hatch": VehicleConfig(
        name="Hot hatch", mass=1300.0, cd=0.33, frontal_area=2.15, crr=0.011,
        wheel_radius=0.32, final_drive=3.9,
        gear_ratios=[3.36, 2.09, 1.47, 1.13, 0.92, 0.76]),
    "Sedan": VehicleConfig(
        name="Sedan", mass=1450.0, cd=0.28, frontal_area=2.25, crr=0.011,
        wheel_radius=0.32, final_drive=3.7,
        gear_ratios=[3.6, 2.0, 1.35, 1.0, 0.82, 0.68]),
    "Sports car": VehicleConfig(
        name="Sports car", mass=1350.0, cd=0.32, frontal_area=2.0, crr=0.011,
        wheel_radius=0.33, final_drive=3.4,
        gear_ratios=[3.2, 2.1, 1.5, 1.15, 0.9, 0.75]),
    "Supercar": VehicleConfig(
        name="Supercar", mass=1480.0, cd=0.33, frontal_area=1.9, crr=0.011,
        wheel_radius=0.34, final_drive=3.6,
        gear_ratios=[3.13, 2.19, 1.63, 1.29, 1.03, 0.84, 0.69]),
    "SUV": VehicleConfig(
        name="SUV", mass=2100.0, cd=0.35, frontal_area=2.8, crr=0.013,
        wheel_radius=0.36, final_drive=3.9,
        gear_ratios=[3.8, 2.1, 1.4, 1.0, 0.78, 0.64]),
    "Pickup truck": VehicleConfig(
        name="Pickup", mass=2400.0, cd=0.42, frontal_area=3.2, crr=0.013,
        wheel_radius=0.38, final_drive=3.7,
        gear_ratios=[3.6, 2.2, 1.5, 1.0, 0.8, 0.67]),
    "Van": VehicleConfig(
        name="Van", mass=2000.0, cd=0.38, frontal_area=3.6, crr=0.013,
        wheel_radius=0.34, final_drive=4.0,
        gear_ratios=[3.9, 2.3, 1.5, 1.05, 0.82, 0.7]),
    "Box truck": VehicleConfig(
        name="Box truck", mass=4500.0, cd=0.50, frontal_area=5.5, crr=0.014,
        wheel_radius=0.44, final_drive=4.6,
        gear_ratios=[5.5, 3.0, 1.9, 1.3, 1.0, 0.78]),
    "Bus / lorry": VehicleConfig(
        name="Bus", mass=9000.0, cd=0.60, frontal_area=6.8, crr=0.015,
        wheel_radius=0.50, final_drive=4.8,
        gear_ratios=[6.5, 3.7, 2.4, 1.6, 1.1, 0.82]),
}


def get_vehicle(name: str) -> VehicleConfig:
    return VEHICLES[name]
