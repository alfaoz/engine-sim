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
    # chassis / drivetrain (for load transfer, traction split, cornering)
    drivetrain: str = "fwd"            # 'fwd' | 'rwd' | 'awd'
    weight_dist_front: float = 0.58    # static fraction of weight on the front axle
    wheelbase: float = 0.0             # m; 0 => auto-estimate from mass
    cg_height: float = 0.0             # m; 0 => auto-estimate

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
    octane: float = 95.0            # fuel RON (petrol); ignored for diesel

    # exhaust geometry (single collector+tailpipe lumped model)
    pipe_length: float = 1.4        # m, header+collector+tailpipe
    pipe_diameter: float = 0.050    # m
    runner_spread: float = 0.18     # fraction of pipe over which cyls inject
    # muffler chamber volume (m^3); 0 => auto from displacement, <0 => straight
    # pipe / open exhaust (no silencer -> raw, loud, bright)
    muffler_volume: float = 0.0
    # number of separate exhaust banks: 0 = auto (infer from the name: V/flat=2,
    # W=4, else inline=1), 1 = forced single collector, 2 = V/flat (left+right
    # heard separately -> bank beat / cross-plane rumble), 4 = W.
    exhaust_banks: int = 0

    # intake tract (runner + airbox) for the 1D induction-acoustics duct
    intake_length: float = 0.45     # m, port -> airbox mouth
    intake_diameter: float = 0.055  # m

    firing_order: List[int] = field(default_factory=list)
    # Optional explicit per-cylinder firing angle (cycle degrees, cyl 1..n).
    # When set it overrides the even-spacing implied by firing_order, so
    # uneven-firing engines (Harley 45deg V-twin 0/315, Ducati L-twin 0/270,
    # 270-crank parallel twins, cross-plane fours, ...) get their real cadence
    # -- which is where their signature beat comes from.
    firing_angles: List[float] = field(default_factory=list)

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
        diesel=True, turbo_boost=135000.0,   # ~1.35 bar (real small TDI boost)
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


# ---- additional cars / performance engines -------------------------------

def _boxer4_turbo():
    return EngineConfig(
        name="2.5L flat-4 boxer turbo (rally)",
        bore=0.0995, stroke=0.079, conrod=0.131,
        compression_ratio=8.2, n_cylinders=4, stroke_cycle=4,
        diesel=False, turbo_boost=95000.0,
        ignition_btdc=18.0, burn_duration=50.0,
        redline_rpm=6700.0, idle_rpm=820.0, inertia=0.20,
        pipe_length=1.8, pipe_diameter=0.060, runner_spread=0.30,
        firing_order=[1, 3, 2, 4],
    )


def _flat6_na():
    return EngineConfig(
        name="3.8L flat-6 (911 NA)",
        bore=0.102, stroke=0.0775, conrod=0.135,
        compression_ratio=12.5, n_cylinders=6, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=48.0,
        redline_rpm=7800.0, idle_rpm=760.0, inertia=0.24,
        pipe_length=1.7, pipe_diameter=0.060, runner_spread=0.30,
        firing_order=[1, 6, 2, 4, 3, 5],
    )


def _vtec_i4():
    return EngineConfig(
        name="2.0L inline-4 (high-rev VTEC)",
        bore=0.087, stroke=0.084, conrod=0.153,
        compression_ratio=11.7, n_cylinders=4, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=50.0,
        redline_rpm=9000.0, idle_rpm=850.0, inertia=0.16,
        pipe_length=1.7, pipe_diameter=0.052, runner_spread=0.24,
        firing_order=[1, 3, 4, 2],
    )


def _bigblock_v8():
    return EngineConfig(
        name="7.4L big-block V8 (OHV)",
        bore=0.10795, stroke=0.1016, conrod=0.165,
        compression_ratio=9.0, n_cylinders=8, stroke_cycle=4,
        ignition_btdc=28.0, burn_duration=56.0,
        redline_rpm=5800.0, idle_rpm=650.0, inertia=0.40,
        pipe_length=2.2, pipe_diameter=0.072, runner_spread=0.30,
        firing_order=[1, 8, 4, 3, 6, 5, 7, 2],
    )


def _flatplane_v8():
    return EngineConfig(
        name="5.2L flat-plane V8 (Voodoo)",
        bore=0.094, stroke=0.093, conrod=0.150,
        compression_ratio=12.0, n_cylinders=8, stroke_cycle=4,
        ignition_btdc=24.0, burn_duration=50.0,
        redline_rpm=8250.0, idle_rpm=850.0, inertia=0.26,
        pipe_length=2.0, pipe_diameter=0.064, runner_spread=0.30,
        firing_order=[1, 5, 4, 8, 3, 7, 2, 6],   # flat-plane (even 90deg)
    )


def _v10_screamer():
    return EngineConfig(
        name="4.8L V10 (LFA screamer)",
        bore=0.088, stroke=0.0797, conrod=0.137,
        compression_ratio=12.0, n_cylinders=10, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=46.0,
        redline_rpm=9000.0, idle_rpm=900.0, inertia=0.26,
        pipe_length=1.9, pipe_diameter=0.062, runner_spread=0.32,
        firing_order=[1, 10, 9, 4, 3, 6, 5, 8, 7, 2],   # 72deg even
    )


def _v12_ferrari():
    return EngineConfig(
        name="6.5L V12 (Ferrari NA)",
        bore=0.094, stroke=0.078, conrod=0.150,
        compression_ratio=13.5, n_cylinders=12, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=46.0,
        redline_rpm=8500.0, idle_rpm=900.0, inertia=0.30,
        pipe_length=2.0, pipe_diameter=0.066, runner_spread=0.34,
        firing_order=[1, 7, 5, 11, 3, 9, 6, 12, 2, 8, 4, 10],   # 60deg even
    )


# ---- additional motorcycles ----------------------------------------------

def _harley_vtwin():
    return EngineConfig(
        name="1.9L 45° V-twin (Harley)",
        bore=0.102, stroke=0.1143, conrod=0.150,
        compression_ratio=10.0, n_cylinders=2, stroke_cycle=4,
        ignition_btdc=18.0, burn_duration=58.0,
        redline_rpm=5800.0, idle_rpm=950.0, inertia=0.13,
        pipe_length=1.5, pipe_diameter=0.050, runner_spread=0.10,
        firing_order=[1, 2], firing_angles=[0.0, 315.0],   # potato-potato
    )


def _ducati_ltwin():
    return EngineConfig(
        name="1.3L 90° L-twin (Ducati)",
        bore=0.116, stroke=0.0608, conrod=0.110,
        compression_ratio=12.6, n_cylinders=2, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=52.0,
        redline_rpm=11000.0, idle_rpm=1100.0, inertia=0.05,
        pipe_length=1.3, pipe_diameter=0.048, runner_spread=0.10,
        firing_order=[1, 2], firing_angles=[0.0, 270.0],
    )


def _parallel_twin_270():
    return EngineConfig(
        name="689cc parallel-twin (270° crank)",
        bore=0.080, stroke=0.0686, conrod=0.117,
        compression_ratio=11.5, n_cylinders=2, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=52.0,
        redline_rpm=10000.0, idle_rpm=1250.0, inertia=0.035,
        pipe_length=1.2, pipe_diameter=0.044, runner_spread=0.10,
        firing_order=[1, 2], firing_angles=[0.0, 270.0],
    )


def _triumph_triple():
    return EngineConfig(
        name="765cc inline-triple (Triumph)",
        bore=0.078, stroke=0.0534, conrod=0.096,
        compression_ratio=12.65, n_cylinders=3, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=50.0,
        redline_rpm=12500.0, idle_rpm=1200.0, inertia=0.03,
        pipe_length=1.2, pipe_diameter=0.044, runner_spread=0.20,
        firing_order=[1, 2, 3],
    )


def _bmw_boxer():
    return EngineConfig(
        name="1.25L boxer-twin (BMW)",
        bore=0.1025, stroke=0.076, conrod=0.138,
        compression_ratio=12.5, n_cylinders=2, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=52.0,
        redline_rpm=9000.0, idle_rpm=1050.0, inertia=0.06,
        pipe_length=1.4, pipe_diameter=0.046, runner_spread=0.12,
        firing_order=[1, 2],   # 360deg crank -> fairly even
    )


def _ducati_v4():
    return EngineConfig(
        name="1.1L V4 (twin-pulse superbike)",
        bore=0.081, stroke=0.0535, conrod=0.096,
        compression_ratio=14.0, n_cylinders=4, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=48.0,
        redline_rpm=14000.0, idle_rpm=1300.0, inertia=0.03,
        pipe_length=1.2, pipe_diameter=0.046, runner_spread=0.18,
        firing_order=[1, 2, 3, 4],
        firing_angles=[0.0, 90.0, 290.0, 380.0],   # Desmosedici "twin pulse"
    )


def _r1_crossplane():
    return EngineConfig(
        name="998cc cross-plane inline-4 (R1)",
        bore=0.079, stroke=0.0509, conrod=0.0885,
        compression_ratio=13.0, n_cylinders=4, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=48.0,
        redline_rpm=14000.0, idle_rpm=1300.0, inertia=0.028,
        pipe_length=1.25, pipe_diameter=0.046, runner_spread=0.20,
        firing_order=[1, 2, 4, 3],
        firing_angles=[0.0, 270.0, 450.0, 540.0],   # cross-plane 270-180-90-180
    )


def _v6_2gr():
    # Toyota 2GR 3.5 V6 (60deg, even-firing): bore 94, stroke 83, CR 10.8
    return EngineConfig(
        name="3.5L V6 (NA, 2GR)",
        bore=0.094, stroke=0.083, conrod=0.147,
        compression_ratio=10.8, n_cylinders=6, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=50.0,
        redline_rpm=6600.0, idle_rpm=680.0, inertia=0.27,
        pipe_length=1.9, pipe_diameter=0.060, runner_spread=0.28,
        firing_order=[1, 4, 2, 5, 3, 6],
    )


def _v8_s63():
    # BMW S63 4.4 V8 hot-vee twin-turbo: bore 89, stroke 88.3, CR 10.1
    return EngineConfig(
        name="4.4L V8 twin-turbo (S63)",
        bore=0.089, stroke=0.0883, conrod=0.146,
        compression_ratio=10.1, n_cylinders=8, stroke_cycle=4,
        diesel=False, turbo_boost=90000.0,
        ignition_btdc=18.0, burn_duration=50.0,
        redline_rpm=7200.0, idle_rpm=700.0, inertia=0.32,
        pipe_length=2.0, pipe_diameter=0.064, runner_spread=0.30,
        firing_order=[1, 5, 4, 8, 6, 3, 7, 2],
    )


def _v10_audi():
    # Audi/Lamborghini 5.2 FSI V10: bore 84.5, stroke 92.8, CR 12.5, ~8700 rpm
    return EngineConfig(
        name="5.2L V10 (R8/Huracan)",
        bore=0.0845, stroke=0.0928, conrod=0.154,
        compression_ratio=12.5, n_cylinders=10, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=46.0,
        redline_rpm=8700.0, idle_rpm=900.0, inertia=0.27,
        pipe_length=1.95, pipe_diameter=0.064, runner_spread=0.32,
        firing_order=[1, 6, 5, 10, 2, 7, 3, 8, 4, 9],
    )


def _v6_vr38():
    # Nissan VR38DETT 3.8 V6 twin-turbo: bore 95.5, stroke 88.4, CR 9.0
    return EngineConfig(
        name="3.8L V6 twin-turbo (GT-R)",
        bore=0.0955, stroke=0.0884, conrod=0.151,
        compression_ratio=9.0, n_cylinders=6, stroke_cycle=4,
        diesel=False, turbo_boost=95000.0,
        ignition_btdc=18.0, burn_duration=50.0,
        redline_rpm=6800.0, idle_rpm=700.0, inertia=0.28,
        pipe_length=1.9, pipe_diameter=0.062, runner_spread=0.28,
        firing_order=[1, 4, 2, 5, 3, 6],
    )


def _boxer4_na():
    # Subaru-style 2.0 flat-4 NA boxer
    return EngineConfig(
        name="2.0L flat-4 boxer (NA)",
        bore=0.084, stroke=0.090, conrod=0.131,
        compression_ratio=10.5, n_cylinders=4, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=52.0,
        redline_rpm=6500.0, idle_rpm=800.0, inertia=0.18,
        pipe_length=1.75, pipe_diameter=0.052, runner_spread=0.26,
        firing_order=[1, 3, 2, 4],
    )


def _turbo4_18():
    # 1.8L inline-4 turbo (modern small turbo petrol)
    return EngineConfig(
        name="1.8L inline-4 turbo",
        bore=0.0805, stroke=0.0885, conrod=0.144,
        compression_ratio=9.6, n_cylinders=4, stroke_cycle=4,
        diesel=False, turbo_boost=70000.0,
        ignition_btdc=18.0, burn_duration=50.0,
        redline_rpm=6500.0, idle_rpm=800.0, inertia=0.18,
        pipe_length=1.7, pipe_diameter=0.052, runner_spread=0.22,
        firing_order=[1, 3, 4, 2],
    )


def _tdi_2000():
    # 2.0L common-rail TDI inline-4 diesel
    return EngineConfig(
        name="2.0 TDI inline-4 (diesel)",
        bore=0.081, stroke=0.0955, conrod=0.144,
        compression_ratio=16.5, n_cylinders=4, stroke_cycle=4,
        diesel=True, turbo_boost=185000.0,   # ~1.85 bar (real 2.0 TDI 150 boost)
        ignition_btdc=6.0, burn_duration=62.0,
        redline_rpm=4800.0, idle_rpm=820.0, inertia=0.23,
        pipe_length=1.85, pipe_diameter=0.052, runner_spread=0.20,
        firing_order=[1, 3, 4, 2],
    )


def _scooter_125():
    # 125cc 4-stroke scooter single (GY6-ish)
    return EngineConfig(
        name="125cc 4-stroke scooter",
        bore=0.0524, stroke=0.0578, conrod=0.090,
        compression_ratio=10.5, n_cylinders=1, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=54.0,
        redline_rpm=9000.0, idle_rpm=1700.0, inertia=0.012,
        pipe_length=0.9, pipe_diameter=0.028, runner_spread=0.0,
        firing_order=[1],
    )


def _grom_125():
    # 125cc 4-stroke mini-bike single (Honda Grom-ish, low-revving)
    return EngineConfig(
        name="125cc mini-bike single",
        bore=0.0524, stroke=0.0578, conrod=0.092,
        compression_ratio=10.0, n_cylinders=1, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=56.0,
        redline_rpm=8500.0, idle_rpm=1500.0, inertia=0.014,
        pipe_length=1.0, pipe_diameter=0.030, runner_spread=0.0,
        firing_order=[1],
    )


def _moto_300_twin():
    # 300cc parallel-twin small bike (180deg crank)
    return EngineConfig(
        name="300cc parallel-twin",
        bore=0.062, stroke=0.0497, conrod=0.090,
        compression_ratio=11.0, n_cylinders=2, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=50.0,
        redline_rpm=12500.0, idle_rpm=1400.0, inertia=0.018,
        pipe_length=1.0, pipe_diameter=0.034, runner_spread=0.14,
        firing_order=[1, 2],
    )


def _moped_50_4t():
    # 50cc 4-stroke moped single
    return EngineConfig(
        name="50cc 4-stroke moped",
        bore=0.039, stroke=0.0418, conrod=0.070,
        compression_ratio=11.0, n_cylinders=1, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=56.0,
        redline_rpm=8500.0, idle_rpm=1800.0, inertia=0.006,
        pipe_length=0.7, pipe_diameter=0.022, runner_spread=0.0,
        firing_order=[1],
    )


def _kei_660_turbo():
    # 660cc kei-car inline-3 turbo (Japan kei class)
    return EngineConfig(
        name="660cc inline-3 turbo (kei)",
        bore=0.064, stroke=0.0683, conrod=0.112,
        compression_ratio=9.2, n_cylinders=3, stroke_cycle=4,
        diesel=False, turbo_boost=55000.0,
        ignition_btdc=18.0, burn_duration=52.0,
        redline_rpm=7000.0, idle_rpm=850.0, inertia=0.10,
        pipe_length=1.3, pipe_diameter=0.040, runner_spread=0.20,
        firing_order=[1, 2, 3],
    )


def _kei_660_na():
    # 660cc kei-car inline-3 NA
    return EngineConfig(
        name="660cc inline-3 (kei NA)",
        bore=0.064, stroke=0.0683, conrod=0.112,
        compression_ratio=11.2, n_cylinders=3, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=52.0,
        redline_rpm=7200.0, idle_rpm=850.0, inertia=0.10,
        pipe_length=1.3, pipe_diameter=0.038, runner_spread=0.20,
        firing_order=[1, 2, 3],
    )


def _citycar_1200():
    # 1.2L inline-4 economy city-car engine
    return EngineConfig(
        name="1.2L inline-4 (city)",
        bore=0.0710, stroke=0.0758, conrod=0.130,
        compression_ratio=11.0, n_cylinders=4, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=52.0,
        redline_rpm=6200.0, idle_rpm=780.0, inertia=0.15,
        pipe_length=1.6, pipe_diameter=0.044, runner_spread=0.22,
        firing_order=[1, 3, 4, 2],
    )


# ---- tractors / agricultural / industrial diesels -------------------------
def _tractor_3cyl():
    return EngineConfig(
        name="3.3L 3-cyl NA tractor diesel",
        bore=0.104, stroke=0.130, conrod=0.205,
        compression_ratio=17.5, n_cylinders=3, stroke_cycle=4,
        diesel=True, turbo_boost=0.0,        # naturally aspirated workhorse
        ignition_btdc=10.0, burn_duration=66.0,
        redline_rpm=2600.0, idle_rpm=800.0, inertia=2.2,
        pipe_length=1.6, pipe_diameter=0.060, runner_spread=0.18,
        firing_order=[1, 2, 3],
    )


def _tractor_4cyl_turbo():
    return EngineConfig(
        name="4.5L 4-cyl turbo tractor diesel",
        bore=0.106, stroke=0.127, conrod=0.210,
        compression_ratio=17.0, n_cylinders=4, stroke_cycle=4,
        diesel=True, turbo_boost=120000.0,
        ignition_btdc=8.0, burn_duration=66.0,
        redline_rpm=2400.0, idle_rpm=800.0, inertia=3.2,
        pipe_length=1.9, pipe_diameter=0.070, runner_spread=0.22,
        firing_order=[1, 3, 4, 2],
    )


def _tractor_6cyl():
    return EngineConfig(
        name="9.0L 6-cyl turbo tractor diesel",
        bore=0.118, stroke=0.136, conrod=0.225,
        compression_ratio=16.5, n_cylinders=6, stroke_cycle=4,
        diesel=True, turbo_boost=180000.0,
        ignition_btdc=6.0, burn_duration=70.0,
        redline_rpm=2200.0, idle_rpm=750.0, inertia=6.0,
        pipe_length=2.4, pipe_diameter=0.090, runner_spread=0.28,
        firing_order=[1, 5, 3, 6, 2, 4],
    )


def _lister_single():
    return EngineConfig(
        name="1.4L slow-rev stationary diesel (Lister-type)",
        bore=0.114, stroke=0.139, conrod=0.230,
        compression_ratio=16.0, n_cylinders=1, stroke_cycle=4,
        diesel=True, turbo_boost=0.0,
        ignition_btdc=12.0, burn_duration=70.0,
        # huge flywheel, ~650 rpm governed -> the classic thump-thump
        redline_rpm=1300.0, idle_rpm=520.0, inertia=2.4,
        pipe_length=1.2, pipe_diameter=0.055, runner_spread=0.0,
        firing_order=[1],
    )


# ---- more motorcycles ------------------------------------------------------
def _cbx_six():
    return EngineConfig(
        name="1.0L inline-6 (Honda CBX)",
        bore=0.0645, stroke=0.0534, conrod=0.098,
        compression_ratio=9.3, n_cylinders=6, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=50.0,
        redline_rpm=9800.0, idle_rpm=1100.0, inertia=0.06,
        pipe_length=1.3, pipe_diameter=0.040, runner_spread=0.22,
        firing_order=[1, 5, 3, 6, 2, 4],
    )


def _h2_two_stroke_triple():
    return EngineConfig(
        name="750cc 2-stroke triple (H2 Mach IV)",
        bore=0.071, stroke=0.0635, conrod=0.120,
        compression_ratio=7.0, n_cylinders=3, stroke_cycle=2,
        diesel=False, turbo_boost=35000.0,      # crankcase scavenge delivery
        ignition_btdc=18.0, burn_duration=44.0,
        evo=92.0, evc=268.0, ivo=115.0, ivc=245.0,
        redline_rpm=8000.0, idle_rpm=1200.0, inertia=0.045,
        pipe_length=1.15, pipe_diameter=0.038, runner_spread=0.20,
        firing_order=[1, 2, 3],
    )


def _ktm_690():
    return EngineConfig(
        name="690cc supermoto single (high-rev)",
        bore=0.105, stroke=0.080, conrod=0.150,
        compression_ratio=12.8, n_cylinders=1, stroke_cycle=4,
        ignition_btdc=26.0, burn_duration=54.0,
        redline_rpm=8500.0, idle_rpm=1450.0, inertia=0.045,
        pipe_length=1.4, pipe_diameter=0.044, runner_spread=0.0,
        firing_order=[1],
    )


def _goldwing_flat6():
    return EngineConfig(
        name="1.8L flat-6 (Gold Wing)",
        bore=0.074, stroke=0.071, conrod=0.125,
        compression_ratio=10.5, n_cylinders=6, stroke_cycle=4,
        ignition_btdc=20.0, burn_duration=52.0,
        redline_rpm=6000.0, idle_rpm=900.0, inertia=0.16,
        pipe_length=1.7, pipe_diameter=0.046, runner_spread=0.24,
        firing_order=[1, 4, 5, 2, 3, 6],
    )


# ---- exotic / many-cylinder ------------------------------------------------
def _supra_2jz():
    return EngineConfig(
        name="3.0L inline-6 turbo (2JZ)",
        bore=0.086, stroke=0.086, conrod=0.142,
        compression_ratio=8.5, n_cylinders=6, stroke_cycle=4,
        turbo_boost=110000.0,
        ignition_btdc=20.0, burn_duration=52.0,
        redline_rpm=7000.0, idle_rpm=750.0, inertia=0.20,
        pipe_length=1.7, pipe_diameter=0.064, runner_spread=0.24,
        firing_order=[1, 5, 3, 6, 2, 4],
    )


def _w16_quad_turbo():
    return EngineConfig(
        name="8.0L W16 quad-turbo (Bugatti)",
        bore=0.086, stroke=0.0857, conrod=0.152,
        compression_ratio=9.0, n_cylinders=16, stroke_cycle=4,
        turbo_boost=160000.0,
        ignition_btdc=18.0, burn_duration=50.0,
        redline_rpm=6800.0, idle_rpm=900.0, inertia=0.42,
        pipe_length=1.9, pipe_diameter=0.090, runner_spread=0.30,
    )


def _v16_na():
    return EngineConfig(
        name="7.0L V16 (NA luxury)",
        bore=0.085, stroke=0.077, conrod=0.145,
        compression_ratio=10.5, n_cylinders=16, stroke_cycle=4,
        ignition_btdc=22.0, burn_duration=50.0,
        redline_rpm=6500.0, idle_rpm=650.0, inertia=0.46,
        pipe_length=2.0, pipe_diameter=0.085, runner_spread=0.30,
    )


def _radial_9():
    return EngineConfig(
        name="17L 9-cyl radial (aircraft)",
        bore=0.130, stroke=0.142, conrod=0.300,
        compression_ratio=6.5, n_cylinders=9, stroke_cycle=4,
        ignition_btdc=25.0, burn_duration=52.0,
        redline_rpm=2400.0, idle_rpm=550.0, inertia=4.0,
        # Open short stacks, no silencer -- a radial is deafening and barks. True
        # per-cylinder stacks (9 pipes) would be most faithful but blow the
        # real-time budget; 3 collector banks of 3 cylinders give the firing-rate
        # bark over the low rumble (the muffled single-collector default turned a
        # 17 L roar into a thin mid-heavy drone).
        pipe_length=0.6, pipe_diameter=0.065, runner_spread=0.30,
        muffler_volume=-0.001, exhaust_banks=3,  # <0 = straight (UI shows -1 L)
        # single-row radial fires every other cylinder -> even 80deg cadence
        firing_order=[1, 3, 5, 7, 9, 2, 4, 6, 8],
    )


PRESETS = {
    # --- small / mopeds / scooters ---
    "50cc 2-stroke single": _moped_50cc,
    "50cc 4-stroke moped": _moped_50_4t,
    "125cc 4-stroke scooter": _scooter_125,
    "125cc mini-bike single": _grom_125,
    "250cc 2-stroke single": _two_stroke_250,
    "300cc parallel-twin": _moto_300_twin,
    # --- motorcycles ---
    "125cc 4-stroke single": _moto_125,
    "650cc thumper single": _thumper_single,
    "690cc supermoto single": _ktm_690,
    "689cc parallel-twin (270°)": _parallel_twin_270,
    "750cc 2-stroke triple (H2)": _h2_two_stroke_triple,
    "1.25L boxer-twin (BMW)": _bmw_boxer,
    "1.3L L-twin (Ducati)": _ducati_ltwin,
    "1.9L V-twin (Harley)": _harley_vtwin,
    "765cc triple (Triumph)": _triumph_triple,
    "600cc inline-4 sportbike": _sportbike_600,
    "998cc cross-plane I4 (R1)": _r1_crossplane,
    "1.0L inline-6 (Honda CBX)": _cbx_six,
    "1.1L V4 superbike (Ducati)": _ducati_v4,
    "1.8L flat-6 (Gold Wing)": _goldwing_flat6,
    # --- cars: small / NA petrol ---
    "660cc inline-3 (kei NA)": _kei_660_na,
    "660cc inline-3 turbo (kei)": _kei_660_turbo,
    "1.0L inline-3": _triple_1000,
    "1.2L inline-4 (city)": _citycar_1200,
    "1.6L inline-4 NA": _na_1600,
    "2.0L flat-4 boxer": _boxer4_na,
    "2.0L VTEC high-rev I4": _vtec_i4,
    "2.5L inline-5": _inline5_2500,
    "3.0L inline-6": _inline6_3000,
    "3.5L V6 (2GR)": _v6_2gr,
    "3.8L flat-6 (911)": _flat6_na,
    "5.0L V8": _v8_5000,
    "6.2L V8 muscle": _v8_muscle_6200,
    "7.4L big-block V8": _bigblock_v8,
    "5.2L flat-plane V8": _flatplane_v8,
    "4.8L V10 (LFA)": _v10_screamer,
    "5.2L V10 (R8/Huracan)": _v10_audi,
    "6.5L V12 (Ferrari)": _v12_ferrari,
    "7.0L V16 (NA luxury)": _v16_na,
    # --- cars: forced induction petrol ---
    "1.8L inline-4 turbo": _turbo4_18,
    "2.0L turbo-4": _gti_2000,
    "2.5L flat-4 turbo (rally)": _boxer4_turbo,
    "3.0L inline-6 turbo (2JZ)": _supra_2jz,
    "3.8L V6 twin-turbo (GT-R)": _v6_vr38,
    "4.4L V8 twin-turbo (S63)": _v8_s63,
    "8.0L W16 quad-turbo (Bugatti)": _w16_quad_turbo,
    # --- diesels ---
    "1.4 TDI diesel-4": _tdi_1400,
    "2.0 TDI diesel-4": _tdi_2000,
    "3.0 V6 TDI diesel": _tdi_v6_3000,
    "6.6 V8 diesel (pickup)": _diesel_v8_66,
    "6.7 i6 diesel (truck)": _diesel_i6_67,
    "12.6 i6 diesel (bus)": _diesel_i6_126,
    # --- tractors / industrial / aircraft ---
    "3.3L 3-cyl tractor diesel": _tractor_3cyl,
    "4.5L 4-cyl turbo tractor": _tractor_4cyl_turbo,
    "9.0L 6-cyl turbo tractor": _tractor_6cyl,
    "1.4L stationary diesel (Lister)": _lister_single,
    "17L 9-cyl radial (aircraft)": _radial_9,
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
        wheel_radius=0.18, final_drive=4.2, gear_ratios=[2.2],
        drivetrain="rwd", weight_dist_front=0.48, wheelbase=1.3, cg_height=0.55),
    "Moped": VehicleConfig(
        name="Moped", mass=95.0, cd=0.9, frontal_area=0.6, crr=0.020,
        wheel_radius=0.20, final_drive=5.2, gear_ratios=[2.8],
        drivetrain="rwd", weight_dist_front=0.47, wheelbase=1.2, cg_height=0.55),
    "Pit bike": VehicleConfig(
        name="Pit bike", mass=85.0, cd=0.7, frontal_area=0.55, crr=0.022,
        wheel_radius=0.22, final_drive=4.2,
        gear_ratios=[2.6, 1.8, 1.35, 1.05],
        drivetrain="rwd", weight_dist_front=0.48, wheelbase=1.1, cg_height=0.55),
    "Small bike (commuter)": VehicleConfig(
        name="Commuter bike", mass=160.0, cd=0.62, frontal_area=0.58, crr=0.019,
        wheel_radius=0.28, final_drive=4.8,
        gear_ratios=[2.7, 1.85, 1.4, 1.15, 0.96, 0.84],
        drivetrain="rwd", weight_dist_front=0.49, wheelbase=1.35, cg_height=0.58),
    "Motorcycle (sport)": VehicleConfig(
        name="Sport bike", mass=240.0, cd=0.58, frontal_area=0.6, crr=0.018,
        # final_drive folds in the primary reduction (~1.9) + chain (~2.9)
        wheel_radius=0.30, final_drive=5.6,
        gear_ratios=[2.62, 1.95, 1.58, 1.36, 1.21, 1.10],
        drivetrain="rwd", weight_dist_front=0.50, wheelbase=1.4, cg_height=0.55),
    "Microcar": VehicleConfig(
        name="Microcar", mass=600.0, cd=0.34, frontal_area=1.7, crr=0.013,
        wheel_radius=0.24, final_drive=4.6,
        gear_ratios=[3.7, 2.1, 1.4, 1.0, 0.82]),
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
        gear_ratios=[3.2, 2.1, 1.5, 1.15, 0.9, 0.75],
        drivetrain="rwd", weight_dist_front=0.50, wheelbase=2.5, cg_height=0.46),
    "Supercar": VehicleConfig(
        name="Supercar", mass=1480.0, cd=0.33, frontal_area=1.9, crr=0.011,
        wheel_radius=0.34, final_drive=3.6,
        gear_ratios=[3.13, 2.19, 1.63, 1.29, 1.03, 0.84, 0.69],
        drivetrain="rwd", weight_dist_front=0.44, wheelbase=2.65, cg_height=0.44),
    "SUV": VehicleConfig(
        name="SUV", mass=2100.0, cd=0.35, frontal_area=2.8, crr=0.013,
        wheel_radius=0.36, final_drive=3.9,
        gear_ratios=[3.8, 2.1, 1.4, 1.0, 0.78, 0.64],
        drivetrain="awd", weight_dist_front=0.54, wheelbase=2.8, cg_height=0.72),
    "Pickup truck (8-spd)": VehicleConfig(
        name="Pickup", mass=2400.0, cd=0.42, frontal_area=3.2, crr=0.013,
        wheel_radius=0.38, final_drive=3.7,
        gear_ratios=[4.7, 3.1, 2.1, 1.67, 1.29, 1.0, 0.84, 0.67],
        drivetrain="rwd", weight_dist_front=0.56, wheelbase=3.4, cg_height=0.75),
    "Van": VehicleConfig(
        name="Van", mass=2000.0, cd=0.38, frontal_area=3.6, crr=0.013,
        wheel_radius=0.34, final_drive=4.0,
        gear_ratios=[3.9, 2.3, 1.5, 1.05, 0.82, 0.7],
        drivetrain="fwd", weight_dist_front=0.57, wheelbase=3.0, cg_height=0.8),
    "Box truck (10-spd)": VehicleConfig(
        name="Box truck", mass=4500.0, cd=0.50, frontal_area=5.5, crr=0.014,
        wheel_radius=0.44, final_drive=4.3,
        gear_ratios=[9.0, 6.5, 4.7, 3.4, 2.5, 1.9, 1.42, 1.1, 0.86, 0.73],
        drivetrain="rwd", weight_dist_front=0.5, wheelbase=3.8, cg_height=1.1),
    "Bus / lorry (12-spd)": VehicleConfig(
        name="Bus", mass=14000.0, cd=0.60, frontal_area=6.8, crr=0.015,
        wheel_radius=0.50, final_drive=3.4,
        gear_ratios=[11.7, 9.2, 7.1, 5.6, 4.4, 3.5,
                     2.8, 2.2, 1.7, 1.35, 1.05, 0.84],
        drivetrain="rwd", weight_dist_front=0.45, wheelbase=5.8, cg_height=1.3),
    "Semi / artic (16-spd)": VehicleConfig(
        name="Semi", mass=26000.0, cd=0.62, frontal_area=9.5, crr=0.0065,
        wheel_radius=0.52, final_drive=3.7,
        gear_ratios=[14.0, 11.5, 9.4, 7.7, 6.2, 5.1, 4.2, 3.4,
                     2.8, 2.3, 1.9, 1.55, 1.25, 1.0, 0.84, 0.73],
        drivetrain="rwd", weight_dist_front=0.35, wheelbase=4.0, cg_height=1.3),
    # --- extra cars ---
    "Kei car": VehicleConfig(
        name="Kei", mass=720.0, cd=0.33, frontal_area=1.9, crr=0.012,
        wheel_radius=0.26, final_drive=4.4,
        gear_ratios=[3.6, 2.05, 1.42, 1.03, 0.82]),
    "Estate / wagon": VehicleConfig(
        name="Estate", mass=1550.0, cd=0.30, frontal_area=2.3, crr=0.011,
        wheel_radius=0.32, final_drive=3.5,
        gear_ratios=[3.6, 2.05, 1.4, 1.0, 0.82, 0.68]),
    "Luxury sedan (8-spd)": VehicleConfig(
        name="Luxury sedan", mass=1900.0, cd=0.26, frontal_area=2.4, crr=0.010,
        wheel_radius=0.34, final_drive=3.1,
        gear_ratios=[4.7, 3.1, 2.1, 1.67, 1.29, 1.0, 0.84, 0.67]),
    "Track car": VehicleConfig(
        name="Track car", mass=1150.0, cd=0.34, frontal_area=1.85, crr=0.012,
        wheel_radius=0.33, final_drive=3.9,
        gear_ratios=[2.9, 2.05, 1.6, 1.3, 1.08, 0.91],
        drivetrain="rwd", weight_dist_front=0.48, wheelbase=2.4, cg_height=0.42),
    "Rally car (AWD)": VehicleConfig(
        name="Rally", mass=1230.0, cd=0.36, frontal_area=2.0, crr=0.016,
        wheel_radius=0.32, final_drive=4.1,
        gear_ratios=[3.5, 2.35, 1.76, 1.38, 1.1, 0.92],
        drivetrain="awd", weight_dist_front=0.56, wheelbase=2.6, cg_height=0.5),
    "Large 4x4 (8-spd)": VehicleConfig(
        name="4x4", mass=2600.0, cd=0.38, frontal_area=3.0, crr=0.014,
        wheel_radius=0.40, final_drive=3.7,
        gear_ratios=[4.7, 3.1, 2.1, 1.67, 1.29, 1.0, 0.84, 0.67],
        drivetrain="awd", weight_dist_front=0.52, wheelbase=2.9, cg_height=0.78),
    "Supercar (7-spd DCT)": VehicleConfig(
        name="Supercar", mass=1500.0, cd=0.34, frontal_area=1.92, crr=0.012,
        wheel_radius=0.34, final_drive=3.55,
        gear_ratios=[3.13, 2.19, 1.63, 1.29, 1.03, 0.84, 0.69],
        drivetrain="rwd", weight_dist_front=0.43, wheelbase=2.65, cg_height=0.44),
    "Hypercar (AWD)": VehicleConfig(
        name="Hypercar", mass=1450.0, cd=0.36, frontal_area=1.95, crr=0.012,
        wheel_radius=0.35, final_drive=3.4,
        gear_ratios=[2.92, 2.04, 1.6, 1.32, 1.09, 0.92, 0.78],
        drivetrain="awd", weight_dist_front=0.45, wheelbase=2.7, cg_height=0.43),
    "Muscle car": VehicleConfig(
        name="Muscle", mass=1750.0, cd=0.36, frontal_area=2.2, crr=0.012,
        wheel_radius=0.34, final_drive=3.55,
        gear_ratios=[2.97, 2.07, 1.43, 1.0, 0.84, 0.66],
        drivetrain="rwd", weight_dist_front=0.54, wheelbase=2.9, cg_height=0.5),
    "Pickup (heavy tow)": VehicleConfig(
        name="HD Pickup", mass=3200.0, cd=0.45, frontal_area=3.4, crr=0.013,
        wheel_radius=0.40, final_drive=3.9,
        gear_ratios=[4.7, 3.1, 2.1, 1.67, 1.29, 1.0, 0.84, 0.67],
        drivetrain="rwd", weight_dist_front=0.58, wheelbase=3.6, cg_height=0.8),
    "Compact SUV": VehicleConfig(
        name="Compact SUV", mass=1650.0, cd=0.33, frontal_area=2.5, crr=0.012,
        wheel_radius=0.34, final_drive=3.7,
        gear_ratios=[3.9, 2.1, 1.4, 1.0, 0.78, 0.64]),
    # --- motorcycles ---
    "Cruiser (V-twin)": VehicleConfig(
        name="Cruiser", mass=380.0, cd=0.70, frontal_area=0.75, crr=0.018,
        wheel_radius=0.40, final_drive=4.6,
        gear_ratios=[3.3, 2.2, 1.6, 1.27, 1.05, 0.89],
        drivetrain="rwd", weight_dist_front=0.47, wheelbase=1.65, cg_height=0.55),
    "Superbike": VehicleConfig(
        name="Superbike", mass=205.0, cd=0.55, frontal_area=0.55, crr=0.017,
        wheel_radius=0.31, final_drive=5.8,
        gear_ratios=[2.6, 2.0, 1.67, 1.44, 1.29, 1.15],
        drivetrain="rwd", weight_dist_front=0.51, wheelbase=1.42, cg_height=0.55),
    "Café racer": VehicleConfig(
        name="Café racer", mass=220.0, cd=0.62, frontal_area=0.62, crr=0.018,
        wheel_radius=0.30, final_drive=5.0,
        gear_ratios=[2.5, 1.7, 1.3, 1.05, 0.9],
        drivetrain="rwd", weight_dist_front=0.49, wheelbase=1.4, cg_height=0.55),
}


def get_vehicle(name: str) -> VehicleConfig:
    return VEHICLES[name]
