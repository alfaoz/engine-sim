"""0-100 km/h acceleration harness — validates that engine power actually
reaches the wheels (clutch + gearing) and that the cars feel right."""
import math
from engine_sim.presets import get_preset, get_vehicle
from engine_sim.sim import EngineSim, SAMPLE_RATE

BLOCK = 512
DT = BLOCK / SAMPLE_RATE


def accel_test(engine, vehicle, target_kmh=100.0, auto_shift=True):
    sim = EngineSim(get_preset(engine))
    sim.step_block(64); sim.load_config(get_preset(engine))
    sim.set_vehicle(get_vehicle(vehicle))
    sim.start_engine()
    for _ in range(40):
        sim.step_block(BLOCK)
        if sim.state == "running":
            break
    for _ in range(70):
        sim.step_block(BLOCK)            # idle settle
    sim.shift_up()                        # into 1st
    sim.set_throttle(1.0)

    t = 0.0
    t60 = None
    shift_cd = 0.0                        # cooldown so the clutch can re-lock
    while t < 30.0:
        sim.step_block(BLOCK)
        t += DT
        shift_cd = max(0.0, shift_cd - DT)
        if auto_shift and sim.rpm > sim.cfg.redline_rpm * 0.95 \
                and sim.gear < sim.vehicle.n_gears() and shift_cd <= 0.0:
            sim.shift_up()
            shift_cd = 0.7
        if t60 is None and sim.speed_kmh() >= 96.5:
            t60 = t
        if sim.speed_kmh() >= target_kmh:
            return t, sim.gear_label(), sim.speed_kmh(), t60
    return None, sim.gear_label(), sim.speed_kmh(), t60


CASES = [
    ("1.0L inline-3", "Hatchback"),
    ("1.6L inline-4 NA", "Hatchback"),
    ("2.0L turbo-4", "Sports car"),
    ("5.0L V8", "Sports car"),
    ("6.2L V8 muscle", "Sedan"),
    ("1.4 TDI diesel-4", "SUV"),
    ("3.0L inline-6", "Sedan"),
]

if __name__ == "__main__":
    print("engine / vehicle                         0-100km/h")
    for eng, veh in CASES:
        t, g, v, t60 = accel_test(eng, veh)
        res = f"{t:5.1f}s" if t else f"DNF ({v:.0f} km/h)"
        print(f"  {eng:24s} {veh:12s}  {res}")
