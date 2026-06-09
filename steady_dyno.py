"""Steady-state dyno: hold the engine at each rpm with a load controller and read
brake torque directly as (gas torque - friction) -- no I*dw/dt differentiation,
so the curve is clean enough to SEE intake ram tuning. This is the measuring
stick for the per-cylinder runner work: a real ram peak shows up here as a hump
in the torque/BMEP curve whose rpm should track runner length.

Usage:  python steady_dyno.py ["5.0L V8" ...]
"""
import math
import sys
import numpy as np
from engine_sim.presets import get_preset, PRESETS
from engine_sim.sim import EngineSim, SAMPLE_RATE
from engine_sim import core

BLOCK = 512
DT = BLOCK / SAMPLE_RATE
FLYWHEEL = 4.0       # big inertia -> slow plant -> the load PI loop is stable
NPTS = 18            # rpm sweep resolution


def _friction(sim, omega):
    return sim.P[core.P_FRIC0] + sim.P[core.P_FRICW] * omega


def steady_point(sim, target_rpm):
    """Hold target_rpm at WOT with a load PI loop, then average brake torque."""
    omega_t = target_rpm * 2.0 * math.pi / 60.0
    sim.set_throttle(1.0)
    load = max(0.0, sim.torque * 0.5)     # seed near current gas torque
    integ = load
    # ---- settle ----
    for _ in range(220):
        err = sim.rpm - target_rpm        # too fast -> add load
        integ = float(np.clip(integ + 0.02 * err, 0.0, 1e6))
        load = float(np.clip(integ + 0.6 * err, 0.0, 1e6))
        sim.set_load(load)
        sim.step_block(BLOCK)
        if sim.state != "running":
            return None
    # ---- measure ----
    tq = []
    rpms = []
    for _ in range(80):
        err = sim.rpm - target_rpm
        integ = float(np.clip(integ + 0.02 * err, 0.0, 1e6))
        load = float(np.clip(integ + 0.6 * err, 0.0, 1e6))
        sim.set_load(load)
        sim.step_block(BLOCK)
        omega = sim.st[core.S_OMEGA]
        tq.append(sim.torque - _friction(sim, omega))   # brake torque
        rpms.append(sim.rpm)
    return float(np.mean(rpms)), float(np.mean(tq)), float(np.std(rpms))


def run(name):
    cfg = get_preset(name)
    sim = EngineSim(cfg)
    sim.step_block(64); sim.load_config(cfg)
    sim.prewarm()      # measure WARM performance
    sim.P[core.P_NOISE] = 0.0          # kill cycle-to-cycle roughness for a clean read

    # crank + idle settle
    sim.start_engine()
    for _ in range(40):
        sim.step_block(BLOCK)
        if sim.state == "running":
            break
    for _ in range(60):
        sim.step_block(BLOCK)
    sim.base_inertia = FLYWHEEL

    Vd = math.pi / 4 * cfg.bore ** 2 * cfg.stroke * cfg.n_cylinders
    nrev = 2.0 if cfg.stroke_cycle == 4 else 1.0
    grid = np.linspace(cfg.idle_rpm * 1.6, cfg.redline_rpm * 0.95, NPTS)

    print(f"\n=== {cfg.name} ===  ({cfg.displacement_cc():.0f}cc, "
          f"redline {cfg.redline_rpm:.0f}, runner {cfg.intake_length:.2f} m)")
    rows = []
    bestT = (0, -1e9); bestP = (0, -1e9)
    for r in grid:
        res = steady_point(sim, r)
        if res is None:
            print(f"  {r:5.0f} rpm   (stalled)")
            continue
        rpm, t, jit = res
        w = rpm * 2 * math.pi / 60
        hp = t * w / 745.7
        bmep = t * (2 * math.pi * nrev) / Vd / 1e5
        rows.append((rpm, t, hp, bmep))
        if t > bestT[1]:
            bestT = (rpm, t)
        if hp > bestP[1]:
            bestP = (rpm, hp)
        bar = "#" * int(max(0, bmep) * 2)
        print(f"  {rpm:5.0f} rpm   {t:6.1f} Nm   {hp:6.1f} hp   "
              f"{bmep:4.1f} bar  {bar}")
    print(f"  >> PEAK TORQUE {bestT[1]:.0f} Nm @ {bestT[0]:.0f},  "
          f"PEAK POWER {bestP[1]:.0f} hp @ {bestP[0]:.0f}")
    return rows


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else list(PRESETS.keys())
    for n in names:
        run(n)
