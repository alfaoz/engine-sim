"""Inertial dyno: free-rev the engine at WOT with a big flywheel and measure
brake torque as I*dw/dt (this nets out friction automatically and uses the exact
dynamics that drive the car). Reports peak torque / power / BMEP."""
import math
import numpy as np
from engine_sim.presets import get_preset, PRESETS
from engine_sim.sim import EngineSim, SAMPLE_RATE
from engine_sim import core

BLOCK = 512
DT = BLOCK / SAMPLE_RATE
FLYWHEEL = 6.0   # big inertia so the rev-up is slow enough to sample finely


def dyno(name):
    cfg = get_preset(name)
    sim = EngineSim(cfg)
    sim.step_block(64); sim.load_config(cfg)

    # crank it properly so combustion is established (the path that works)
    sim.start_engine()
    for _ in range(40):
        sim.step_block(BLOCK)
        if sim.state == "running":
            break
    for _ in range(60):              # settle at idle
        sim.step_block(BLOCK)

    # now bolt on a big flywheel and go WOT so the rev-up samples finely
    sim.P[core.P_INERTIA] = FLYWHEEL
    sim.set_throttle(1.0)

    Vd = math.pi / 4 * cfg.bore ** 2 * cfg.stroke * cfg.n_cylinders
    nrev = 2.0 if cfg.stroke_cycle == 4 else 1.0

    samples = []
    for _ in range(4000):
        w0 = sim.st[1]
        sim.step_block(BLOCK)
        w1 = sim.st[1]
        rpm = w1 * 60 / (2 * math.pi)
        dwdt = (w1 - w0) / DT
        torque = FLYWHEEL * dwdt          # brake torque (friction already netted)
        if sim.state == "running":
            samples.append((rpm, torque))
        if rpm > cfg.redline_rpm * 0.985 or sim.state == "off":
            break

    # bin by rpm, smooth
    arr = np.array(samples)
    rpms = arr[:, 0]; tq = arr[:, 1]
    grid = np.linspace(cfg.idle_rpm * 1.5, cfg.redline_rpm * 0.97, 12)
    tq_g = np.interp(grid, rpms, tq)
    rows = []
    bestT = (0, 0.); bestP = (0, 0.)
    for r, t in zip(grid, tq_g):
        w = r * 2 * math.pi / 60
        hp = t * w / 745.7
        bmep = t * (2 * math.pi * nrev) / Vd / 1e5
        rows.append((r, t, hp, bmep))
        if t > bestT[1]:
            bestT = (r, t)
        if hp > bestP[1]:
            bestP = (r, hp)
    print(f"\n=== {cfg.name} ===  ({cfg.displacement_cc():.0f}cc)")
    for r, t, hp, b in rows:
        print(f"  {r:5.0f} rpm   {t:6.1f} Nm   {hp:6.1f} hp   {b:4.1f} bar")
    print(f"  >> PEAK {bestT[1]:.0f} Nm @ {bestT[0]:.0f},  "
          f"{bestP[1]:.0f} hp @ {bestP[0]:.0f}")
    return bestT[1], bestP[1]


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] if len(sys.argv) > 1 else list(PRESETS.keys())
    for n in names:
        dyno(n)
