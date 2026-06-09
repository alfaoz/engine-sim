"""CPU benchmark: wall-clock ms to compute one 512-sample audio block, on the
heaviest presets (most cylinders / cells). The real-time budget is one block of
audio = 512/48000 = 10.67 ms; we must stay well under it (the audio callback has
to finish a block faster than it plays). Times the FULL path the audio thread
runs (control update + simulate_block), engine running at WOT.

Usage:  python bench.py ["6.5L V12 (Ferrari NA)" ...]
"""
import sys
import time
import numpy as np
from engine_sim.presets import get_preset, PRESETS
from engine_sim.sim import EngineSim, SAMPLE_RATE, BLOCK

BUDGET_MS = BLOCK / SAMPLE_RATE * 1000.0
HEAVY = ["6.5L V12 (Ferrari)", "12.6 i6 diesel (bus)",
         "5.0L V8", "5.2L V10 (R8/Huracan)", "1.0L inline-3"]


def bench(name, nblocks=600):
    cfg = get_preset(name)
    sim = EngineSim(cfg)
    sim.step_block(64); sim.load_config(cfg)     # compile
    sim.prewarm()
    sim.start_engine()
    for _ in range(80):
        sim.step_block(BLOCK)
        if sim.state == "running":
            break
    sim.set_throttle(1.0)
    for _ in range(60):                           # spin up under load
        sim.step_block(BLOCK)
    # timed run
    best = 1e9
    samples = []
    for _ in range(5):                            # take the best of a few sweeps
        t0 = time.perf_counter()
        for _ in range(nblocks):
            sim.step_block(BLOCK)
        dt = (time.perf_counter() - t0) / nblocks * 1000.0
        samples.append(dt)
        best = min(best, dt)
    med = float(np.median(samples))
    print(f"  {cfg.name:30s} {cfg.n_cylinders:2d}cyl {sim.N:3d}cell  "
          f"{best:6.3f} ms best / {med:6.3f} med   "
          f"{med / BUDGET_MS * 100:5.1f}% RT   ({sim.rpm:.0f} rpm)")
    return med


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else HEAVY
    print(f"real-time budget = {BUDGET_MS:.2f} ms/block\n")
    for n in names:
        if n in PRESETS:
            bench(n)
        else:
            print(f"  (no preset '{n}')")
