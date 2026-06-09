"""Headless sanity check — no audio, no UI. Exercises start/idle/rev-limit and
a full vehicle launch through the gears."""
import time
import numpy as np
from engine_sim.presets import PRESETS, get_preset, get_vehicle
from engine_sim.sim import EngineSim, SAMPLE_RATE

BLOCK = 512


def advance(sim, seconds):
    n = int(SAMPLE_RATE * seconds)
    audio = np.zeros(n)
    for i in range(0, n, BLOCK):
        m = min(BLOCK, n - i)
        audio[i:i + m] = sim.step_block(m)
    return audio


def crank_until_running(sim, max_s=3.0):
    sim.start_engine()
    t = 0.0
    while sim.state != "running" and t < max_s:
        advance(sim, 0.1)
        t += 0.1
    return sim.state == "running"


def run_preset(name):
    cfg = get_preset(name)
    sim = EngineSim(cfg)
    sim.step_block(64)            # warmup compile
    sim.load_config(cfg)
    sim.prewarm()                 # test warm idle/rev (cold start is a UI feature)
    print(f"\n=== {cfg.name} ===  ({cfg.displacement_cc():.0f}cc, "
          f"{cfg.stroke_cycle}-stroke, {sim.N}-cell exhaust)")

    started = crank_until_running(sim)
    advance(sim, 1.0)             # let it settle at idle
    idle_rpm = sim.rpm
    advance(sim, 0.5)
    idle_audio = advance(sim, 0.5)

    sim.set_throttle(1.0)         # WOT in neutral -> should rev to limiter
    rev = advance(sim, 2.0)
    wot_rpm = sim.rpm

    finite = np.all(np.isfinite(rev)) and np.all(np.isfinite(idle_audio))
    print(f"  started: {started}, idle held: {idle_rpm:6.0f} rpm "
          f"(target {cfg.idle_rpm:.0f})")
    print(f"  WOT neutral: {wot_rpm:6.0f} rpm (redline {cfg.redline_rpm:.0f})")
    print(f"  idle audio rms: {np.sqrt(np.mean(idle_audio**2)):.3f}, "
          f"WOT audio rms: {np.sqrt(np.mean(rev**2)):.3f}, finite: {finite}")
    ok = (started and finite and abs(idle_rpm - cfg.idle_rpm) < 600
          and wot_rpm > cfg.idle_rpm * 1.5)
    return ok


def test_drive():
    print("\n=== DRIVE: 1.0L triple in a Hatchback ===")
    sim = EngineSim(get_preset("1.0L inline-3"))
    sim.step_block(64); sim.load_config(get_preset("1.0L inline-3"))
    sim.prewarm()
    sim.set_vehicle(get_vehicle("Hatchback"))
    crank_until_running(sim)
    advance(sim, 1.0)
    sim.shift_up()                # into 1st
    sim.set_throttle(1.0)
    for g in range(1, 6):
        advance(sim, 2.0)
        print(f"  gear {sim.gear_label()}: {sim.rpm:5.0f} rpm, "
              f"{sim.speed_kmh():5.1f} km/h")
        if sim.rpm > cfg_redline(sim) * 0.9 and sim.gear < 5:
            sim.shift_up()
    moved = sim.speed_kmh() > 30.0
    print(f"  reached {sim.speed_kmh():.1f} km/h -> {'OK' if moved else 'FAIL'}")
    return moved


def cfg_redline(sim):
    return sim.cfg.redline_rpm


if __name__ == "__main__":
    ok = True
    for name in PRESETS:
        try:
            ok &= run_preset(name)
        except Exception:
            import traceback; traceback.print_exc(); ok = False
    try:
        ok &= test_drive()
    except Exception:
        import traceback; traceback.print_exc(); ok = False
    print("\n" + ("ALL OK" if ok else "PROBLEMS FOUND"))
