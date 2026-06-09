"""Idle quality test: crank, settle, blip the throttle, release, and watch the
idle recover. Reports how close idle holds to target and how much it hunts
(oscillation range as it settles) -- the heavy engines are the hard cases."""
import numpy as np
from engine_sim.presets import get_preset
from engine_sim.sim import EngineSim, SAMPLE_RATE

BLOCK = 512


def adv(sim, sec):
    n = int(SAMPLE_RATE * sec)
    out = []
    for i in range(0, n, BLOCK):
        sim.step_block(min(BLOCK, n - i))
        out.append(sim.rpm)
    return np.array(out)


def test(name):
    cfg = get_preset(name)
    sim = EngineSim(cfg)
    sim.step_block(64); sim.load_config(cfg)
    sim.prewarm()                # test warm idle quality (cold start is separate)
    sim.start_engine()
    for _ in range(80):
        sim.step_block(BLOCK)
        if sim.state == "running":
            break
    pre = adv(sim, 4.0)                       # settle at idle (no blip yet)
    idle0 = pre[-30:].mean()
    idle_hunt = float(pre[-180:].max() - pre[-180:].min())  # steady idle ripple
    sim.set_throttle(0.5); adv(sim, 0.8)     # blip
    sim.set_throttle(0.0)
    tr = adv(sim, 6.0)                        # recovery after tip-out
    tgt = cfg.idle_rpm
    recov = tr[-240:]                         # last ~2.4 s (settled)
    rhunt = float(recov.max() - recov.min())  # residual hunting after recovery
    print(f"  {cfg.name:30s} tgt {tgt:5.0f}  idle {idle0:5.0f} "
          f"(err {idle0-tgt:+5.0f}, ripple±{idle_hunt/2:3.0f})  "
          f"tip-out dip {tr.min():5.0f}  recover {recov.mean():5.0f} hunt±{rhunt/2:4.0f}")


if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or ["1.0L inline-3", "5.0L V8", "6.2L V8 muscle",
                             "7.4L big-block V8", "6.5L V12 (Ferrari)",
                             "12.6 i6 diesel (bus)", "1.4 TDI diesel-4"]
    for n in names:
        test(n)
