"""Offline audio render — the listening/measurement instrument for sound work.

Renders an engine preset through the real sim (no audio device) and writes
16-bit WAVs plus level/spectral stats, so changes to the sound path can be
A/B'd by ear and by numbers.

Scenarios:
  idle    : warm idle
  wot     : wide-open throttle held in neutral (limiter/governor)
  sweep   : slow pedal ramp 0 -> 1 in neutral (rpm sweep through the range)
  blip    : idle, three throttle blips, decel pops

Usage:
  python render.py                          # default engine, all scenarios
  python render.py --engine "1.0L inline-3" --scenario sweep
  python render.py --list                   # show preset names
  python render.py --all-engines --scenario wot
"""
import argparse
import os
import wave

import numpy as np

from engine_sim.presets import PRESETS, get_preset
from engine_sim.sim import EngineSim, SAMPLE_RATE

BLOCK = 512
OUTDIR = "renders"


def advance(sim, seconds, sink=None):
    n = int(SAMPLE_RATE * seconds)
    buf = np.zeros(n)
    for i in range(0, n, BLOCK):
        m = min(BLOCK, n - i)
        buf[i:i + m] = sim.step_block(m)
    if sink is not None:
        sink.append(buf)
    return buf


def start(sim, max_s=4.0):
    sim.start_engine()
    t = 0.0
    while sim.state != "running" and t < max_s:
        advance(sim, 0.1)
        t += 0.1
    if sim.state != "running":
        raise RuntimeError("engine failed to start")
    advance(sim, 1.5)          # settle at idle


def scenario_idle(sim, sink):
    advance(sim, 6.0, sink)


def scenario_wot(sim, sink):
    sim.set_throttle(1.0)
    advance(sim, 5.0, sink)
    sim.set_throttle(0.0)


def scenario_sweep(sim, sink):
    secs = 8.0
    steps = int(secs / 0.1)
    for i in range(steps):
        sim.set_throttle((i + 1) / steps)
        advance(sim, 0.1, sink)
    sim.set_throttle(0.0)


def scenario_blip(sim, sink):
    advance(sim, 1.5, sink)
    for _ in range(3):
        sim.set_throttle(0.9)
        advance(sim, 0.7, sink)
        sim.set_throttle(0.0)
        advance(sim, 1.6, sink)


SCENARIOS = {
    "idle": scenario_idle,
    "wot": scenario_wot,
    "sweep": scenario_sweep,
    "blip": scenario_blip,
}


def write_wav(path, audio):
    x = np.clip(audio, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def band_levels(audio):
    """RMS level (dBFS) in octave-ish bands — a coarse spectral fingerprint."""
    n = len(audio)
    if n < 8192:
        return {}
    spec = np.abs(np.fft.rfft(audio * np.hanning(n))) / n
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    bands = [(20, 80), (80, 200), (200, 500), (500, 1200),
             (1200, 3000), (3000, 8000), (8000, 20000)]
    out = {}
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        p = np.sqrt(np.sum(spec[m] ** 2))
        out[f"{lo}-{hi}"] = 20.0 * np.log10(p + 1e-12)
    return out


def render(name, scen, outdir):
    cfg = get_preset(name)
    sim = EngineSim(cfg)
    sim.step_block(64)
    sim.load_config(cfg)
    sim.prewarm()
    start(sim)
    sink = []
    SCENARIOS[scen](sim, sink)
    audio = np.concatenate(sink)
    finite = bool(np.all(np.isfinite(audio)))
    if not finite:
        audio = np.nan_to_num(audio)
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)) + 1e-12)
    crest = 20.0 * np.log10(peak / (rms + 1e-12))
    safe = name.replace("/", "-").replace(" ", "_")
    path = os.path.join(outdir, f"{safe}__{scen}.wav")
    write_wav(path, audio)
    print(f"  {scen:5s}  rms {rms:6.3f}  peak {peak:5.2f}  crest {crest:4.1f} dB"
          f"  finite {finite}  -> {path}")
    bl = band_levels(audio)
    if bl:
        print("         bands(dB): " +
              "  ".join(f"{k}:{v:6.1f}" for k, v in bl.items()))
    return rms, finite


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="1.0L inline-3")
    ap.add_argument("--scenario", default="all",
                    choices=["all"] + list(SCENARIOS))
    ap.add_argument("--all-engines", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    if args.list:
        for n in PRESETS:
            print(n)
        return

    os.makedirs(args.outdir, exist_ok=True)
    engines = list(PRESETS) if args.all_engines else [args.engine]
    scens = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in engines:
        print(f"\n== {name} ==")
        for sc in scens:
            try:
                render(name, sc, args.outdir)
            except Exception as e:
                print(f"  {sc:5s}  FAILED: {e}")


if __name__ == "__main__":
    main()
