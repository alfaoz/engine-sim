# SOUND_FINDINGS.md — what the sound-ladder attempt learned (2026-06-10)

For any fresh session attempting exhaust-sound work: read `fable-fuckup.md`
first (process rules), then this (what was tried since, how each step
actually SOUNDED to the user, and which diagnoses were wrong). Branch:
`sound-ladder`, one commit per change; per-rung renders in `renders_ab/`.

## The architecture that exists now (all committed, user verdict mixed)

Audio = monopole radiation `d(mdot_exit)/dt / 4πr` at a listener 2 m away
(FS = 20 Pa = 120 dB SPL) + ground-image source (tailpipe 0.3 m over
asphalt, ear 1.5 m) + Levine-Schwinger-style open end (lows reflect, highs
radiate, corner ka=1 from exit radius) + Darcy wall friction (f=0.035,
quadratic) + absorptive packing in silencer-chamber cells (0.006/substep
≈ 2/3 per pass) + pipe chemistry crackle (below). Synthetic layers
remaining: clatter (gated noise → 2 resonators), de-hash LPF 4.5 kHz,
body resonator (now OFF by default), induction mix 0.15.

## The "tubey / tinny / plastic tube" complaint — NOT fixed by 4 attempts

The user reported "tinny/plasticky", later "engine sounds like it's coming
out of a shitty tinny tube", repeatedly, AFTER each of these:

1. **Radiation tap** (`21cd362`): physically right, but d/dt tilts +6 dB/oct
   → made things BRIGHTER/thinner. The tap alone does not fix tube-ness;
   it exposes it.
2. **Physical termination** (`fd4272c`): killed trapped HF comb (V8 WOT
   1200–3000 Hz −19 dB) — still "tubey" per user.
3. **Darcy friction** (`497eb05`): damps mid quarter-wave comb under load —
   still "so tube-y" per user.
4. **Ground image** (`a52dfd7`): +6 dB lows, mid comb — still tubey.
5. Latest pair, **verdict pending**: body resonator off by default +
   absorptive chamber packing (`444cad9`, `f25807f`).

Hypotheses NOT yet tried for tube-ness, in rough order of suspicion:
- The whole output is still ONE coherent 1D path: a single pipe into a
  single point source. Real cars radiate from tailpipe + shell + block +
  intake with incoherent phase. Any single-path model may read "tube".
- The de-hash 4.5 kHz LPF (fixed 2-pole) imposes a static "small speaker"
  envelope on everything.
- The induction duct (constant-area tube, 0.15 mix, always on) literally
  adds a tube honk; never A/B'd by the user with mix at 0.
- The quarter-wave comb at SMALL signal (idle) survives Darcy (quadratic
  → no small-signal loss) and flat damp 0.0015 is tiny: idle may stay
  comb-y. Frequency-dependent visco-thermal loss (~√f) was never added.
- No diffuse field at all: zero reverb/reflections beyond the one ground
  bounce.

## Crackle: took FOUR iterations; current form works mechanically

What the user rejected, in order:
1. Dice rolls (`pop`/`afterfire` RNG): random bangs during WOT — "excessive
   and fake". Cars don't crackle at WOT (no O2 in rich exhaust).
2. Chemistry v1 (trace-trigger): deflagration on >1e-12 kg inventories →
   pops at the firing rate everywhere — "continuous and shit".
3. Chemistry v2 (fired-cycle trigger gate): inventory still unbounded (cold
   rich running banks mg/cycle) → machine-gun on every overrun — "no fix
   whatsoever". LESSON: bound the INVENTORY, not the trigger.
4. Chemistry v3 (EVO-edge ignition): pops landed on the firing grid —
   "periodic... fires exactly where the engine would fire, not grumbly
   randomish". LESSON: valve-locked timing always sounds like the engine,
   never like burble. Turbulence is genuinely stochastic; refusing RNG on
   principle was wrong — the post-mortem bans tuned constants, not
   randomness.

Current (accepted-in-principle, level TBD): poppable fuel = never-flamed
only (misfire fraction (1-comp) + fuel-cut wall film, Aquino X-τ X=0.3
τ=0.6 s), ignition = per-sample Poisson at eddy rate |u|/D with random
pocket size, lean flammability gate (4.7% mass of ignition cell), AFT
energy cap, washout by real flow. Default survival fraction 0.05 (catted);
0.3 = race exhaust and made EVERY car crackle.

## Clatter: the most-rejected layer (still synthetic)

- It is envelope-gated noise → two bore-scaled low-Q resonators. The user
  has called it fake/too loud in every form. Constraints learned:
  - 4+ cyl petrol must be ZERO (an LFA is silk). A 0.10 "floor for
    presence" was rejected twice.
  - Level must fade hard with rpm: (1.5·idle/rpm)² now; before that it
    GREW with firing rate and buried everything ("static" on a boosted
    tractor at 2200 rpm). A comment claimed an rpm fade existed; it didn't.
  - It must ride the listener gain (mic distance); an absolute anchor
    left it +13 dB after the tap change ("all I hear is clatter").
  - User keeps the checkbox OFF but explicitly said keep the layer in code.
- The wanted thing remains: real block-radiated engine sound for cars,
  derived (event-timed mechanics is the rework's failure zone — see
  fable-fuckup piston-slap chatter). Whoever tries: ONE mechanism, alone,
  behind a toggle, subtle.

## Other landmines hit

- **Loudness reference**: with FS=20 Pa@2 m, levels are physical (stock 1.6
  WOT ≈ 87 dB SPL ≈ 0.023 FS) — that reads QUIET vs old builds. Expect
  "everything is silent" reactions; mic slider compensates. Idle/WOT
  dynamic range is real now (~25 dB).
- **Engine-off duct blow-up** (UI screenshot, 25 MPa = clamp ceiling):
  caused by no ring blowby (parked cylinder stayed pressurized forever) +
  ram-column 500 m/s cap ignoring valve choking. Both fixed physically
  (`be97a2a`). Symptom of the class: "sound with engine off" = something
  wedged against solver clamps.
- **Diesel idle 800↔1400 sine**: integral-dominant governor limit cycle +
  petrol-style +55% cold fast-idle on a diesel. Fixed: +12% diesel cold
  raise, P-dominant gains (`fd0241c`).
- **Muffler packing magnitude**: per-substep losses compound over ~180
  substeps of chamber dwell; 0.06 annihilated the output (1e-5 survival).
  Size per-substep losses from DWELL TIME, not gut feeling.
- All-rungs-stacked listening: when several rungs land between listens,
  the user cannot attribute regressions and trust erodes — same lesson as
  fable-fuckup §3, still true even with per-rung commits.

## Process rules in force (user-amended)

Thumbs-down handling: retune if a retune is safe → different approach if
one exists → full revert ONLY if the mechanism is unnecessary. Checkpoint
before first edit; one audible change per commit; renders per rung; only
the user's ears approve. The user tests in the LIVE UI, cold starts, real
driving — offline prewarmed renders miss cold/transient behaviour (three
bugs were invisible in renders: cold rich crackle, idle limit cycle,
engine-off blow-up).
