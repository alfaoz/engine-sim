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

## 2026-06-10 session 2 (rungs 10+): what landed, what's queued

Landed (each its own commit, gates green, USER EARS PENDING on all):
- `0ec9780` idle governor pole placement (SIMC from the real plant) —
  fixes 2JZ ±135 rpm hunt; NOT a sound change but un-contaminates idle
  listening. The old inertia-only schedule had ki ~6x too hot.
- `3d76e4c` fuel meter (P_FUELUSED -> L/h + L/100km UI readouts + trace).
- `73effa8` **per-bank tailpipe radiation** — each bank's d(mdot)/dt
  radiates from its own tailpipe position (dual exits 0.8 m apart, own
  direct + ground-image rays, shared multi-tap ring). Single-bank presets
  bit-equal to old output; V8/V12/W16 are the A/B. This is the first
  attack on the "one coherent point source" tube hypothesis.
- `ca61e65` knock detonation physical (end gas burns at sqrt(gRT)/bore
  rate + 3x Woschni on knocked cycles). 95 RON untouched / 50 RON knocks
  + overheats / 5 RON can't run. No audio layer added; flows through dq.

User verdicts (first listen, after rungs 10-13): per-bank radiation OK
(W16 high-pitch pre-existing, not worsened); induction "a tad roboticy
but ok" (suspect list, not convicted); "city 1.2 sounds racey even
with muffler" -> drove the next two rungs.

Landed since (one commit each, gates green, EARS PENDING):
- `155875a` pre-ignition wrap bug (user's 1-RON-at-140km/h find).
- `536bbfe` hot-surface ignition >=280 C wall (user's 410-C-metal find).
- `bb12c13` crackle default 0.
- `cd4a112` visco-thermal sqrt(f) wall loss REPLACES the flat exhaust
  damp (flat 0.0015 was backwards: Q~4 at the fundamental, Q~100 mids).
  Idle ~+5 dB livelier broadband -- needs the muffler rung with it.
- `45f92eb` staged two-chamber silencer (1:1.618, pass bands never
  align): city 1.2 idle midband -14 dB, purr intact; V8 WOT ~92 dB SPL
  with more rumble. LESSONS: transitions sized in cells (sub-cell cones
  = p*dA impulses, clipped the V8), slope-limit 1.6x/cell, expansion
  ratio capped 20:1 buildable (45:1 blew the moped tail cone at WOT),
  muffler floor scales with engine, packing fills the shell span
  positionally (area rule left the neck as a high-Q mirror cavity).

Second listen verdicts: city 1.2 idle GOOD; overrun/rev-release still
racey -> diagnosed NOT exhaust: DFCO pulled MAP to 3 kPa and every EVO
slammed atmosphere into the vacuum (overrun was LOUDER than firing,
rms 0.049 vs 0.019). Muffler builder UX: user read 0 as "no muffler"
(it means auto box) -> relabeled.

Landed since:
- `cc4e836` decel dashpot: DFCO holds idle-like MAP (~20 kPa) via the
  governor's own trim * sqrt(rpm/idle). Overrun >1.2 kHz hash -4..-6 dB,
  decay is burble not vacuum slam. EARS PENDING.
- `291be11` launch clutch slips to hold rpm (TCU-style) + 2 s post-start
  DFCO inhibit. Fixes the 1.0L hatch knife-edge (DNF'd accel.py at
  baseline!); hatch min-rpm 670 vs 230, all accel times within noise.
  NOTE: first idle settle is ~2 s slower now (the accidental post-start
  fuel cut used to brake the overshoot); steady state unchanged-good
  (5.0L ±15, V12 ±9). idle_test's 4 s window reads the tail as ripple.

Third round (user: release at 3200/2500 still racey; higher rpm OK):
- Measured truth: overrun was AS LOUD as fired at same rpm; energy at
  the gas-exchange pulse rate. TWO wrong things found:
  1. MAP-hold dashpot was over-derived (ECUs hold the idle VALVE; real
     overrun vacuum deepens with rpm) -> `78c4112` holds idle trim
     instead: MAP 5-16 kPa falling, braking kept, slam softened.
  2. Stock cars had NO CAT -- the main HF attenuator of any production
     exhaust -> `2c511c0` cat brick: 5x face swell + LINEAR Poiseuille
     channel damping (32 nu/d^2, 400 cpsi, no free constants). LINEAR
     loss finally bites at small signal where Darcy can't. Front/rear
     silencers also separated (real layout). Release-at-3200 now
     QUIETER+darker than fired hold; V8 rumble kept; accel identical.
- Dead end logged: splitting the boxes for "drone" (quarter-wave lock)
  did NOT change the release bands -- the ring was gas-exchange pulse
  rate, not the pipe column. Don't re-chase column drone for overrun.

Fourth round — THE REPRODUCTION (and two process failures):
- Sound-lab Muffler checkbox had been wired to the dead P_MUFFLP filter
  since the geometry silencer landed: the user's A/B instrument was
  disconnected and nobody noticed. Fixed `45db16a`: toggle now swaps
  real geometry live (ON->OFF measured +21 dB). Induction checkbox A/B
  by user: raceyness NOT induction. Clatter is 0 on 4-cyl petrol.
- ALL my overrun probes were bench-neutral coasts (rpm falls through
  the band in <1 s). Driving IN GEAR (the user's condition): sustained
  overrun at 3300 rpm measures +10 dB LOUDER than WOT power at the
  same revs (rms 0.056 vs 0.017), brighter in every band. Real cars
  are 15-20 dB QUIETER on overrun. The symptom was always reproducible
  -- in the right conditions. BENCH PROBES DO NOT COUNT FOR OVERRUN.
- Exhaust sources were still once-per-sample impulses (the intake buzz
  bug, never fixed on the exhaust side) -> spread `6ce65c1`; small.

DIAGNOSIS standing: at EVO on deep-vacuum overrun the pipe (1 atm)
slams ~0.3 g into the ~8 kPa cylinder, the piston pushes it back out
-- a ±0.3 g/cycle oscillating pump at the firing rate, injected
DIRECTLY into the shared 1D pipe cell. A real engine has the exhaust
PORT + PRIMARY RUNNER volume (~0.3-0.5 L/cyl) between valve and
collector that low-passes this slosh; we have zero buffer. Candidate
fix = lumped per-cylinder exhaust port/runner (the deferred
per-cylinder-primaries plan in cheap 1-state form, like intake
RAM_MODE 1). Re-voices every preset (it is the header) -> goes in
ALONE, user listens.

Fifth round — the slosh band defeated four geometric attacks:
- Port buffer built (`27ad510`, default OFF): failed its gate — slosh
  fundamental (~110 Hz) is BELOW the V/(A*c) corner (~240 Hz); it
  deadened FIRED blowdown instead. Box separation, packing-off, and
  full-length boxes: 80-200 Hz overrun band pinned at 42-44 dB in
  every config INCLUDING MUFFLER OFF. Mass conservation sets it: EVO
  backflow pumps ±0.2 g/cycle and a <=2 m pipe is transparent at
  110 Hz. The model is internally consistent (~90 dB SPL at 2 m from
  that pump). A real car's overrun is quieter mainly via (a) higher
  overrun MAP (~15-20 kPa real vs our 5-8 — between my two dashpot
  attempts; no derivable law found yet, deferred) and (b) softer slam
  harmonics. STOP attacking this with in-pipe geometry.
- Long-box rung kept (`27ad510`): TL ~ sin^2(kL); two-box staging only
  when both >=0.35 m, else ONE real muffler. Re-voices V8s + city.

USER REPORT (live, post-cat): muffled V8 "really robotic".
Hypothesis: the cat stripped the 3-8 kHz air (-9 dB) that masked the
STATIC 4-tap comb (2 pipes x direct+ground-image, all fixed delays);
a bare harmonic stack + fixed comb = robotic. TEST FOR USER: drag the
mic-distance slider while listening — the image delay (and comb
spacing) changes with distance; if the robotic character shifts
pitch/depth with the slider, the static comb is convicted. Fix
direction then: diffuse/scattered ground image (asphalt + underbody
scatter, not a perfect mirror), NOT more pipe work.

Queued next:
1. USER: mic-distance comb test on the muffled V8 (above) + verdict on
   the long-box V8 voice (one real muffler now, was two stubs).
2. If comb convicted: diffuse ground-image reflection (physical
   scattering) as the next single rung.
2. Muffler shell radiation (mass-law transmission from chamber cells) —
   derived replacement for the rejected synthetic body resonator.
3. Per-cylinder exhaust primaries -> collector (full header geometry) —
   last, alone, behind a toggle, V12 bench gate.
4. Open: W16 high-pitched (pre-existing; suspect: 4-way cell split /
   coarse grid); induction "roboticy" (constant-area duct, fixed LPF).

## Process rules in force (user-amended)

Thumbs-down handling: retune if a retune is safe → different approach if
one exists → full revert ONLY if the mechanism is unnecessary. Checkpoint
before first edit; one audible change per commit; renders per rung; only
the user's ears approve. The user tests in the LIVE UI, cold starts, real
driving — offline prewarmed renders miss cold/transient behaviour (three
bugs were invisible in renders: cold rich crackle, idle limit cycle,
engine-off blow-up).
