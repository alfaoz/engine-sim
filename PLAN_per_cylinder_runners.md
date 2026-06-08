# Plan (deferred): per-cylinder 1D intake runners for real ram tuning

## Why
The dyno torque-curve *shape* is currently set almost entirely by friction +
valve-area choke, so every engine looks similar (flat-ish torque, gentle
decline). Real engines get an engine-specific torque **peak** from intake
**ram / Helmholtz tuning** — the intake pressure wave launched by each valve
event reflects off the plenum/airbox and arrives back at the valve in time to
ram extra charge in, at an rpm set by the runner length. That peak is what
makes a 911 flat-6 peak torque ~5000 rpm and a truck diesel ~1600.

## What was already tried (and why it failed) — context for next time
- **Phase 1 (DONE, kept):** intake valve area scaled with redline
  (`engine_sim/sim.py` `_build`, `P_AIN`/`P_AEX = 0.135 * Apist * sqrt(redline/6000)`).
  Gives high-rpm VE rolloff; rolloff rpm follows stroke geometry. Differentiates
  diesels (peak low, roll off) from revvers (hold flat, power to redline). No
  power penalty. This is the current state.
- **Phase 2 attempt (REVERTED):** routed breathing through a single *shared* 1D
  runner (cylinders draw from the runner head cell; runner mouth open to the
  plenum). Result: the one lumped runner can't both supply bulk airflow AND
  sustain strong ram waves — it **choked power** (~+0.7 s on 0-100) and produced
  **no visible VE peak**. Reverted to: breathing from the plenum (power-safe),
  runner is a SOUND-only model excited by the real valve draw.

## The deferred design: one 1D runner PER cylinder
Give each cylinder its own short 1D intake runner (Euler pipe), all opening into
a shared plenum at their mouths; the plenum is fed by the throttle from the
boost/airbox reservoir (unchanged bulk-flow path = power stays safe).

- State: `rho_in[c]`, `mom_in[c]`, `Ene_in[c]` become `[ncyl][Ni]` (or a flat
  `[ncyl*Ni]`). Each runner is short (per-cylinder length, e.g. 0.2–0.5 m).
- Each cylinder breathes from ITS runner head cell (carries that runner's ram
  wave); runner mouth open BC = plenum pressure (MAP). Because each runner only
  serves one cylinder, the wave is strong (high velocity in a small runner) and
  the plenum still buffers bulk flow → ram peak without choking.
- Ram peak rpm set per engine by runner length (`intake_length` already exists
  in `EngineConfig`; make it meaningful and per-preset: long runner = low-rpm
  torque, short = high-rpm power). Variable-length/“tuned” intakes could later
  switch length vs rpm.
- Reuse the generic `gas_step(... p_open, T_open ...)` already in `core.py`
  (it takes an arbitrary open-end reservoir pressure — that's the plenum).
- Induction sound: sum/most-pulsed runner mouth (or a shared plenum tap) → the
  induction note; also where the turbo compressor whistle would live.

## Validation gates (must all pass before keeping)
1. `accel.py` 0-100 times unchanged vs current Phase-1 build (no power choke).
2. Steady-state VE vs rpm shows a real PEAK whose rpm tracks runner length
   (test 911 short runner peaks high, truck long runner peaks low).
3. `test_physics.py` ALL OK, everything finite.
4. Real-time: per-cylinder runners add `ncyl` extra 1D solves — re-run the
   512-sample block benchmark on the heaviest presets (V12, bus i6); must stay
   under the 10.6 ms budget. If tight, use very short runners / few cells, or
   only enable for <=8 cylinders.

## CPU note
The single shared-runner cost was ~negligible (Ni≈40, 2 substeps). `ncyl` runners
multiply that by cylinder count; a V12 = 12× → must benchmark. Likely fine with
small Ni (~24) but verify.

## Related
- Builds on the SOUND-only runner already in `core.py` (search `air_draw`,
  `rho_in`, `P_IN_NCELLS`).
- Would also make the turbo **compressor whistle** and **BOV** emergent (they
  radiate through the intake), which is currently absent.
