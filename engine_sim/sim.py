"""Orchestration: turn an EngineConfig into the flat parameter/state arrays the
numba core consumes, drive it from a real-time audio callback, and expose live
controls + telemetry to the UI."""
import math
import threading
import numpy as np

from . import core
from .core import P_NPARAMS, simulate_block
from .presets import EngineConfig, VehicleConfig, get_vehicle

GRAV = 9.81
RHO_AIR = 1.20
TWO_PI_60 = 2.0 * math.pi / 60.0   # rpm -> rad/s
TRACTION_MU = 1.00                 # tyre grip limit (drive force <= mu*m*g)
DRIVELINE_FN = 18.0                # driveline torsional natural freq (Hz)
DRIVELINE_ZETA = 0.7               # driveline damping ratio

R_AIR = 287.0
GAMMA_CYL = 1.34
GAMMA_EX = 1.33
PATM = 101325.0
TATM = 300.0
TINT = 315.0

SAMPLE_RATE = 48000
BLOCK = 512
MIC_R = 2.0            # listener distance from the tailpipe/engine (m)
P_FULLSCALE = 20.0     # digital full scale = 20 Pa = 120 dB SPL (pain threshold)
H_SRC = 0.3            # tailpipe height over the ground (m)
H_EAR = 1.5            # listener ear height (m)
C_AIR = 343.0          # ambient sound speed for the ground-path delay (m/s)
N_SCOPE = 720          # cylinder-pressure scope resolution (per cycle)
TAMB = 293.0           # ambient air / cold-soak temperature (K, ~20 C)
GOV_TF = 0.12          # idle-governor rpm measurement filter time constant (s)
T_THERMOSTAT = 361.0   # thermostat opening temperature (K, ~88 C)

# Per-cylinder intake-runner ram model:
#   0 = off (breathe straight from the plenum / MAP -- the Phase-1 baseline)
#   1 = lumped inertial column (1 state/cyl, no PDE, ~free CPU)
#   2 = 1D Euler wave runners (Nr cells/cyl; ram emerges from wave reflection)
RAM_MODE = 1
RUN_NCELLS = 28      # per-cylinder runner cells for mode 2


class EngineSim:
    def __init__(self, cfg: EngineConfig):
        self.lock = threading.Lock()
        self.stream = None
        self.device = None          # output device index (None = system default)
        self.volume = 0.6
        self.pedal = 0.0            # accelerator pedal 0..1
        self.load = 0.0             # extra accessory load (Nm)
        self.rpm = 0.0
        self.rpm_prev = 0.0         # previous rpm (idle governor derivative)
        self.rpm_rate = 0.0         # filtered rpm rate, rpm/s
        self.rpm_gov = 0.0          # low-passed idle speed the governor regulates
        self._rpm_gov_prev = 0.0    # for the governor's (filtered) rate term
        self.torque = 0.0
        self.audio_tap = np.zeros(2048, dtype=np.float64)   # ring of last output
        self._tap_pos = 0
        self._cb_buf = np.zeros(BLOCK, dtype=np.float64)     # reused audio out buffer

        # vehicle / control state
        self.vehicle = get_vehicle("Bench / neutral (free rev)")
        self.gear = 0              # 0 = neutral
        self.state = "off"         # off | cranking | running
        self.v = 0.0               # vehicle speed (m/s)
        self.brake = 0.0           # brake pedal 0..1
        self.idle_i = 0.0          # idle governor integrator
        self.shift_timer = 0.0     # clutch-out window during a shift
        self.crank_timer = 0.0
        self.crank_assist = 0.0
        self.run_time = 0.0        # seconds since the engine caught
        self.clutch_cap = 200.0
        self.boost = 0.0           # actual turbo boost (Pa), spools with lag
        self.thr_smooth = 0.0      # slewed throttle (intake/fuel dynamics)
        self.clutch_engage = 0.0   # 0..1 automatic-clutch engagement
        self.base_inertia = 0.2    # engine-only rotating inertia
        self.grade = 0.0           # road gradient (rad); driver/scenario input
        self.knock_retard = 0.0    # ECU knock-retard (deg of timing pulled)
        self.knock = 0.0           # telemetry: knock intensity this block
        self.fuel_rate = 0.0       # smoothed fuel mass flow (kg/s)
        self.fuel_total_kg = 0.0   # fuel used since this engine was loaded (kg)
        self.wheel_rpm = 0.0
        self.traction_control = True   # cut power on wheelspin (defeatable)
        self.tc_cut = 0.0          # current TC power cut (0..1), telemetry
        self.tyre_slip = True      # slip-ratio tyre model; off = rigid grip cap
        # audio source mix (for A/B-ing real vs coloration)
        # the duct's valve-draw source is spread over the CFL substeps (it was a
        # per-sample impulse into one cell, which buzzed); a real airbox +
        # throttled boundary would still be more physical than the reservoir end.
        self.mix_induction = 0.15  # induction mix ON by default (kept modest:
        #                            it runs hot/clips at higher levels)
        # synthetic body resonator OFF by default: it is a high-Q tube-flavoured
        # coloration, redundant (and "tubey") now that the real chamber
        # geometry, radiation tap and ground image supply the body physically.
        self.mix_body = 0.0
        self.mix_sat = 1.0
        # new audio sources (toggleable so issues can be A/B'd / soloed).
        # clatter ON by default: it's the engine's raw MECHANICAL noise (the
        # combustion pressure-rise ringing the block/valvetrain), the presence the
        # bare exhaust lacks -- without it idle sounds like just a resonant pipe
        # ("a balloon"). knock (a fault ping) stays off; muffler on.
        self.mix_clatter = True    # combustion/mechanical clatter (engine body noise)
        self.mix_knock = False     # spark-knock "ping" (petrol)
        self.mix_muffler = True    # muffler HF transmission loss
        self.mic_r = MIC_R         # listener distance (m); the UI can move it
        self._clatter_on = 0.0
        self._knock_on = 0.0
        self._muff_on = 0.0

        self.load_config(cfg)

    @property
    def throttle(self):
        return self.pedal

    # ------------------------------------------------------------------ build
    def load_config(self, cfg: EngineConfig):
        with self.lock:
            self.cfg = cfg
            self.fuel_rate = 0.0
            self.fuel_total_kg = 0.0
            self._build()

    @property
    def fuel_lph(self):
        """Smoothed fuel consumption, litres/hour (pump density: petrol
        ~0.74 kg/L, diesel ~0.84 kg/L)."""
        rho = 0.84 if self.cfg.diesel else 0.74
        return self.fuel_rate / rho * 3600.0

    def _build(self):
        cfg = self.cfg
        P = np.zeros(P_NPARAMS, dtype=np.float64)
        ncyl = cfg.n_cylinders
        cyc = 4.0 * math.pi if cfg.stroke_cycle == 4 else 2.0 * math.pi
        cyc_deg = 720.0 if cfg.stroke_cycle == 4 else 360.0
        self.cyc_deg = cyc_deg

        Apist = math.pi / 4.0 * cfg.bore ** 2
        Vdisp = Apist * cfg.stroke
        Vclear = Vdisp / (cfg.compression_ratio - 1.0)

        P[core.P_NCYL] = ncyl
        P[core.P_CYCLE] = cyc
        P[core.P_R] = cfg.stroke / 2.0
        P[core.P_L] = cfg.conrod
        P[core.P_APIST] = Apist
        P[core.P_VCLEAR] = Vclear
        P[core.P_VDISP] = Vdisp
        P[core.P_GAMMA] = GAMMA_CYL
        P[core.P_RGAS] = R_AIR
        P[core.P_CV] = R_AIR / (GAMMA_CYL - 1.0)
        P[core.P_LHV] = 43.0e6 if cfg.diesel else 44.0e6
        P[core.P_AFR] = 14.7
        # indicated->brake losses we don't fully model (blowby, incomplete burn) --
        # calibrated so BMEP lands in the real range (NA ~10-12 bar). Raised from
        # 0.78 now that Woschni models the in-cylinder heat loss EXPLICITLY, so this
        # catch-all no longer has to hide as much heat-transfer approximation.
        P[core.P_COMBEFF] = 0.84 * cfg.power_tune
        P[core.P_IGN] = math.radians(cyc_deg - cfg.ignition_btdc)
        P[core.P_BURN] = math.radians(cfg.burn_duration)
        P[core.P_WIEBE_A] = 5.0
        P[core.P_WIEBE_M] = 2.0
        P[core.P_TINT] = TINT
        P[core.P_PATM] = PATM
        P[core.P_TATM] = TATM
        P[core.P_THROTTLE] = self.throttle
        P[core.P_IVO] = math.radians(cfg.ivo)
        P[core.P_IVC] = math.radians(cfg.ivc)
        P[core.P_EVO] = math.radians(cfg.evo)
        P[core.P_EVC] = math.radians(cfg.evc)
        port = cfg.stroke_cycle == 2
        if port:
            P[core.P_AIN] = 0.26 * Apist
            P[core.P_AEX] = 0.22 * Apist
        else:
            # 4-stroke valve area scaled to the engine's design speed: a
            # high-revving engine has bigger valves (breathes to its redline),
            # a low-revving diesel has smaller ones (chokes early). Combined with
            # valve-area ~ bore^2 vs displacement ~ bore^2*stroke, this makes the
            # VE roll off -- and therefore peak torque land -- at an engine-
            # specific rpm instead of the same flat curve for everyone.
            rev_scale = (cfg.redline_rpm / 6000.0) ** 0.5
            P[core.P_AIN] = 0.135 * Apist * rev_scale
            P[core.P_AEX] = 0.135 * Apist * rev_scale
        # closed-throttle idle-air leak (fraction of full throttle area). A high-rev
        # engine has oversized valves AND throttle, so a fixed fractional leak feeds
        # too much air to idle low -- scale the leak down by the valve-oversize so
        # the governor keeps authority to pull idle down to target on every engine.
        rev_scale_idle = (cfg.redline_rpm / 6000.0) ** 0.5 if cfg.stroke_cycle == 4 else 1.0
        self.idle_floor = 0.0008 / max(1.0, rev_scale_idle) ** 2
        P[core.P_CD] = 0.70
        P[core.P_INERTIA] = cfg.inertia
        # Friction from FMEP (friction mean effective pressure) x swept volume,
        # so friction torque scales with displacement like a real engine:
        #   T_fric = FMEP * Vd_total / divisor,  FMEP = FMEP0 + slope*omega
        # (~0.6 bar static rising to ~2 bar at speed => ~15-20% of peak torque).
        disp_l = Vdisp * ncyl * 1e3
        self.disp_l = disp_l
        Vd_total = Vdisp * ncyl
        divisor = 4.0 * math.pi if cfg.stroke_cycle == 4 else 2.0 * math.pi
        fmep0 = 52000.0 if cfg.diesel else 44000.0   # diesels rub a bit more
        P[core.P_FRIC0] = fmep0 * Vd_total / divisor
        # Viscous (rpm-dependent) friction. The law is linear in rpm, so at a
        # bike's 14k redline it over-predicts FMEP (~4.4 bar vs a realistic
        # ~2.5) -- that both robs power and dumps huge heat into the oil. Soften
        # it for high-revving engines so FMEP@redline stays ~2.5 bar; engines at
        # or below ~7500 rpm (all the cars/diesels) are unchanged.
        rev_soft = min(1.0, 7500.0 / cfg.redline_rpm)
        P[core.P_FRICW] = 270.0 * Vd_total / divisor * rev_soft
        P[core.P_LOAD] = self.load
        # starter motor torque: sized to crank past peak compression, so it scales
        # with displacement AND compression ratio (a high-CR big-bore single/twin
        # resists cranking far more than its displacement alone implies -- now that
        # cold air has its real gamma~1.4 the compression pressure is honest).
        self.starter_torque = 28.0 + 40.0 * disp_l * (0.55 + 0.05 * cfg.compression_ratio)
        # clutch torque capacity ~ 1.5-2x peak engine torque (disp_l is litres)
        self.clutch_cap = 320.0 * disp_l + 50.0

        # ---- thermal model (3 lumped masses: metal, coolant, oil) ----
        # Heat IN is the real combustion-to-wall loss the core already computes
        # (P_QWALL); heat OUT is the radiator (thermostat- and airflow-gated). The
        # warm-up transient, its time constant, and all cold effects EMERGE. Heat
        # capacities and conductances scale with displacement, so a small engine
        # warms in a minute and a big diesel takes many.
        dl = max(0.2, disp_l)
        self.C_metal = 2600.0 * dl        # J/K  (heads/block; fast)
        self.C_cool = 6000.0 * dl         # J/K  (coolant)
        self.C_oil = 3500.0 * dl          # J/K  (oil; slowest, drives friction)
        # metal->coolant is now firm enough that the coolant actually tracks the
        # metal (the old value left a ~145 s lag so coolant never warmed); the
        # metal still runs hotter than the coolant like a real head.
        self.h_mc = 130.0 * dl            # W/K  metal -> coolant
        self.h_mo = 32.0 * dl             # W/K  metal -> oil
        self.h_rad = 240.0 * dl           # W/K  coolant -> radiator (gated)
        self.h_oa = 6.0 * dl              # W/K  oil -> ambient (minor)
        self.h_oc = 80.0 * dl             # W/K  oil cooler: oil -> coolant
        self.base_fric0 = P[core.P_FRIC0]
        self.base_fricw = P[core.P_FRICW]
        self.base_afr = 14.7
        self.friction_mult = 1.0
        self.idle_target = cfg.idle_rpm   # raised when cold (ECU fast idle)
        self.warm_frac = 0.0
        # cold-soak start (ambient); call prewarm() to begin hot
        self.T_metal = TAMB
        self.T_cool = TAMB
        self.T_oil = TAMB
        self.coolant_C = TAMB - 273.15

        # ---- intake manifold (filling/emptying plenum) ----
        # plenum volume ~ one engine displacement; throttle bore sized so WOT
        # leaves only a small pumping (manifold) depression near atmospheric.
        self.man_vol = max(2.0e-4, Vd_total)
        # full-open throttle flow area, scaled to the engine's intake valve area
        self.thr_area_max = 0.55 * P[core.P_AIN] * ncyl
        P[core.P_MANVOL] = self.man_vol
        P[core.P_TMAN] = TINT
        P[core.P_PBOOST] = PATM
        P[core.P_THRAREA] = self.thr_area_max
        # ---- driveline / vehicle (filled in live each block) ----
        P[core.P_RATIO] = 0.0
        P[core.P_RW] = 0.3
        P[core.P_VMASS] = 0.0
        P[core.P_CRR] = 0.0
        P[core.P_CDA] = 0.0
        P[core.P_BRAKEF] = 0.0
        P[core.P_TCAP] = 0.0
        P[core.P_KDL] = 1.0e4
        P[core.P_CDL] = 50.0
        P[core.P_MU] = TRACTION_MU
        P[core.P_GRAVITY] = GRAV
        P[core.P_GAMMAEX] = GAMMA_EX
        P[core.P_REX] = R_AIR
        P[core.P_CVEX] = R_AIR / (GAMMA_EX - 1.0)

        # ---- exhaust bank count (decided here so the grid can be sized for it) ----
        nbanks = int(cfg.exhaust_banks)
        if nbanks < 1:
            nm = cfg.name.lower()
            if "w16" in nm or "w12" in nm:
                nbanks = 4
            elif any(k in nm for k in ("v4", "v6", "v8", "v10", "v12", "v16",
                                       "v-twin", "l-twin", "flat", "boxer")):
                nbanks = 2
            else:
                nbanks = 1
        nbanks = max(1, min(nbanks, ncyl))
        self.nbanks = nbanks
        P[core.P_NBANKS] = float(nbanks)

        # exhaust grid. Each bank runs its OWN 1D solver, so to stay real-time we
        # split a fixed total cell budget across the banks (a 2-bank V8 uses ~half
        # the cells per pipe). Total work then stays ~that of a single pipe, which
        # is what keeps multi-bank engines from underrunning ("periodic cutout").
        # cell budget: the 2nd-order MUSCL solver resolves the wave action at a
        # much coarser grid than the old first-order scheme, so a bigger dx keeps
        # the fidelity while cutting compute. Cost scales ~N^2 (more cells AND more
        # CFL substeps), so this is what keeps even long-pipe single-bank engines
        # (e.g. the big tractor diesel) comfortably inside the real-time budget.
        base_cells = min(240, max(80, cfg.pipe_length / 0.012))
        N = int(np.clip(base_cells / nbanks, 56, 240))
        dx = cfg.pipe_length / N
        P[core.P_NCELLS] = N
        P[core.P_DX] = dx
        P[core.P_PIPELEN] = cfg.pipe_length
        pipe_area = math.pi / 4.0 * cfg.pipe_diameter ** 2
        P[core.P_PIPEAREA] = pipe_area

        # ---- muffler chamber acoustics (geometry-derived, lumped) ----
        # The silencer behaves as a Helmholtz resonator: the tailpipe "neck" of
        # air bouncing on the chamber's gas spring gives the exhaust its low
        # boom/drone, f_H = c/2pi * sqrt(A_neck/(V_chamber*L_neck)). The chamber
        # also kills high frequencies (transmission loss) -- a bigger box is
        # darker. Both come from the volume, not a tuned constant.
        # The silencer is now REAL geometry: the area profile A(x) built below
        # carries an expansion chamber whose volume = the muffler volume, so the
        # high-frequency transmission loss and the level drop (a straight pipe is
        # several times louder) EMERGE from the quasi-1D wave solver. The old
        # muffler one-pole LP (P_MUFFLP) is therefore redundant and disabled. A
        # light per-engine body/Helmholtz resonance (P_BODYFREQ) is kept as voicing
        # so the low boom stays present even where the single chamber notches it.
        self._straight = cfg.muffler_volume < 0.0
        if cfg.stroke_cycle == 2:
            # 2-stroke: the tuned expansion chamber (built in the area profile) IS
            # the exhaust -- it's a resonator, not an attenuating silencer, so no
            # muffler loudness compensation and no synthetic Helmholtz body.
            self._straight = False
            self._muff_vol = 0.0
            self._A_ch = pipe_area
            P[core.P_BODYFREQ] = 0.0
        elif self._straight:
            self._muff_vol = 0.0                         # open/straight pipe
            self._A_ch = pipe_area                       # no chamber
            P[core.P_BODYFREQ] = 0.0                     # raw quarter-wave only
        else:
            # default box ~8x displacement; the floor scales with the engine
            # (a moped carries ~a 1 L can, not a 3 L sedan box -- the flat
            # 3 L floor put a 43:1 expansion on a 2 cm pipe and blew up the
            # 50cc's tail cone at WOT)
            self._muff_vol = cfg.muffler_volume if cfg.muffler_volume > 0.0 \
                else max(1.2e-3, 8.0 * Vd_total)
            # expansion-chamber cross-section (the chamber zone spans ~26% of
            # the pipe length incl. cones; see _exhaust_area_profile for the
            # two-chamber split); the solver makes the silencing/TL from
            # this, no tuned filter.
            L_ch = 0.26 * cfg.pipe_length
            # expansion ratio capped at a BUILDABLE 20:1 (4.5x diameter --
            # real shells; 45:1 was a numerical hand grenade on small pipes)
            self._A_ch = float(np.clip(self._muff_vol / max(L_ch, 1e-3),
                                       3.0 * pipe_area, 20.0 * pipe_area))
            c_ex = math.sqrt(GAMMA_EX * R_AIR * 700.0)   # hot-gas sound speed
            L_neck = 0.18                                # effective tailpipe neck
            f_H = c_ex / (2.0 * math.pi) * math.sqrt(
                pipe_area / (self._muff_vol * L_neck))
            P[core.P_BODYFREQ] = float(np.clip(f_H, 45.0, 220.0))
        # loudness compensation: a bigger chamber attenuates more, so without this
        # a big-muffler engine would be far quieter than a small one. Normalising
        # by the expansion ratio keeps perceived loudness consistent across presets
        # while PRESERVING the chamber's spectral shaping (the TL notches).
        self._muff_ratio = self._A_ch / pipe_area
        self._turbo_muffles = cfg.turbo_boost > 0.0
        P[core.P_MUFFLP] = 0.0                           # HF loss now from geometry
        self._muff_on = 0.0

        # 1D intake duct (runner + airbox), coarser grid than exhaust to bound
        # CPU; 2-strokes use crankcase scavenge, so no duct (Ni=0).
        if cfg.stroke_cycle == 4:
            in_len = cfg.intake_length
            Ni = int(min(160, max(40, in_len / 0.012)))
            dx_in = in_len / Ni
            # duct area scales with the engine's intake-valve area (a big engine
            # has a big intake) so induction gas velocity -- and therefore the
            # induction sound level -- is roughly engine-size independent.
            in_area = max(math.pi / 4.0 * cfg.intake_diameter ** 2,
                          1.8 * P[core.P_AIN] * ncyl)
            # per-cylinder intake runner (inertia ram). Geometry only: a primary
            # runner cross-section ~ 0.40*bore, length = the preset's runner
            # length. The ram-tuning rpm then EMERGES from this geometry; it is
            # not a fitted curve.
            run_dia = 0.40 * cfg.bore
            P[core.P_RUN_AREA] = math.pi / 4.0 * run_dia ** 2
            # Runner LENGTH sized to the engine's design speed, the way real
            # manufacturers do it: a high-revving engine uses SHORT runners so the
            # ram peak lands high; a low-speed / diesel engine uses LONG runners so
            # it peaks low. One physical design rule (length ~ 1/redline, longer
            # for diesels), not a per-engine fit -- the peak rpm then EMERGES.
            run_len = 0.30 * (6500.0 / cfg.redline_rpm)
            if cfg.diesel:
                run_len *= 1.9            # diesels: long intake tracts, low peak
            P[core.P_RUN_LEN] = min(0.85, max(0.12, run_len))
            P[core.P_RAM_MODE] = float(RAM_MODE)
            P[core.P_RUN_NCELLS] = float(RUN_NCELLS)
        else:
            Ni, dx_in, in_area, in_len = 0, 1.0, 1.0, 0.0
            P[core.P_RUN_AREA] = 0.0
            P[core.P_RUN_LEN] = 0.0
            P[core.P_RAM_MODE] = 0.0
            P[core.P_RUN_NCELLS] = 0.0
        self.Ni = Ni
        self.Nr = RUN_NCELLS if (cfg.stroke_cycle == 4 and RAM_MODE == 2) else 1
        P[core.P_IN_NCELLS] = Ni
        P[core.P_IN_DX] = dx_in
        P[core.P_IN_AREA] = in_area
        P[core.P_INGAIN] = self.mix_induction   # induction mix into the output
        P[core.P_BODYGAIN] = self.mix_body      # body/thump resonator mix
        P[core.P_SAT] = self.mix_sat            # tanh saturation on/off

        # ---- turbocharger sizing (boost emerges from the shaft in the core) ----
        turbo = cfg.turbo_boost > 0.0 and cfg.stroke_cycle == 4
        P[core.P_TURBO] = 1.0 if turbo else 0.0
        if turbo:
            # shaft inertia scales with engine size -> bigger turbos spool slower
            P[core.P_TRB_INERTIA] = 2.0e-5 + 3.0e-5 * disp_l
            P[core.P_TRB_RIMP] = 0.026          # compressor impeller radius (m)
            P[core.P_TRB_ETA] = 0.62 if cfg.diesel else 0.55  # turbine harvest
            P[core.P_WGATE_PR] = (PATM + cfg.turbo_boost) / PATM
            P[core.P_ICOOL] = 0.65              # intercooler effectiveness
            P[core.P_TRB_CELL] = float(int(0.18 * N))   # turbine taps here
        else:
            P[core.P_TRB_INERTIA] = 1.0
            P[core.P_TRB_RIMP] = 0.0
            P[core.P_TRB_ETA] = 0.0
            P[core.P_WGATE_PR] = 1.0
            P[core.P_ICOOL] = 0.0
            P[core.P_TRB_CELL] = 0.0
        P[core.P_DT] = 1.0 / SAMPLE_RATE
        P[core.P_SR] = SAMPLE_RATE
        # P_HT is now the Woschni heat-transfer gain (dimensionless); the
        # correlation supplies the bore/pressure/velocity/temperature dependence.
        P[core.P_HT] = 1.0
        if cfg.diesel:
            P[core.P_HT] = 1.15      # higher wall loss (cooler walls, CI mixing)
        P[core.P_WALLT] = self.T_metal      # wall temp tracks the metal (thermal model)
        P[core.P_DIESEL] = 1.0 if cfg.diesel else 0.0
        # exhaust wall loss is now the visco-thermal model (below); the flat
        # per-substep damp is a manual extra, default OFF (UI "exh damping")
        P[core.P_DAMP] = 0.0
        # ---- Kirchhoff visco-thermal boundary-layer loss (exhaust) ----
        # alpha(f) = sqrt(pi f nu)/(r c) * (1 + (g-1)/sqrt(Pr)) Np/m at hot
        # exhaust-gas conditions (Sutherland viscosity at ~700 K). The 1D
        # solver realizes it as a one-pole high-shelf momentum loss with
        # per-cell rate K/sqrt(A); K and the corner fc are set by matching
        # the shelf to alpha(f)*c at the pipe's own quarter-wave fundamental
        # f0 = c/4L and at 4 kHz (algebra below, no free constants):
        #   loss(f) = k f^2/(f^2+fc^2);  loss(f0) = a0,  loss(F) = a0*sqrt(F/f0)
        #   => fc^2 = f0^2 F^2 (R-1)/(F^2 - R f0^2),  R = sqrt(F/f0)
        T_vt = 700.0
        mu_vt = (1.716e-5 * (T_vt / 273.15) ** 1.5
                 * (273.15 + 110.4) / (T_vt + 110.4))
        c_vt = math.sqrt(GAMMA_EX * R_AIR * T_vt)
        nu_vt = mu_vt * R_AIR * T_vt / PATM            # mu / rho
        r_vt = math.sqrt(pipe_area / math.pi)
        f0_vt = c_vt / (4.0 * cfg.pipe_length)
        F_vt = 4000.0
        a0_vt = (math.sqrt(math.pi * f0_vt * nu_vt) / (r_vt * c_vt)
                 * (1.0 + (GAMMA_EX - 1.0) / math.sqrt(0.72))) * c_vt
        R_vt = math.sqrt(F_vt / f0_vt)
        fc2_vt = (f0_vt ** 2 * F_vt ** 2 * (R_vt - 1.0)
                  / (F_vt ** 2 - R_vt * f0_vt ** 2))
        k_vt = a0_vt * (f0_vt ** 2 + fc2_vt) / f0_vt ** 2
        P[core.P_VTK] = k_vt * math.sqrt(pipe_area)    # rate = K/sqrt(A_cell)
        P[core.P_VTW] = 2.0 * math.pi * math.sqrt(fc2_vt)
        P[core.P_OUTGAIN] = self.out_gain()
        P[core.P_REDLINE] = cfg.redline_rpm
        P[core.P_RUNNING] = 0.0     # engine starts off — user cranks it
        P[core.P_STARTER] = 0.0
        P[core.P_LPF] = 0.55        # ~6.5 kHz one-pole output lowpass
        P[core.P_NOISE] = 0.06      # combustion cycle-to-cycle roughness
        P[core.P_POP] = 0.0         # unused (pops are now pipe chemistry)
        P[core.P_AFTERFIRE] = 0.0   # unburnt-fuel survival fraction (set live)
        # unburnt-fuel survival fraction. Default 0: a stock catted exhaust
        # oxidizes effectively all surplus HC -- no crackle unless the user
        # dials it in. The UI "crackle" slider sets it (0.3+ = race system).
        self.backfire = 0.0
        # fuelling / ignition / mechanical noise (set live each block)
        P[core.P_FUELCUT] = 0.0     # DFCO injection cut
        P[core.P_PHI] = 1.0         # commanded equivalence ratio (petrol)
        # Mechanical presence through the block. Diesels knock (CI) regardless
        # of size. Petrol clatter belongs to small/characterful engines
        # (singles, twins, triples) and is ZERO on a refined 4+ cylinder
        # petrol (an LFA does not clatter -- the 0.10 floor for cars was
        # tried twice and user-rejected both times). Additionally the level
        # FADES WITH RPM in _control_update: the knocks are an idle/low-rpm
        # character that recedes into the combustion roar as revs climb.
        if cfg.diesel:
            self._clatter_on = 0.5 * float(np.clip(4.0 / ncyl, 0.5, 1.0))
        else:
            self._clatter_on = 0.34 * float(np.clip((4 - ncyl) / 3.0, 0.0, 1.0))
        self._knock_on = 0.0 if cfg.diesel else 0.30
        P[core.P_CLATTER] = self._clatter_on if self.mix_clatter else 0.0
        P[core.P_OCTANE] = cfg.octane
        P[core.P_KNOCKAUD] = self._knock_on if self.mix_knock else 0.0
        P[core.P_KNOCK] = 0.0
        self.knock_retard = 0.0
        # vehicle chassis params (overwritten live in _control_update); bench-safe
        P[core.P_GRADE] = 0.0
        P[core.P_STEER] = 0.0
        P[core.P_WD_DRIVE] = 0.5
        P[core.P_XFER] = 0.0
        P[core.P_JWHEEL] = 1.5
        P[core.P_WHEELBASE] = 0.0
        P[core.P_WD_FRONT] = 0.58
        P[core.P_CGH] = 0.5
        P[core.P_DIFF] = 0.0
        P[core.P_TYRESLIP] = 1.0 if self.tyre_slip else 0.0

        # ---- exhaust banks: assign cylinders (count decided up by the grid) ----
        # split cylinders 1..n evenly across banks (1-4 left, 5-8 right, ...)
        cyl_bank = np.zeros(ncyl, dtype=np.int64)
        for c in range(ncyl):
            cyl_bank[c] = min(nbanks - 1, c * nbanks // ncyl)
        self.cyl_bank = cyl_bank
        # No bank-count trim and no muffler loudness compensation any more:
        # with the radiation tap, two tailpipes really are louder than one and
        # a big silencer really is quieter -- preset loudness is now physical.

        self.P = P
        self.N = N

        # cylinder phase offsets. If the preset gives explicit per-cylinder
        # firing angles (uneven-firing engines), use them verbatim; otherwise
        # space the firing order evenly around the cycle.
        phase = np.zeros(ncyl)
        if cfg.firing_angles and len(cfg.firing_angles) == ncyl:
            for c in range(ncyl):
                phase[c] = (-math.radians(cfg.firing_angles[c])) % cyc
        else:
            step = cyc / ncyl
            order = cfg.firing_order if cfg.firing_order else list(range(1, ncyl + 1))
            for j, cyl in enumerate(order):
                phase[cyl - 1] = (-j * step) % cyc
        self.phase = phase

        # injection cells: spread each bank's cylinders over the head end of
        # THAT bank's pipe
        inj = np.zeros(ncyl, dtype=np.int64)
        spread = cfg.runner_spread
        per_bank = [0] * nbanks
        for c in range(ncyl):
            per_bank[cyl_bank[c]] += 1
        seen = [0] * nbanks
        for c in range(ncyl):
            b = cyl_bank[c]; cnt = per_bank[b]
            frac = 0.0 if cnt <= 1 else (seen[b] / (cnt - 1)) * spread
            inj[c] = int(min(0.4, frac) * N)
            seen[b] += 1
        self.inj = inj

        # state — engine off (not spinning) until the user cranks it.
        # st = [crank_angle, omega_eng, vehicle_v, driveline_windup, manifold_mass]
        self.st = np.zeros(core.S_NSTATE)
        self.st[core.S_MMAN] = PATM * self.man_vol / (R_AIR * TINT)
        self.state = "off"
        self.v = 0.0
        self.idle_i = 0.0
        self.shift_timer = 0.0
        self.crank_timer = 0.0
        self.crank_assist = 0.0
        self.run_time = 0.0
        self.boost = 0.0
        self.thr_smooth = 0.0
        self.clutch_engage = 0.0
        self.fuelcut = False        # DFCO state (hysteresis)
        self.lambda_phase = 0.0     # closed-loop lambda dither phase
        self.base_inertia = cfg.inertia
        Vmid = Vclear + 0.5 * Vdisp
        self.cyl_T = np.full(ncyl, TINT)
        self.cyl_m = np.full(ncyl, PATM * Vmid / (R_AIR * TINT))
        # per-cylinder knock state: [T_ivc, P_ivc, Livengood-Wu integral, knocked]
        # per-cylinder trapped reference + knock state:
        # [T_ivc, P_ivc, Livengood-Wu integral, knocked, V_ivc, spare]
        self.cyl_knk = np.zeros((ncyl, 6))
        # pipe chemistry: per-cylinder [fuel_frac, air_frac, port wall-film kg,
        # fired-this-cycle] + per-bank inventories [unburnt fuel, free-O2 air]
        self.cyl_chem = np.zeros((ncyl, 4))
        self.pipe_chem = np.zeros((nbanks, 2))

        rho0 = PATM / (R_AIR * TATM)
        self.rho = np.full((nbanks, N), rho0)
        self.mom = np.zeros((nbanks, N))
        self.Ene = np.full((nbanks, N), PATM / (GAMMA_EX - 1.0))

        # 1D intake duct gas state (cool air at rest); min length 1 if disabled
        nin = max(1, Ni)
        rho_in0 = PATM / (R_AIR * TINT)
        self.rho_in = np.full(nin, rho_in0)
        self.mom_in = np.zeros(nin)
        self.Ene_in = np.full(nin, PATM / (GAMMA_CYL - 1.0))

        # per-cylinder intake-runner ram state: column mass flow (starts at rest),
        # fresh-charge accumulator
        self.run_mdot = np.zeros(ncyl)
        self.fresh = np.zeros(ncyl)

        # mode-2 per-cylinder 1D runner cells (cool air at rest, at MAP/plenum).
        Nr = self.Nr
        self.rho_r = np.full((ncyl, Nr), rho_in0)
        self.mom_r = np.zeros((ncyl, Nr))
        self.Ene_r = np.full((ncyl, Nr), PATM / (GAMMA_CYL - 1.0))
        self.src_rm = np.zeros((ncyl, Nr))
        self.src_re = np.zeros((ncyl, Nr))

        # ---- quasi-1D cross-section profiles A(x) for the MUSCL-Hancock solver.
        # The exhaust profile (header -> collector -> muffler -> tailpipe) is built
        # by _exhaust_area_profile so silencer transmission loss and tuning emerge
        # from geometry. Intake duct and runners are (for now) constant-area.
        self.ex_area = self._exhaust_area_profile(N, pipe_area)
        self.ex_aface = self._faces(self.ex_area)
        self.ex_wk = np.zeros((15, N))
        self.in_area = np.full(nin, in_area)
        self.in_aface = self._faces(self.in_area)
        self.in_wk = np.zeros((15, nin))
        run_a = P[core.P_RUN_AREA] if P[core.P_RUN_AREA] > 0.0 else 1e-4
        self.run_area_a = np.full(Nr, run_a)
        self.run_aface = self._faces(self.run_area_a)
        self.run_wk = np.zeros((15, Nr))
        # per-pipe open-end boundary filter state (Levine-Schwinger split):
        # [slow part of p - p_open, slow part of rho - rho_open]
        self.bnd_ex = np.zeros((nbanks, 2))
        self.bnd_in = np.zeros(2)
        self.bnd_run = np.zeros((ncyl, 2))
        # absorptive muffler packing: extra per-cell damping ONLY inside the
        # silencer chamber (where A(x) opens past the pipe bore). A real
        # dissipative muffler's glass-pack absorbs ~50-90% of a wave per pass;
        # a purely reflective chamber keeps the coherent ring ("tube"). Not
        # for straight pipes or 2-stroke chambers (those SHOULD ring).
        # per-substep loss sized from dwell: a wave spends ~L_ch/c ~ 1 ms in
        # the chamber (~180 CFL substeps), so 0.006/substep absorbs ~2/3 of a
        # pass -- glass-pack territory. (0.06 annihilated the output: ~1e-5
        # survival per pass.)
        self.pk_ex = np.zeros(N)
        if cfg.stroke_cycle == 4 and not self._straight:
            # packing fills the SHELL span (0.50..0.88+cones), neck included:
            # a real box's internals run wall to wall. The old area-based
            # rule left the inter-chamber neck unpacked -- a high-Q cavity
            # between two near-perfect mirrors.
            xc_pk = (np.arange(N) + 0.5) / N
            self.pk_ex[(xc_pk > 0.50) & (xc_pk < 0.94)] = 0.006
        self.pk_in = np.zeros(max(1, Ni))
        self.pk_run = np.zeros(self.Nr)

        self.scope_p = np.full(N_SCOPE, PATM)
        # output filter state: exhaust DC block (0,1), de-hash LPF (2,3),
        # body/thump resonator (4,5); induction DC block (6,7), de-hash LPF (8,9);
        # peak-limiter envelope (10) + gain (11); combustion-clatter resonators
        # (12,13) and (14,15); knock ping (16,17); muffler chamber LP (18);
        # clatter impact envelope (19); 20 unused (per-bank exhaust mdot
        # state lives in mdot_prev); prev intake mdot for the induction
        # derivative (21); radiation delay-ring write index (22)
        self.filt = np.zeros(24)
        # shared multi-tap radiation delay ring: every tailpipe's direct ray
        # and ground-image ray writes at its path delay (max ~90 samples at
        # the 0.25 m minimum mic distance with the widest pipe offset)
        self.gnd = np.zeros(256)
        # per-bank radiation paths [direct delay, direct gain, image delay,
        # image gain] + per-bank previous exit mdot (derivative state)
        self.rad = np.zeros((nbanks, 4))
        self.mdot_prev = np.zeros(nbanks)
        # visco-thermal shelf state: slow momentum average per exhaust cell
        self.vt_ex = np.zeros((nbanks, N))
        self._apply_radiation()
        self.filt[11] = 1.0          # limiter gain starts at unity
        self.P[core.P_MAP] = PATM

    @staticmethod
    def _faces(area):
        """Face areas (N+1) from cell areas (N): interior faces are the average
        of the two neighbours; the two ends take the end-cell area."""
        N = len(area)
        af = np.empty(N + 1)
        af[0] = area[0]
        af[N] = area[N - 1]
        for k in range(1, N):
            af[k] = 0.5 * (area[k - 1] + area[k])
        return af

    def _exhaust_area_profile(self, N, pipe_area):
        """Cross-section A(x) along the exhaust, head cell (x=0, at the ports) ->
        tailpipe outlet (x=1). The quasi-1D solver turns this geometry into real
        wave action: an expansion chamber reflects and traps energy (the silencer
        boom + high-frequency transmission loss) and the cones tune a 2-stroke.

        Control points (xpos, area) are linearly interpolated -- area transitions
        are physical cones, and the chamber volume is sized to the muffler volume
        so the boom frequency tracks the real box size, no tuned constant."""
        cfg = self.cfg
        L = cfg.pipe_length
        xc = (np.arange(N) + 0.5) / N
        if cfg.stroke_cycle == 2:
            # 2-stroke tuned pipe: header -> diverging cone -> belly -> converging
            # cone -> stinger. The belly/cone reflections are the "braaap" and the
            # scavenging resonance; the tiny stinger sets back-pressure.
            belly = pipe_area * 6.0
            stinger = pipe_area * 0.42
            xs = [0.0, 0.12, 0.45, 0.58, 0.86, 1.0]
            ars = [pipe_area, pipe_area, belly, belly, stinger, stinger]
        elif self._straight:
            # open / straight pipe: a mild collector step then constant bore
            # (raw quarter-wave resonator, bright and loud -- no silencing)
            xs = [0.0, 0.30, 1.0]
            ars = [pipe_area, pipe_area, pipe_area]
        else:
            # 4-stroke with silencer: header+collector -> TWO expansion
            # chambers of unequal length -> tailpipe. A single chamber has
            # transmission-loss ZEROS (pass bands) at k*L_ch = n*pi -- for a
            # city car that put the first one at ~770 Hz, right in the
            # midband, and the firing comb sailed through it ("sounds racey
            # even with a muffler"). Real OEM silencers stage chambers of
            # unequal lengths so the pass bands never align: each chamber's
            # zero falls in the other's stop band. Lengths split 1:1.618
            # (incommensurate -> no common harmonics), joined by a pipe-bore
            # neck. All geometry -- the solver makes the staged TL.
            # EVERY transition is sized in CELLS (>= 2.5) so a 31:1 area
            # change is never steeper than the grid can carry: a sub-cell
            # cone turns the quasi-1D p*dA wall force into a violent
            # single-cell impulse (the first cut clipped a 2-bank V8 whose
            # 83-cell grid made the 1%-wide cones sub-cell).
            A_ch = self._A_ch                            # sized in _build
            A_tail = 0.8 * pipe_area
            s = max(0.03, 2.5 / N)        # one resolvable transition width
            flat = 0.38 - 5.0 * s         # chamber zone 0.50..0.88 minus cones
            if flat > 4.0 * s:
                # two staged chambers, lengths 1 : 1.618
                L1 = flat / 2.618
                L2 = flat - L1
                p1 = 0.50 + 2.0 * s       # cone up done -> ch1
                p2 = p1 + L1              # ch1 done
                p3 = p2 + s               # cone down -> neck
                p4 = p3 + s               # neck done
                p5 = p4 + s               # cone up -> ch2
                p6 = p5 + L2              # = 0.88, ch2 done
                xs = [0.0, 0.50, p1, p2, p3, p4, p5, p6,
                      p6 + 2.0 * s, 1.0]
                ars = [pipe_area, pipe_area, A_ch, A_ch, pipe_area,
                       pipe_area, A_ch, A_ch, A_tail, A_tail]
            else:
                # grid too coarse for staging (heavily split multi-bank
                # exotics): single chamber, transitions still >= 2.5 cells
                xs = [0.0, 0.50, 0.50 + 2.0 * s, 0.88, 0.88 + 2.0 * s, 1.0]
                ars = [pipe_area, pipe_area, A_ch, A_ch, A_tail, A_tail]
        prof = np.interp(xc, xs, ars)
        # slope limit: adjacent cells never differ by more than 1.6x, so no
        # cone is ever steeper than the grid resolves (the quasi-1D p*dA
        # wall force across a sub-cell 40:1 step is an impulse, not a cone)
        for k in range(1, N):
            if prof[k] > 1.6 * prof[k - 1]:
                prof[k] = 1.6 * prof[k - 1]
        for k in range(N - 2, -1, -1):
            if prof[k] > 1.6 * prof[k + 1]:
                prof[k] = 1.6 * prof[k + 1]
        return prof

    # ------------------------------------------------------------- live tuning
    def _radiation_geom(self):
        """Direct and ground-image path lengths from the listener geometry:
        tailpipe at H_SRC over asphalt, ear at H_EAR, mic_r horizontal."""
        d = self.mic_r if self.mic_r > 0.25 else 0.25
        r = math.sqrt(d * d + (H_EAR - H_SRC) ** 2)
        r_img = math.sqrt(d * d + (H_EAR + H_SRC) ** 2)
        return r, r_img

    def out_gain(self):
        """The ONE output-gain formula (build and live UI both use it):
        monopole radiation over the direct path, full scale = P_FULLSCALE Pa."""
        r, _ = self._radiation_geom()
        return 1.0 / (4.0 * math.pi * r * P_FULLSCALE)

    def _apply_radiation(self):
        """Per-bank radiation paths from the listener geometry. Each bank's
        tailpipe sits at its own lateral offset (dual exits flank the rear
        of the car ~0.8 m apart; more pipes spread across the same span; a
        single pipe is on the centreline = the old geometry exactly). Every
        pipe gets a direct ray and a ground-image ray: gain = r_ref/r_path,
        delay = path difference vs the centreline reference that defines
        outgain."""
        r, _ = self._radiation_geom()
        self.P[core.P_OUTGAIN] = 1.0 / (4.0 * math.pi * r * P_FULLSCALE)
        d = self.mic_r if self.mic_r > 0.25 else 0.25
        nb = self.nbanks
        span = 0.8 if nb > 1 else 0.0      # typical dual-exit spacing (m)
        for b in range(nb):
            y = (b - 0.5 * (nb - 1)) * (span / max(nb - 1, 1))
            r_dir = math.sqrt(d * d + y * y + (H_EAR - H_SRC) ** 2)
            r_img = math.sqrt(d * d + y * y + (H_EAR + H_SRC) ** 2)
            self.rad[b, 0] = round((r_dir - r) / C_AIR * SAMPLE_RATE)
            self.rad[b, 1] = r / r_dir
            self.rad[b, 2] = round((r_img - r) / C_AIR * SAMPLE_RATE)
            self.rad[b, 3] = r / r_img

    def set_mic_distance(self, r):
        self.mic_r = float(np.clip(r, 0.25, 20.0))
        self._apply_radiation()

    def set_throttle(self, v):
        self.pedal = float(np.clip(v, 0.0, 1.0))

    def set_load(self, v):
        self.load = float(max(0.0, v))

    def set_brake(self, v):
        self.brake = float(np.clip(v, 0.0, 1.0))

    def set_grade(self, v):
        """Road gradient as a slope fraction (rise/run), e.g. 0.1 = 10% uphill."""
        self.grade = float(np.clip(math.atan(v), -0.35, 0.35))

    def set_param(self, idx, value):
        self.P[idx] = value

    def set_vehicle(self, vcfg: VehicleConfig):
        self.vehicle = vcfg
        if self.gear > vcfg.n_gears():
            self.gear = 0
        self.v = 0.0

    # ---------------------------------------------------------------- controls
    def start_engine(self):
        if self.state == "off":
            self.state = "cranking"
            self.crank_timer = 0.0

    def stop_engine(self):
        self.state = "off"

    def shift_up(self):
        if self.gear < self.vehicle.n_gears():
            self.gear += 1
            self.shift_timer = 0.12      # clutch declutches for this window

    def shift_down(self):
        if self.gear > 0:
            self.gear -= 1
            self.shift_timer = 0.12

    def gear_label(self):
        return "N" if self.gear == 0 else str(self.gear)

    def speed_kmh(self):
        return self.v * 3.6

    # ----------------------------------------------------- control update
    def _control_update(self, dt):
        """Block-rate DRIVER + ACTUATOR layer only. It sets the inputs the
        physics core integrates (throttle area, feed pressure, fuelling, gear
        ratio, clutch capacity, road-load coefficients) and runs the handful of
        things that genuinely ARE controllers on a real car: the starter, the
        idle-air governor (the ECU's idle valve), the turbo, and the automatic
        clutch actuator. All mechanical coupling and vehicle motion is physics,
        integrated per-sample in the core."""
        cfg = self.cfg
        veh = self.vehicle
        rpm = self.rpm
        idle = cfg.idle_rpm
        P = self.P

        # filtered rpm rate (rpm/s) for the idle governor's derivative term
        if dt > 0.0:
            self.rpm_rate += ((rpm - self.rpm_prev) / dt - self.rpm_rate) * 0.3
        self.rpm_prev = rpm

        # ---- governor idle speed: a low-passed rpm (the MEAN idle speed). A
        # low-inertia, uneven-firing engine (cross-plane I4, V-twin, V4) swings the
        # instantaneous block rpm by hundreds; regulating that raw value used to
        # saturate the loop on every firing spike and ratchet the idle UP. The
        # governor instead tracks this filtered speed and a rate derived from it. ----
        a_gov = min(1.0, dt / GOV_TF)
        self.rpm_gov += (rpm - self.rpm_gov) * a_gov
        gov_rate = (self.rpm_gov - self._rpm_gov_prev) / dt if dt > 0.0 else 0.0
        self._rpm_gov_prev = self.rpm_gov

        if self.shift_timer > 0.0:
            self.shift_timer = max(0.0, self.shift_timer - dt)

        # ---- engine state machine + air/fuel demand (0..1) ----
        starter = 0.0
        running = 0.0
        demand = 0.0
        if self.state == "off":
            self.idle_i = 0.0
        elif self.state == "cranking":
            running = 1.0
            starter = self.starter_torque
            demand = 0.25
            self.crank_timer += dt
            if rpm > 0.85 * idle:
                self.state = "running"
                self.idle_i = 0.08          # seed near the idle-hold demand
                self.crank_assist = 0.6     # starter overlap window (s)
                self.run_time = 0.0         # time since catch (DFCO inhibit)
            elif self.crank_timer > 3.0:
                self.state = "off"          # failed to catch
        else:  # running
            running = 1.0
            # starter overlap: keep the cranking motor assisting for a short window
            # after catch, tapering out as rpm rises toward idle. A real starter
            # stays meshed until the engine clearly runs; this bridges the first
            # few compression strokes that would otherwise stall a low-inertia,
            # high-CR single/twin before combustion has stabilised.
            if self.crank_assist > 0.0:
                self.crank_assist = max(0.0, self.crank_assist - dt)
                if rpm < 1.15 * idle:
                    starter = self.starter_torque * float(
                        np.clip((1.15 * idle - rpm) / (1.15 * idle), 0.0, 1.0))
            if self.pedal < 0.03:
                # idle-air governor: proportional-dominated PI. The rpm plant is
                # an integrator, so a gentle P term does the regulating and a
                # slow integral only trims the steady idle-air bias. For petrol
                # this is the idle-air/throttle area; for diesel it trims fuel.
                # The PLANT (manifold + combustion) is real physics -- no hacks.
                # PID idle governor. The plant (manifold->torque->rpm) lags, so
                # a derivative term on rpm rate damps the overshoot. The TARGET is
                # raised when cold (ECU fast idle), set by the thermal model.
                # PI-D idle governor on the FILTERED idle speed (rpm_gov). The
                # plant (idle-air -> manifold -> torque -> inertia) is a lagged
                # integrator, so a proportional term regulates, a rate (derivative)
                # term damps the manifold lag, and a slow integral trims the steady
                # idle air with CONDITIONAL anti-windup (it stops integrating only
                # when that would push further into a saturated actuator -- so it no
                # longer ratchets up on a noisy, low-inertia engine). The PLANT is
                # real physics; this is just the ECU's idle valve / fuel trim.
                idle_t = self.idle_target
                err = (idle_t - self.rpm_gov) / idle_t
                rate_norm = gov_rate / idle_t          # per-second, normalised
                # ---- gains by pole placement on the REAL idle plant ----------
                # The loop is: demand -> (fuel, or throttle area -> manifold
                # fill) -> indicated torque -> rpm integrator (1/inertia) ->
                # GOV_TF measurement filter. Every element of that chain is
                # known here at runtime, so kp/ki/kd come from the SIMC tuning
                # rule for an integrating process (Skogestad 2003: with the
                # closed-loop time constant chosen equal to the effective
                # delay theta, Kc = 1/(2*k'*theta), tau_I = 8*theta; the
                # derivative cancels the one dominant lag, series-PID
                # tau_D = tau2). The old schedule adapted to inertia ONLY and
                # ignored the lags entirely -- any preset whose manifold +
                # filter lags put the 180-degree crossover near the loop
                # bandwidth limit-cycled (the 2JZ hunted +-135 rpm at 0.8 Hz).
                w_i = idle_t * math.pi / 30.0           # idle target (rad/s)
                # friction torque the idle must hold, from the LIVE values (the
                # cold friction multiplier raises it -> firmer cold-idle loop)
                T_fric = P[core.P_FRIC0] + P[core.P_FRICW] * w_i
                cycr = P[core.P_CYCLE]                  # cycle length (rad)
                Vd_tot = P[core.P_VDISP] * cfg.n_cylinders
                eta_v = 0.9                             # typical NA low-speed VE
                theta_d = 0.5 * cycr / w_i              # intake->power delay
                if cfg.diesel:
                    # fuel -> torque is direct (no manifold in the fuel path)
                    # and linear in demand up to the smoke limit, so dT/du is
                    # the smoke-limited torque from the trapped air at
                    # atmospheric MAP. Air-standard Otto efficiency is an
                    # upper bound on eta_i -> over-estimates the plant gain ->
                    # a conservative (gentler) loop.
                    air_cyc = eta_v * PATM / (R_AIR * P[core.P_TMAN]) * Vd_tot
                    eta_i = 1.0 - cfg.compression_ratio ** (1.0 - GAMMA_CYL)
                    T_max = (eta_i * P[core.P_COMBEFF] * P[core.P_LHV]
                             * (air_cyc / 23.0) / cycr)
                    dT_du = max(T_max, T_fric / 0.6)    # idles below 60% fuel
                    tau2 = GOV_TF                       # only lag: the filter
                    theta = theta_d + 0.04              # common-rail smoothing
                elif cfg.stroke_cycle == 2:
                    # crankcase feed pressure is linear in demand (the PBOOST
                    # law: 0.03 + 0.97*u), and indicated torque ~ feed
                    # pressure, so dT/du = 0.97 * full-feed torque.
                    air_cyc = eta_v * PATM / (R_AIR * P[core.P_TMAN]) * Vd_tot
                    eta_i = 1.0 - cfg.compression_ratio ** (1.0 - GAMMA_CYL)
                    T_max = (eta_i * P[core.P_COMBEFF] * P[core.P_LHV]
                             * (air_cyc / P[core.P_AFR]) / cycr)
                    dT_du = max(0.97 * T_max, T_fric / 0.6)
                    tau2 = GOV_TF                       # throttle is wide open
                    theta = theta_d
                else:
                    # 4-stroke petrol: flow through the nearly-closed plate is
                    # choked (MAP ~0.35 atm at idle, far below the 0.53
                    # critical ratio), so mdot -- and with it indicated torque
                    # -- is proportional to plate area A(u) = floor+(1-fl)*u^2.
                    # At the steady idle point T_ind(u0) = T_fric, so
                    # dT/du = T_fric * A'(u0)/A(u0): the loop gain needs NO
                    # efficiency estimate at all. u0 comes from matching the
                    # engine's swallow rate at idle vacuum to the choked
                    # throttle flow.
                    mdot_eng = (eta_v * (0.35 * PATM) / (R_AIR * P[core.P_TMAN])
                                * Vd_tot * w_i / cycr)
                    # choked mass flux: p0*sqrt(g/(R*T0))*(2/(g+1))^((g+1)/(2(g-1)))
                    flux = (PATM * math.sqrt(1.4 / (R_AIR * TAMB))
                            * (2.0 / 2.4) ** 3.0)
                    a0 = mdot_eng / (P[core.P_CD] * flux * self.thr_area_max)
                    fl = self.idle_floor
                    u0 = math.sqrt(max((min(a0, 0.5) - fl) / (1.0 - fl), 4e-4))
                    dT_du = (T_fric * 2.0 * u0 * (1.0 - fl)
                             / (fl + (1.0 - fl) * u0 * u0))
                    # manifold fill time constant: plenum volume over the
                    # engine's volumetric swallow rate at idle speed
                    tau2 = self.man_vol / (eta_v * Vd_tot * w_i / cycr)
                    theta = theta_d + GOV_TF            # filter lumped as delay
                # normalised plant: d(rpm/idle)/dt = k' * demand
                k_pl = dT_du / (cfg.inertia * w_i)
                theta = max(theta, 0.02)
                Kc = 1.0 / (2.0 * k_pl * theta)
                tau_i = 8.0 * theta
                # series -> parallel PID; broad clips are actuator sanity
                # bounds for degenerate presets, not tuning.
                kp = float(np.clip(Kc * (1.0 + tau2 / tau_i), 0.02, 2.5))
                ki = float(np.clip(Kc / tau_i, 0.005, 2.0))
                kd = float(np.clip(Kc * tau2, 0.002, 0.6))
                imax = 0.9 if cfg.diesel else 0.85
                raw = self.idle_i + kp * err - kd * rate_norm
                demand = float(np.clip(raw, 0.0, 0.95))
                # conditional integration: skip only when winding further into a stop
                sat_hi = demand >= 0.95 and err > 0.0
                sat_lo = demand <= 0.0 and err < 0.0
                if not (sat_hi or sat_lo):
                    self.idle_i = float(np.clip(self.idle_i + err * dt * ki,
                                                -0.05, imax))
            else:
                demand = self.pedal
                self.idle_i = float(np.clip(self.idle_i, 0.04, 0.7))
            if rpm < 0.2 * idle:
                self.state = "off"          # stalled

        # ---- decel fuel cut (DFCO) -------------------------------------------
        # On a closed throttle well above idle the ECU cuts injection entirely:
        # real engine braking, and a genuinely lean exhaust (free O2) so the
        # overrun pops are earned. Hysteresis resumes fuel before rpm falls back
        # to idle so the governor catches the engine without a stumble.
        idle_t = self.idle_target
        if self.state == "running":
            self.run_time += dt
            # DFCO inhibit for the first seconds after catch (production ECU
            # behavior): the post-start rpm overshoot is NOT an overrun, and
            # cutting fuel into it destabilises the settle.
            if self.pedal < 0.03 and rpm > idle_t * 1.4 and self.run_time > 2.0:
                self.fuelcut = True
            elif self.fuelcut and (self.pedal >= 0.03 or rpm < idle_t * 1.15):
                self.fuelcut = False
                self.idle_i = 0.10          # seed idle-air so it doesn't dip on resume
        if self.fuelcut:
            # decel dashpot / idle-follower: through DFCO the ECU keeps the
            # idle valve at its learned trim (idle AIRFLOW, not idle MAP --
            # the vacuum genuinely deepens with rpm on a real overrun, which
            # is also where engine braking comes from). The sealed plate
            # used to pull 3 kPa; the idle trim gives ~idle_MAP*idle/rpm,
            # softening the EVO backflow slam while keeping real braking.
            # (A first cut held MAP constant via trim*sqrt(rpm/idle): too
            # much air -- it halved engine braking and measured as loud as
            # firing. ECUs hold the valve, not the depression.)
            demand = max(demand, self.idle_i)
        if self.state != "running":
            self.fuelcut = False
        P[core.P_FUELCUT] = 1.0 if self.fuelcut else 0.0

        # ---- traction control: ease power when the driven wheel spins faster
        # than the road. Modern performance cars all have it; without it a
        # powerful RWD car just lights up the tyres and accelerates slowly.
        tc_ratio = (veh.gear_ratios[self.gear - 1] * veh.final_drive
                    if 1 <= self.gear <= veh.n_gears() else 0.0)
        if (self.traction_control and self.state == "running"
                and tc_ratio != 0.0 and self.v > 0.5):
            wheel_surf = self.st[core.S_OMEGA_W] * veh.wheel_radius
            slip = (wheel_surf - self.v) / max(self.v, 2.5)
            if slip > 0.12:
                self.tc_cut = min(0.92, self.tc_cut + dt * 10.0 * (slip - 0.12))
            else:
                self.tc_cut = max(0.0, self.tc_cut - dt * 5.0)
        else:
            self.tc_cut = 0.0
        demand *= (1.0 - self.tc_cut)

        # ---- ECU knock control: pull timing when the core reports end-gas
        # knock, restore it slowly when clean. The result is a small limit cycle
        # sitting right at the knock line -- exactly how a real knock-retard loop
        # caps timing under boost / low octane / lugging, costing a little power.
        if self.knock > 0.0:
            self.knock_retard = min(14.0, self.knock_retard + dt * 45.0)
        else:
            self.knock_retard = max(0.0, self.knock_retard - dt * 3.0)

        # ---- spark advance + fuelling (petrol; diesel is CI / smoke-limited) --
        if not cfg.diesel:
            # Spark advance map: centrifugal (rises with rpm) + vacuum (light
            # load) advance, anchored so WOT near peak-power rpm stays at the
            # preset's reference timing -- the dyno calibration is preserved and
            # only part-load / off-peak timing varies (where it should). The ECU
            # knock-retard is then subtracted.
            peak = 0.7 * cfg.redline_rpm
            vac = 10.0 * (1.0 - demand) ** 1.3
            cent = 6.0 * ((rpm - peak) / max(1.0, peak))
            adv = float(np.clip(cfg.ignition_btdc + vac + cent - self.knock_retard,
                                5.0, 45.0))
            P[core.P_IGN] = math.radians(self.cyc_deg - adv)

            # Equivalence-ratio command: stoich at light cruise (for the cat),
            # enriching toward ~12.5:1 (phi~1.18) under high load for peak power,
            # extra rich when cold (vaporization), with a small closed-loop
            # lambda dither at warm part-throttle cruise.
            phi = 1.0
            if demand > 0.75:
                phi += (demand - 0.75) / 0.25 * 0.18
            phi += 0.25 * (1.0 - self.warm_frac)
            if (self.state == "running" and self.warm_frac > 0.8
                    and 0.03 <= self.pedal < 0.5):
                self.lambda_phase += dt * 2.0 * math.pi * 1.4
                phi += 0.035 * math.sin(self.lambda_phase)
            P[core.P_PHI] = phi

        # ---- afterfire / crackle ----
        # self.backfire is the fraction of the cylinders' unburnt surplus fuel
        # that survives into the pipe un-oxidized (exhaust/cat design). Whether
        # and when it POPS is decided in the core by the pipe's real fuel + O2
        # inventories and temperature -- there is no pop probability any more.
        P[core.P_AFTERFIRE] = self.backfire if not cfg.diesel else 0.0

        # ---- intake feed: set throttle area + PRE-compressor feed pressure ----
        # Turbo boost is no longer scripted here -- it emerges in the core from
        # the turbo shaft's power balance. We only supply the feed pressure
        # (atmosphere, or 2-stroke crankcase) and the throttle area.
        if cfg.diesel:
            # max-speed governor: a diesel does NOT bounce a hard rev-cut limiter
            # (that warbles/buzzes, esp. a high-torque low-redline tractor). It
            # smoothly pulls fuel over a droop band as it approaches the governed
            # speed, so it just stops accelerating cleanly at redline.
            droop = max(80.0, 0.06 * cfg.redline_rpm)
            gov = float(np.clip((rpm - (cfg.redline_rpm - droop)) / droop, 0.0, 1.0))
            demand *= (1.0 - gov)
            P[core.P_PBOOST] = PATM
            P[core.P_THRAREA] = self.thr_area_max      # no throttle plate
            # common-rail injection responds fast (ms); a small lag only
            self.thr_smooth += (demand - self.thr_smooth) * min(1.0, dt / 0.04)
            P[core.P_THROTTLE] = self.thr_smooth       # diesel fuelling command
        elif cfg.stroke_cycle == 2:
            # 2-stroke crankcase scavenge: throttle-dependent, no turbo shaft
            P[core.P_PBOOST] = PATM * (0.03 + 0.97 * demand) + cfg.turbo_boost * demand
            P[core.P_THRAREA] = self.thr_area_max
            P[core.P_THROTTLE] = demand
        else:
            P[core.P_PBOOST] = PATM
            # progressive throttle plate: area ~ demand^2 gives fine resolution
            # near idle (gentle loop gain) and full bore at WOT, like a real body.
            # The idle leak (idle_floor) is scaled per engine so even big-valve,
            # high-rev bikes can be governed down to their target idle.
            P[core.P_THRAREA] = self.thr_area_max * (
                self.idle_floor + (1.0 - self.idle_floor) * demand * demand)
            P[core.P_THROTTLE] = demand

        # ---- driveline / vehicle inputs for the coupled core integration ----
        r_w = veh.wheel_radius
        has_veh = veh.mass > 5.0
        ratio = 0.0
        if 1 <= self.gear <= veh.n_gears():
            ratio = veh.gear_ratios[self.gear - 1] * veh.final_drive

        # automatic clutch actuator: how much the clutch is "let out" (0..1).
        # Disengaged in neutral / shifting / off; locks up once the wheels turn
        # fast enough; feeds in with throttle for a launch; opens to protect the
        # engine from stalling. This is the real auto-clutch/TCU, not a physics
        # fudge -- the torque it then transmits is pure Coulomb friction.
        idle_w = idle * TWO_PI_60
        omega_eng = self.st[core.S_OMEGA]
        omega_in = (self.v / r_w) * ratio if ratio != 0.0 else 0.0
        rolling = omega_in > 0.9 * idle_w
        if (ratio == 0.0 or not has_veh or self.state != "running"
                or self.shift_timer > 0.0):
            engage_target = 0.0
        elif rolling:
            engage_target = 1.0                        # rolling -> lock up
        else:
            # standstill launch: slip the clutch to HOLD the engine at a
            # launch speed that rises with pedal (what a TCU / driver's left
            # foot does). Engagement follows the engine's surplus over the
            # hold band, so a strong engine locks through quickly while a
            # weak one is held slipping near its launch rpm instead of being
            # dragged below its torque band and stalled (the old pedal-only
            # feed bogged a 1.0L hatch to ~250 rpm on every WOT launch and
            # lived on a knife edge). Continuous, no panic-release cliff.
            launch_w = idle_w * (1.2 + 1.3 * self.pedal)
            engage_target = self.pedal * float(np.clip(
                (omega_eng - 1.1 * idle_w) / (launch_w - 1.1 * idle_w),
                0.0, 1.0))
        # ease the clutch in slowly for a standstill launch (so the engine revs
        # up rather than bogging); snap shut quickly once the wheels are rolling
        if engage_target <= self.clutch_engage:
            rate = 9.0
        elif rolling:
            rate = 9.0
        else:
            rate = 2.2                                  # from-rest launch slip-in
        self.clutch_engage = float(np.clip(
            self.clutch_engage + (engage_target - self.clutch_engage)
            * min(1.0, dt * rate), 0.0, 1.0))

        # torsional driveline stiffness/damping set for a fixed natural
        # frequency against the gear's reflected inertia (driveline compliance)
        if ratio != 0.0 and has_veh:
            i_in = veh.mass * (r_w / ratio) ** 2
            i_red = 1.0 / (1.0 / self.base_inertia + 1.0 / i_in)
            wn = 2.0 * math.pi * DRIVELINE_FN
            P[core.P_KDL] = i_red * wn * wn
            P[core.P_CDL] = 2.0 * DRIVELINE_ZETA * i_red * wn

        P[core.P_RATIO] = ratio
        P[core.P_RW] = r_w
        P[core.P_VMASS] = veh.mass
        P[core.P_CRR] = veh.crr
        P[core.P_CDA] = 0.5 * RHO_AIR * veh.cd * veh.frontal_area
        P[core.P_BRAKEF] = self.brake * veh.mass * 8.0
        P[core.P_TCAP] = self.clutch_cap * self.clutch_engage
        P[core.P_MU] = TRACTION_MU

        # ---- chassis geometry: load transfer, drive split, cornering ----
        wd_front = veh.weight_dist_front
        wb = veh.wheelbase if veh.wheelbase > 0.0 else max(1.8, 2.0 + veh.mass / 3000.0)
        cgh = veh.cg_height if veh.cg_height > 0.0 else 0.45
        hL = cgh / wb                       # load-transfer factor
        dtrain = getattr(veh, "drivetrain", "fwd")
        if dtrain == "rwd":
            wd_drive = 1.0 - wd_front; xfer = hL
        elif dtrain == "awd":
            wd_drive = 1.0; xfer = 0.0
        else:                                # fwd: accel UNLOADS the driven axle
            wd_drive = wd_front; xfer = -hL
        P[core.P_WD_FRONT] = wd_front
        P[core.P_WD_DRIVE] = wd_drive
        P[core.P_XFER] = xfer
        P[core.P_WHEELBASE] = wb
        P[core.P_CGH] = cgh
        # driven wheel + driveline rotational inertia (scales with vehicle size)
        P[core.P_JWHEEL] = 1.0 + veh.mass * 8.0e-4
        P[core.P_GRADE] = self.grade

        # ---- clatter rpm fade: mechanical knocks are an idle/low-rpm sound;
        # as revs climb the combustion/exhaust roar masks them. Full level up
        # to ~2x idle, then receding as 1/rpm (the code always CLAIMED this
        # behaviour in a comment; now it actually does it).
        fade_rpm = float(np.clip((1.5 * idle / max(rpm, 1.0)) ** 2, 0.0, 1.0))
        P[core.P_CLATTER] = (self._clatter_on * fade_rpm
                             if self.mix_clatter else 0.0)

        # ---- engine bookkeeping ----
        P[core.P_LOAD] = self.load
        P[core.P_INERTIA] = self.base_inertia
        P[core.P_RUNNING] = running
        P[core.P_STARTER] = starter

    # ------------------------------------------------------------------ stepping
    def step_block(self, n, out=None):
        # the audio callback passes a preallocated buffer so the latency-critical
        # path does no per-block allocation; simulate_block overwrites every sample
        if out is None or len(out) != n:
            out = np.zeros(n, dtype=np.float64)
        with self.lock:
            self._control_update(n / SAMPLE_RATE)
            t = simulate_block(n, self.P, self.st, self.cyl_m, self.cyl_T,
                               self.phase, self.inj, self.cyl_bank,
                               self.rho, self.mom,
                               self.Ene, self.rho_in, self.mom_in, self.Ene_in,
                               self.run_mdot, self.fresh, self.cyl_knk,
                               self.cyl_chem, self.pipe_chem,
                               self.rho_r, self.mom_r, self.Ene_r,
                               self.src_rm, self.src_re,
                               out, self.scope_p, N_SCOPE, self.filt, self.gnd,
                               self.rad, self.mdot_prev, self.vt_ex,
                               self.ex_area, self.ex_aface, self.ex_wk,
                               self.in_area, self.in_aface, self.in_wk,
                               self.run_area_a, self.run_aface, self.run_wk,
                               self.bnd_ex, self.bnd_in, self.bnd_run,
                               self.pk_ex, self.pk_in, self.pk_run)
            self.torque = float(t)
            self.rpm = self.st[core.S_OMEGA] * 60.0 / (2.0 * math.pi)
            self.v = float(self.st[core.S_V])
            self.boost = float(self.P[core.P_BOOSTOUT])      # emergent boost (Pa)
            self.turbo_rpm = self.st[core.S_TURBO] * 60.0 / (2.0 * math.pi)
            self.knock = float(self.P[core.P_KNOCK])         # knock telemetry
            self.wheel_rpm = self.st[core.S_OMEGA_W] * 60.0 / (2.0 * math.pi)
            # fuel consumption: the core reports the injected mass per block;
            # smooth the rate over ~1 s so few-cylinder engines (whose blocks
            # catch 0 or 1 injection events) read steady instead of strobing.
            dt_blk = n / SAMPLE_RATE
            fuel_kg = float(self.P[core.P_FUELUSED])
            self.fuel_total_kg += fuel_kg
            self.fuel_rate += ((fuel_kg / dt_blk) - self.fuel_rate) \
                * min(1.0, dt_blk / 1.0)
            self._thermal_update(dt_blk)
        return out

    def _thermal_update(self, dt):
        """Advance the 3-mass thermal model and apply the cold effects. Heat IN is
        the combustion-to-wall loss the core just computed (P_QWALL) plus friction
        heat into the oil; heat OUT is the radiator, gated by a thermostat and by
        airflow. Metal temperature feeds back as the cylinder wall temp, so cold
        running (more heat lost from the charge) is self-consistent physics."""
        P = self.P
        omega = self.st[core.S_OMEGA]
        q_comb = float(P[core.P_QWALL])             # combustion heat into the metal
        # friction work dissipated this block -> heats the oil
        fric_pow = ((self.base_fric0 + self.base_fricw * omega)
                    * self.friction_mult * omega)
        q_fric = max(0.0, fric_pow) * dt
        # inter-mass conduction
        q_mc = self.h_mc * (self.T_metal - self.T_cool) * dt
        q_mo = self.h_mo * (self.T_metal - self.T_oil) * dt
        q_oc = self.h_oc * (self.T_oil - self.T_cool) * dt   # oil cooler -> coolant
        # thermostat: shut (small bypass) until ~T_THERMOSTAT, then opens
        thermo = float(np.clip((self.T_cool - (T_THERMOSTAT - 8.0)) / 12.0,
                               0.04, 1.0))
        airflow = 1.0 + 0.035 * self.v              # ram air through the radiator
        q_rad = self.h_rad * max(0.0, self.T_cool - TAMB) * thermo * airflow * dt
        q_oa = self.h_oa * max(0.0, self.T_oil - TAMB) * dt
        self.T_metal += (q_comb - q_mc - q_mo) / self.C_metal
        self.T_cool += (q_mc + q_oc - q_rad) / self.C_cool
        self.T_oil += (q_mo + q_fric - q_oa - q_oc) / self.C_oil
        self.T_metal = float(np.clip(self.T_metal, TAMB, 1300.0))
        self.T_cool = float(np.clip(self.T_cool, TAMB, 400.0))
        self.T_oil = float(np.clip(self.T_oil, TAMB, 420.0))
        P[core.P_WALLT] = self.T_metal              # wall temp for the next block

        # ---- emergent cold effects (warm fraction 0=cold .. 1=at temp) ----
        span = T_THERMOSTAT - TAMB
        wf = float(np.clip((self.T_cool - TAMB) / span, 0.0, 1.0))
        owf = float(np.clip((self.T_oil - TAMB) / span, 0.0, 1.0))
        self.warm_frac = wf
        self.coolant_C = self.T_cool - 273.15
        # cold oil is viscous -> higher FMEP (drag); ~3x at cold soak -> 1x warm
        self.friction_mult = 1.0 + 2.3 * (1.0 - owf) ** 1.4
        P[core.P_FRIC0] = self.base_fric0 * self.friction_mult
        P[core.P_FRICW] = self.base_fricw * self.friction_mult
        # ECU raises the idle target when cold (fast idle) for stable warm-up.
        # Petrol fast-idles high (~+55%, carb/EFI warm-up enrichment); a
        # diesel has no throttle losses or cat to light and idles cold at
        # barely above its warm speed (~+12%) -- the old petrol-style +55%
        # put a 850-rpm TDI at 1320 cold, which read as a broken idle.
        raise_frac = 0.12 if self.cfg.diesel else 0.55
        self.idle_target = self.cfg.idle_rpm * (1.0 + raise_frac * (1.0 - wf))
        # NB: cold enrichment is now applied as a richer phi command in
        # _control_update (the equivalence-ratio fuelling map), not by mutating
        # the stoichiometric AFR here.

    def prewarm(self):
        """Bring the engine straight up to operating temperature (skip warm-up)."""
        self.T_metal = 450.0              # matches the legacy fixed wall temp
        self.T_cool = T_THERMOSTAT
        self.T_oil = T_THERMOSTAT - 3.0
        self.friction_mult = 1.0
        self.warm_frac = 1.0
        self.coolant_C = self.T_cool - 273.15
        self.idle_target = self.cfg.idle_rpm
        self.P[core.P_WALLT] = self.T_metal
        self.P[core.P_FRIC0] = self.base_fric0
        self.P[core.P_FRICW] = self.base_fricw
        self.P[core.P_AFR] = self.base_afr

    def warmup(self):
        """Trigger numba compilation off the audio thread (engine stays off)."""
        save = self.state
        self.state = "off"
        self.step_block(64)
        self.load_config(self.cfg)
        self.state = save

    # -------------------------------------------------------------------- audio
    def _callback(self, outdata, frames, time_info, status):
        # Allocation-free hot path: reuse a preallocated buffer and do the guard /
        # soft-limit in place. Per-block heap churn here (or in step_block) shows
        # up as periodic buffer underruns -- the audio dropping out at a regular
        # rate -- so the whole callback avoids creating temporaries.
        cb = self._cb_buf if frames == len(self._cb_buf) else None
        try:
            out = self.step_block(frames, out=cb)
        except Exception:
            outdata.fill(0.0)
            return
        np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.multiply(out, self.volume, out=out)
        np.tanh(out, out=out)                       # soft limiter, in place
        outdata[:, 0] = out                         # float64 -> float32 on assign
        # store tail for the waveform display (ring buffer; plain slice copies)
        tp = self._tap_pos
        n = len(out)
        buf = self.audio_tap
        if tp + n <= len(buf):
            buf[tp:tp + n] = out
        else:
            k = len(buf) - tp
            buf[tp:] = out[:k]
            buf[:n - k] = out[k:]
        self._tap_pos = (tp + n) % len(buf)

    def start_audio(self):
        import sounddevice as sd
        if self.stream is not None:
            return
        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            blocksize=BLOCK, device=self.device, callback=self._callback)
        self.stream.start()

    def set_device(self, device):
        """Select the output device (index or None=default); restart if live."""
        was_on = self.is_audio_on()
        if was_on:
            self.stop_audio()
        self.device = device
        if was_on:
            self.start_audio()

    def stop_audio(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def is_audio_on(self):
        return self.stream is not None
