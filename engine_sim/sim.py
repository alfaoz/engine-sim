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
N_SCOPE = 720          # cylinder-pressure scope resolution (per cycle)


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
        self.torque = 0.0
        self.audio_tap = np.zeros(2048, dtype=np.float64)   # ring of last output
        self._tap_pos = 0

        # vehicle / control state
        self.vehicle = get_vehicle("Bench / neutral (free rev)")
        self.gear = 0              # 0 = neutral
        self.state = "off"         # off | cranking | running
        self.v = 0.0               # vehicle speed (m/s)
        self.brake = 0.0           # brake pedal 0..1
        self.idle_i = 0.0          # idle governor integrator
        self.shift_timer = 0.0     # clutch-out window during a shift
        self.crank_timer = 0.0
        self.clutch_cap = 200.0
        self.boost = 0.0           # actual turbo boost (Pa), spools with lag
        self.thr_smooth = 0.0      # slewed throttle (intake/fuel dynamics)
        self.clutch_engage = 0.0   # 0..1 automatic-clutch engagement
        self.base_inertia = 0.2    # engine-only rotating inertia
        # audio source mix (for A/B-ing real vs coloration)
        # induction OFF by default: the current 1D-duct coupling is a numerical
        # hack (per-sample impulse into one cell) that buzzes -- it needs a real
        # airbox + throttled boundary before it earns a place in the mix.
        self.mix_induction = 0.0
        self.mix_body = 0.35
        self.mix_sat = 1.0

        self.load_config(cfg)

    @property
    def throttle(self):
        return self.pedal

    # ------------------------------------------------------------------ build
    def load_config(self, cfg: EngineConfig):
        with self.lock:
            self.cfg = cfg
            self._build()

    def _build(self):
        cfg = self.cfg
        P = np.zeros(P_NPARAMS, dtype=np.float64)
        ncyl = cfg.n_cylinders
        cyc = 4.0 * math.pi if cfg.stroke_cycle == 4 else 2.0 * math.pi
        cyc_deg = 720.0 if cfg.stroke_cycle == 4 else 360.0

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
        # indicated->brake losses we don't fully model (blowby, incomplete burn,
        # heat) — calibrated so BMEP lands in the real range (NA ~10-12 bar)
        P[core.P_COMBEFF] = 0.78 * cfg.power_tune
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
        P[core.P_FRICW] = 270.0 * Vd_total / divisor
        P[core.P_LOAD] = self.load
        # starter motor torque and clutch capacity scale with engine size
        self.starter_torque = 25.0 + 30.0 * disp_l
        # clutch torque capacity ~ 1.5-2x peak engine torque (disp_l is litres)
        self.clutch_cap = 320.0 * disp_l + 50.0

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

        # exhaust grid
        N = int(min(360, max(90, cfg.pipe_length / 0.008)))
        dx = cfg.pipe_length / N
        P[core.P_NCELLS] = N
        P[core.P_DX] = dx
        P[core.P_PIPELEN] = cfg.pipe_length
        P[core.P_PIPEAREA] = math.pi / 4.0 * cfg.pipe_diameter ** 2

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
        else:
            Ni, dx_in, in_area, in_len = 0, 1.0, 1.0, 0.0
        self.Ni = Ni
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
        P[core.P_HT] = 6.0 * (Apist / 0.005)
        P[core.P_WALLT] = 450.0
        P[core.P_DIESEL] = 1.0 if cfg.diesel else 0.0
        P[core.P_DAMP] = 0.0015
        P[core.P_OUTGAIN] = 1.0 / 2600.0
        P[core.P_REDLINE] = cfg.redline_rpm
        P[core.P_RUNNING] = 0.0     # engine starts off — user cranks it
        P[core.P_STARTER] = 0.0
        P[core.P_LPF] = 0.55        # ~6.5 kHz one-pole output lowpass
        P[core.P_NOISE] = 0.06      # combustion cycle-to-cycle roughness
        P[core.P_POP] = 0.0         # overrun backfire intensity (set live)
        self.backfire = 1.0         # user multiplier for backfire amount

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

        # injection cells: spread cylinders over the head end of the pipe
        inj = np.zeros(ncyl, dtype=np.int64)
        spread = cfg.runner_spread
        for c in range(ncyl):
            frac = 0.0 if ncyl == 1 else (c / (ncyl - 1)) * spread
            inj[c] = int(min(0.4, frac) * N)
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
        self.boost = 0.0
        self.thr_smooth = 0.0
        self.clutch_engage = 0.0
        self.base_inertia = cfg.inertia
        Vmid = Vclear + 0.5 * Vdisp
        self.cyl_T = np.full(ncyl, TINT)
        self.cyl_m = np.full(ncyl, PATM * Vmid / (R_AIR * TINT))

        rho0 = PATM / (R_AIR * TATM)
        self.rho = np.full(N, rho0)
        self.mom = np.zeros(N)
        self.Ene = np.full(N, PATM / (GAMMA_EX - 1.0))

        # 1D intake duct gas state (cool air at rest); min length 1 if disabled
        nin = max(1, Ni)
        rho_in0 = PATM / (R_AIR * TINT)
        self.rho_in = np.full(nin, rho_in0)
        self.mom_in = np.zeros(nin)
        self.Ene_in = np.full(nin, PATM / (GAMMA_CYL - 1.0))

        self.scope_p = np.full(N_SCOPE, PATM)
        # output filter state: exhaust DC block (0,1), de-hash LPF (2,3),
        # body/thump resonator (4,5); induction DC block (6,7), de-hash LPF (8,9)
        self.filt = np.zeros(12)
        self.P[core.P_MAP] = PATM

    # ------------------------------------------------------------- live tuning
    def set_throttle(self, v):
        self.pedal = float(np.clip(v, 0.0, 1.0))

    def set_load(self, v):
        self.load = float(max(0.0, v))

    def set_brake(self, v):
        self.brake = float(np.clip(v, 0.0, 1.0))

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
            if rpm > 0.7 * idle:
                self.state = "running"
                self.idle_i = 0.08          # seed near the idle-hold demand
            elif self.crank_timer > 3.0:
                self.state = "off"          # failed to catch
        else:  # running
            running = 1.0
            if self.pedal < 0.03:
                # idle-air governor: proportional-dominated PI. The rpm plant is
                # an integrator, so a gentle P term does the regulating and a
                # slow integral only trims the steady idle-air bias. For petrol
                # this is the idle-air/throttle area; for diesel it trims fuel.
                # The PLANT (manifold + combustion) is real physics -- no hacks.
                # PID idle governor. The plant (manifold->torque->rpm) lags, so
                # a derivative term on rpm rate damps the overshoot that would
                # otherwise make a heavy, low-idle engine hunt; the integral
                # trims the steady idle-air bias.
                # heavy engines have a much slower rotational plant, so the
                # integral (which would otherwise wind up and slow-hunt) is eased
                # off with inertia; P and D stay fixed.
                ki = 0.40 * float(np.clip((0.2 / max(0.01, cfg.inertia)) ** 0.5,
                                          0.18, 1.0))
                err = (idle - rpm) / idle
                demand = float(np.clip(
                    self.idle_i + 0.16 * err - 3.0e-5 * self.rpm_rate, 0.0, 0.9))
                if 0.0 < demand < 0.9:      # integrate only when not saturated
                    self.idle_i = float(np.clip(self.idle_i + err * dt * ki,
                                                0.0, 0.7))
            else:
                demand = self.pedal
                self.idle_i = float(np.clip(self.idle_i, 0.04, 0.7))
            if rpm < 0.2 * idle:
                self.state = "off"          # stalled

        # ---- overrun backfire intensity (closed throttle, elevated rpm) ----
        pop = 0.0
        if self.state == "running" and self.pedal < 0.06 and rpm > idle * 1.8:
            pop = float(np.clip((rpm - idle * 1.8) / max(1.0, cfg.redline_rpm),
                                0.0, 1.0)) * self.backfire
        P[core.P_POP] = pop

        # ---- intake feed: set throttle area + PRE-compressor feed pressure ----
        # Turbo boost is no longer scripted here -- it emerges in the core from
        # the turbo shaft's power balance. We only supply the feed pressure
        # (atmosphere, or 2-stroke crankcase) and the throttle area.
        if cfg.diesel:
            P[core.P_PBOOST] = PATM
            P[core.P_THRAREA] = self.thr_area_max      # no throttle plate
            # injection delivery has a little lag of its own
            self.thr_smooth += (demand - self.thr_smooth) * min(1.0, dt / 0.12)
            P[core.P_THROTTLE] = self.thr_smooth       # diesel fuelling command
        elif cfg.stroke_cycle == 2:
            # 2-stroke crankcase scavenge: throttle-dependent, no turbo shaft
            P[core.P_PBOOST] = PATM * (0.03 + 0.97 * demand) + cfg.turbo_boost * demand
            P[core.P_THRAREA] = self.thr_area_max
            P[core.P_THROTTLE] = demand
        else:
            P[core.P_PBOOST] = PATM
            # progressive throttle plate: area ~ demand^2 gives fine resolution
            # near idle (gentle loop gain) and full bore at WOT, like a real body
            P[core.P_THRAREA] = self.thr_area_max * (0.0008 + 0.9992 * demand * demand)
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
        elif omega_eng < 0.8 * idle_w:
            engage_target = 0.0                        # near stall -> slip free
        else:
            engage_target = self.pedal                 # launch: feed with throttle
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

        # ---- engine bookkeeping ----
        P[core.P_LOAD] = self.load
        P[core.P_INERTIA] = self.base_inertia
        P[core.P_RUNNING] = running
        P[core.P_STARTER] = starter

    # ------------------------------------------------------------------ stepping
    def step_block(self, n):
        out = np.zeros(n, dtype=np.float64)
        with self.lock:
            self._control_update(n / SAMPLE_RATE)
            t = simulate_block(n, self.P, self.st, self.cyl_m, self.cyl_T,
                               self.phase, self.inj, self.rho, self.mom,
                               self.Ene, self.rho_in, self.mom_in, self.Ene_in,
                               out, self.scope_p, N_SCOPE, self.filt)
            self.torque = float(t)
            self.rpm = self.st[core.S_OMEGA] * 60.0 / (2.0 * math.pi)
            self.v = float(self.st[core.S_V])
            self.boost = float(self.P[core.P_BOOSTOUT])      # emergent boost (Pa)
            self.turbo_rpm = self.st[core.S_TURBO] * 60.0 / (2.0 * math.pi)
        return out

    def warmup(self):
        """Trigger numba compilation off the audio thread (engine stays off)."""
        save = self.state
        self.state = "off"
        self.step_block(64)
        self.load_config(self.cfg)
        self.state = save

    # -------------------------------------------------------------------- audio
    def _callback(self, outdata, frames, time_info, status):
        try:
            out = self.step_block(frames)
        except Exception:
            outdata.fill(0.0)
            return
        # guard against any non-finite value reaching the sound card
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # soft limiter
        y = np.tanh(out * self.volume)
        outdata[:, 0] = y.astype(np.float32)
        # store tail for waveform display
        tp = self._tap_pos
        n = len(y)
        buf = self.audio_tap
        if tp + n <= len(buf):
            buf[tp:tp + n] = y
        else:
            k = len(buf) - tp
            buf[tp:] = y[:k]
            buf[:n - k] = y[k:]
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
