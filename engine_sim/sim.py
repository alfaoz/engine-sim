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
        self.volume = 0.6
        self.pedal = 0.0            # accelerator pedal 0..1
        self.load = 0.0             # extra accessory load (Nm)
        self.rpm = 0.0
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
        P[core.P_AIN] = (0.26 if port else 0.16) * Apist
        P[core.P_AEX] = (0.22 if port else 0.14) * Apist
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

        # cylinder phase offsets from firing order (even spacing)
        step = cyc / ncyl
        phase = np.zeros(ncyl)
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

        # state — engine off (not spinning) until the user cranks it
        self.st = np.array([0.0, 0.0])
        self.state = "off"
        self.v = 0.0
        self.idle_i = 0.0
        self.shift_timer = 0.0
        self.crank_timer = 0.0
        self.boost = 0.0
        self.thr_smooth = 0.0
        Vmid = Vclear + 0.5 * Vdisp
        self.cyl_T = np.full(ncyl, TINT)
        self.cyl_m = np.full(ncyl, PATM * Vmid / (R_AIR * TINT))

        rho0 = PATM / (R_AIR * TATM)
        self.rho = np.full(N, rho0)
        self.mom = np.zeros(N)
        self.Ene = np.full(N, PATM / (GAMMA_EX - 1.0))

        self.scope_p = np.full(N_SCOPE, PATM)
        self.filt = np.zeros(4)     # output filter state (DC block + lowpass)
        self._set_map(0.0)

    def _set_map(self, thr):
        # self.boost already holds the (spooled) boost pressure for this block
        cfg = self.cfg
        if cfg.diesel:
            self.P[core.P_MAP] = PATM + self.boost
        elif cfg.stroke_cycle == 2:
            # 2-stroke: crankcase delivery is throttle-dependent (instant, no
            # turbo lag), and tiny at closed throttle so it can idle
            self.P[core.P_MAP] = PATM * (0.03 + 0.97 * thr) + self.boost
        else:
            # low closed-throttle floor; idle governor opens it as needed
            self.P[core.P_MAP] = PATM * (0.12 + 0.88 * thr) + self.boost

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
            self.shift_timer = 0.15

    def shift_down(self):
        if self.gear > 0:
            self.gear -= 1
            self.shift_timer = 0.15

    def gear_label(self):
        return "N" if self.gear == 0 else str(self.gear)

    def speed_kmh(self):
        return self.v * 3.6

    # ----------------------------------------------------- control update
    def _control_update(self, dt):
        """Block-rate engine control + drivetrain + vehicle dynamics.
        Sets P_THROTTLE / P_LOAD / P_RUNNING / P_STARTER and integrates speed."""
        cfg = self.cfg
        veh = self.vehicle
        rpm = self.rpm
        idle = cfg.idle_rpm

        if self.shift_timer > 0.0:
            self.shift_timer = max(0.0, self.shift_timer - dt)

        # ---- engine state machine ----
        starter = 0.0
        running = 0.0
        eff_throttle = 0.0
        if self.state == "off":
            pass
        elif self.state == "cranking":
            running = 1.0
            starter = self.starter_torque
            eff_throttle = 0.35
            self.crank_timer += dt
            if rpm > 0.7 * idle:
                self.state = "running"
                self.idle_i = 0.18      # seed so throttle doesn't slam shut
            elif self.crank_timer > 3.0:
                self.state = "off"     # failed to catch
        else:  # running
            running = 1.0
            if self.pedal < 0.03:
                # fast PI idle governor (must catch decay of a big low-idle V8)
                err = (idle - rpm) / 1000.0
                self.idle_i += err * dt * 9.0
                self.idle_i = float(np.clip(self.idle_i, 0.0, 0.8))
                eff_throttle = float(np.clip(1.2 * err + self.idle_i, 0.0, 0.85))
            else:
                eff_throttle = self.pedal
                self.idle_i = max(self.idle_i * (1.0 - dt), 0.06)  # hold a seed
            if rpm < 0.22 * idle:
                self.state = "off"          # stalled

        # overrun backfire: closed throttle at elevated rpm (engine braking)
        pop = 0.0
        if self.state == "running" and self.pedal < 0.06 and rpm > idle * 1.8:
            pop = float(np.clip((rpm - idle * 1.8) / max(1.0, cfg.redline_rpm),
                                0.0, 1.0)) * self.backfire
        self.P[core.P_POP] = pop

        # ---- drivetrain: clutch + gearbox ----
        omega_eng = self.st[1]
        ratio = 0.0
        if self.gear >= 1 and self.gear <= veh.n_gears():
            ratio = veh.gear_ratios[self.gear - 1] * veh.final_drive

        T_clutch = 0.0
        if ratio > 0.0 and self.shift_timer <= 0.0 and self.state != "off":
            omega_wheel_eng = (self.v / veh.wheel_radius) * ratio
            slip = omega_eng - omega_wheel_eng
            # Two inertias (engine, vehicle) coupled by a clutch.
            cap = self.clutch_cap
            launch_w = (0.50 if cfg.diesel else 0.72) * cfg.redline_rpm * TWO_PI_60
            launching = self.pedal > 0.1 and omega_wheel_eng < launch_w

            if launching:
                # Slipping clutch as an engine-speed regulator targeting the
                # powerband: transmit torque proportional to how far the engine
                # is ABOVE launch_w (drives the car, drags down any over-rev),
                # and 0 when below (lets it spin up freely -> can't stall).
                T_clutch = float(np.clip(cap * (omega_eng - launch_w) / 60.0,
                                         0.0, cap))
            else:
                # Engaged: stiff coupling locks engine to wheels (small slip ->
                # large torque), giving rigid drive, cruise and engine braking.
                T_clutch = float(np.clip(cap * math.tanh(slip / 1.0), -cap, cap))

        # ---- vehicle longitudinal dynamics ----
        if veh.mass > 5.0:
            drive_force = (T_clutch * ratio / veh.wheel_radius) if ratio > 0 else 0.0
            roll = veh.crr * veh.mass * GRAV
            drag = 0.5 * RHO_AIR * veh.cd * veh.frontal_area * self.v * self.v
            brake_f = self.brake * veh.mass * 9.0       # up to ~0.9 g braking
            moving = self.v > 0.02
            resist = (roll + drag + brake_f) if moving else 0.0
            net_f = drive_force - resist
            self.v = max(0.0, self.v + net_f / veh.mass * dt)

        # ---- intake dynamics: throttle/fuel slew + turbo spool ----
        # the slew models tip-in fuel/intake lag; bypass it at idle so the
        # governor keeps direct authority (otherwise the two loops hunt/stall)
        tau_thr = 0.22 if cfg.diesel else 0.05      # diesels ramp fuel lazily
        if self.pedal < 0.03:
            self.thr_smooth = eff_throttle
        else:
            self.thr_smooth += (eff_throttle - self.thr_smooth) * min(1.0, dt / tau_thr)
        thr = self.thr_smooth

        if cfg.turbo_boost > 0.0 and cfg.stroke_cycle == 4:
            # turbocharger: boost needs exhaust energy (rpm) and lags (spool)
            spool = float(np.clip(
                (rpm - idle) / max(1.0, 0.5 * cfg.redline_rpm - idle), 0.0, 1.0))
            target = cfg.turbo_boost * thr * spool
            if target < self.boost:
                tau = 0.20                          # wastegate/decay is quick
            else:
                tau = 0.55 if cfg.diesel else 0.40  # spool-up lag
            self.boost += (target - self.boost) * min(1.0, dt / tau)
        else:
            self.boost = cfg.turbo_boost * thr      # 2-stroke crankcase: instant

        # ---- write into the sim parameter array ----
        self.P[core.P_THROTTLE] = thr
        self._set_map(thr)
        self.P[core.P_LOAD] = T_clutch + self.load
        self.P[core.P_RUNNING] = running
        self.P[core.P_STARTER] = starter

    # ------------------------------------------------------------------ stepping
    def step_block(self, n):
        out = np.zeros(n, dtype=np.float64)
        with self.lock:
            self._control_update(n / SAMPLE_RATE)
            t = simulate_block(n, self.P, self.st, self.cyl_m, self.cyl_T,
                               self.phase, self.inj, self.rho, self.mom,
                               self.Ene, out, self.scope_p, N_SCOPE, self.filt)
            self.torque = float(t)
            self.rpm = self.st[1] * 60.0 / (2.0 * math.pi)
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
            blocksize=BLOCK, callback=self._callback)
        self.stream.start()

    def stop_audio(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def is_audio_on(self):
        return self.stream is not None
