"""Dear PyGui control panel + live instrumentation for the engine simulator."""
import math
import threading
from collections import deque
import numpy as np
import dearpygui.dearpygui as dpg

from . import core
from .presets import (PRESETS, get_preset, EngineConfig,
                      VEHICLES, get_vehicle)
from .sim import EngineSim, SAMPLE_RATE

GEX = 1.33
RAIR = 287.0
PATM = 101325.0


class EngineUI:
    def __init__(self):
        self.preset_names = list(PRESETS.keys())
        self.vehicle_names = list(VEHICLES.keys())
        self.out_device_map = {"System default": None}
        try:
            import sounddevice as sd
            for i, d in enumerate(sd.query_devices()):
                if d.get('max_output_channels', 0) > 0:
                    self.out_device_map[f"{i}: {d['name']}"] = i
        except Exception:
            pass
        self.out_device_names = list(self.out_device_map.keys())
        self.sim = EngineSim(get_preset(self.preset_names[0]))
        self._building = False
        self._kb_throttle = 0.0
        self._dyno_thread = None
        self._dyno_result = None      # (rpm[], torque[], hp[], peakT, peakP)
        self._dyno_running = False
        # rolling time-history traces (one point per UI frame)
        self.HIST = 360
        self.h_t = deque([0.0] * self.HIST, maxlen=self.HIST)      # time axis
        self.h_map = deque([0.0] * self.HIST, maxlen=self.HIST)
        self.h_boost = deque([0.0] * self.HIST, maxlen=self.HIST)
        self.h_turbo = deque([0.0] * self.HIST, maxlen=self.HIST)  # krpm
        self.h_rpm = deque([0.0] * self.HIST, maxlen=self.HIST)
        self._frame = 0
        # live torque/power vs rpm, accumulated (peak per rpm bin) while driving
        self.LIVE_NB = 60
        self.live_tq = np.full(self.LIVE_NB, np.nan)
        self.live_hp = np.full(self.LIVE_NB, np.nan)
        self._tq_s = 0.0      # time-smoothed torque (cycle-average) for live plot
        self._hp_s = 0.0

    # ------------------------------------------------------------ config glue
    def current_edited_config(self) -> EngineConfig:
        g = dpg.get_value
        cyc = 4 if g("cycle") == "4-stroke" else 2
        return EngineConfig(
            name=self.sim.cfg.name,
            bore=g("bore") / 1000.0,
            stroke=g("stroke") / 1000.0,
            conrod=g("conrod") / 1000.0,
            compression_ratio=g("cr"),
            n_cylinders=int(g("ncyl")),
            stroke_cycle=cyc,
            diesel=g("diesel"),
            turbo_boost=g("boost") * 1000.0,
            ignition_btdc=g("ign"),
            burn_duration=g("burn"),
            evo=g("evo"), evc=g("evc"), ivo=g("ivo"), ivc=g("ivc"),
            redline_rpm=g("redline"), idle_rpm=g("idle"),
            inertia=g("inertia"),
            pipe_length=g("pipelen"), pipe_diameter=g("pipedia") / 1000.0,
            runner_spread=g("spread"),
            firing_order=self.sim.cfg.firing_order,
        )

    def push_config_to_widgets(self, cfg: EngineConfig):
        self._building = True
        dpg.set_value("bore", cfg.bore * 1000)
        dpg.set_value("stroke", cfg.stroke * 1000)
        dpg.set_value("conrod", cfg.conrod * 1000)
        dpg.set_value("cr", cfg.compression_ratio)
        dpg.set_value("ncyl", cfg.n_cylinders)
        dpg.set_value("cycle", "4-stroke" if cfg.stroke_cycle == 4 else "2-stroke")
        dpg.set_value("diesel", cfg.diesel)
        dpg.set_value("boost", cfg.turbo_boost / 1000)
        dpg.set_value("ign", cfg.ignition_btdc)
        dpg.set_value("burn", cfg.burn_duration)
        dpg.set_value("evo", cfg.evo); dpg.set_value("evc", cfg.evc)
        dpg.set_value("ivo", cfg.ivo); dpg.set_value("ivc", cfg.ivc)
        dpg.set_value("redline", cfg.redline_rpm)
        dpg.set_value("idle", cfg.idle_rpm)
        dpg.set_value("inertia", cfg.inertia)
        dpg.set_value("pipelen", cfg.pipe_length)
        dpg.set_value("pipedia", cfg.pipe_diameter * 1000)
        dpg.set_value("spread", cfg.runner_spread)
        self._building = False
        self._update_info()

    def _update_info(self):
        cfg = self.current_edited_config()
        dpg.set_value("info_disp",
                      f"Displacement: {cfg.displacement_cc():.0f} cc  "
                      f"({cfg.cylinder_cc():.0f} cc x {cfg.n_cylinders})")

    # ------------------------------------------------------------- callbacks
    def on_preset(self, sender, name):
        was_on = self.sim.is_audio_on()
        self.sim.stop_audio()
        self.sim.load_config(get_preset(name))
        self.push_config_to_widgets(self.sim.cfg)
        if was_on:
            self.sim.warmup()
            self.sim.start_audio()
            dpg.set_value("audio_btn_state", True)

    def on_rebuild(self, *_):
        if self._building:
            return
        was_on = self.sim.is_audio_on()
        self.sim.stop_audio()
        self.sim.load_config(self.current_edited_config())
        self._update_info()
        if was_on:
            self.sim.warmup()
            self.sim.start_audio()

    def on_geometry_edit(self, *_):
        if not self._building:
            self._update_info()

    def on_throttle(self, s, v):
        self.sim.set_throttle(v / 100.0)

    def on_load(self, s, v):
        self.sim.set_load(v)

    def on_brake(self, s, v):
        self.sim.set_brake(v / 100.0)

    def on_backfire(self, s, v):
        self.sim.backfire = v

    def on_vehicle(self, s, name):
        self.sim.set_vehicle(get_vehicle(name))

    def on_start(self, *_):
        self.sim.start_engine()

    def on_stop(self, *_):
        self.sim.stop_engine()

    def on_shift_up(self, *_):
        self.sim.shift_up()

    def on_shift_down(self, *_):
        self.sim.shift_down()

    def on_volume(self, s, v):
        self.sim.volume = v

    def on_device(self, s, name):
        self.sim.set_device(self.out_device_map.get(name))

    def on_mix_induction(self, s, on):
        self.sim.mix_induction = 0.35 if on else 0.0
        self.sim.P[core.P_INGAIN] = self.sim.mix_induction

    def on_mix_body(self, s, on):
        self.sim.mix_body = 0.35 if on else 0.0
        self.sim.P[core.P_BODYGAIN] = self.sim.mix_body

    def on_mix_sat(self, s, on):
        self.sim.mix_sat = 1.0 if on else 0.0
        self.sim.P[core.P_SAT] = self.sim.mix_sat

    def on_clear_live(self, *_):
        self.live_tq[:] = np.nan
        self.live_hp[:] = np.nan

    def on_lock_graphs(self, s, locked):
        # locked => no mouse zoom/pan on any plot (autoscale still runs)
        for tag in ("plot_pipe", "plot_cyl", "plot_wav", "plot_intk",
                    "plot_live", "plot_turbo", "plot_map", "plot_dyno"):
            dpg.configure_item(tag, no_inputs=bool(locked))

    def on_live_timing(self, *_):
        # ignition / burn / redline / output can update without a rebuild
        cfg = self.sim.cfg
        cyc_deg = 720.0 if cfg.stroke_cycle == 4 else 360.0
        self.sim.P[core.P_IGN] = math.radians(cyc_deg - dpg.get_value("ign"))
        self.sim.P[core.P_BURN] = math.radians(dpg.get_value("burn"))
        self.sim.P[core.P_REDLINE] = dpg.get_value("redline")
        self.sim.P[core.P_OUTGAIN] = 1.0 / dpg.get_value("outscale")
        self.sim.P[core.P_DAMP] = dpg.get_value("damp")

    def on_audio_toggle(self, s, v):
        if v:
            self.sim.warmup()
            self.sim.start_audio()
        else:
            self.sim.stop_audio()

    def blip(self, *_):
        # momentary throttle stab handled by setting slider; here just full
        self.sim.set_throttle(1.0)
        dpg.set_value("throttle", 100)

    # --------------------------------------------------------------- dyno sweep
    def on_dyno(self, *_):
        if self._dyno_running:
            return
        self._dyno_running = True
        dpg.configure_item("dyno_btn", enabled=False)
        dpg.set_value("dyno_status", "running sweep...")
        cfg = self.sim.cfg
        self._dyno_thread = threading.Thread(
            target=self._run_dyno, args=(cfg,), daemon=True)
        self._dyno_thread.start()

    def _run_dyno(self, cfg):
        """Inertial dyno on a private sim instance (doesn't touch live audio):
        free-rev at WOT with a big flywheel, brake torque = I*dw/dt."""
        try:
            import math as _m
            sim = EngineSim(cfg)
            sim.step_block(64)
            sim.load_config(cfg)
            sim.start_engine()
            for _ in range(60):
                sim.step_block(self.sim_block())
                if sim.state == "running":
                    break
            for _ in range(60):
                sim.step_block(self.sim_block())
            # Heavy flywheel scaled to engine size: the slow rev-up gives many
            # samples per rpm and lets the turbo track quasi-steadily (a light
            # flywheel revs too fast -> noisy curve + boost lag). Tiny engines
            # still need a small wheel or they'd take an age to spin up.
            Vd = (_m.pi / 4 * cfg.bore ** 2 * cfg.stroke * cfg.n_cylinders)
            disp_l = Vd * 1e3
            FLY = float(min(9.0, max(0.35, 3.5 * disp_l)))
            sim.base_inertia = FLY
            sim.set_throttle(1.0)
            dt = self.sim_block() / SAMPLE_RATE
            lo = cfg.idle_rpm * 1.4
            hi = cfg.redline_rpm * 0.97
            nb = 40
            bin_sum = np.zeros(nb)
            bin_cnt = np.zeros(nb)
            for _ in range(9000):
                w0 = sim.st[core.S_OMEGA]
                sim.step_block(self.sim_block())
                w1 = sim.st[core.S_OMEGA]
                rpm = w1 * 60 / (2 * _m.pi)
                tq = FLY * (w1 - w0) / dt
                if sim.state == "running" and lo <= rpm <= hi:
                    b = int((rpm - lo) / (hi - lo) * nb)
                    if 0 <= b < nb:
                        bin_sum[b] += tq
                        bin_cnt[b] += 1.0
                if rpm > hi or sim.state == "off":
                    break
            if bin_cnt.sum() < 8:
                self._dyno_result = ([], [], [], 0.0, 0.0, 0.0, 0.0)
                return
            grid = lo + (np.arange(nb) + 0.5) / nb * (hi - lo)
            # average torque per rpm bin, fill empties by interpolation
            tqg = np.where(bin_cnt > 0, bin_sum / np.maximum(bin_cnt, 1), np.nan)
            ok = ~np.isnan(tqg)
            tqg = np.interp(grid, grid[ok], tqg[ok])
            # smooth
            k = np.ones(5) / 5.0
            tqg = np.convolve(tqg, k, mode="same")
            hp = tqg * (grid * 2 * np.pi / 60) / 745.7
            iT = int(np.argmax(tqg)); iP = int(np.argmax(hp))
            self._dyno_result = (grid.tolist(), tqg.tolist(), hp.tolist(),
                                 float(tqg[iT]), float(grid[iT]),
                                 float(hp[iP]), float(grid[iP]))
        except Exception as e:
            self._dyno_result = ([], [], [], 0.0, 0.0, 0.0, 0.0)
            print("dyno error:", e)

    def sim_block(self):
        return 512

    def _apply_dyno_result(self):
        if self._dyno_result is None:
            return
        grid, tqg, hp, pkT, pkTrpm, pkP, pkPrpm = self._dyno_result
        self._dyno_result = None
        self._dyno_running = False
        dpg.configure_item("dyno_btn", enabled=True)
        if not grid:
            dpg.set_value("dyno_status", "sweep failed")
            return
        dpg.set_value("dyno_tq", [grid, tqg])
        dpg.set_value("dyno_hp", [grid, hp])
        dpg.fit_axis_data("dyno_x")
        dpg.fit_axis_data("dyno_y")
        dpg.set_value("dyno_status",
                      f"peak {pkT:.0f} Nm @ {pkTrpm:.0f},  "
                      f"{pkP:.0f} hp @ {pkPrpm:.0f}")

    # ----------------------------------------------------------- firing diagram
    def _draw_firing(self):
        sim = self.sim
        ncyl = sim.cfg.n_cylinders
        cyc = 4.0 * math.pi if sim.cfg.stroke_cycle == 4 else 2.0 * math.pi
        theta = sim.st[core.S_THETA]
        dpg.delete_item("fire_draw", children_only=True)
        cols = min(ncyl, 6)
        rows = (ncyl + cols - 1) // cols
        cw, ch = 222, 92
        r = min(16, (cw / cols) * 0.38, (ch / max(1, rows)) * 0.38)
        ign = sim.cfg.ignition_btdc
        burn = sim.cfg.burn_duration
        cyc_deg = 720.0 if sim.cfg.stroke_cycle == 4 else 360.0
        # 4-stroke phase windows (cycle deg from firing TDC): power/exh/int/comp
        for c in range(ncyl):
            col = c % cols
            row = c // cols
            cx = (col + 0.5) * (cw / cols)
            cy = (row + 0.5) * (ch / rows)
            cph = ((theta + sim.phase[c]) % cyc)
            deg = cph / cyc * cyc_deg
            # stroke color
            if sim.cfg.stroke_cycle == 4:
                if deg < 180:   col_rgb = (230, 90, 60)    # power
                elif deg < 360: col_rgb = (110, 110, 120)  # exhaust
                elif deg < 540: col_rgb = (70, 150, 230)   # intake
                else:           col_rgb = (170, 150, 70)   # compression
            else:
                col_rgb = (230, 90, 60) if deg < 180 else (90, 130, 200)
            # combustion flash: bright when within the burn window after ignition
            d_from_ign = (deg - (cyc_deg - ign)) % cyc_deg
            flashing = d_from_ign < burn * 1.4
            if flashing:
                f = 1.0 - d_from_ign / (burn * 1.4)
                col_rgb = (255, int(200 + 55 * f), int(60 + 40 * f))
            dpg.draw_circle((cx, cy), r, fill=col_rgb,
                            color=(20, 20, 20), parent="fire_draw")
            dpg.draw_text((cx - 4, cy - 6), str(c + 1), size=12,
                          color=(10, 10, 10), parent="fire_draw")

    # ------------------------------------------------------------------- build
    def build(self):
        dpg.create_context()
        dpg.create_viewport(title="Engine Simulator — physics + 1D exhaust gas dynamics",
                            width=1480, height=900)

        with dpg.window(tag="main"):
            # ---- top control bar ----
            with dpg.group(horizontal=True):
                dpg.add_text("Engine:")
                dpg.add_combo(self.preset_names, default_value=self.preset_names[0],
                              width=220, callback=self.on_preset, tag="preset")
                dpg.add_text("Vehicle:")
                dpg.add_combo(self.vehicle_names, default_value=self.vehicle_names[0],
                              width=190, callback=self.on_vehicle, tag="vehicle")
                dpg.add_checkbox(label="SOUND", callback=self.on_audio_toggle,
                                 tag="audio_btn_state")
                dpg.add_text("Out:")
                dpg.add_combo(self.out_device_names, default_value="System default",
                              width=210, callback=self.on_device, tag="outdev")
                dpg.add_text("Vol")
                dpg.add_slider_float(tag="vol", default_value=0.6, min_value=0.0,
                                     max_value=1.5, width=110, callback=self.on_volume)
                dpg.add_button(label="Blip", callback=self.blip)
                dpg.add_checkbox(label="Lock graphs", default_value=True,
                                 callback=self.on_lock_graphs, tag="lock_graphs")
                dpg.add_text("  Mix:")
                dpg.add_checkbox(label="Induction", default_value=False,
                                 callback=self.on_mix_induction, tag="mix_ind")
                dpg.add_checkbox(label="Body", default_value=True,
                                 callback=self.on_mix_body, tag="mix_body")
                dpg.add_checkbox(label="Sat", default_value=True,
                                 callback=self.on_mix_sat, tag="mix_sat")
                dpg.add_text("", tag="info_disp")

            dpg.add_separator()

            with dpg.group(horizontal=True):
                # ---- left: drive controls / readouts ----
                with dpg.child_window(width=235, height=720):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="START (Enter)", width=110,
                                       callback=self.on_start)
                        dpg.add_button(label="STOP", width=90, callback=self.on_stop)
                    dpg.add_text("state: off", tag="state_val", color=(255, 220, 120))
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="< Down (Q)", width=100,
                                       callback=self.on_shift_down)
                        dpg.add_button(label="Up (E) >", width=100,
                                       callback=self.on_shift_up)
                    with dpg.group(horizontal=True):
                        dpg.add_text("Gear")
                        dpg.add_text("N", tag="gear_val", color=(255, 230, 120))
                        dpg.add_text("   Speed")
                        dpg.add_text("0 km/h", tag="speed_val", color=(160, 220, 255))
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("GAS (W)")
                            dpg.add_slider_float(tag="throttle", default_value=0.0,
                                                 min_value=0.0, max_value=100.0,
                                                 vertical=True, height=180, width=60,
                                                 callback=self.on_throttle)
                        with dpg.group():
                            dpg.add_text("BRAKE (S)")
                            dpg.add_slider_float(tag="brake", default_value=0.0,
                                                 min_value=0.0, max_value=100.0,
                                                 vertical=True, height=180, width=60,
                                                 callback=self.on_brake)
                    dpg.add_text("Accessory load (Nm)")
                    dpg.add_slider_float(tag="load", default_value=0.0, min_value=0.0,
                                         max_value=400.0, width=200,
                                         callback=self.on_load)
                    dpg.add_separator()
                    dpg.add_text("RPM", color=(120, 200, 255))
                    dpg.add_text("0", tag="rpm_val")
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("Power", color=(255, 220, 120))
                            dpg.add_text("0", tag="pwr_val")
                            dpg.add_text("Torque (Nm)", color=(120, 255, 160))
                            dpg.add_text("0", tag="trq_val")
                            dpg.add_text("BMEP (bar)", color=(160, 230, 160))
                            dpg.add_text("0", tag="bmep_val")
                            dpg.add_text("MAP (kPa)", color=(255, 200, 120))
                            dpg.add_text("0", tag="map_val")
                        with dpg.group():
                            dpg.add_text("Boost (kPa)", color=(255, 180, 120))
                            dpg.add_text("0", tag="boost_val")
                            dpg.add_text("VE (%)", color=(150, 220, 255))
                            dpg.add_text("0", tag="ve_val")
                            dpg.add_text("EGT (°C)", color=(255, 150, 120))
                            dpg.add_text("0", tag="egt_val")
                            dpg.add_text("Peak cyl (bar)", color=(255, 160, 160))
                            dpg.add_text("0", tag="pk_val")
                    dpg.add_text("λ (lambda)", color=(200, 200, 255))
                    dpg.add_text("0", tag="lam_val")
                    dpg.add_separator()
                    dpg.add_text("FIRING", color=(255, 200, 150))
                    with dpg.drawlist(width=222, height=92, tag="fire_draw"):
                        pass
                    dpg.add_text("", tag="fire_lbl", color=(170, 170, 170))

                # ---- middle: plots (2-column grid of compact graphs) ----
                with dpg.child_window(width=800, height=760):
                    pw = 388          # per-plot width in the 2-up rows
                    ph = 168          # compact plot height

                    with dpg.group(horizontal=True):
                        with dpg.plot(label="Exhaust pipe pressure (wave -> sound)",
                                      height=ph, width=pw, no_inputs=True,
                                      tag="plot_pipe"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="position (m)")
                            with dpg.plot_axis(dpg.mvYAxis, label="kPa gauge",
                                               tag="pipe_y"):
                                dpg.add_line_series([0], [0], tag="pipe_series")
                        with dpg.plot(label="Cylinder #1 pressure vs crank",
                                      height=ph, width=pw, no_inputs=True,
                                      tag="plot_cyl"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="cycle angle (deg)")
                            with dpg.plot_axis(dpg.mvYAxis, label="bar",
                                               tag="cyl_y"):
                                dpg.add_line_series([0], [0], tag="cyl_series")

                    with dpg.group(horizontal=True):
                        with dpg.plot(label="Tailpipe audio waveform",
                                      height=ph, width=pw, no_inputs=True,
                                      tag="plot_wav"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="sample")
                            with dpg.plot_axis(dpg.mvYAxis, label="amp",
                                               tag="wav_y"):
                                dpg.add_line_series([0], [0], tag="wav_series")
                        with dpg.plot(label="Intake duct pressure (induction wave)",
                                      height=ph, width=pw, no_inputs=True,
                                      tag="plot_intk"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="position (m)")
                            with dpg.plot_axis(dpg.mvYAxis, label="kPa gauge",
                                               tag="intk_y"):
                                dpg.add_line_series([0], [0], tag="intk_series")

                    with dpg.group(horizontal=True):
                        with dpg.plot(label="Live torque & power vs RPM",
                                      height=ph, width=pw, no_inputs=True,
                                      tag="plot_live"):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="RPM", tag="live_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="Nm / hp",
                                               tag="live_y"):
                                dpg.add_line_series([0], [0], tag="live_tq_s",
                                                    label="torque")
                                dpg.add_line_series([0], [0], tag="live_hp_s",
                                                    label="power")
                        with dpg.plot(label="Boost & turbo speed vs time",
                                      height=ph, width=pw, no_inputs=True,
                                      tag="plot_turbo"):
                            dpg.add_plot_legend()
                            dpg.add_plot_axis(dpg.mvXAxis, label="frame", tag="turbo_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="kPa / krpm",
                                               tag="turbo_y"):
                                dpg.add_line_series([0], [0], tag="boost_s",
                                                    label="boost kPa")
                                dpg.add_line_series([0], [0], tag="turbo_s",
                                                    label="turbo krpm")

                    with dpg.group(horizontal=True):
                        with dpg.plot(label="Manifold pressure (MAP) vs time",
                                      height=ph, width=pw, no_inputs=True,
                                      tag="plot_map"):
                            dpg.add_plot_axis(dpg.mvXAxis, label="frame", tag="map_x")
                            with dpg.plot_axis(dpg.mvYAxis, label="kPa", tag="map_y"):
                                dpg.add_line_series([0], [0], tag="map_s")
                        with dpg.group():
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="RUN DYNO", width=100,
                                               callback=self.on_dyno, tag="dyno_btn")
                                dpg.add_button(label="clear live", width=80,
                                               callback=self.on_clear_live)
                            dpg.add_text("WOT sweep", tag="dyno_status",
                                         color=(170, 170, 170))
                            with dpg.plot(label="Dyno — torque & power vs RPM",
                                          height=ph - 36, width=pw, no_inputs=True,
                                          tag="plot_dyno"):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="RPM",
                                                  tag="dyno_x")
                                with dpg.plot_axis(dpg.mvYAxis, label="Nm / hp",
                                                   tag="dyno_y"):
                                    dpg.add_line_series([0], [0], tag="dyno_tq",
                                                        label="torque")
                                    dpg.add_line_series([0], [0], tag="dyno_hp",
                                                        label="power")

                # ---- right: engine design parameters ----
                with dpg.child_window(width=300, height=720):
                    dpg.add_text("GEOMETRY", color=(180, 180, 255))
                    self._num("bore", "Bore (mm)", 80, 20, 140)
                    self._num("stroke", "Stroke (mm)", 80, 20, 140)
                    self._num("conrod", "Conrod (mm)", 140, 50, 250)
                    self._num("cr", "Compression ratio", 10, 5, 24)
                    dpg.add_input_int(label="Cylinders", tag="ncyl", default_value=4,
                                      min_value=1, max_value=12, width=120,
                                      callback=self.on_geometry_edit)
                    dpg.add_radio_button(("4-stroke", "2-stroke"), tag="cycle",
                                         default_value="4-stroke", horizontal=True,
                                         callback=self.on_geometry_edit)
                    dpg.add_checkbox(label="Diesel (compression ignition)",
                                     tag="diesel", callback=self.on_geometry_edit)
                    self._num("boost", "Turbo boost (kPa)", 0, 0, 250)
                    dpg.add_separator()
                    dpg.add_text("COMBUSTION / VALVES", color=(255, 200, 150))
                    self._num("ign", "Ignition BTDC (deg)", 22, 0, 60, live=True)
                    self._num("burn", "Burn duration (deg)", 55, 20, 120, live=True)
                    self._num("evo", "EVO (deg)", 145, 90, 220)
                    self._num("evc", "EVC (deg)", 375, 320, 420)
                    self._num("ivo", "IVO (deg)", 355, 300, 420)
                    self._num("ivc", "IVC (deg)", 585, 500, 640)
                    dpg.add_separator()
                    dpg.add_text("EXHAUST", color=(150, 255, 220))
                    self._num("pipelen", "Pipe length (m)", 1.5, 0.3, 3.0)
                    self._num("pipedia", "Pipe dia (mm)", 50, 18, 90)
                    self._num("spread", "Runner spread", 0.2, 0.0, 0.4)
                    dpg.add_separator()
                    dpg.add_text("DYNAMICS", color=(220, 220, 220))
                    self._num("redline", "Redline (rpm)", 7000, 3000, 12000, live=True)
                    self._num("idle", "Idle (rpm)", 850, 400, 2000)
                    self._num("inertia", "Flywheel inertia", 0.18, 0.01, 1.2)
                    dpg.add_separator()
                    dpg.add_text("AUDIO/SIM TUNING", color=(200, 200, 200))
                    self._num("outscale", "Output scale (Pa/unit)", 2600, 300, 20000,
                              live=True)
                    self._num("damp", "Exhaust damping", 0.0015, 0.0, 0.02, live=True)
                    dpg.add_slider_float(tag="backfire", label="Backfire (overrun)",
                                         default_value=1.0, min_value=0.0,
                                         max_value=3.0, width=120,
                                         callback=self.on_backfire)
                    dpg.add_separator()
                    dpg.add_button(label="REBUILD ENGINE", width=-1,
                                   callback=self.on_rebuild,
                                   tag="rebuild_btn")

        self.push_config_to_widgets(self.sim.cfg)

        with dpg.handler_registry():
            dpg.add_key_press_handler(dpg.mvKey_E, callback=self.on_shift_up)
            dpg.add_key_press_handler(dpg.mvKey_Q, callback=self.on_shift_down)
            dpg.add_key_press_handler(dpg.mvKey_Return, callback=self.on_start)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("main", True)

    def _num(self, tag, label, default, mn, mx, live=False):
        cb = self.on_live_timing if live else self.on_geometry_edit
        dpg.add_input_float(tag=tag, label=label, default_value=float(default),
                            min_value=float(mn), max_value=float(mx), width=120,
                            step=0, callback=cb, on_enter=False)

    # ------------------------------------------------------------------- frame
    def update_plots(self):
        sim = self.sim

        # keyboard throttle: hold W or Up arrow. Note: we keep writing through
        # the entire release ramp (including the final 0) so the throttle can't
        # get stuck at a small residual that would disable the idle governor.
        if dpg.is_key_down(dpg.mvKey_W) or dpg.is_key_down(dpg.mvKey_Up):
            self._kb_throttle = min(1.0, self._kb_throttle + 0.08)
            sim.set_throttle(self._kb_throttle)
            dpg.set_value("throttle", self._kb_throttle * 100.0)
        elif self._kb_throttle > 0.0:
            self._kb_throttle = max(0.0, self._kb_throttle - 0.12)
            sim.set_throttle(self._kb_throttle)
            dpg.set_value("throttle", self._kb_throttle * 100.0)

        # keyboard brake: hold S or Down arrow
        if dpg.is_key_down(dpg.mvKey_S) or dpg.is_key_down(dpg.mvKey_Down):
            sim.set_brake(1.0)
            dpg.set_value("brake", 100.0)
        elif sim.brake > 0.0:
            sim.set_brake(0.0)
            dpg.set_value("brake", 0.0)

        # advance sim in UI thread only when audio isn't driving it
        if not sim.is_audio_on():
            sim.step_block(1500)

        # drive readouts
        dpg.set_value("state_val", f"state: {sim.state}")
        dpg.set_value("gear_val", sim.gear_label())
        dpg.set_value("speed_val", f"{sim.speed_kmh():.0f} km/h")

        # ---- readouts / live metrics ----
        cfg = sim.cfg
        omega = sim.rpm * 2.0 * math.pi / 60.0
        hp = sim.torque * omega / 745.7
        kw = sim.torque * omega / 1000.0
        nrev = 2.0 if cfg.stroke_cycle == 4 else 1.0
        vd_tot = max(1e-9, sim.disp_l * 1e-3)        # litres -> m^3
        bmep = sim.torque * (2.0 * math.pi * nrev) / vd_tot / 1e5
        MAP = sim.P[core.P_MAP]
        Tman = max(1.0, sim.P[core.P_TMAN])
        ve = MAP * 300.0 / (PATM * Tman) * 100.0     # vs atmospheric density
        # exhaust gas temp from the hot cells near the head
        N = sim.N
        k0, k1 = max(1, N // 20), max(2, N // 6)
        rho = sim.rho[k0:k1]; mom = sim.mom[k0:k1]; Ene = sim.Ene[k0:k1]
        with np.errstate(all='ignore'):
            u = mom / rho
            p = (GEX - 1.0) * (Ene - 0.5 * rho * u * u)
            egt = np.nanmean(p / (rho * RAIR)) - 273.0
        if not np.isfinite(egt):
            egt = 0.0
        boost = max(0.0, getattr(sim, "boost", 0.0)) / 1000.0
        # lambda: petrol burns ~stoich; diesel runs lean by fuelling
        if sim.state != "running":
            lam = float("nan")
        elif cfg.diesel:
            thr = max(0.02, sim.P[core.P_THROTTLE])
            lam = (23.0 / thr) / 14.5
        else:
            lam = 1.0
        dpg.set_value("rpm_val", f"{sim.rpm:7.0f}")
        dpg.set_value("pwr_val", f"{hp:.0f} hp ({kw:.0f} kW)")
        dpg.set_value("trq_val", f"{sim.torque:7.1f}")
        dpg.set_value("bmep_val", f"{bmep:6.1f}")
        dpg.set_value("map_val", f"{MAP/1000:6.1f}")
        dpg.set_value("boost_val", f"{boost:6.1f}")
        dpg.set_value("ve_val", f"{ve:5.0f}")
        dpg.set_value("egt_val", f"{egt:5.0f}")
        dpg.set_value("pk_val", f"{sim.scope_p.max()/1e5:6.1f}")
        dpg.set_value("lam_val", "—" if not np.isfinite(lam) else f"{lam:4.2f}")

        # exhaust pipe pressure profile (the wave)
        N = sim.N
        rho = sim.rho[:N]; mom = sim.mom[:N]; Ene = sim.Ene[:N]
        with np.errstate(all='ignore'):
            u = mom / rho
            p = (GEX - 1.0) * (Ene - 0.5 * rho * u * u)
        p = np.nan_to_num(p, nan=PATM, posinf=PATM, neginf=PATM)
        xs = np.linspace(0, sim.cfg.pipe_length, N)
        dpg.set_value("pipe_series", [xs.tolist(), ((p - PATM) / 1000.0).tolist()])
        dpg.fit_axis_data("pipe_y")

        # cylinder pressure scope
        ns = len(sim.scope_p)
        cyc_deg = 720.0 if sim.cfg.stroke_cycle == 4 else 360.0
        cx = np.linspace(0, cyc_deg, ns)
        dpg.set_value("cyl_series", [cx.tolist(), (sim.scope_p / 1e5).tolist()])
        dpg.fit_axis_data("cyl_y")

        # audio waveform
        tap = sim.audio_tap
        pos = sim._tap_pos
        wav = np.concatenate([tap[pos:], tap[:pos]])[-1024:]
        dpg.set_value("wav_series", [list(range(len(wav))), wav.tolist()])
        dpg.set_axis_limits("wav_y", -1.05, 1.05)

        # intake duct pressure profile (induction wave; static unless mixed in)
        Ni = sim.Ni
        if Ni > 0:
            ri = sim.rho_in[:Ni]; mi = sim.mom_in[:Ni]; ei = sim.Ene_in[:Ni]
            with np.errstate(all='ignore'):
                ui = mi / ri
                pin = (1.34 - 1.0) * (ei - 0.5 * ri * ui * ui)
            pin = np.nan_to_num(pin, nan=PATM, posinf=PATM, neginf=PATM)
            xi = np.linspace(0, sim.cfg.intake_length, Ni)
            dpg.set_value("intk_series", [xi.tolist(),
                                          ((pin - PATM) / 1000.0).tolist()])
            dpg.fit_axis_data("intk_y")

        # live torque & power vs RPM. Only sample during a clean WOT pull (full
        # throttle, running, clutch locked / not shifting) and average per rpm
        # bin -- otherwise partial-throttle, engine-braking and shift transients
        # turn it into noise. This converges toward the real power curve.
        # time-smooth torque/power over several cycles so few-cylinder engines
        # (whose per-block torque is only a slice of a cycle) don't spike it
        self._tq_s += (sim.torque - self._tq_s) * 0.06
        self._hp_s += (hp - self._hp_s) * 0.06
        wot_pull = (sim.state == "running" and sim.pedal > 0.85
                    and sim.shift_timer <= 0.0 and sim.rpm > cfg.idle_rpm * 1.2)
        if wot_pull:
            b = int(sim.rpm / max(1.0, cfg.redline_rpm) * self.LIVE_NB)
            if 0 <= b < self.LIVE_NB:
                if np.isnan(self.live_tq[b]):
                    self.live_tq[b] = self._tq_s
                    self.live_hp[b] = self._hp_s
                else:
                    self.live_tq[b] += (self._tq_s - self.live_tq[b]) * 0.15
                    self.live_hp[b] += (self._hp_s - self.live_hp[b]) * 0.15
        lx = (np.arange(self.LIVE_NB) + 0.5) / self.LIVE_NB * cfg.redline_rpm
        m = ~np.isnan(self.live_tq)
        if m.any():
            dpg.set_value("live_tq_s", [lx[m].tolist(), self.live_tq[m].tolist()])
            dpg.set_value("live_hp_s", [lx[m].tolist(), self.live_hp[m].tolist()])
            dpg.fit_axis_data("live_x"); dpg.fit_axis_data("live_y")

        # rolling time-history traces (boost / turbo speed / MAP)
        self._frame += 1
        self.h_boost.append(boost)
        self.h_turbo.append(getattr(sim, "turbo_rpm", 0.0) / 1000.0)
        self.h_map.append(MAP / 1000.0)
        fx = list(range(len(self.h_boost)))
        dpg.set_value("boost_s", [fx, list(self.h_boost)])
        dpg.set_value("turbo_s", [fx, list(self.h_turbo)])
        dpg.fit_axis_data("turbo_x"); dpg.fit_axis_data("turbo_y")
        dpg.set_value("map_s", [fx, list(self.h_map)])
        dpg.fit_axis_data("map_x"); dpg.fit_axis_data("map_y")

        # firing diagram + dyno result hand-off from the worker thread
        self._draw_firing()
        if self._dyno_result is not None:
            self._apply_dyno_result()

    def run(self):
        self.build()
        while dpg.is_dearpygui_running():
            self.update_plots()
            dpg.render_dearpygui_frame()
        self.sim.stop_audio()
        dpg.destroy_context()


def main():
    EngineUI().run()
