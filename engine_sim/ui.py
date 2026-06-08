"""Dear PyGui control panel + live instrumentation for the engine simulator."""
import math
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
        self.sim = EngineSim(get_preset(self.preset_names[0]))
        self.rpm_hist = deque([0.0] * 240, maxlen=240)
        self.trq_hist = deque([0.0] * 240, maxlen=240)
        self._building = False
        self._kb_throttle = 0.0

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

    # ------------------------------------------------------------------- build
    def build(self):
        dpg.create_context()
        dpg.create_viewport(title="Engine Simulator — physics + 1D exhaust gas dynamics",
                            width=1280, height=860)

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
                dpg.add_text("Vol")
                dpg.add_slider_float(tag="vol", default_value=0.6, min_value=0.0,
                                     max_value=1.5, width=120, callback=self.on_volume)
                dpg.add_button(label="Blip", callback=self.blip)
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
                    dpg.add_text("Torque (Nm)", color=(120, 255, 160))
                    dpg.add_text("0", tag="trq_val")
                    dpg.add_text("MAP (kPa)", color=(255, 200, 120))
                    dpg.add_text("0", tag="map_val")
                    dpg.add_text("Peak cyl (bar)", color=(255, 160, 160))
                    dpg.add_text("0", tag="pk_val")

                # ---- middle: plots ----
                with dpg.child_window(width=720, height=720):
                    with dpg.plot(label="Exhaust pipe pressure (the travelling wave -> sound)",
                                  height=210, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="position (m)")
                        with dpg.plot_axis(dpg.mvYAxis, label="gauge pressure (kPa)",
                                           tag="pipe_y"):
                            dpg.add_line_series([0], [0], tag="pipe_series")
                    with dpg.plot(label="Cylinder #1 pressure vs crank angle",
                                  height=210, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="cycle angle (deg)")
                        with dpg.plot_axis(dpg.mvYAxis, label="pressure (bar)",
                                           tag="cyl_y"):
                            dpg.add_line_series([0], [0], tag="cyl_series")
                    with dpg.plot(label="Tailpipe audio waveform",
                                  height=210, width=-1):
                        dpg.add_plot_axis(dpg.mvXAxis, label="sample")
                        with dpg.plot_axis(dpg.mvYAxis, label="amplitude",
                                           tag="wav_y"):
                            dpg.add_line_series([0], [0], tag="wav_series")

                # ---- right: engine design parameters ----
                with dpg.child_window(width=300, height=720):
                    dpg.add_text("GEOMETRY", color=(180, 180, 255))
                    self._num("bore", "Bore (mm)", 80, 20, 140)
                    self._num("stroke", "Stroke (mm)", 80, 20, 140)
                    self._num("conrod", "Conrod (mm)", 140, 50, 250)
                    self._num("cr", "Compression ratio", 10, 5, 24)
                    dpg.add_input_int(label="Cylinders", tag="ncyl", default_value=4,
                                      min_value=1, max_value=8, width=120,
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

        # keyboard throttle: hold W or Up arrow
        if dpg.is_key_down(dpg.mvKey_W) or dpg.is_key_down(dpg.mvKey_Up):
            self._kb_throttle = min(1.0, self._kb_throttle + 0.08)
        elif self._kb_throttle > 0.0:
            self._kb_throttle = max(0.0, self._kb_throttle - 0.12)
        if self._kb_throttle > 0.001:
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

        # readouts
        dpg.set_value("rpm_val", f"{sim.rpm:7.0f}")
        dpg.set_value("trq_val", f"{sim.torque:7.1f}")
        dpg.set_value("map_val", f"{sim.P[core.P_MAP]/1000:6.1f}")
        dpg.set_value("pk_val", f"{sim.scope_p.max()/1e5:6.1f}")

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

    def run(self):
        self.build()
        while dpg.is_dearpygui_running():
            self.update_plots()
            dpg.render_dearpygui_frame()
        self.sim.stop_audio()
        dpg.destroy_context()


def main():
    EngineUI().run()
