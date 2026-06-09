"""Dear PyGui control panel + live instrumentation for the engine simulator."""
import math
import os
import json
import threading
import dataclasses
from collections import deque
import numpy as np
import dearpygui.dearpygui as dpg

from . import core
from .presets import (PRESETS, get_preset, EngineConfig,
                      VEHICLES, get_vehicle)
from .sim import EngineSim, SAMPLE_RATE

CUSTOM_FILE = os.path.join(os.path.dirname(__file__), "..", "custom_engines.json")

GEX = 1.33
RAIR = 287.0
PATM = 101325.0


class EngineUI:
    def __init__(self):
        self.custom_engines = {}        # name -> EngineConfig (user-saved)
        self._load_custom_from_disk()
        self.preset_names = list(PRESETS.keys()) + list(self.custom_engines.keys())
        self._is_custom = False
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
            octane=g("octane"), power_tune=g("ptune"),
            pipe_length=g("pipelen"), pipe_diameter=g("pipedia") / 1000.0,
            runner_spread=g("spread"),
            intake_length=g("intlen"), intake_diameter=g("intdia") / 1000.0,
            muffler_volume=g("muffvol") / 1000.0,
            firing_order=self.sim.cfg.firing_order,
            firing_angles=self.sim.cfg.firing_angles,
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
        dpg.set_value("octane", cfg.octane)
        dpg.set_value("muffvol", cfg.muffler_volume * 1000.0)
        if dpg.does_item_exist("intlen"):
            dpg.set_value("intlen", cfg.intake_length)
            dpg.set_value("intdia", cfg.intake_diameter * 1000.0)
            dpg.set_value("ptune", cfg.power_tune)
        self._building = False
        self._is_custom = cfg.name in self.custom_engines
        if dpg.does_item_exist("custom_lbl"):
            tag = "custom" if self._is_custom else "stock"
            dpg.set_value("custom_lbl", f"{tag}: {cfg.name}")
            dpg.configure_item("custom_lbl", color=(150, 220, 150))
        self._update_firing_label()
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
        self.sim.load_config(self._get_cfg(name))
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
            self._mark_custom()

    def on_throttle(self, s, v):
        self.sim.set_throttle(v / 100.0)

    def on_load(self, s, v):
        self.sim.set_load(v)

    def on_brake(self, s, v):
        self.sim.set_brake(v / 100.0)

    def on_backfire(self, s, v):
        self.sim.backfire = v

    def on_prewarm(self, *_):
        self.sim.prewarm()

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
        self.sim.mix_induction = 0.15 if on else 0.0
        self.sim.P[core.P_INGAIN] = self.sim.mix_induction

    def on_mix_body(self, s, on):
        self.sim.mix_body = 0.35 if on else 0.0
        self.sim.P[core.P_BODYGAIN] = self.sim.mix_body

    def on_mix_sat(self, s, on):
        self.sim.mix_sat = 1.0 if on else 0.0
        self.sim.P[core.P_SAT] = self.sim.mix_sat

    def on_mix_clatter(self, s, on):
        self.sim.mix_clatter = bool(on)
        self.sim.P[core.P_CLATTER] = self.sim._clatter_on if on else 0.0

    def on_mix_knock(self, s, on):
        self.sim.mix_knock = bool(on)
        self.sim.P[core.P_KNOCKAUD] = self.sim._knock_on if on else 0.0

    def on_mix_muffler(self, s, on):
        self.sim.mix_muffler = bool(on)
        self.sim.P[core.P_MUFFLP] = self.sim._muff_on if on else 0.0

    def on_grade(self, s, v):
        self.sim.set_grade(v / 100.0)          # slider is % slope

    def on_tyre_slip(self, s, on):
        self.sim.tyre_slip = bool(on)
        self.sim.P[core.P_TYRESLIP] = 1.0 if on else 0.0

    def on_octane(self, s, v):
        self.sim.cfg.octane = float(v)
        self.sim.P[core.P_OCTANE] = float(v)   # live (knock model reads it)

    # -------------------------------------------------- custom engines / timing
    def _get_cfg(self, name):
        if name in self.custom_engines:
            return dataclasses.replace(self.custom_engines[name])   # a copy
        return get_preset(name)

    def _load_custom_from_disk(self):
        try:
            with open(CUSTOM_FILE) as f:
                data = json.load(f)
            for n, d in data.items():
                self.custom_engines[n] = EngineConfig(**d)
        except Exception:
            pass

    def _save_custom_to_disk(self):
        try:
            with open(CUSTOM_FILE, "w") as f:
                json.dump({n: dataclasses.asdict(c)
                           for n, c in self.custom_engines.items()}, f, indent=2)
        except Exception as e:
            print("save failed:", e)

    def _mark_custom(self):
        self._is_custom = True
        if dpg.does_item_exist("custom_lbl"):
            dpg.set_value("custom_lbl", "CUSTOM (unsaved)")
            dpg.configure_item("custom_lbl", color=(255, 200, 120))

    def _update_firing_label(self):
        if not dpg.does_item_exist("firing_lbl"):
            return
        cfg = self.sim.cfg
        if cfg.firing_angles:
            dpg.set_value("firing_lbl", f"uneven firing: {cfg.firing_angles}")
        else:
            fo = cfg.firing_order or list(range(1, cfg.n_cylinders + 1))
            dpg.set_value("firing_lbl", f"firing order: {fo}")

    def on_auto_timing(self, *_):
        """Regenerate a clean even-firing order for the current cylinder count
        (and drop any stale uneven-firing angles), then rebuild. Use this after
        changing the number of cylinders so the timing matches."""
        ncyl = int(dpg.get_value("ncyl"))
        self.sim.cfg.firing_order = list(range(1, ncyl + 1))
        self.sim.cfg.firing_angles = []
        self.on_rebuild()
        self._update_firing_label()

    def on_save_engine(self, *_):
        cfg = self.current_edited_config()
        name = (dpg.get_value("custom_name") or "").strip()
        if not name:
            name = f"Custom {len(self.custom_engines) + 1}"
        cfg.name = name
        self.custom_engines[name] = cfg
        self._save_custom_to_disk()
        if name not in self.preset_names:
            self.preset_names.append(name)
            dpg.configure_item("preset", items=self.preset_names)
        dpg.set_value("preset", name)
        self._is_custom = False
        dpg.set_value("custom_lbl", f"saved: {name}")
        dpg.configure_item("custom_lbl", color=(150, 220, 150))

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
        self._mark_custom()

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
        """Steady-state dyno on a private sim instance (doesn't touch live audio).
        Hold WOT at each rpm with a load PI loop and read brake torque directly
        (gas torque - friction) -- the inertial I*dw/dt method smeared the curve
        into a flat plateau (it caught the off-idle transient in low-rpm bins).
        This gives the true humped VE/torque curve."""
        try:
            import math as _m
            B = 512
            dt_blk = B / SAMPLE_RATE
            sim = EngineSim(cfg)
            sim.step_block(64)
            sim.load_config(cfg)
            sim.prewarm()
            sim.P[core.P_NOISE] = 0.0       # clean read (no cycle roughness)
            sim.start_engine()
            for _ in range(40):
                sim.step_block(B)
                if sim.state == "running":
                    break
            for _ in range(60):
                sim.step_block(B)
            sim.base_inertia = 4.0          # big flywheel -> stable load loop
            lo = cfg.idle_rpm * 1.5
            hi = cfg.redline_rpm * 0.97
            npts = 16
            grid = np.linspace(lo, hi, npts)
            rpms = []
            tqs = []
            for target in grid:
                sim.set_throttle(1.0)
                integ = max(0.0, sim.torque * 0.5)
                # settle the load loop onto this rpm
                for _ in range(150):
                    err = sim.rpm - target
                    integ = float(np.clip(integ + 0.02 * err, 0.0, 1e6))
                    load = float(np.clip(integ + 0.6 * err, 0.0, 1e6))
                    sim.set_load(load)
                    sim.step_block(B)
                    if sim.state != "running":
                        break
                if sim.state != "running":
                    sim.start_engine()
                    for _ in range(40):
                        sim.step_block(B)
                        if sim.state == "running":
                            break
                    continue
                # measure brake torque (gas - friction), averaged
                tacc = []
                racc = []
                for _ in range(60):
                    err = sim.rpm - target
                    integ = float(np.clip(integ + 0.02 * err, 0.0, 1e6))
                    load = float(np.clip(integ + 0.6 * err, 0.0, 1e6))
                    sim.set_load(load)
                    sim.step_block(B)
                    fr = sim.P[core.P_FRIC0] + sim.P[core.P_FRICW] * sim.st[core.S_OMEGA]
                    tacc.append(sim.torque - fr)
                    racc.append(sim.rpm)
                rpms.append(float(np.mean(racc)))
                tqs.append(float(np.mean(tacc)))
            if len(rpms) < 4:
                self._dyno_result = ([], [], [], 0.0, 0.0, 0.0, 0.0)
                return
            grid = np.array(rpms)
            tqg = np.array(tqs)
            tqg = np.convolve(tqg, np.ones(3) / 3.0, mode="same")
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
    def _apply_theme(self):
        """Global spacing/padding so panels and plots breathe (less cramped)."""
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 10)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 7, 4)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 7)
                dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 7, 5)
                dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 8, 3)
            with dpg.theme_component(dpg.mvPlot):
                # tight padding so the data fills the panel (less dead border)
                dpg.add_theme_style(dpg.mvPlotStyleVar_PlotPadding, 2, 2,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LabelPadding, 2, 1,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LegendPadding, 4, 3,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_LegendInnerPadding, 2, 2,
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MousePosPadding, 4, 4,
                                    category=dpg.mvThemeCat_Plots)
        dpg.bind_theme(theme)

    def build(self):
        dpg.create_context()
        self._apply_theme()
        dpg.create_viewport(title="Engine Simulator - physics + 1D exhaust gas dynamics",
                            width=1860, height=920)

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
                dpg.add_checkbox(label="Tyre slip", default_value=True,
                                 callback=self.on_tyre_slip, tag="tyre_slip")
                dpg.add_text("  Mix:")
                dpg.add_checkbox(label="Induction", default_value=True,
                                 callback=self.on_mix_induction, tag="mix_ind")
                dpg.add_checkbox(label="Body", default_value=True,
                                 callback=self.on_mix_body, tag="mix_body")
                dpg.add_checkbox(label="Sat", default_value=True,
                                 callback=self.on_mix_sat, tag="mix_sat")
                dpg.add_checkbox(label="Clatter", default_value=False,
                                 callback=self.on_mix_clatter, tag="mix_clatter")
                dpg.add_checkbox(label="Knock", default_value=False,
                                 callback=self.on_mix_knock, tag="mix_knock")
                dpg.add_checkbox(label="Muffler", default_value=True,
                                 callback=self.on_mix_muffler, tag="mix_muffler")

            dpg.add_separator()

            with dpg.group(horizontal=True):
                # ---- left: drive controls / readouts ----
                with dpg.child_window(width=250, height=800):
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="START (Enter)", width=110,
                                       callback=self.on_start)
                        dpg.add_button(label="STOP", width=90, callback=self.on_stop)
                    dpg.add_text("state: off", tag="state_val", color=(255, 220, 120))
                    dpg.add_text("", tag="info_disp", color=(150, 150, 150))
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
                        dpg.add_text("Drivetrain", color=(200, 220, 200))
                        dpg.add_text("-", tag="drive_val", color=(255, 230, 120))
                    dpg.add_text("Incline (%)")
                    dpg.add_slider_float(tag="grade", default_value=0.0,
                                         min_value=-25.0, max_value=25.0, width=200,
                                         callback=self.on_grade)
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
                    with dpg.group(horizontal=True):
                        dpg.add_text("RPM", color=(120, 200, 255))
                        dpg.add_text("0", tag="rpm_val")
                    # fixed-column metric table -> values never shift the layout
                    rows = [
                        (("Pwr", (255, 220, 120), "pwr_val", ""),
                         ("Bst", (255, 180, 120), "boost_val", "kPa")),
                        (("Trq", (120, 255, 160), "trq_val", "Nm"),
                         ("VE", (150, 220, 255), "ve_val", "%")),
                        (("BMEP", (160, 230, 160), "bmep_val", "bar"),
                         ("EGT", (255, 150, 120), "egt_val", "C")),
                        (("MAP", (255, 200, 120), "map_val", "kPa"),
                         ("Pcyl", (255, 160, 160), "pk_val", "bar")),
                        (("Lam", (200, 200, 255), "lam_val", ""), None),
                    ]
                    with dpg.table(header_row=False, borders_innerH=False,
                                   borders_outerH=False, borders_innerV=False,
                                   borders_outerV=False, policy=dpg.mvTable_SizingFixedFit):
                        dpg.add_table_column(init_width_or_weight=40, width_fixed=True)
                        dpg.add_table_column(init_width_or_weight=78, width_fixed=True)
                        dpg.add_table_column(init_width_or_weight=42, width_fixed=True)
                        dpg.add_table_column(init_width_or_weight=78, width_fixed=True)
                        for left, right in rows:
                            with dpg.table_row():
                                for cell in (left, right):
                                    if cell is None:
                                        dpg.add_text(""); dpg.add_text("")
                                        continue
                                    lbl, col, tag, unit = cell
                                    dpg.add_text(lbl, color=col)
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("--", tag=tag)
                                        if unit:
                                            dpg.add_text(unit, color=(110, 110, 110))
                    # ---- temperature (thermostat gauge + prewarm) ----
                    dpg.add_separator()
                    with dpg.group(horizontal=True):
                        dpg.add_text("TEMP", color=(255, 170, 120))
                        dpg.add_text("cold", tag="warm_lbl", color=(150, 200, 255))
                        dpg.add_button(label="Prewarm", width=78,
                                       callback=self.on_prewarm)
                    dpg.add_progress_bar(tag="temp_bar", default_value=0.0,
                                         width=-1, overlay="20 C")
                    with dpg.group(horizontal=True):
                        self._metric("Oil ", "oil_val", (220, 200, 140), "C")
                        self._metric("Metl", "metal_val", (230, 170, 140), "C")
                    dpg.add_separator()
                    dpg.add_text("FIRING", color=(255, 200, 150))
                    with dpg.drawlist(width=222, height=92, tag="fire_draw"):
                        pass
                    # colour legend for the stroke phases drawn above
                    with dpg.group(horizontal=True):
                        dpg.add_text("Power", color=(230, 90, 60))
                        dpg.add_text("Exh", color=(150, 150, 160))
                        dpg.add_text("Int", color=(70, 150, 230))
                    with dpg.group(horizontal=True):
                        dpg.add_text("Comp", color=(190, 165, 80))
                        dpg.add_text("Fire!", color=(255, 215, 90))
                    dpg.add_text("", tag="fire_lbl", color=(170, 170, 170))

                # ---- middle: plots (2-column grid of compact graphs) ----
                with dpg.child_window(width=860, height=800):
                    pw = 410          # per-plot width in the 2-up rows
                    ph = 176          # compact plot height

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
                            with dpg.plot(label="Dyno - torque & power vs RPM",
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
                with dpg.child_window(width=320, height=800):
                    dpg.add_text("GEOMETRY", color=(180, 180, 255))
                    self._num("bore", "Bore (mm)", 80, 20, 140)
                    self._num("stroke", "Stroke (mm)", 80, 20, 140)
                    self._num("conrod", "Conrod (mm)", 140, 50, 250)
                    self._num("cr", "Compression ratio", 10, 5, 24)
                    dpg.add_input_int(label="Cylinders", tag="ncyl", default_value=4,
                                      min_value=1, max_value=16, width=120,
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
                    dpg.add_input_float(tag="octane", label="Fuel octane (RON)",
                                        default_value=95.0, min_value=80.0,
                                        max_value=110.0, width=120, step=0,
                                        callback=self.on_octane, on_enter=False)
                    dpg.add_separator()
                    dpg.add_text("EXHAUST", color=(150, 255, 220))
                    self._num("pipelen", "Pipe length (m)", 1.5, 0.3, 3.0)
                    self._num("pipedia", "Pipe dia (mm)", 50, 18, 90)
                    self._num("spread", "Runner spread", 0.2, 0.0, 0.4)
                    self._num("muffvol", "Muffler vol (L, <0 straight)", 0, -1, 60)
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
                                         default_value=0.0, min_value=0.0,
                                         max_value=3.0, width=120,
                                         callback=self.on_backfire)
                    dpg.add_separator()
                    dpg.add_button(label="REBUILD ENGINE", width=-1,
                                   callback=self.on_rebuild,
                                   tag="rebuild_btn")

                # ---- far right: custom engine / timing / advanced ----
                with dpg.child_window(width=290, height=800):
                    dpg.add_text("CUSTOM ENGINE", color=(255, 220, 150))
                    dpg.add_text("stock", tag="custom_lbl", color=(150, 200, 150),
                                 wrap=270)
                    dpg.add_text("Edit any value -> becomes custom. Name + Save:",
                                 color=(140, 140, 140), wrap=270)
                    dpg.add_input_text(tag="custom_name", hint="my engine name",
                                       width=-1)
                    dpg.add_button(label="SAVE ENGINE", width=-1,
                                   callback=self.on_save_engine)
                    dpg.add_separator()
                    dpg.add_text("TIMING / CYLINDERS", color=(255, 200, 150))
                    dpg.add_text("After changing cylinder count, click:",
                                 color=(140, 140, 140), wrap=270)
                    dpg.add_button(label="AUTO-ADJUST TIMING", width=-1,
                                   callback=self.on_auto_timing)
                    dpg.add_text("firing order: --", tag="firing_lbl",
                                 color=(170, 170, 170), wrap=270)
                    dpg.add_separator()
                    dpg.add_text("ADVANCED (not on main panel)",
                                 color=(180, 180, 255))
                    self._num("intlen", "Intake length (m)", 0.45, 0.1, 1.0)
                    self._num("intdia", "Intake dia (mm)", 55, 20, 110)
                    self._num("ptune", "Power tune (x)", 1.0, 0.5, 1.6)
                    dpg.add_separator()
                    dpg.add_text("VEHICLE", color=(200, 220, 200))
                    dpg.add_text("--", tag="veh_info", color=(180, 180, 180),
                                 wrap=270)

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

    def _metric(self, label, tag, color, unit=""):
        """One compact readout line: coloured label, value, dim unit."""
        with dpg.group(horizontal=True):
            dpg.add_text(label, color=color)
            dpg.add_text("--", tag=tag)
            if unit:
                dpg.add_text(unit, color=(105, 105, 105))

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
        dpg.set_value("rpm_val", f"{sim.rpm:.0f}")
        dpg.set_value("pwr_val", f"{hp:.0f}hp/{kw:.0f}kW")
        dpg.set_value("trq_val", f"{sim.torque:.0f}")
        dpg.set_value("bmep_val", f"{bmep:.1f}")
        dpg.set_value("map_val", f"{MAP/1000:.0f}")
        dpg.set_value("boost_val", f"{boost:.1f}")
        dpg.set_value("ve_val", f"{ve:.0f}")
        dpg.set_value("egt_val", f"{egt:.0f}")
        dpg.set_value("pk_val", f"{sim.scope_p.max()/1e5:.0f}")
        dpg.set_value("lam_val", "--" if not np.isfinite(lam) else f"{lam:.2f}")

        # temperature gauge (coolant) + oil/metal + warm-up state
        wf = getattr(sim, "warm_frac", 1.0)
        dpg.set_value("temp_bar", float(wf))
        dpg.configure_item("temp_bar", overlay=f"{sim.coolant_C:.0f} C coolant")
        dpg.set_value("oil_val", f"{sim.T_oil - 273.15:.0f}")
        dpg.set_value("metal_val", f"{sim.T_metal - 273.15:.0f}")
        dpg.set_value("warm_lbl", "warm" if wf > 0.95 else "warming")

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

        # drivetrain readout + vehicle info panel
        veh = sim.vehicle
        dpg.set_value("drive_val", getattr(veh, "drivetrain", "-").upper())
        if dpg.does_item_exist("veh_info"):
            dpg.set_value("veh_info",
                          f"{veh.name}: {veh.mass:.0f} kg, "
                          f"{getattr(veh, 'drivetrain', '?').upper()}, "
                          f"{veh.n_gears()}-spd, wheel {veh.wheel_radius:.2f} m")

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
