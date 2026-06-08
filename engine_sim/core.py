"""
Real-time engine core: crank kinematics + 0D filling/emptying cylinder
thermodynamics + 1D compressible (Euler) exhaust gas dynamics.

Everything in the hot path is numba-njit compiled so a whole audio block can be
advanced sample-by-sample at 48 kHz in real time. The pressure wave arriving at
the open (tailpipe) end of the 1D exhaust IS the audio signal.

State layout
------------
st (float64[2])      : [crank_angle_rad, omega_rad_s]
cyl_m (float64[n])   : trapped gas mass per cylinder (kg)
cyl_T (float64[n])   : gas temperature per cylinder (K)
rho/mom/Ene (float64[N]) : 1D exhaust conservative variables per cell
phase (float64[n])   : crank-angle offset of each cylinder in cycle coords (rad)
inj  (int64[n])      : exhaust pipe cell index each cylinder dumps into

Parameters live in a flat float64 array `P` indexed by the P_* constants so the
UI can mutate them live without recompiling.
"""
import numpy as np
from numba import njit

# ---- parameter indices -----------------------------------------------------
P_NCYL        = 0
P_CYCLE       = 1   # cycle angle in rad: 4*pi for 4-stroke, 2*pi for 2-stroke
P_R           = 2   # crank radius = stroke/2
P_L           = 3   # conrod length
P_APIST       = 4   # piston area
P_VCLEAR      = 5   # clearance volume
P_VDISP       = 6   # displaced volume per cylinder
P_GAMMA       = 7   # ratio of specific heats (cylinder)
P_RGAS        = 8   # specific gas constant
P_CV          = 9
P_LHV         = 10  # fuel lower heating value J/kg
P_AFR         = 11  # stoichiometric air/fuel ratio
P_COMBEFF     = 12
P_IGN         = 13  # ignition angle in cycle coords (rad)
P_BURN        = 14  # burn duration (rad)
P_WIEBE_A     = 15
P_WIEBE_M     = 16
P_MAP         = 17  # intake manifold pressure (Pa)
P_TINT        = 18  # intake charge temperature (K)
P_PATM        = 19
P_TATM        = 20
P_THROTTLE    = 21
P_IVO         = 22
P_IVC         = 23
P_EVO         = 24
P_EVC         = 25
P_AIN         = 26  # max intake valve flow area
P_AEX         = 27  # max exhaust valve flow area
P_CD          = 28  # discharge coefficient
P_INERTIA     = 29  # rotating inertia
P_FRIC0       = 30  # constant friction torque
P_FRICW       = 31  # viscous friction coeff (per rad/s)
P_LOAD        = 32  # external load torque
P_GAMMAEX     = 33  # exhaust gas gamma
P_REX         = 34  # exhaust gas R
P_CVEX        = 35
P_PIPELEN     = 36
P_PIPEAREA    = 37
P_NCELLS      = 38
P_DX          = 39
P_DT          = 40
P_HT          = 41  # heat-transfer lumped coefficient
P_WALLT       = 42  # cylinder wall temperature
P_DIESEL      = 43  # 0 gasoline, 1 diesel
P_SR          = 44  # sample rate
P_DAMP        = 45  # exhaust wall damping (per cell fraction)
P_OUTGAIN     = 46  # audio output gain
P_REDLINE     = 47  # rev-limiter rpm (fuel cut above)
P_RUNNING     = 48  # 1 = fuel/combustion enabled, 0 = engine off
P_STARTER     = 49  # starter motor torque (Nm), applied while cranking
P_LPF         = 50  # output one-pole lowpass coefficient (0..1)
P_NOISE       = 51  # combustion roughness / flow-noise amount
P_POP         = 52  # overrun backfire intensity (0..1)

# --- intake manifold (real filling/emptying plenum) ---
P_MANVOL      = 53  # intake manifold (plenum) volume (m^3)
P_THRAREA     = 54  # effective throttle flow area (m^2): pedal + idle bypass
P_PBOOST      = 55  # plenum-feed pressure (Pa) = atmosphere + turbo boost
P_TMAN        = 56  # manifold charge temperature (K), intercooled ~ const

# --- driveline + vehicle (coupled per-sample, no controllers) ---
P_RATIO       = 57  # total gear ratio (engine:wheel), 0 = neutral / open
P_RW          = 58  # wheel radius (m)
P_VMASS       = 59  # vehicle mass (kg); <=5 => bench (free rev)
P_CRR         = 60  # rolling resistance coefficient
P_CDA         = 61  # 0.5 * rho_air * Cd * frontal_area  (aero drag factor)
P_BRAKEF      = 62  # brake force (N) at the contact patch
P_TCAP        = 63  # clutch torque capacity (Nm); 0 = clutch open
P_KDL         = 64  # driveline torsional stiffness (Nm/rad)
P_CDL         = 65  # driveline torsional damping (Nm/(rad/s))
P_MU          = 66  # tyre traction coefficient (drive force <= mu*m*g)
P_GRAVITY     = 67  # gravitational acceleration (m/s^2)

# --- 1D intake duct (induction acoustics) ---
P_IN_NCELLS   = 68  # intake duct cell count (0 = disabled)
P_IN_DX       = 69  # intake duct cell size (m)
P_IN_AREA     = 70  # intake duct cross-sectional area (m^2)
P_INGAIN      = 71  # induction mix level into the output

# --- turbocharger (real shaft + turbine + compressor) ---
P_TURBO       = 72  # 1 = turbocharged, 0 = NA
P_TRB_INERTIA = 73  # turbo shaft polar inertia (kg m^2)
P_TRB_RIMP    = 74  # compressor impeller radius (m) -> tip speed
P_TRB_ETA     = 75  # effective turbine harvest coefficient
P_WGATE_PR    = 76  # wastegate pressure ratio (caps boost)
P_ICOOL       = 77  # intercooler effectiveness (0..1)
P_TRB_CELL    = 78  # exhaust cell index the turbine taps
P_BOOSTOUT    = 79  # (out) actual boost pressure (Pa, gauge) for the UI

# --- audio source toggles (for A/B-ing what's real vs coloration) ---
P_BODYGAIN    = 80  # body/thump resonator mix (0 = off)
P_SAT         = 81  # 1 = tanh saturation, 0 = hard clip only

P_NPARAMS     = 84

# --- state vector layout (st) ---
S_THETA       = 0   # crank angle (rad)
S_OMEGA       = 1   # engine angular velocity (rad/s)
S_V           = 2   # vehicle speed (m/s)
S_PHI         = 3   # driveline torsional wind-up angle (rad)
S_MMAN        = 4   # intake manifold trapped gas mass (kg)
S_TURBO       = 5   # turbo shaft angular velocity (rad/s)
S_AIRFLOW     = 6   # low-passed engine air mass flow (kg/s) for the turbo
S_NSTATE      = 7

TWO_PI = 2.0 * np.pi


@njit(cache=True, fastmath=True, inline='always')
def _wrap(a, period):
    a = a % period
    if a < 0.0:
        a += period
    return a


@njit(cache=True, fastmath=True, inline='always')
def cyl_volume(theta_crank, r, L, Apist, Vclear):
    """Cylinder volume and dV/dtheta from slider-crank geometry.
    theta_crank measured from TDC (0 -> TDC, pi -> BDC)."""
    s_ = np.sin(theta_crank)
    c_ = np.cos(theta_crank)
    root = np.sqrt(L * L - r * r * s_ * s_)
    s = (r + L) - (r * c_ + root)            # piston displacement from TDC
    dsdth = r * s_ * (1.0 + r * c_ / root)   # ds/dtheta
    V = Vclear + Apist * s
    dVdth = Apist * dsdth
    return V, dVdth


@njit(cache=True, fastmath=True, inline='always')
def open_frac(cp, op, cl, cyc):
    """Valve/port open fraction (0..1) at cycle phase cp for an event that
    opens at `op` and closes at `cl` (cycle coords, rad). Smooth sine lift."""
    # distance from open, modulo cycle
    span = cl - op
    if span < 0.0:
        span += cyc
    x = cp - op
    if x < 0.0:
        x += cyc
    if x > span or span <= 0.0:
        return 0.0
    return np.sin(np.pi * x / span)


@njit(cache=True, fastmath=True, inline='always')
def wiebe(frac, a, m):
    if frac <= 0.0:
        return 0.0
    if frac >= 1.0:
        return 1.0
    return 1.0 - np.exp(-a * frac ** (m + 1.0))


@njit(cache=True, fastmath=True, inline='always')
def orifice_mdot(p_up, T_up, p_down, area, Cd, g, R):
    """Compressible mass flow (kg/s) through a restriction, upstream -> downstream.
    Returns positive magnitude; caller assigns direction."""
    if area <= 0.0 or p_up <= 0.0:
        return 0.0
    pr = p_down / p_up
    if pr > 1.0:
        pr = 1.0
    crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    if pr < crit:
        pr = crit  # choked
    term = (2.0 * g / (g - 1.0)) * (pr ** (2.0 / g) - pr ** ((g + 1.0) / g))
    if term < 0.0:
        term = 0.0
    return Cd * area * p_up / np.sqrt(R * T_up) * np.sqrt(term)


@njit(cache=True, fastmath=True)
def gas_step(rho, mom, Ene, N, dx, dt, g, R, patm, Tatm, p_open, T_open, damp,
             src_m, src_E, cell_vol):
    """One explicit finite-volume (Rusanov) step of the 1D Euler equations
    with a reflecting head wall (cell 0) and an open tail (cell N-1) that
    radiates into a reservoir at p_open/T_open (atmosphere for the exhaust
    tailpipe, the boost/airbox pressure for the intake mouth). src_m / src_E
    are per-cell mass (kg) / energy (J) sources for this step; they are
    converted to densities by the cell volume before being applied."""
    rho_atm = patm / (R * Tatm)
    E_atm = patm / (g - 1.0)
    rho_open = p_open / (R * T_open)

    # apply cylinder sources first (mass/energy -> density via cell volume)
    for i in range(N):
        if src_m[i] != 0.0 or src_E[i] != 0.0:
            rho[i] += src_m[i] / cell_vol
            Ene[i] += src_E[i] / cell_vol
            if rho[i] < 1e-6:
                rho[i] = 1e-6

    # primitive vars
    u = np.empty(N)
    p = np.empty(N)
    a = np.empty(N)
    for i in range(N):
        ui = mom[i] / rho[i]
        pi = (g - 1.0) * (Ene[i] - 0.5 * rho[i] * ui * ui)
        if pi < 1.0:
            pi = 1.0
        u[i] = ui
        p[i] = pi
        a[i] = np.sqrt(g * pi / rho[i])

    # fluxes at interfaces 0..N (N+1 faces); use ghost cells for BCs
    fr = np.empty(N + 1)
    fm = np.empty(N + 1)
    fe = np.empty(N + 1)
    for f in range(N + 1):
        # left/right states
        if f == 0:
            # reflecting head wall: ghost = mirror of cell 0 with -u
            rL = rho[0]; uL = -u[0]; pL = p[0]; aL = a[0]
            rR = rho[0]; uR = u[0]; pR = p[0]; aR = a[0]
        elif f == N:
            # open tail: ghost held at the reservoir pressure (radiating)
            rL = rho[N - 1]; uL = u[N - 1]; pL = p[N - 1]; aL = a[N - 1]
            rR = rho_open; uR = u[N - 1]; pR = p_open
            aR = np.sqrt(g * pR / rR)
        else:
            rL = rho[f - 1]; uL = u[f - 1]; pL = p[f - 1]; aL = a[f - 1]
            rR = rho[f];     uR = u[f];     pR = p[f];     aR = a[f]

        EL = pL / (g - 1.0) + 0.5 * rL * uL * uL
        ER = pR / (g - 1.0) + 0.5 * rR * uR * uR
        # physical fluxes
        FrL = rL * uL;            FrR = rR * uR
        FmL = rL * uL * uL + pL;  FmR = rR * uR * uR + pR
        FeL = (EL + pL) * uL;     FeR = (ER + pR) * uR
        smax = max(abs(uL) + aL, abs(uR) + aR)
        fr[f] = 0.5 * (FrL + FrR) - 0.5 * smax * (rR - rL)
        fm[f] = 0.5 * (FmL + FmR) - 0.5 * smax * (rR * uR - rL * uL)
        fe[f] = 0.5 * (FeL + FeR) - 0.5 * smax * (ER - EL)

    # update
    lam = dt / dx
    for i in range(N):
        rho[i] += -lam * (fr[i + 1] - fr[i])
        mom[i] += -lam * (fm[i + 1] - fm[i])
        Ene[i] += -lam * (fe[i + 1] - fe[i])
        # gentle wall damping toward rest (acoustic/viscous losses)
        mom[i] -= damp * mom[i]
        # --- robust clamps (also catch any non-finite value) ---
        ri = rho[i]
        if not (ri == ri) or ri < 1e-4:      # NaN or too low
            ri = 1e-4
        if ri > 30.0 * rho_atm:
            ri = 30.0 * rho_atm
        rho[i] = ri
        mi = mom[i]
        if not (mi == mi):
            mi = 0.0
        # cap velocity
        umax = 1200.0
        if mi > umax * ri:
            mi = umax * ri
        elif mi < -umax * ri:
            mi = -umax * ri
        mom[i] = mi
        kin = 0.5 * mi * mi / ri
        Eint = Ene[i] - kin
        if not (Eint == Eint) or Eint < E_atm * 1e-3:
            Eint = E_atm * 1e-3
        # bound temperature -> bounds sound speed -> keeps the explicit step
        # stable through blowdown/backfire spikes
        T_i = (g - 1.0) * Eint / (ri * R)
        if T_i > 2800.0:
            Eint = ri * R * 2800.0 / (g - 1.0)
        Ene[i] = kin + Eint


@njit(cache=True, fastmath=True)
def simulate_block(n_samples, P, st, cyl_m, cyl_T, phase, inj,
                   rho, mom, Ene, rho_in, mom_in, Ene_in,
                   out_audio, scope_p, n_scope, filt):
    """Advance the full engine by n_samples audio samples. Fills out_audio with
    the voiced tailpipe (exhaust) signal plus the intake-mouth induction signal,
    and records cylinder-0 pressure vs crank into scope_p for the UI. Returns
    mean engine torque. rho_in/mom_in/Ene_in are the 1D intake-duct gas state."""
    ncyl = int(P[P_NCYL])
    cyc = P[P_CYCLE]
    r = P[P_R]; L = P[P_L]; Apist = P[P_APIST]; Vclear = P[P_VCLEAR]
    Vdisp = P[P_VDISP]
    g = P[P_GAMMA]; R = P[P_RGAS]; cv = P[P_CV]
    cp = cv + R
    LHV = P[P_LHV]; AFR = P[P_AFR]; ceff = P[P_COMBEFF]
    ign = P[P_IGN]; burn = P[P_BURN]; wa = P[P_WIEBE_A]; wm = P[P_WIEBE_M]
    Tint = P[P_TINT]
    patm = P[P_PATM]; Tatm = P[P_TATM]
    throttle = P[P_THROTTLE]
    IVO = P[P_IVO]; IVC = P[P_IVC]; EVO = P[P_EVO]; EVC = P[P_EVC]
    Ain = P[P_AIN]; Aex = P[P_AEX]; Cd = P[P_CD]
    inertia = P[P_INERTIA]; fric0 = P[P_FRIC0]; fricw = P[P_FRICW]
    load = P[P_LOAD]

    # intake manifold (filling/emptying plenum). Pfeed/Tfeed are the
    # pre-compressor feed conditions; Pboost/Tman are set per sample (turbo).
    Vman = P[P_MANVOL]; thr_area = P[P_THRAREA]
    Pfeed = P[P_PBOOST]; Tfeed = P[P_TMAN]
    Pboost = Pfeed; Tman = Tfeed
    four_stroke = cyc > 3.0 * np.pi      # 4-stroke spans 4*pi, 2-stroke 2*pi

    # driveline + vehicle
    ratio = P[P_RATIO]; rw = P[P_RW]; vmass = P[P_VMASS]
    crr = P[P_CRR]; cda = P[P_CDA]; brakef = P[P_BRAKEF]
    Tcap = P[P_TCAP]; kdl = P[P_KDL]; cdl = P[P_CDL]
    mu = P[P_MU]; grav = P[P_GRAVITY]
    has_veh = vmass > 5.0
    coupled = has_veh and ratio != 0.0 and Tcap > 0.0
    gex = P[P_GAMMAEX]; Rex = P[P_REX]; cvex = P[P_CVEX]
    N = int(P[P_NCELLS]); dx = P[P_DX]; dt = P[P_DT]
    # 1D intake duct
    Ni = int(P[P_IN_NCELLS]); dx_in = P[P_IN_DX]
    in_area = P[P_IN_AREA]; ingain = P[P_INGAIN]
    # turbocharger
    turbo = P[P_TURBO] > 0.5
    J_trb = P[P_TRB_INERTIA]; r_imp = P[P_TRB_RIMP]
    eta_t = P[P_TRB_ETA]; wgate_pr = P[P_WGATE_PR]
    icool = P[P_ICOOL]; trb_cell = int(P[P_TRB_CELL])
    ht = P[P_HT]; wallT = P[P_WALLT]
    diesel = P[P_DIESEL] > 0.5
    damp = P[P_DAMP]
    outgain = P[P_OUTGAIN]
    redline = P[P_REDLINE]
    running = P[P_RUNNING] > 0.5
    starter = P[P_STARTER]
    lpf = P[P_LPF]
    noise = P[P_NOISE]
    pop = P[P_POP]
    pipe_area = P[P_PIPEAREA]
    cell_vol = dx * pipe_area
    # energy per overrun re-ignition event, scaled to ONE cell's volume so the
    # local pressure rise stays bounded (~a few atm) and can't blow up the 1D
    # solver into a checkerboard, however big the engine or fine the grid.
    pop_burst = patm * cell_vol * 6.0
    if lpf <= 0.0:
        lpf = 1.0

    # ---- output voicing (muffler + body), state-variable filters ----
    # a 2-pole de-hash lowpass tames the numerical HF "tinny/computery" edge;
    # a low resonator adds the exhaust-system body/thump; soft saturation warms
    # transients. Coefficients are precomputed from the sample rate.
    sr = P[P_SR]
    f_main = 2.0 * np.sin(np.pi * 5200.0 / sr)   # de-hash LPF corner
    d_main = 1.25                                # ~Butterworth damping
    f_body = 2.0 * np.sin(np.pi * 95.0 / sr)     # body/thump resonance
    d_body = 0.22                                # high-Q -> resonant body
    body_gain = P[P_BODYGAIN]
    sat_on = P[P_SAT] > 0.5
    f_in = 2.0 * np.sin(np.pi * 3600.0 / sr)     # induction de-hash LPF corner
    # the intake duct gets moderate damping (airbox absorption) to tame the
    # high-order organ-pipe modes while keeping the low induction honk.
    damp_in = 0.025

    # gas substeps to satisfy CFL; amax must exceed the temperature-capped
    # sound speed (~sqrt(g*R*2800) ~ 1040 m/s) so backfire spikes stay stable
    amax = 1150.0
    nsub = int(amax * dt / (0.8 * dx)) + 1
    dt_gas = dt / nsub

    src_m = np.zeros(N)
    src_E = np.zeros(N)

    # intake duct: its own CFL substepping (cooler -> slower sound -> usually
    # fewer substeps than the hot exhaust). Allocate min size 1 when disabled.
    if Ni > 0:
        nsub_in = int(500.0 * dt / (0.8 * dx_in)) + 1
        dt_gas_in = dt / nsub_in
        cell_vol_in = dx_in * in_area
    else:
        nsub_in = 1
        dt_gas_in = dt
        cell_vol_in = 1.0
    src_in_m = np.zeros(Ni if Ni > 0 else 1)
    src_in_E = np.zeros(Ni if Ni > 0 else 1)

    torque_acc = 0.0
    map_acc = 0.0
    scope_period = cyc  # one full cycle for cylinder 0

    theta = st[S_THETA]
    omega = st[S_OMEGA]
    v = st[S_V]
    phi = st[S_PHI]
    m_man = st[S_MMAN]
    omega_trb = st[S_TURBO]
    air_ema = st[S_AIRFLOW]

    # turbomachinery constants
    cp_ex = cvex + Rex
    sigma = 0.9         # compressor work (slip) factor
    eta_c = 0.70        # compressor isentropic efficiency
    c_trb_fric = 2.0e-9 if turbo else 0.0   # bearing drag coeff (per omega^2 power)
    g_ratio = g / (g - 1.0)
    gm1_g = (g - 1.0) / g
    boost_out = 0.0

    for s in range(n_samples):
        dth = omega * dt

        # ---- turbo: compressor pressure ratio from shaft tip speed ----
        # PR comes from Euler turbomachinery (work = sigma*U^2), so boost is an
        # emergent function of shaft speed -- there is no scripted spool curve.
        if turbo:
            U = omega_trb * r_imp
            pr_c = (1.0 + sigma * U * U / (cp * Tfeed)) ** g_ratio
            if pr_c < 1.0:
                pr_c = 1.0
            Pboost = Pfeed * pr_c
            T_comp = Tfeed * pr_c ** gm1_g
            Tman = T_comp - icool * (T_comp - Tfeed)   # intercooler cools charge
            boost_out = Pboost - patm
        else:
            pr_c = 1.0

        # ---- intake manifold pressure (the real plenum, not a throttle map) ----
        if four_stroke:
            MAP = m_man * R * Tman / Vman
            if MAP < 1.0:
                MAP = 1.0
            # Breathing draws from the plenum (filled by the throttle from the
            # feed/boost reservoir) -- this sets power and is NOT routed through
            # the 1D runner (that coupling chokes flow). The runner below is a
            # SOUND model only.
            if Pboost >= MAP:
                mdot_thr = orifice_mdot(Pboost, Tman, MAP, thr_area, Cd, g, R)
            else:
                mdot_thr = -orifice_mdot(MAP, Tman, Pboost, thr_area, Cd, g, R)
            m_man += mdot_thr * dt
        else:
            # 2-stroke: crankcase scavenge delivery is per-stroke, no plenum
            MAP = Pboost
        map_acc += MAP

        # per-cylinder thermodynamics over this sample
        Tgas_torque = 0.0
        air_draw = 0.0          # net intake-valve mass flow this sample (duct src)
        for k in range(N):
            src_m[k] = 0.0
            src_E[k] = 0.0

        for c in range(ncyl):
            cph = _wrap(theta + phase[c], cyc)        # cycle phase
            cph_prev = _wrap(theta + phase[c] - dth, cyc)
            crank = _wrap(cph, TWO_PI)                # for volume (periodic 2pi)
            V, dVdth = cyl_volume(crank, r, L, Apist, Vclear)
            m = cyl_m[c]; T = cyl_T[c]
            Pc = m * R * T / V
            if Pc < 1.0:
                Pc = 1.0

            dQ = 0.0
            # combustion heat release (valves closed window around TDC firing)
            f_now = (cph - ign) / burn
            f_pre = (cph_prev - ign) / burn
            # handle wrap of ign near end of cycle
            if f_now < -0.5:
                f_now += cyc / burn
            if f_pre < -0.5:
                f_pre += cyc / burn
            dxb = wiebe(f_now, wa, wm) - wiebe(f_pre, wa, wm)
            rpm_now = omega * 9.5492966   # 60/(2pi)
            if dxb > 0.0 and rpm_now < redline and running:
                # fuel energy available this cycle (mass trapped is ~constant now)
                if diesel:
                    # fuel metered by throttle, capped by the smoke limit (a
                    # diesel runs lean even at full load -> realistic BMEP)
                    air = m
                    fuel_max = air / 23.0
                    fuel = throttle * fuel_max
                else:
                    # meter fuel by FRESH air delivered (~ manifold pressure), not
                    # total trapped mass — trapped exhaust residual carries no O2.
                    # This is what lets a 2-stroke idle instead of self-charging.
                    fresh = MAP * Vdisp / (R * Tman)
                    air = m if m < fresh else fresh
                    fuel = air / AFR
                Qcyc = fuel * LHV * ceff
                # cycle/flame roughness so cycles aren't mathematically identical
                rough = 1.0 + noise * np.random.standard_normal()
                if rough < 0.0:
                    rough = 0.0
                dQ += Qcyc * dxb * rough

            # heat transfer to walls (lumped)
            dQ -= ht * (T - wallT) * dt

            # ---- valve flows ----
            dm_in = 0.0
            dm_ex = 0.0
            dH = 0.0
            fin = open_frac(cph, IVO, IVC, cyc)
            if fin > 0.0:
                area = Ain * fin
                if MAP >= Pc:
                    md = orifice_mdot(MAP, Tman, Pc, area, Cd, g, R) * dt
                    dm_in += md
                    dH += md * cp * Tman
                    air_draw += md            # induction: mass pulled from tract
                    if four_stroke:
                        m_man -= md           # charge drawn out of the plenum
                else:
                    md = orifice_mdot(Pc, T, MAP, area, Cd, g, R) * dt
                    dm_in -= md
                    dH -= md * cp * T
                    air_draw -= md            # reversion: pushed back into tract
                    if four_stroke:
                        m_man += md           # reversion back into the plenum
                if four_stroke and m_man < 1e-9:
                    m_man = 1e-9
            # rising edge of exhaust valve opening (once per cycle per cylinder)
            dd = cph - cph_prev
            if dd < 0.0:
                dd += cyc
            rel = EVO - cph_prev
            if rel < 0.0:
                rel += cyc
            evo_edge = rel <= dd

            fex = open_frac(cph, EVO, EVC, cyc)
            if fex > 0.0:
                area = Aex * fex
                # exhaust pipe head cell pressure (where this cyl dumps)
                ci = inj[c]
                ui = mom[ci] / rho[ci]
                Pp = (gex - 1.0) * (Ene[ci] - 0.5 * rho[ci] * ui * ui)
                if Pp < 1.0:
                    Pp = 1.0
                if Pc >= Pp:
                    md = orifice_mdot(Pc, T, Pp, area, Cd, g, R) * dt
                    dm_ex += md
                    dH -= md * cp * T
                    # deposit into pipe cell
                    src_m[ci] += md
                    src_E[ci] += md * cp * T   # stagnation enthalpy in
                else:
                    # reversion: pipe gas back into cylinder
                    Tp = Pp / (rho[ci] * Rex)
                    md = orifice_mdot(Pp, Tp, Pc, area, Cd, gex, Rex) * dt
                    dm_ex -= md
                    dH += md * cp * Tp
                    src_m[ci] -= md
                    src_E[ci] -= md * cp * Tp

                # ---- overrun backfire (physically gated) ----
                # When the engine is lean-on-overrun (closed throttle pumping
                # air -> free O2 in the exhaust) the unburnt fuel that left with
                # this cylinder's charge auto-ignites in the hot pipe. It can
                # fire at most ONCE per exhaust event (this EVO edge), so the
                # crackle is locked to the firing rate, not the sample rate.
                if pop > 0.0 and evo_edge and np.random.random() < pop * 0.5:
                    src_E[ci] += pop_burst * (0.6 + 0.8 * np.random.random())

            # ---- update cylinder state (energy & mass balance) ----
            U = m * cv * T
            work = Pc * dVdth * dth
            U_new = U + dQ - work + dH
            m_new = m + dm_in - dm_ex
            if m_new < 1e-8:
                m_new = 1e-8
            T_new = U_new / (m_new * cv)
            if T_new < 200.0:
                T_new = 200.0
            if T_new > 4000.0:
                T_new = 4000.0
            cyl_m[c] = m_new
            cyl_T[c] = T_new

            # gas torque on crank = (P - Pcrankcase) * piston force * arm
            Tgas_torque += (Pc - patm) * Apist * (dVdth / Apist)  # = (Pc-patm)*dVdth

            # scope: record cylinder 0 pressure over its cycle
            if c == 0:
                idx = int(cph / scope_period * n_scope)
                if idx < 0:
                    idx = 0
                if idx >= n_scope:
                    idx = n_scope - 1
                scope_p[idx] = Pc

        # ---- exhaust gas dynamics substeps (avg tailpipe = anti-alias) ----
        Pt_acc = 0.0
        for _ in range(nsub):
            gas_step(rho, mom, Ene, N, dx, dt_gas, gex, Rex, patm, Tatm,
                     patm, Tatm, damp, src_m, src_E, cell_vol)
            # sources are an impulse for the whole sample; apply only once
            for k in range(N):
                src_m[k] = 0.0
                src_E[k] = 0.0
            ut = mom[N - 1] / rho[N - 1]
            Pt = (gex - 1.0) * (Ene[N - 1] - 0.5 * rho[N - 1] * ut * ut)
            Pt_acc += Pt

        # ---- intake duct gas dynamics (induction sound is emergent) ----
        # The duct is a 1D Euler pipe excited at the head (cell 0) by the REAL
        # per-cylinder intake-valve mass flow (pulsed by valve lift -- the actual
        # induction source, not a steady throttle impulse) and open at the mouth
        # (cell Ni-1) to the airbox/boost reservoir, which refills the draw. The
        # pressure wave radiating from the mouth IS the induction "suck"/honk,
        # and it rises with boost -> the turbo's intake character. Loudness
        # scales with airflow automatically (big draw at WOT, tiny at idle).
        Pin_acc = 0.0
        if Ni > 0 and ingain > 0.0:
            # sound-only runner: excited by the real valve draw, open at the
            # mouth to the boost/airbox reservoir which refills it.
            src_in_m[0] = -air_draw
            src_in_E[0] = -air_draw * cp * Tman
            for _ in range(nsub_in):
                gas_step(rho_in, mom_in, Ene_in, Ni, dx_in, dt_gas_in, g, R,
                         patm, Tatm, Pboost, Tman, damp_in, src_in_m, src_in_E,
                         cell_vol_in)
                for k in range(Ni):
                    src_in_m[k] = 0.0
                    src_in_E[k] = 0.0
                uin = mom_in[Ni - 1] / rho_in[Ni - 1]
                Pin = (g - 1.0) * (Ene_in[Ni - 1] - 0.5 * rho_in[Ni - 1] * uin * uin)
                Pin_acc += Pin

        # ---- turbocharger shaft dynamics ----
        # Turbine harvests exhaust enthalpy flux; compressor work loads the
        # shaft; the shaft inertia gives emergent spool/lag; the wastegate caps
        # boost by dumping turbine drive. Boost (above) and lag both EMERGE from
        # this power balance -- no scripted spool curve, no boost-vs-rpm table.
        if turbo:
            # engine air mass flow (low-passed) drives both turbine & compressor
            air = mdot_thr if mdot_thr > 0.0 else 0.0
            air_ema += (air - air_ema) * 0.02
            # turbine drive from the exhaust thermal/flow state at its cell
            rt = rho[trb_cell]
            ut = mom[trb_cell] / rt
            pt = (gex - 1.0) * (Ene[trb_cell] - 0.5 * rt * ut * ut)
            if pt < 1.0:
                pt = 1.0
            Tt = pt / (rt * Rex)
            dT = Tt - Tatm
            if dT < 0.0:
                dT = 0.0
            P_turb = eta_t * air_ema * cp_ex * dT
            # wastegate: once boost reaches the set ratio, bleed the turbine
            if pr_c >= wgate_pr:
                P_turb = 0.0
            # compressor absorbs work to make the boost it is currently making
            P_comp = air_ema * cp * Tfeed * (pr_c ** gm1_g - 1.0) / eta_c
            P_fric = c_trb_fric * omega_trb * omega_trb * omega_trb
            w_eff = omega_trb if omega_trb > 100.0 else 100.0
            omega_trb += (P_turb - P_comp - P_fric) / (J_trb * w_eff) * dt
            if omega_trb < 0.0:
                omega_trb = 0.0
            # remove the harvested enthalpy from the pipe (back-pressure), bounded
            ext = P_turb * dt
            cap = 0.3 * Ene[trb_cell] * cell_vol
            if ext > cap:
                ext = cap
            Ene[trb_cell] -= ext / cell_vol

        # ============================================================
        # COUPLED MECHANICS — engine, driveline and vehicle integrated
        # together every sample. The clutch + driveline is a torsional
        # spring/damper whose torque saturates at the clutch capacity
        # (Coulomb friction). Rigid drive, launch slip and engine braking
        # all emerge from this one law; there are no mode switches.
        # ============================================================
        # net torque at the engine shaft, before the clutch
        T_eng = Tgas_torque - (fric0 + fricw * omega) - load + starter

        if not coupled:
            # open driveline: engine spins free, vehicle coasts on road load
            omega += T_eng / inertia * dt
            if omega < 0.0:
                omega = 0.0
            phi = 0.0
            if has_veh and v > 0.0:
                f_res = crr * vmass * grav + cda * v * v + brakef
                v -= f_res / vmass * dt
                if v < 0.0:
                    v = 0.0
        else:
            omega_in = (v / rw) * ratio      # gearbox-input speed (driven plate)
            slip = omega - omega_in
            phi += slip * dt                 # driveline wind-up
            T_cl = kdl * phi + cdl * slip
            # clutch slips once demanded torque exceeds its clamp capacity
            if T_cl > Tcap:
                T_cl = Tcap
                phi = (Tcap - cdl * slip) / kdl
            elif T_cl < -Tcap:
                T_cl = -Tcap
                phi = (-Tcap - cdl * slip) / kdl
            # tyre grip: contact patch can't pass more than mu*m*g -> wheelspin,
            # which unloads the clutch so the engine flares (emergent, not faked)
            f_drive = T_cl * ratio / rw
            f_max = mu * vmass * grav
            if f_drive > f_max:
                f_drive = f_max
                T_cl = f_drive * rw / ratio
                phi = (T_cl - cdl * slip) / kdl
            elif f_drive < -f_max:
                f_drive = -f_max
                T_cl = f_drive * rw / ratio
                phi = (T_cl - cdl * slip) / kdl
            # engine reacts against the clutch torque
            omega += (T_eng - T_cl) / inertia * dt
            if omega < 0.0:
                omega = 0.0
            # vehicle: drive force minus road load (load only opposes motion)
            if v > 0.0:
                f_res = crr * vmass * grav + cda * v * v + brakef
            else:
                f_res = 0.0
            v += (f_drive - f_res) / vmass * dt
            if v < 0.0:
                v = 0.0

        theta += dth
        if theta > 1e7:
            theta = _wrap(theta, cyc)
        torque_acc += Tgas_torque

        # ---- audio output: voiced tailpipe gauge pressure ----
        raw = Pt_acc / nsub - patm
        # DC blocker (removes the static offset/thump)
        hp = raw - filt[0] + 0.999 * filt[1]
        filt[0] = raw
        filt[1] = hp
        x = hp * outgain
        # 2-pole de-hash lowpass (state-variable) -> kills the tinny HF edge
        lp1 = filt[2]; bp1 = filt[3]
        hpf = x - lp1 - d_main * bp1
        bp1 += f_main * hpf
        lp1 += f_main * bp1
        filt[2] = lp1; filt[3] = bp1
        # low resonator (bandpass) adds exhaust-system body / thump
        lp2 = filt[4]; bp2 = filt[5]
        hpf2 = lp1 - lp2 - d_body * bp2
        bp2 += f_body * hpf2
        lp2 += f_body * bp2
        filt[4] = lp2; filt[5] = bp2
        sig = lp1 + body_gain * bp2

        # ---- induction (intake-mouth) signal: DC-block + de-hash, then mix ----
        # NB: this path MUST be de-hashed like the exhaust, else the 1D duct's
        # numerical HF hash leaks through as an "electric buzz" that grows with
        # rpm. A 2-pole lowpass (SVF) removes it and softens the intake tone.
        if Ni > 0:
            raw_in = Pin_acc / nsub_in - patm
            hp_in = raw_in - filt[6] + 0.999 * filt[7]
            filt[6] = raw_in
            filt[7] = hp_in
            xin = hp_in * outgain
            lpi = filt[8]; bpi = filt[9]
            hpi = xin - lpi - d_main * bpi
            bpi += f_in * hpi
            lpi += f_in * bpi
            filt[8] = lpi; filt[9] = bpi
            sig += ingain * lpi

        # soft saturation (optional): warmth + glues the backfire transients
        if sat_on:
            out_audio[s] = np.tanh(1.25 * sig)
        else:
            y = sig
            if y > 1.0:
                y = 1.0
            elif y < -1.0:
                y = -1.0
            out_audio[s] = y

    st[S_THETA] = theta
    st[S_OMEGA] = omega
    st[S_V] = v
    st[S_PHI] = phi
    st[S_MMAN] = m_man
    st[S_TURBO] = omega_trb
    st[S_AIRFLOW] = air_ema
    P[P_MAP] = map_acc / n_samples    # report manifold pressure for the UI
    P[P_BOOSTOUT] = boost_out         # report actual boost (Pa, gauge)

    return torque_acc / n_samples
