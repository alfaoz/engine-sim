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
P_NPARAMS     = 64

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
def gas_step(rho, mom, Ene, N, dx, dt, g, R, patm, Tatm, damp,
             src_m, src_E, cell_vol):
    """One explicit finite-volume (Rusanov) step of the 1D Euler equations
    with a reflecting head wall (cell 0) and a partially-radiating open tail
    (cell N-1). src_m / src_E are per-cell mass (kg) / energy (J) sources for
    this step (from cylinder blowdown); they are converted to densities by the
    cell volume before being applied."""
    rho_atm = patm / (R * Tatm)
    E_atm = patm / (g - 1.0)

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
            # open tail: ghost held near atmosphere (radiating)
            rL = rho[N - 1]; uL = u[N - 1]; pL = p[N - 1]; aL = a[N - 1]
            rR = rho_atm; uR = u[N - 1]; pR = patm
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
                   rho, mom, Ene, out_audio, scope_p, n_scope, filt):
    """Advance the full engine by n_samples audio samples. Fills out_audio with
    tailpipe pressure (Pa, relative to atmosphere) and records cylinder-0
    pressure vs crank into scope_p for the UI. Returns mean engine torque."""
    ncyl = int(P[P_NCYL])
    cyc = P[P_CYCLE]
    r = P[P_R]; L = P[P_L]; Apist = P[P_APIST]; Vclear = P[P_VCLEAR]
    Vdisp = P[P_VDISP]
    g = P[P_GAMMA]; R = P[P_RGAS]; cv = P[P_CV]
    cp = cv + R
    LHV = P[P_LHV]; AFR = P[P_AFR]; ceff = P[P_COMBEFF]
    ign = P[P_IGN]; burn = P[P_BURN]; wa = P[P_WIEBE_A]; wm = P[P_WIEBE_M]
    MAP = P[P_MAP]; Tint = P[P_TINT]
    patm = P[P_PATM]; Tatm = P[P_TATM]
    throttle = P[P_THROTTLE]
    IVO = P[P_IVO]; IVC = P[P_IVC]; EVO = P[P_EVO]; EVC = P[P_EVC]
    Ain = P[P_AIN]; Aex = P[P_AEX]; Cd = P[P_CD]
    inertia = P[P_INERTIA]; fric0 = P[P_FRIC0]; fricw = P[P_FRICW]
    load = P[P_LOAD]
    gex = P[P_GAMMAEX]; Rex = P[P_REX]; cvex = P[P_CVEX]
    N = int(P[P_NCELLS]); dx = P[P_DX]; dt = P[P_DT]
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
    pop_burst = patm * Vdisp * 0.25    # energy per backfire event
    pipe_area = P[P_PIPEAREA]
    cell_vol = dx * pipe_area
    if lpf <= 0.0:
        lpf = 1.0

    # gas substeps to satisfy CFL; amax must exceed the temperature-capped
    # sound speed (~sqrt(g*R*2800) ~ 1040 m/s) so backfire spikes stay stable
    amax = 1150.0
    nsub = int(amax * dt / (0.8 * dx)) + 1
    dt_gas = dt / nsub

    src_m = np.zeros(N)
    src_E = np.zeros(N)

    torque_acc = 0.0
    scope_period = cyc  # one full cycle for cylinder 0

    for s in range(n_samples):
        theta = st[0]
        omega = st[1]
        dth = omega * dt

        # per-cylinder thermodynamics over this sample
        Tgas_torque = 0.0
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
                    fresh = MAP * Vdisp / (R * Tint)
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
                    md = orifice_mdot(MAP, Tint, Pc, area, Cd, g, R) * dt
                    dm_in += md
                    dH += md * cp * Tint
                else:
                    md = orifice_mdot(Pc, T, MAP, area, Cd, g, R) * dt
                    dm_in -= md
                    dH -= md * cp * T
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
                # overrun backfire: unburnt charge lights off in the hot pipe
                if pop > 0.0 and np.random.random() < pop * 0.0008:
                    src_E[ci] += pop_burst

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
            gas_step(rho, mom, Ene, N, dx, dt_gas, gex, Rex, patm, Tatm, damp,
                     src_m, src_E, cell_vol)
            # sources are an impulse for the whole sample; apply only once
            for k in range(N):
                src_m[k] = 0.0
                src_E[k] = 0.0
            ut = mom[N - 1] / rho[N - 1]
            Pt = (gex - 1.0) * (Ene[N - 1] - 0.5 * rho[N - 1] * ut * ut)
            Pt_acc += Pt

        # ---- rotational dynamics ----
        T_load = fric0 + fricw * omega + load
        net = Tgas_torque - T_load + starter
        omega += net / inertia * dt
        if omega < 0.0:
            omega = 0.0   # engine can come to a full stop (stall)
        theta += dth
        if theta > 1e7:
            theta = _wrap(theta, cyc)
        st[0] = theta
        st[1] = omega
        torque_acc += Tgas_torque

        # ---- audio output: filtered tailpipe gauge pressure ----
        raw = Pt_acc / nsub - patm
        xprev = filt[0]
        yprev = filt[1]
        hp = raw - xprev + 0.999 * yprev   # DC blocker (removes offset/thump)
        filt[0] = raw
        filt[1] = hp
        lp = filt[2] + lpf * (hp - filt[2])  # one-pole lowpass (de-fizz)
        filt[2] = lp
        out_audio[s] = lp * outgain

    return torque_acc / n_samples
