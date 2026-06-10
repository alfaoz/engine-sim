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
P_DAMP        = 45  # pipe wall-loss scale (1 = physical friction/heat loss)
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

# --- per-cylinder intake runner (inertia ram tuning) ---
P_RUN_LEN     = 82  # runner length (m); 0 = ram model off (breathe from MAP)
P_RUN_AREA    = 83  # runner cross-sectional area (m^2)
P_RAM_MODE    = 84  # 0 = off (MAP), 1 = lumped column, 2 = 1D wave runners
P_RUN_NCELLS  = 85  # per-cylinder 1D runner cell count (mode 2)
P_QWALL       = 86  # (out) combustion heat rejected to the walls this block (J)

# --- fuelling / ignition / mechanical-noise (Phase 1 realism) ---
P_FUELCUT     = 87  # 1 = injection cut (DFCO / overrun), combustion disabled
P_PHI         = 88  # commanded fuel/air equivalence ratio (petrol fuelling)
P_CLATTER     = 89  # mechanical combustion-clatter mix (diesel knock, etc.)

# --- knock / combustion (Phase 2) ---
P_KNOCK       = 90  # (out) knock intensity accumulated this block
P_OCTANE      = 91  # fuel octane (RON) for the end-gas autoignition model
P_KNOCKAUD    = 92  # knock "ping" audio mix level

# --- vehicle: load transfer + road (Phase 2) + tyre slip (Phase 3) ---
P_GRADE       = 93  # road gradient angle (rad); + = uphill
P_WD_DRIVE    = 94  # static fraction of weight on the driven axle
P_XFER        = 95  # longitudinal load-transfer factor h/L (+RWD, -FWD, 0 AWD)
P_JWHEEL      = 96  # driven wheel + driveline rotational inertia (kg m^2)

# --- lateral dynamics (Phase 4) ---
P_STEER       = 97  # front road-wheel steer angle (rad)
P_WHEELBASE   = 98  # wheelbase (m)
P_WD_FRONT    = 99  # static front weight fraction
P_CGH         = 100 # CG height (m)
P_DIFF        = 101 # 0 = open diff, 1 = fully locked (LSD bias in between)

# --- exhaust muffler (geometry-derived lumped acoustics) ---
P_BODYFREQ    = 102 # exhaust-system body/Helmholtz resonance (Hz)
P_MUFFLP      = 103 # muffler chamber HF transmission-loss corner (Hz); 0 = off

P_TYRESLIP    = 104 # 1 = slip-ratio tyre + wheel DOF, 0 = rigid grip cap (dyno)

P_NBANKS      = 105 # number of separate exhaust banks (V/W engines -> 2+)

P_AFTERFIRE   = 106 # rich-running afterfire intensity (over-fuel flames/pops)

P_NPARAMS     = 107

# --- state vector layout (st) ---
S_THETA       = 0   # crank angle (rad)
S_OMEGA       = 1   # engine angular velocity (rad/s)
S_V           = 2   # vehicle speed (longitudinal, m/s)
S_PHI         = 3   # driveline torsional wind-up angle (rad)
S_MMAN        = 4   # intake manifold trapped gas mass (kg)
S_TURBO       = 5   # turbo shaft angular velocity (rad/s)
S_AIRFLOW     = 6   # low-passed engine air mass flow (kg/s) for the turbo
S_OMEGA_W     = 7   # driven-wheel angular velocity (rad/s) [tyre slip DOF]
S_VY          = 8   # lateral (body-frame) velocity (m/s) [lateral dynamics]
S_YAW         = 9   # yaw rate (rad/s) [lateral dynamics]
S_HEADING     = 10  # heading angle (rad) [lateral dynamics]
S_NSTATE      = 11

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
def tyre_mf(s, B, C):
    """Normalised Pacejka 'magic formula' shape (peak 1.0): force factor vs slip.
    Rises with slip to a peak then falls -- gives real wheelspin/lockup and the
    cornering grip limit. s is slip ratio (longitudinal) or slip angle (lateral)."""
    return np.sin(C * np.arctan(B * s))


@njit(cache=True, fastmath=True, inline='always')
def wiebe(frac, a, m):
    if frac <= 0.0:
        return 0.0
    if frac >= 1.0:
        return 1.0
    return 1.0 - np.exp(-a * frac ** (m + 1.0))


@njit(cache=True, fastmath=True, inline='always')
def _u_of_T(T, cv0, b):
    """Specific internal energy with a temperature-dependent heat capacity
    cv(T)=cv0+b*T (so gamma drops as the charge heats, ~1.40 cold air -> ~1.29
    hot combustion products). u = integral of cv dT from 0 K = (cv0+0.5 b T) T."""
    return (cv0 + 0.5 * b * T) * T


@njit(cache=True, fastmath=True, inline='always')
def _h_of_T(T, cv0, b, R):
    """Specific enthalpy h = u + R*T for the same cv(T) model (for the valve
    enthalpy transport: hotter gas carries more energy per kg than a fixed cp)."""
    return (cv0 + 0.5 * b * T + R) * T


@njit(cache=True, fastmath=True, inline='always')
def _T_of_u(u, cv0, b):
    """Invert u(T) (a quadratic in T) for the new temperature after the energy
    balance -- closed form, no iteration: 0.5 b T^2 + cv0 T - u = 0."""
    if u < 0.0:
        u = 0.0
    return (-cv0 + np.sqrt(cv0 * cv0 + 2.0 * b * u)) / b


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


@njit(cache=True, fastmath=True, inline='always')
def _minmod(a, b):
    """Minmod slope limiter: returns the smaller-magnitude argument if they
    share a sign, else 0. This is what keeps the 2nd-order reconstruction from
    overshooting (Gibbs ringing) across the sharp blowdown fronts."""
    if a * b <= 0.0:
        return 0.0
    return a if abs(a) < abs(b) else b


@njit(cache=True, fastmath=True, inline='always')
def _hllc(rL, uL, pL, rR, uR, pR, g):
    """HLLC approximate Riemann solver (Toro). Returns the interface flux
    (Fmass, Fmom, Fene). Unlike Rusanov/local-Lax-Friedrichs it carries the
    contact wave, so the hot-gas slug from blowdown stays a sharp front instead
    of being smeared over a dozen cells -- which is exactly the wave action that
    voices the exhaust. Constant gamma g (combustion products ~ fixed comp)."""
    aL = np.sqrt(g * pL / rL)
    aR = np.sqrt(g * pR / rR)
    EL = pL / (g - 1.0) + 0.5 * rL * uL * uL
    ER = pR / (g - 1.0) + 0.5 * rR * uR * uR
    # Davis/Einfeldt wave-speed estimates (robust, positively conservative)
    SL = uL - aL
    sr_ = uR - aR
    if sr_ < SL:
        SL = sr_
    SR = uR + aR
    sl_ = uL + aL
    if sl_ > SR:
        SR = sl_
    if SL >= 0.0:
        return rL * uL, rL * uL * uL + pL, (EL + pL) * uL
    if SR <= 0.0:
        return rR * uR, rR * uR * uR + pR, (ER + pR) * uR
    # contact (star) wave speed
    denom = rL * (SL - uL) - rR * (SR - uR)
    Sstar = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / denom
    if Sstar >= 0.0:
        F0 = rL * uL
        F1 = rL * uL * uL + pL
        F2 = (EL + pL) * uL
        fac = rL * (SL - uL) / (SL - Sstar)
        U0 = fac
        U1 = fac * Sstar
        U2 = fac * (EL / rL + (Sstar - uL) * (Sstar + pL / (rL * (SL - uL))))
        return (F0 + SL * (U0 - rL),
                F1 + SL * (U1 - rL * uL),
                F2 + SL * (U2 - EL))
    F0 = rR * uR
    F1 = rR * uR * uR + pR
    F2 = (ER + pR) * uR
    fac = rR * (SR - uR) / (SR - Sstar)
    U0 = fac
    U1 = fac * Sstar
    U2 = fac * (ER / rR + (Sstar - uR) * (Sstar + pR / (rR * (SR - uR))))
    return (F0 + SR * (U0 - rR),
            F1 + SR * (U1 - rR * uR),
            F2 + SR * (U2 - ER))


@njit(cache=True, fastmath=True)
def gas_step_q(rho, mom, Ene, N, dx, dt, g, R, patm, Tatm, p_open, T_open,
               src_m, src_E, area, aface, wk, fric, cool, dlin, wscale, Twall,
               hf, refl_st, refl_alpha):
    """One MUSCL-Hancock step of the QUASI-1D Euler equations (variable area
    A(x)) with an HLLC flux. Second-order in space (minmod-limited primitive
    reconstruction) and time (Hancock half-step predictor). Replaces the old
    first-order Rusanov gas_step: far less numerical diffusion, so real
    quarter-wave tuning, collector cross-talk and silencer transmission loss
    emerge from the geometry instead of being faked with output filters.

    area[i]  : cross-sectional area of cell i (m^2)   -- header/collector/muffler
    aface[k] : area at face k (N+1 faces); momentum sees a p*dA/dx wall force
    wk       : (>=15, N) scratch (primitives, slopes, evolved L/R, dU accumulators)

    Wall losses are PHYSICAL, not a flat momentum bleed:
    fric[i]  : turbulent Darcy wall friction f/(2D) per cell (1/m); the loss is
               ~rho*u|u|, so it bites the strong blowdown fronts and the mean
               flow but leaves small-signal acoustics nearly untouched.
    dlin     : linear (laminar boundary-layer) momentum loss per substep. The
               quadratic friction vanishes for small signals, so WITHOUT this
               the pipe modes never fully die. Kept at the PHYSICAL
               visco-thermal scale (a pulse loses only a few % per pipe
               transit); the mode ringing is mainly drained by the open-end
               radiation leak, which is the real mechanism. An over-strong
               value here guts the transmitted firing pulses (-13 dB measured
               at 0.0012) and the engine orders drown under flow noise.
    cool[i]  : wall heat-loss coefficient (1/m); the rate is cool[i]*|u| --
               the Reynolds analogy (heat transfer rides on forced convection,
               like the friction it mirrors). Flowing exhaust cools along the
               pipe -> a real axial sound-speed gradient; a STOPPED pipe stops
               exchanging, so a hot silencer can't pump a chimney flow that
               radiates as endless static.
    hf[i]    : grid-scale visco-thermal smoothing (one Laplacian pass) standing
               in for the boundary-layer HF dissipation a 1D scheme can't
               carry. PER CELL: elevated inside an absorptive muffler chamber
               (fibre packing eats high frequencies), small in bare pipe.

    Reflecting head wall at face 0. The tail face N is a Levine-Schwinger-style
    RADIATING open end: an unflanged pipe reflects low frequencies (pressure
    release at the reservoir p_open/T_open -- the quarter-wave tuning) but
    radiates frequencies above ka~1 instead of reflecting them. refl_st (1-elem
    persistent state) low-passes the tail gauge pressure at the corner set by
    the exit radius (refl_alpha per substep); the slow part is reflected, the
    fast part passes out through the ghost. Mean outflow weakens the reflection
    further (jet absorption, Munt) -> the pipe stops ringing like a sealed tube.
    """
    rho_open = p_open / (R * T_open)
    E_atm = patm / (g - 1.0)
    rho_atm = patm / (R * Tatm)

    # ---- cylinder / port sources (mass kg, energy J) -> densities via cell vol
    for i in range(N):
        if src_m[i] != 0.0 or src_E[i] != 0.0:
            vol = area[i] * dx
            rho[i] += src_m[i] / vol
            Ene[i] += src_E[i] / vol
            if rho[i] < 1e-6:
                rho[i] = 1e-6

    r = wk[0]; u = wk[1]; p = wk[2]
    sr = wk[3]; su = wk[4]; sp = wk[5]
    rLs = wk[6]; uLs = wk[7]; pLs = wk[8]
    rRs = wk[9]; uRs = wk[10]; pRs = wk[11]
    # dU accumulators reuse the preallocated scratch (rows 12-14) -- allocating
    # these per call would be ~12k mallocs per audio block and stall the callback
    # (periodic dropouts). They are zeroed in the primitive loop below.
    d0 = wk[12]; d1 = wk[13]; d2 = wk[14]

    for i in range(N):
        ui = mom[i] / rho[i]
        pi = (g - 1.0) * (Ene[i] - 0.5 * rho[i] * ui * ui)
        if pi < 1.0:
            pi = 1.0
        r[i] = rho[i]; u[i] = ui; p[i] = pi
        d0[i] = 0.0; d1[i] = 0.0; d2[i] = 0.0

    # ---- minmod-limited primitive slopes (zero at the two boundary cells) ----
    sr[0] = 0.0; su[0] = 0.0; sp[0] = 0.0
    sr[N - 1] = 0.0; su[N - 1] = 0.0; sp[N - 1] = 0.0
    for i in range(1, N - 1):
        sr[i] = _minmod(r[i] - r[i - 1], r[i + 1] - r[i])
        su[i] = _minmod(u[i] - u[i - 1], u[i + 1] - u[i])
        sp[i] = _minmod(p[i] - p[i - 1], p[i + 1] - p[i])

    # ---- Hancock predictor: reconstruct to faces, evolve cell by dt/2 ----
    hdtx = 0.5 * dt / dx
    for i in range(N):
        rl = r[i] - 0.5 * sr[i]; ul = u[i] - 0.5 * su[i]; pl = p[i] - 0.5 * sp[i]
        rr = r[i] + 0.5 * sr[i]; ur = u[i] + 0.5 * su[i]; pr = p[i] + 0.5 * sp[i]
        if rl < 1e-6: rl = 1e-6
        if rr < 1e-6: rr = 1e-6
        if pl < 1.0: pl = 1.0
        if pr < 1.0: pr = 1.0
        # conservative + physical flux at each face value
        EL = pl / (g - 1.0) + 0.5 * rl * ul * ul
        ER = pr / (g - 1.0) + 0.5 * rr * ur * ur
        FL0 = rl * ul; FL1 = rl * ul * ul + pl; FL2 = (EL + pl) * ul
        FR0 = rr * ur; FR1 = rr * ur * ur + pr; FR2 = (ER + pr) * ur
        # ΔU = (dt/2/dx)*(F_left - F_right), applied to BOTH interface states
        e0 = hdtx * (FL0 - FR0); e1 = hdtx * (FL1 - FR1); e2 = hdtx * (FL2 - FR2)
        UL0 = rl + e0; UL1 = rl * ul + e1; UL2 = EL + e2
        UR0 = rr + e0; UR1 = rr * ur + e1; UR2 = ER + e2
        if UL0 < 1e-6: UL0 = 1e-6
        if UR0 < 1e-6: UR0 = 1e-6
        uli = UL1 / UL0; uri = UR1 / UR0
        pli = (g - 1.0) * (UL2 - 0.5 * UL0 * uli * uli)
        pri = (g - 1.0) * (UR2 - 0.5 * UR0 * uri * uri)
        if pli < 1.0: pli = 1.0
        if pri < 1.0: pri = 1.0
        rLs[i] = UL0; uLs[i] = uli; pLs[i] = pli   # cell i's LEFT-face state
        rRs[i] = UR0; uRs[i] = uri; pRs[i] = pri   # cell i's RIGHT-face state

    # ---- face fluxes (HLLC) and quasi-1D conservative update ----
    # interior faces 1..N-1: between cell f-1 (right state) and cell f (left)
    for f in range(1, N):
        af = aface[f]
        F0, F1, F2 = _hllc(rRs[f - 1], uRs[f - 1], pRs[f - 1],
                           rLs[f], uLs[f], pLs[f], g)
        F0 *= af; F1 *= af; F2 *= af
        d0[f - 1] -= F0; d1[f - 1] -= F1; d2[f - 1] -= F2
        d0[f] += F0; d1[f] += F1; d2[f] += F2
    # head wall (face 0): reflecting -> mirror cell 0's left state with -u
    af0 = aface[0]
    F0, F1, F2 = _hllc(rLs[0], -uLs[0], pLs[0], rLs[0], uLs[0], pLs[0], g)
    d0[0] += F0 * af0; d1[0] += F1 * af0; d2[0] += F2 * af0
    # tail (face N): Levine-Schwinger RADIATING open end (see docstring). The
    # one-pole state refl_st splits the tail gauge pressure: low frequencies see
    # the reservoir (pressure-release reflection -> tuning), high frequencies
    # see their own extrapolation (transmitted -> radiated away). Outflow Mach
    # lets part of the low band through too (jet absorption).
    afN = aface[N]
    pg = pRs[N - 1] - p_open
    refl_st[0] += refl_alpha * (pg - refl_st[0])
    lpg = refl_st[0]
    rr_t = rRs[N - 1]
    a_t = np.sqrt(g * pRs[N - 1] / rr_t)
    # low-frequency leakage: even below ka~1 an open end radiates ~(ka)^2 of
    # each bounce, so the reflection is 0.90, not 1.0 -- without this the
    # quarter-wave modes have infinite Q and ring as steady tones.
    mfac = 0.10 + 2.5 * uRs[N - 1] / a_t
    if mfac < 0.10:
        mfac = 0.10
    elif mfac > 0.90:
        mfac = 0.90
    p_g = p_open + (pg - lpg) + mfac * lpg
    if p_g < 1.0:
        p_g = 1.0
    if uRs[N - 1] >= 0.0:
        T_g = pRs[N - 1] / (rr_t * R)     # outflow: ghost is the leaving hot gas
    else:
        T_g = T_open                      # inflow: reservoir air drawn back in
    rho_g = p_g / (R * T_g)
    if rho_g < 1e-6:
        rho_g = 1e-6
    F0, F1, F2 = _hllc(rRs[N - 1], uRs[N - 1], pRs[N - 1],
                       rho_g, uRs[N - 1], p_g, g)
    d0[N - 1] -= F0 * afN; d1[N - 1] -= F1 * afN; d2[N - 1] -= F2 * afN

    # ---- apply: dU = dt/Vol * (flux balance + p*dA momentum source) ----
    for i in range(N):
        vol = area[i] * dx
        inv = dt / vol
        rho[i] += inv * d0[i]
        # quasi-1D pressure-on-walls force: p_i * (A_{i+1/2} - A_{i-1/2})
        mom[i] += inv * (d1[i] + p[i] * (aface[i + 1] - aface[i]))
        Ene[i] += inv * d2[i]

    # ---- grid-scale visco-thermal smoothing (boundary-layer + packing) ----
    # conservative FACE-FLUX viscosity on VELOCITY and specific internal
    # energy. (Not a Laplacian on the conserved densities: smoothing rho*u
    # across a muffler cone's area jump makes a spurious wall force that
    # chokes the mean flow and can stall the engine.) Negligible below the
    # grid's resolved band, dissipative at the cell scale -- the role the
    # acoustic boundary layer plays, plus (where hf[i] is elevated) the fibre
    # packing of an absorptive silencer chamber. Uses the PRE-update
    # primitives in r/u/p (explicit diffusion).
    for f in range(1, N):
        epsf = 0.5 * (hf[f] + hf[f - 1])
        if epsf > 0.0:
            Amin = area[f] if area[f] < area[f - 1] else area[f - 1]
            mf = 0.5 * (r[f] + r[f - 1]) * Amin * dx     # face slug mass
            X = epsf * (u[f] - u[f - 1]) * mf            # momentum exchanged
            XE = (epsf * (p[f] / r[f] - p[f - 1] / r[f - 1])
                  * mf / (g - 1.0))                      # heat exchanged
            vol_l = area[f - 1] * dx
            vol_r = area[f] * dx
            mom[f - 1] += X / vol_l
            mom[f] -= X / vol_r
            Ene[f - 1] += XE / vol_l
            Ene[f] -= XE / vol_r

    E_w = R * Twall / (g - 1.0)        # wall-temperature internal energy / rho
    for i in range(N):
        # ---- robust clamps (also catch any non-finite value) ----
        ri = rho[i]
        if not (ri == ri) or ri < 1e-4:
            ri = 1e-4
        if ri > 30.0 * rho_atm:
            ri = 30.0 * rho_atm
        rho[i] = ri
        mi = mom[i]
        if not (mi == mi):
            mi = 0.0
        umax = 1200.0
        if mi > umax * ri:
            mi = umax * ri
        elif mi < -umax * ri:
            mi = -umax * ri
        # turbulent wall friction (Darcy): dmom = -f/(2D) * rho*u|u| * dt.
        # Quadratic in velocity, so it brakes the blowdown jet and the mean
        # flow without flattening the small-signal acoustics.
        ui = mi / ri
        mi -= dt * wscale * fric[i] * mi * (ui if ui >= 0.0 else -ui)
        mi -= dlin * mi                  # laminar/visco floor (kills ringing)
        mom[i] = mi
        kin = 0.5 * mi * mi / ri
        Eint = Ene[i] - kin
        if not (Eint == Eint) or Eint < E_atm * 1e-3:
            Eint = E_atm * 1e-3
        # wall heat loss (Reynolds analogy: rate ~ cool*|u|): the flowing gas
        # cools toward the pipe wall -> axial temperature (sound-speed)
        # gradient; zero exchange at standstill.
        au = ui if ui >= 0.0 else -ui
        Eint -= dt * wscale * cool[i] * au * (Eint - ri * E_w)
        if Eint < E_atm * 1e-3:
            Eint = E_atm * 1e-3
        T_i = (g - 1.0) * Eint / (ri * R)
        if T_i > 2800.0:
            Eint = ri * R * 2800.0 / (g - 1.0)
        Ene[i] = kin + Eint


@njit(cache=True, fastmath=True)
def simulate_block(n_samples, P, st, cyl_m, cyl_T, phase, inj, cyl_bank,
                   rho, mom, Ene, rho_in, mom_in, Ene_in,
                   run_mdot, fresh, cyl_knk, cyl_trim, cyl_var,
                   rho_r, mom_r, Ene_r, src_rm, src_re,
                   out_audio, scope_p, n_scope, filt,
                   pa_ex, fa_ex, wk_ex,
                   pa_in, fa_in, wk_in,
                   pa_run, fa_run, wk_run,
                   fric_ex, cool_ex, hf_ex, fric_in, cool_in, hf_in,
                   fric_run, cool_run, hf_run, bc_ex, bc_in, bc_run):
    """Advance the full engine by n_samples audio samples. Fills out_audio with
    the voiced tailpipe (exhaust) signal plus the intake-mouth induction signal,
    and records cylinder-0 pressure vs crank into scope_p for the UI. Returns
    mean engine torque. rho_in/mom_in/Ene_in are the 1D intake-duct gas state.

    Per-cylinder intake-runner ram (inertia tuning):
      run_mdot[c] : mass flow in cylinder c's intake runner (kg/s), a momentum
                    state -- the air column's inertia is what rams charge in.
      fresh[c]    : fresh air mass inducted this cycle (kg); meters the fuel so
                    the ram VE gain actually shows up as torque.

    ex_area/ex_aface (N / N+1) : exhaust pipe cross-section per cell / per face
    (header -> collector -> muffler -> tailpipe); the quasi-1D solver turns this
    geometry into real transmission loss and tuning. in_*/run_* are the same for
    the intake duct and the per-cylinder 1D runners. ex_wk/in_wk/run_wk are
    (>=15, N) MUSCL-Hancock scratch buffers (one per distinct pipe length)."""
    ncyl = int(P[P_NCYL])
    cyc = P[P_CYCLE]
    r = P[P_R]; L = P[P_L]; Apist = P[P_APIST]; Vclear = P[P_VCLEAR]
    Vdisp = P[P_VDISP]
    g = P[P_GAMMA]; R = P[P_RGAS]; cv = P[P_CV]
    cp = cv + R
    # temperature-dependent cylinder heat capacity cv(T)=cv0_c+b_c*T, anchored so
    # cv(1500 K) == the configured constant cv (preserves the power calibration)
    # while gamma falls realistically from cold intake air to hot products.
    b_c = 0.125
    cv0_c = cv - b_c * 1500.0
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
    grade = P[P_GRADE]                       # road gradient (rad)
    wd_drive = P[P_WD_DRIVE]                 # static weight frac on driven axle
    xfer = P[P_XFER]                         # longitudinal load-transfer h/L
    j_wheel = P[P_JWHEEL]                    # driven-wheel rotational inertia
    if j_wheel < 1e-3:
        j_wheel = 1.5
    tyre_slip = P[P_TYRESLIP] > 0.5          # 0 = rigid grip cap (dyno feel)
    # lateral / chassis
    steer = P[P_STEER]; wheelbase = P[P_WHEELBASE]
    wd_front = P[P_WD_FRONT]; cgh = P[P_CGH]; diff_lock = P[P_DIFF]
    has_veh = vmass > 5.0
    coupled = has_veh and ratio != 0.0 and Tcap > 0.0
    gex = P[P_GAMMAEX]; Rex = P[P_REX]; cvex = P[P_CVEX]
    N = int(P[P_NCELLS]); dx = P[P_DX]; dt = P[P_DT]
    # 1D intake duct
    Ni = int(P[P_IN_NCELLS]); dx_in = P[P_IN_DX]
    in_area = P[P_IN_AREA]; ingain = P[P_INGAIN]
    # per-cylinder intake runner (inertia ram), DIRECT-COLUMN model. Geometry
    # only -- no tuning: the runner air slug (length L_eff = runner length +
    # textbook 0.85*R open-end correction, area = runner area) has momentum.
    # d(mdot)/dt = (A/L_eff)*(P_plenum - P_cyl - dP_valve). The column's inertia
    # is what keeps packing charge toward IVC (ram); the valve is the only
    # restriction. There is NO buffer-volume constant to fit.
    run_len = P[P_RUN_LEN]; run_area = P[P_RUN_AREA]
    ram_mode = int(P[P_RAM_MODE])
    ram1 = ram_mode == 1 and run_area > 0.0 and four_stroke   # lumped column
    ram2 = ram_mode == 2 and run_area > 0.0 and four_stroke   # 1D wave runners
    ram_on = ram1 or ram2
    if ram1:
        run_leff = run_len + 0.85 * np.sqrt(run_area / np.pi)
        inert_k = run_area / run_leff        # inertance coupling (kg/s per Pa.s)
        ram_amin = 0.02 * Ain                # below this valve area, column halts
        col_mmax = 500.0 * run_area          # column speed cap (m/s) * area
    else:
        run_leff = 1.0; inert_k = 0.0; ram_amin = 0.0; col_mmax = 0.0
    # mode 2: per-cylinder 1D Euler runner. Each cylinder breathes from its OWN
    # runner head cell (which carries that runner's pressure wave); the runner
    # mouth (tail cell) is open to the plenum at MAP. The ram peak rpm and its
    # strength emerge purely from wave-speed x runner-length -- no fitted const.
    Nr = int(P[P_RUN_NCELLS])
    if ram2 and Nr > 1:
        dx_r = run_len / Nr
        amax_r = 480.0                       # cool-intake sound-speed headroom
        nsub_r = int(amax_r * dt / (0.8 * dx_r)) + 1
        dt_gas_r = dt / nsub_r
    else:
        ram2 = False; dx_r = 1.0; nsub_r = 1; dt_gas_r = dt
    # turbocharger
    turbo = P[P_TURBO] > 0.5
    J_trb = P[P_TRB_INERTIA]; r_imp = P[P_TRB_RIMP]
    eta_t = P[P_TRB_ETA]; wgate_pr = P[P_WGATE_PR]
    icool = P[P_ICOOL]; trb_cell = int(P[P_TRB_CELL])
    ht = P[P_HT]; wallT = P[P_WALLT]
    diesel = P[P_DIESEL] > 0.5
    wall_loss = P[P_DAMP]             # pipe wall-loss scale (1 = physical)
    outgain = P[P_OUTGAIN]
    redline = P[P_REDLINE]
    running = P[P_RUNNING] > 0.5
    starter = P[P_STARTER]
    lpf = P[P_LPF]
    noise = P[P_NOISE]
    pop = P[P_POP]
    afterfire = P[P_AFTERFIRE]   # rich-running afterfire (over-fuel flames/pops)
    fuelcut = P[P_FUELCUT] > 0.5     # DFCO / overrun injection cut
    phi_cmd = P[P_PHI]               # commanded equivalence ratio (petrol)
    if phi_cmd <= 0.0:
        phi_cmd = 1.0
    clatter_mix = P[P_CLATTER]       # mechanical combustion-clatter level
    octane = P[P_OCTANE]             # fuel RON for the knock model
    if octane < 1.0:
        octane = 95.0
    knock_aud = P[P_KNOCKAUD]        # knock "ping" audio mix
    knock_acc = 0.0                  # knock intensity reported to the ECU
    # end-gas margin: the unburned zone runs cooler than the mean charge temp
    # (wall heat loss, turbulence) so raw mean-T autoignition over-predicts. This
    # scale lifts the delay so pump-fuel engines sit just below knock at design
    # load and only detonate when lugged, over-boosted, or fed low octane.
    knock_scale = 4.0
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
    # solver-bandwidth lowpass: the grid resolves waves down to ~7 cells per
    # wavelength; above that is numerical hash, not physics, so the corner is
    # COMPUTED from the cell size (a finer grid genuinely sounds brighter).
    f_bw = 560.0 / (7.0 * P[P_DX])
    if f_bw < 2800.0:
        f_bw = 2800.0
    elif f_bw > 7500.0:
        f_bw = 7500.0
    f_main = 2.0 * np.sin(np.pi * f_bw / sr)
    d_main = 1.25                                # ~Butterworth damping
    # crest control is a STATIC soft-clip in the per-sample output stage (no
    # feedback limiter -- that pumped on the strong periodic diesel pulses).
    # body/Helmholtz resonance: frequency comes from the real muffler chamber
    # geometry (set in _build), not an arbitrary constant.
    body_hz = P[P_BODYFREQ]
    if body_hz < 20.0:
        body_hz = 95.0
    f_body = 2.0 * np.sin(np.pi * body_hz / sr)
    d_body = 0.22                                # high-Q -> resonant body
    # muffler chamber high-frequency transmission loss (one-pole LP); the corner
    # drops as the chamber grows, so a big silencer is darker than a sports can.
    muff_hz = P[P_MUFFLP]
    muff_lp = 0.0
    if muff_hz > 0.0:
        muff_lp = 1.0 - np.exp(-2.0 * np.pi * muff_hz / sr)
    body_gain = P[P_BODYGAIN]
    sat_on = P[P_SAT] > 0.5
    # mechanical combustion-clatter resonators: the combustion pressure-rise
    # rings the block/head structure. The resonance is now tied to engine size
    # (a small stiff bike block rings higher, a big diesel lower) instead of a
    # fixed pitch, and the resonators decay faster (lower Q) so the knocks stay
    # DISCRETE even at a fast petrol firing rate instead of smearing into a buzz.
    bore_m = np.sqrt(4.0 * Apist / np.pi)
    # ---- Woschni heat-transfer constants (precomputed) ----
    # h = wos_scale*3.26*B^-0.2*p[kPa]^0.8*T^-0.55*w^0.8 ; w = C1*Sp + C2*dp*term.
    # wos_scale (=P_HT) is the one calibration knob; everything else is the real
    # Woschni correlation, so wall heat loss now tracks pressure, speed and load.
    stroke_m = 2.0 * r                       # piston stroke
    bore_pow = bore_m ** (-0.2)
    wos_c2 = 3.24e-3                          # combustion velocity coefficient
    wos_scale = ht                           # P_HT repurposed as the Woschni gain
    PI_INV = 1.0 / np.pi
    clatter_hz = 145.0 / bore_m
    if clatter_hz < 850.0:
        clatter_hz = 850.0
    elif clatter_hz > 2500.0:
        clatter_hz = 2500.0
    # ---- modal body: the block/head structure as a small bank of modes ----
    # Mechanical sound is made by EVENTS (piston slap, valve seating, injector
    # ticks, the combustion pressure rise) striking a resonant structure. Six
    # low-Q modes at non-harmonic ratios of the bore-scaled base frequency
    # stand in for the block's dense modal field; the irregular event train
    # supplies the realism, not gated noise. (Lumped structure model -- this is
    # the one part of the sound that is structure-borne, not gas dynamics.)
    NMODE = 6
    mode_f = np.empty(NMODE)
    mode_d = np.empty(NMODE)
    mode_w = np.empty(NMODE)
    for _m in range(NMODE):
        if _m == 0:
            rat = 1.0; mode_d[_m] = 0.28; mode_w[_m] = 1.0
        elif _m == 1:
            rat = 1.62; mode_d[_m] = 0.30; mode_w[_m] = 0.80
        elif _m == 2:
            rat = 2.41; mode_d[_m] = 0.33; mode_w[_m] = 0.62
        elif _m == 3:
            rat = 3.55; mode_d[_m] = 0.36; mode_w[_m] = 0.48
        elif _m == 4:
            rat = 5.10; mode_d[_m] = 0.40; mode_w[_m] = 0.36
        else:
            rat = 7.30; mode_d[_m] = 0.45; mode_w[_m] = 0.26
        fmh = clatter_hz * rat
        if fmh > 9000.0:
            fmh = 9000.0
        mode_f[_m] = 2.0 * np.sin(np.pi * fmh / sr)
    # very short contact-noise burst per impact (~1.2 ms): the micro-rattle of
    # a real impact's contact chaos, NOT a sustained gated-noise wash
    clk_decay = np.exp(-1.0 / (0.0012 * sr))
    # mechanical event scalings (Pa-equivalent excitation):
    rL_slap = r / L                      # lateral force fraction of gas force
    # cold piston clearance multiplier (slap is loud on a cold block)
    cold_slap = 1.0 + 2.2 * (420.0 - wallT) / 150.0
    if cold_slap < 1.0:
        cold_slap = 1.0
    elif cold_slap > 3.2:
        cold_slap = 3.2
    slap_k = 0.008 * rL_slap             # slap impulse vs gas pressure (Pa-eq)
    slap_arm = 60000.0                   # re-arm threshold (0.6 bar side load)
    seat_in_k = 420.0 * np.sqrt(Ain)     # intake valve seating ~ omega*lift
    seat_ex_k = 540.0 * np.sqrt(Aex)     # exhaust seats harder (smaller, hotter)
    inj_k = 2600.0 if diesel else 0.0    # injector tick (events/cycle = ncyl)
    # spark-knock "ping": the end-gas detonation rings the chamber. Lower and
    # softer than before (~4.5 kHz, lower Q) so it reads as a rattle, not a
    # harsh digital sine.
    f_knk = 2.0 * np.sin(np.pi * 4500.0 / sr); d_knk = 0.30
    f_in = 2.0 * np.sin(np.pi * 3600.0 / sr)     # induction de-hash LPF corner

    # gas substeps to satisfy CFL; amax must exceed the temperature-capped
    # sound speed (~sqrt(g*R*2800) ~ 1040 m/s) so backfire spikes stay stable
    amax = 1150.0
    nsub = int(amax * dt / (0.8 * dx)) + 1
    dt_gas = dt / nsub

    # ---- open-end (Levine-Schwinger) termination corners ----
    # reflection low-pass corner f_c = c/(2*pi*a_exit): below it the open end
    # reflects (tuning), above it the pipe radiates. alpha is the one-pole
    # coefficient per GAS SUBSTEP: 2*pi*f_c*dt = c*dt/a.
    r_exit_ex = np.sqrt(fa_ex[N] / np.pi)
    c_ex_ref = np.sqrt(gex * Rex * 650.0)        # warm tailpipe sound speed
    alpha_ex = 1.0 - np.exp(-c_ex_ref * dt_gas / r_exit_ex)
    # radiation-efficiency lowpass (per SAMPLE): below ka~1 the orifice is a
    # monopole into the room; above it the jet beams forward and an off-axis
    # listener stops gaining the derivative's +6 dB/oct -- one pole at the same
    # Levine-Schwinger corner keeps lows physical and tames the HF "click".
    arad_ex = 1.0 - np.exp(-c_ex_ref * dt / r_exit_ex)
    d_exit_ex = 2.0 * r_exit_ex      # orifice diameter for the Strouhal peak
    # pipe-wall temperature for the wall heat loss (exhaust runs warm)
    Twall_ex = 520.0

    # separate exhaust banks (V/W engines): rho/mom/Ene are (nbanks, N); each
    # bank is its own collector pipe and radiates from its own tailpipe. The
    # summed tailpipes give the bank beat / cross-plane rumble of a real V8.
    nbanks = int(P[P_NBANKS])
    if nbanks < 1:
        nbanks = 1
    src_m = np.zeros((nbanks, N))
    src_E = np.zeros((nbanks, N))

    # intake duct: its own CFL substepping (cooler -> slower sound -> usually
    # fewer substeps than the hot exhaust). Allocate min size 1 when disabled.
    if Ni > 0:
        nsub_in = int(500.0 * dt / (0.8 * dx_in)) + 1
        dt_gas_in = dt / nsub_in
        r_exit_in = np.sqrt(fa_in[Ni] / np.pi)
        alpha_in = 1.0 - np.exp(-347.0 * dt_gas_in / r_exit_in)
        arad_in = 1.0 - np.exp(-347.0 * dt / r_exit_in)
    else:
        nsub_in = 1
        dt_gas_in = dt
        alpha_in = 0.5
        arad_in = 0.5
    Twall_in = 320.0                 # intake walls near ambient (loss ~off)
    if ram2:
        r_exit_r = np.sqrt(fa_run[Nr] / np.pi)
        alpha_run = 1.0 - np.exp(-347.0 * dt_gas_r / r_exit_r)
    else:
        alpha_run = 0.5
    src_in_m = np.zeros(Ni if Ni > 0 else 1)
    src_in_E = np.zeros(Ni if Ni > 0 else 1)

    torque_acc = 0.0
    map_acc = 0.0
    qwall_acc = 0.0      # combustion heat rejected to the walls this block (J)
    scope_period = cyc  # one full cycle for cylinder 0

    theta = st[S_THETA]
    omega = st[S_OMEGA]
    v = st[S_V]
    phi = st[S_PHI]
    m_man = st[S_MMAN]
    omega_trb = st[S_TURBO]
    air_ema = st[S_AIRFLOW]
    omega_w = st[S_OMEGA_W]      # driven-wheel speed (tyre slip DOF)
    vy = st[S_VY]                # lateral velocity (bicycle model)
    yaw = st[S_YAW]             # yaw rate
    heading = st[S_HEADING]

    # ---- vehicle / tyre constants (precomputed) ----
    v_eps = 2.5                  # slip-ratio low-speed regulariser (m/s)
    B_long = 10.0; C_long = 1.6  # longitudinal tyre magic-formula shape
    B_lat = 8.0; C_lat = 1.5     # lateral (cornering) tyre shape
    cos_grade = np.cos(grade); sin_grade = np.sin(grade)
    g_eff = grav * cos_grade     # normal-load gravity component on a slope
    awd = wd_drive > 0.95        # all-wheel drive (split torque both axles)
    rear_driven = xfer >= 0.0    # RWD: accel transfers load ONTO the driven axle
    lateral_on = wheelbase > 0.5 and has_veh
    if lateral_on:
        a_cg = (1.0 - wd_front) * wheelbase   # CG -> front axle
        b_cg = wd_front * wheelbase           # CG -> rear axle
        Iz = vmass * a_cg * b_cg              # yaw inertia (radius_gyr^2 = a*b)
        if Iz < 1.0:
            Iz = 1.0
    else:
        a_cg = 1.0; b_cg = 1.0; Iz = 1.0
    ax_long = 0.0                # longitudinal accel (lagged, for load transfer)

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
        clk_hit = 0.0           # impulsive mechanical impacts this sample (Pa-eq)
        knock_exc = 0.0         # end-gas knock excitation this sample
        air_draw = 0.0          # net intake-valve mass flow this sample (duct src)
        for b in range(nbanks):
            for k in range(N):
                src_m[b, k] = 0.0
                src_E[b, k] = 0.0
        if ram2:
            for c in range(ncyl):
                src_rm[c, 0] = 0.0      # head-cell source per runner (valve draw)
                src_re[c, 0] = 0.0

        for c in range(ncyl):
            cph = _wrap(theta + phase[c], cyc)        # cycle phase
            cph_prev = _wrap(theta + phase[c] - dth, cyc)
            crank = _wrap(cph, TWO_PI)                # for volume (periodic 2pi)
            V, dVdth = cyl_volume(crank, r, L, Apist, Vclear)
            m = cyl_m[c]; T = cyl_T[c]
            Pc = m * R * T / V
            if Pc < 1.0:
                Pc = 1.0

            # valve open fractions (hoisted: shared by the Woschni heat transfer
            # and the valve-flow integration below)
            fin = open_frac(cph, IVO, IVC, cyc)
            fex = open_frac(cph, EVO, EVC, cyc)

            # ---- trapped reference captured at IVC (ALL engines): the Woschni
            # motored-pressure term and the end-gas knock model both read it ----
            ddk = cph - cph_prev
            if ddk < 0.0:
                ddk += cyc
            rel_ivc = IVC - cph_prev
            if rel_ivc < 0.0:
                rel_ivc += cyc
            ivc_edge = rel_ivc <= ddk
            if ivc_edge:
                cyl_knk[c, 0] = T      # trapped temperature
                cyl_knk[c, 1] = Pc     # trapped pressure
                cyl_knk[c, 4] = V      # trapped volume (for motored pressure)
                # ---- per-CYCLE combustion quality (drawn once per cycle) ----
                # Real cycle-to-cycle IMEP variation is a slow wander (mixture
                # prep, residual gas), not white noise: an AR(1) walk whose
                # stationary std equals the configured roughness, correlated
                # over ~8 cycles. On top: the cylinder's STATIC trim (injector/
                # compression spread -- the engine's permanent half-order
                # signature) and a cold/lean partial-burn lottery (the stumble
                # of a cold or starved engine).
                w = 0.88 * cyl_var[c, 0] + 0.475 * noise * np.random.standard_normal()
                cyl_var[c, 0] = w
                q_cyc = cyl_trim[c] * (1.0 + w)
                pmis = 0.0
                if wallT < 350.0:
                    pmis += 0.04 * (350.0 - wallT) / 60.0
                if not diesel and phi_cmd < 0.78:
                    pmis += (0.78 - phi_cmd) * 1.5
                if pmis > 0.0 and np.random.random() < pmis:
                    q_cyc *= 0.45 + 0.4 * np.random.random()   # partial burn
                if q_cyc < 0.0:
                    q_cyc = 0.0
                cyl_var[c, 1] = q_cyc

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
            if dxb > 0.0 and rpm_now < redline and running and not fuelcut:
                # fuel energy available this cycle (mass trapped is ~constant now)
                if diesel:
                    # fuel metered by throttle, capped by the smoke limit (a
                    # diesel runs lean even at full load -> realistic BMEP)
                    air = m
                    fuel_max = air / 23.0
                    fuel = throttle * fuel_max
                    Qcyc = fuel * LHV * ceff
                else:
                    if ram_on:
                        # meter by the fresh air the runner ACTUALLY inducted this
                        # cycle -- so the inertia-ram VE gain shows up as torque.
                        air = fresh[c]
                        if air < 0.0:
                            air = 0.0
                        if air > m:
                            air = m
                    else:
                        # no runner model: estimate fresh charge from manifold
                        # pressure (trapped residual carries no O2). Lets 2T idle.
                        fr = MAP * Vdisp / (R * Tman)
                        air = m if m < fr else fr
                    # equivalence-ratio fuelling. AFR is stoichiometric; phi_cmd is
                    # the ECU's commanded richness. Beyond stoich (phi>1) the extra
                    # fuel finds no O2 so the released energy is capped (burn_phi);
                    # a small completeness bump peaks just rich of stoich (why peak
                    # power is made ~12.5:1), and a lean charge burns less and
                    # eventually misfires. At phi=1 comp=1 so the WOT calibration
                    # (stoich) is preserved exactly.
                    stoich_fuel = air / AFR
                    burn_phi = phi_cmd if phi_cmd < 1.0 else 1.0
                    if phi_cmd <= 1.0:
                        comp = 1.0 - 0.8 * (1.0 - phi_cmd) * (1.0 - phi_cmd)
                        if phi_cmd < 0.6:
                            comp *= phi_cmd / 0.6      # very lean -> misfire toward 0
                    else:
                        d = phi_cmd - 1.0
                        comp = 1.0 + 0.5 * d - 2.2 * d * d
                    if comp < 0.0:
                        comp = 0.0
                    Qcyc = stoich_fuel * burn_phi * LHV * ceff * comp
                # per-cycle quality (static trim x AR(1) wander x partial-burn,
                # drawn at IVC) plus a SMALL per-sample flame-front roughness
                rough = cyl_var[c, 1] * (1.0 + 0.25 * noise
                                         * np.random.standard_normal())
                if rough < 0.0:
                    rough = 0.0
                dq_fuel = Qcyc * dxb * rough
                dQ += dq_fuel
                # combustion impact: the heat-release pressure rise (g-1)*dQ/V is
                # the force that hammers the structure (sharp for diesel CI). It
                # feeds the impulse accumulator, not a sustained tone.
                cb_imp = (g - 1.0) * dq_fuel / V
                clk_hit += cb_imp

            # ---- end-gas knock model (petrol spark-ignition only) -----------
            # The unburned end gas is compressed isentropically by the piston and
            # the advancing flame; if it auto-ignites before the flame arrives it
            # detonates -> knock. A Livengood-Wu integral over the Douaud-Eyzat
            # ignition delay (a standard gasoline correlation) decides when. Knock
            # is promoted by high pressure (boost/CR), high temperature, advanced
            # timing (more end-gas dwell at high P), and low octane -- all emergent.
            if not diesel:
                if ivc_edge:                        # intake valve just closed
                    cyl_knk[c, 2] = 0.0             # reset the knock integral
                    cyl_knk[c, 3] = 0.0             # reset knocked-this-cycle flag
                if running and not fuelcut and cyl_knk[c, 3] < 0.5:
                    Tivc = cyl_knk[c, 0]; Pivc = cyl_knk[c, 1]
                    if Pivc > 1.0 and Pc > Pivc:
                        Tu = Tivc * (Pc / Pivc) ** ((g - 1.0) / g)
                        if Tu > 320.0:
                            # delay (s): 0.01768*(ON/100)^3.402*P[atm]^-1.7*exp(3800/Tu)
                            tau = (knock_scale * 0.01768 * (octane * 0.01) ** 3.402
                                   * (Pc / 101325.0) ** (-1.7) * np.exp(3800.0 / Tu))
                            cyl_knk[c, 2] += dt / tau
                            if cyl_knk[c, 2] >= 1.0:
                                unburned = 1.0 - wiebe(f_now, wa, wm)
                                if unburned > 0.05:
                                    ki = unburned * Pc * 1.0e-6   # ~MPa of end gas
                                    knock_acc += ki
                                    knock_exc += ki
                                    cyl_knk[c, 3] = 1.0           # one event / cycle

            # ---- mechanical impact events (excite the modal body) ----
            # These are TIMED by the simulation state, not synthesized: the
            # event train's irregular cadence is what reads as "mechanical".
            if clatter_mix > 0.0:
                # piston slap: the lateral gas force F_gas*tan(rod angle) flips
                # side across TDC/BDC and throws the piston over its clearance;
                # impact ~ gas pressure at the crossing, louder on a cold bore.
                # ARMED only above a pressure threshold: during gas exchange
                # Pc hovers at ~patm and the raw sign chatters every few
                # samples, which reads as continuous static, not impacts.
                lat = (Pc - patm) * np.sin(crank)
                if lat > slap_arm:
                    s_now = 1.0
                elif lat < -slap_arm:
                    s_now = -1.0
                else:
                    s_now = 0.0
                if s_now != 0.0:
                    if s_now * cyl_knk[c, 5] < 0.0:
                        dpp = Pc - patm
                        if dpp < 0.0:
                            dpp = -dpp
                        clk_hit += slap_k * dpp * cold_slap
                    cyl_knk[c, 5] = s_now
                # valve seating ticks: seat velocity ~ engine speed
                rel_vc = EVC - cph_prev
                if rel_vc < 0.0:
                    rel_vc += cyc
                if rel_vc <= ddk:
                    clk_hit += seat_ex_k * omega
                if ivc_edge:
                    clk_hit += seat_in_k * omega
                # injector tick at start of injection (diesel common rail)
                if inj_k > 0.0 and running and not fuelcut:
                    rel_ig = ign - cph_prev
                    if rel_ig < 0.0:
                        rel_ig += cyc
                    if rel_ig <= ddk:
                        clk_hit += inj_k * (0.25 + 0.75 * throttle)

            # ---- Woschni in-cylinder heat transfer ----
            # h_c (W/m^2K) rises with charge pressure, gas velocity (piston speed +
            # a combustion-driven term from the pressure rise above the motored
            # trace) and falls with temperature. Multiplied by the instantaneous
            # exposed wall area (crown + head + bared liner). So wall loss now
            # grows with load and speed -- warm-up, the cold-running enrichment and
            # the exhaust energy all follow real physics, not a fixed coefficient.
            Sp = stroke_m * omega * PI_INV          # mean piston speed
            V_ivc = cyl_knk[c, 4]; P_ivc = cyl_knk[c, 1]; T_ivc = cyl_knk[c, 0]
            if V_ivc > 1e-12 and P_ivc > 1.0:
                p_mot = P_ivc * (V_ivc / V) ** g     # motored (no-burn) pressure
                dpm = Pc - p_mot
                if dpm < 0.0:
                    dpm = 0.0
                w_wos = 2.28 * Sp + wos_c2 * dpm * Vdisp * T_ivc / (P_ivc * V_ivc)
            else:
                w_wos = 2.28 * Sp
            if fin > 0.0 or fex > 0.0:
                w_wos += 3.9 * Sp                    # gas-exchange velocity (C1=6.18)
            if w_wos < 0.1:
                w_wos = 0.1
            h_wos = (wos_scale * 3.26 * bore_pow * (Pc * 1.0e-3) ** 0.8
                     * T ** (-0.55) * w_wos ** 0.8)
            x_pist = (V - Vclear) / Apist            # piston travel from TDC
            A_wall = 2.0 * Apist + np.pi * bore_m * x_pist
            q_w = h_wos * A_wall * (T - wallT) * dt
            dQ -= q_w
            qwall_acc += q_w          # this heat banks into the metal (sim side)

            # ---- valve flows ----
            dm_in = 0.0
            dm_ex = 0.0
            dH = 0.0
            dd = cph - cph_prev
            if dd < 0.0:
                dd += cyc
            # the fresh-charge accumulator resets at intake-valve opening, then
            # sums the real inducted air over the intake event (for fuel metering)
            rel_i = IVO - cph_prev
            if rel_i < 0.0:
                rel_i += cyc
            if ram_on and rel_i <= dd:
                fresh[c] = 0.0

            if ram2:
                # ---- 1D wave runner: breathe from THIS cylinder's runner head
                # cell (it carries the runner's pressure wave); the head-cell
                # draw is deposited as a source, advanced by gas_step below. ----
                if fin > 0.0:
                    rh = rho_r[c, 0]
                    uh = mom_r[c, 0] / rh
                    P_rh = (g - 1.0) * (Ene_r[c, 0] - 0.5 * rh * uh * uh)
                    if P_rh < 1.0:
                        P_rh = 1.0
                    T_rh = P_rh / (rh * R)
                    area = Ain * fin
                    if P_rh >= Pc:
                        md = orifice_mdot(P_rh, T_rh, Pc, area, Cd, g, R) * dt
                        dm_in += md
                        dH += md * _h_of_T(T_rh, cv0_c, b_c, R)
                        air_draw += md
                        fresh[c] += md
                        src_rm[c, 0] -= md          # mass leaves the runner head
                        src_re[c, 0] -= md * _h_of_T(T_rh, cv0_c, b_c, R)
                        m_man -= md                 # bulk charge drains the plenum
                        if m_man < 1e-9:
                            m_man = 1e-9
                    else:
                        md = orifice_mdot(Pc, T, P_rh, area, Cd, g, R) * dt
                        dm_in -= md
                        dH -= md * _h_of_T(T, cv0_c, b_c, R)
                        air_draw -= md
                        fresh[c] -= md
                        src_rm[c, 0] += md          # reversion back into the runner
                        src_re[c, 0] += md * _h_of_T(T, cv0_c, b_c, R)
                        m_man += md
            elif ram1:
                # ---- inertia-ram runner: integrate the air column's momentum ----
                A_v = Ain * fin
                if A_v > ram_amin:
                    mdot = run_mdot[c]
                    rho_up = MAP / (R * Tman)
                    CA = Cd * A_v
                    qd = mdot / CA
                    dP_valve = qd * (qd if qd >= 0.0 else -qd) / (2.0 * rho_up)
                    # column momentum: plenum pressure minus cylinder minus the
                    # valve loss needed to pass the current flow
                    mdot += dt * inert_k * (MAP - Pc - dP_valve)
                    if mdot > col_mmax:
                        mdot = col_mmax
                    elif mdot < -col_mmax:
                        mdot = -col_mmax
                    run_mdot[c] = mdot
                    md = mdot * dt
                    if md >= 0.0:
                        dm_in += md
                        dH += md * _h_of_T(Tman, cv0_c, b_c, R)
                        air_draw += md        # induction: mass pulled from tract
                        fresh[c] += md        # real fresh air trapped this cycle
                        if four_stroke:
                            m_man -= md
                            if m_man < 1e-9:
                                m_man = 1e-9
                    else:
                        dm_in += md           # md < 0: reversion out of cylinder
                        dH += md * _h_of_T(T, cv0_c, b_c, R)
                        air_draw += md
                        fresh[c] += md
                        if four_stroke:
                            m_man -= md       # back into the plenum
                else:
                    run_mdot[c] *= 0.6        # valve ~shut: column stagnates
            elif fin > 0.0:
                # ---- plain plenum breathing (no runner model) ----
                area = Ain * fin
                if MAP >= Pc:
                    md = orifice_mdot(MAP, Tman, Pc, area, Cd, g, R) * dt
                    dm_in += md
                    dH += md * _h_of_T(Tman, cv0_c, b_c, R)
                    air_draw += md            # induction: mass pulled from tract
                    if four_stroke:
                        m_man -= md           # charge drawn out of the plenum
                else:
                    md = orifice_mdot(Pc, T, MAP, area, Cd, g, R) * dt
                    dm_in -= md
                    dH -= md * _h_of_T(T, cv0_c, b_c, R)
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

            if fex > 0.0:
                area = Aex * fex
                # exhaust pipe head cell pressure (in THIS cylinder's bank pipe)
                cb = cyl_bank[c]
                ci = inj[c]
                ui = mom[cb, ci] / rho[cb, ci]
                Pp = (gex - 1.0) * (Ene[cb, ci] - 0.5 * rho[cb, ci] * ui * ui)
                if Pp < 1.0:
                    Pp = 1.0
                if Pc >= Pp:
                    md = orifice_mdot(Pc, T, Pp, area, Cd, g, R) * dt
                    dm_ex += md
                    dH -= md * _h_of_T(T, cv0_c, b_c, R)
                    # deposit into the bank's pipe cell
                    src_m[cb, ci] += md
                    src_E[cb, ci] += md * _h_of_T(T, cv0_c, b_c, R)   # stagnation enthalpy in
                else:
                    # reversion: pipe gas back into cylinder
                    Tp = Pp / (rho[cb, ci] * Rex)
                    md = orifice_mdot(Pp, Tp, Pc, area, Cd, gex, Rex) * dt
                    dm_ex -= md
                    dH += md * _h_of_T(Tp, cv0_c, b_c, R)
                    src_m[cb, ci] -= md
                    src_E[cb, ci] -= md * _h_of_T(Tp, cv0_c, b_c, R)

                # ---- afterfire: unburnt fuel igniting in the hot pipe ----
                # At most ONCE per exhaust event (this EVO edge) so the crackle is
                # locked to the firing rate, not the sample rate. Two mechanisms:
                #  (a) LEAN overrun -- closed throttle pumps air, free O2 in the
                #      pipe ignites the residual fuel (the decel "pops & bangs").
                #  (b) RICH running (phi>1) -- an over-fuelled engine dumps unburnt
                #      fuel out with the charge; in a hot pipe it deflagrates and
                #      throws flames (Harley/open-radial character). Probability and
                #      burst size grow with the richness and need a hot pipe.
                if evo_edge:
                    Tp_cell = Pp / (rho[cb, ci] * Rex)
                    if pop > 0.0 and np.random.random() < pop * 0.5:
                        src_E[cb, ci] += pop_burst * (0.6 + 0.8 * np.random.random())
                    rich = phi_cmd - 1.0
                    if (afterfire > 0.0 and rich > 0.0 and Tp_cell > 650.0
                            and running and not fuelcut
                            and np.random.random() < afterfire * rich * 3.0):
                        src_E[cb, ci] += (pop_burst * rich * 1.5
                                          * (0.5 + np.random.random()))

            # ---- update cylinder state (energy & mass balance) ----
            # internal energy uses the temperature-dependent cv(T); the new temp is
            # recovered by inverting u(T) (closed form) so the higher heat capacity
            # of hot products correctly limits the peak temperature.
            U = m * _u_of_T(T, cv0_c, b_c)
            work = Pc * dVdth * dth
            U_new = U + dQ - work + dH
            m_new = m + dm_in - dm_ex
            if m_new < 1e-8:
                m_new = 1e-8
            T_new = _T_of_u(U_new / m_new, cv0_c, b_c)
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
        # Each bank is advanced as its own 1D pipe. The tap is the orifice MASS
        # FLUX (summed over banks): what a microphone hears from an open pipe
        # end is the monopole field radiated by the fluctuating efflux, not the
        # in-duct pressure -- the radiated signal is formed below as d(mdot)/dt.
        mdot_acc = 0.0
        for _ in range(nsub):
            for b in range(nbanks):
                gas_step_q(rho[b], mom[b], Ene[b], N, dx, dt_gas, gex, Rex,
                           patm, Tatm, patm, Tatm, src_m[b], src_E[b],
                           pa_ex, fa_ex, wk_ex, fric_ex, cool_ex, 0.00012,
                           wall_loss, Twall_ex, hf_ex, bc_ex[b:b + 1], alpha_ex)
            # sources are an impulse for the whole sample; apply only once
            for b in range(nbanks):
                for k in range(N):
                    src_m[b, k] = 0.0
                    src_E[b, k] = 0.0
            for b in range(nbanks):
                mdot_acc += mom[b, N - 1]      # rho*u at the exit cell

        # ---- orifice JET noise: the turbulent rush of the exhaust jet ----
        # Lighthill scaling: far-field jet pressure ~ rho*U^2 * M^3 * D/r --
        # the M^3 acoustic-efficiency factor is what makes a low-speed jet
        # essentially SILENT and a near-sonic blowdown snarl. The spectrum is
        # humped at the Strouhal frequency f = 0.2*U/D. Both track the
        # instantaneous exit velocity, so the rasp breathes with the firing
        # pulses; there is no flat noise floor to bury the engine orders.
        q_jet = 0.0
        ue_pk = 0.0
        for b in range(nbanks):
            ue = mom[b, N - 1] / rho[b, N - 1]
            au = ue if ue >= 0.0 else -ue
            q_jet += rho[b, N - 1] * au * au
            if au > ue_pk:
                ue_pk = au
        M_j = ue_pk / c_ex_ref
        if M_j > 1.0:
            M_j = 1.0
        a_jet = q_jet * M_j * M_j * M_j * d_exit_ex   # Pa at r_mic = 1 m
        f_st = 0.2 * ue_pk / d_exit_ex
        if f_st < 120.0:
            f_st = 120.0
        elif f_st > 8500.0:
            f_st = 8500.0
        fj = 2.0 * np.sin(np.pi * f_st / sr)
        lpj = filt[37]; bpj = filt[38]
        hj = a_jet * np.random.standard_normal() - lpj - 1.2 * bpj
        bpj += fj * hj
        lpj += fj * bpj
        filt[37] = lpj; filt[38] = bpj

        # ---- per-cylinder 1D intake runners (mode 2): advance each runner with
        # its valve draw as a head-cell source and the mouth open to the plenum
        # (MAP). The reflected wave that ram peaks each runner is emergent. ----
        if ram2:
            for c in range(ncyl):
                rrho = rho_r[c]; rmom = mom_r[c]; rEne = Ene_r[c]
                srm = src_rm[c]; sre = src_re[c]
                for _ in range(nsub_r):
                    gas_step_q(rrho, rmom, rEne, Nr, dx_r, dt_gas_r, g, R,
                               patm, Tatm, MAP, Tman, srm, sre,
                               pa_run, fa_run, wk_run, fric_run, cool_run,
                               0.010, wall_loss, Twall_in, hf_run,
                               bc_run[c:c + 1], alpha_run)
                    srm[0] = 0.0
                    sre[0] = 0.0

        # ---- intake duct gas dynamics (induction sound is emergent) ----
        # The duct is a 1D Euler pipe excited at the head (cell 0) by the REAL
        # per-cylinder intake-valve mass flow (pulsed by valve lift -- the actual
        # induction source, not a steady throttle impulse) and open at the mouth
        # (cell Ni-1) to the airbox/boost reservoir, which refills the draw. The
        # pressure wave radiating from the mouth IS the induction "suck"/honk,
        # and it rises with boost -> the turbo's intake character. Loudness
        # scales with airflow automatically (big draw at WOT, tiny at idle).
        mdot_in_acc = 0.0
        if Ni > 0 and ingain > 0.0:
            # sound-only runner: excited by the real valve draw, open at the
            # mouth to the boost/airbox reservoir which refills it.
            src_in_m[0] = -air_draw
            src_in_E[0] = -air_draw * cp * Tman
            for _ in range(nsub_in):
                gas_step_q(rho_in, mom_in, Ene_in, Ni, dx_in, dt_gas_in, g, R,
                           patm, Tatm, Pboost, Tman, src_in_m, src_in_E,
                           pa_in, fa_in, wk_in, fric_in, cool_in, 0.012,
                           wall_loss, Twall_in, hf_in, bc_in, alpha_in)
                for k in range(Ni):
                    src_in_m[k] = 0.0
                    src_in_E[k] = 0.0
                mdot_in_acc += mom_in[Ni - 1]    # mouth mass flux (per area)

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
            # (bank 0; a single turbine fed off one collector is fine here)
            rt = rho[0, trb_cell]
            ut = mom[0, trb_cell] / rt
            pt = (gex - 1.0) * (Ene[0, trb_cell] - 0.5 * rt * ut * ut)
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
            trb_vol = pa_ex[trb_cell] * dx
            cap = 0.3 * Ene[0, trb_cell] * trb_vol
            if ext > cap:
                ext = cap
            Ene[0, trb_cell] -= ext / trb_vol

        # ============================================================
        # COUPLED MECHANICS — engine, driveline and vehicle integrated
        # together every sample. The clutch + driveline is a torsional
        # spring/damper whose torque saturates at the clutch capacity
        # (Coulomb friction). Rigid drive, launch slip and engine braking
        # all emerge from this one law; there are no mode switches.
        # ============================================================
        # net torque at the engine shaft, before the clutch
        T_eng = Tgas_torque - (fric0 + fricw * omega) - load + starter

        if not has_veh:
            # bench: engine spins free (free-rev dyno)
            omega += T_eng / inertia * dt
            if omega < 0.0:
                omega = 0.0
            phi = 0.0
        else:
            vx = v
            if not tyre_slip:
                omega_w = vx / rw if rw > 0.0 else 0.0   # rigid: wheel = road
            # ---- engine <-> clutch <-> gearbox-input (driven by the WHEEL) ----
            if coupled:
                omega_in = omega_w * ratio        # gearbox input = wheel x ratio
                slip_cl = omega - omega_in
                phi += slip_cl * dt               # driveline wind-up
                T_cl = kdl * phi + cdl * slip_cl
                if T_cl > Tcap:                   # clutch saturates (Coulomb)
                    T_cl = Tcap
                    phi = (Tcap - cdl * slip_cl) / kdl
                elif T_cl < -Tcap:
                    T_cl = -Tcap
                    phi = (-Tcap - cdl * slip_cl) / kdl
                omega += (T_eng - T_cl) / inertia * dt
                if omega < 0.0:
                    omega = 0.0
                T_wheel = T_cl * ratio            # torque delivered to drive wheels
            else:
                omega += T_eng / inertia * dt     # declutched: engine free
                if omega < 0.0:
                    omega = 0.0
                phi = 0.0
                T_wheel = 0.0
                omega_w = vx / rw if rw > 0.0 else 0.0   # wheel tracks the road

            # ---- normal loads with quasi-static longitudinal transfer ----
            # accel pitches load rearward (ax_long>0 -> onto a RWD driven axle,
            # off an FWD one). Uses last sample's accel (lagged) to stay explicit.
            N_fr = vmass * (g_eff * wd_front - ax_long * xfer)
            N_re = vmass * (g_eff * (1.0 - wd_front) + ax_long * xfer)
            if N_fr < 0.0:
                N_fr = 0.0
            if N_re < 0.0:
                N_re = 0.0
            if awd:
                N_drive = N_fr + N_re
            elif rear_driven:
                N_drive = N_re
            else:
                N_drive = N_fr

            # ---- driven-axle force (straight-line / dyno: no cornering) ----
            f_grip = mu * N_drive
            if tyre_slip:
                # slip-ratio tyre: Fx = mu*N*MF(slip); the wheel is its own DOF,
                # so wheelspin (and recovery) emerge. MF saturates at the grip
                # limit so no friction-circle is needed.
                vabs = (vx if vx >= 0.0 else -vx) + v_eps
                kappa = (omega_w * rw - vx) / vabs
                Fx = f_grip * tyre_mf(kappa, B_long, C_long)
                if coupled:
                    omega_w += (T_wheel - Fx * rw) / j_wheel * dt
                    if omega_w < 0.0:
                        omega_w = 0.0
            else:
                # rigid grip cap (dyno feel): deliver the wheel torque straight
                # to the contact patch, clamped at the traction limit. No wheel
                # DOF, no wheelspin.
                Fx = T_wheel / rw
                if Fx > f_grip:
                    Fx = f_grip
                elif Fx < -f_grip:
                    Fx = -f_grip

            # ---- body longitudinal motion ----
            sgn = 1.0 if vx >= 0.0 else -1.0
            roll = crr * vmass * g_eff if vx > 0.01 else 0.0
            drag = cda * vx * vx
            Fx_net = (Fx - drag * sgn - roll * sgn - brakef * sgn
                      - vmass * grav * sin_grade)
            ax_long = Fx_net / vmass
            vx += ax_long * dt
            if vx < 0.0:
                vx = 0.0
                if not coupled:
                    omega_w = 0.0
            v = vx

        theta += dth
        if theta > 1e7:
            theta = _wrap(theta, cyc)
        torque_acc += Tgas_torque

        # ---- audio output: monopole radiation from the tailpipe orifice ----
        # far-field pressure of an open pipe end at mic distance r_mic:
        #   p(t) = d(mdot_exit)/dt / (4*pi*r_mic)        [mass-flux monopole]
        # The derivative is what physically happens between duct and room: DC
        # and infrasound vanish, the spectrum tilts +6 dB/oct, and the "inside
        # a sealed tube" coloration goes away. r_mic = 1 m.
        mdot_now = mdot_acc * fa_ex[N] / nsub
        p_rad = (mdot_now - filt[0]) * (sr * 0.0795775)   # 1/(4*pi)*sr
        filt[0] = mdot_now
        # radiation-efficiency / off-axis lowpass (see arad_ex above)
        filt[20] += arad_ex * (p_rad - filt[20])
        raw = filt[20] + bpj
        # gentle DC blocker (numerical drift safety; the derivative already
        # removed the physical DC)
        hp = raw - filt[21] + 0.995 * filt[1]
        filt[21] = raw
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

        # ---- mechanical structure-borne sound: impacts ring the modal body --
        # clk_hit is this sample's summed impact excitation (combustion
        # pressure rise + piston slap + valve seats + injector ticks, all
        # timed by the physics above). Each impact strikes the 6-mode block
        # bank directly and gates a ~1 ms contact-noise burst -- discrete
        # mechanical events whose RATE rises with rpm, not a steady buzz.
        if clatter_mix > 0.0:
            kick = clk_hit * outgain * 5.0e-4
            env = filt[36] * clk_decay
            if kick > env:                               # peak-follow
                env = kick
            filt[36] = env
            drive = kick + 0.25 * env * np.random.standard_normal()
            acc_m = 0.0
            for m_ in range(NMODE):
                lp_m = filt[24 + 2 * m_]; bp_m = filt[25 + 2 * m_]
                hp_m = drive - lp_m - mode_d[m_] * bp_m
                bp_m += mode_f[m_] * hp_m
                lp_m += mode_f[m_] * bp_m
                filt[24 + 2 * m_] = lp_m
                filt[25 + 2 * m_] = bp_m
                acc_m += mode_w[m_] * bp_m
            sig += clatter_mix * acc_m

        # ---- spark knock "ping" (end-gas detonation rings the chamber) ----
        if knock_aud > 0.0:
            kexc = knock_exc * 0.6
            lpk = filt[16]; bpk = filt[17]
            hk = kexc - lpk - d_knk * bpk
            bpk += f_knk * hk
            lpk += f_knk * bpk
            filt[16] = lpk; filt[17] = bpk
            sig += knock_aud * bpk

        # ---- muffler chamber HF transmission loss (one-pole LP on exhaust) ----
        if muff_lp > 0.0:
            ml = filt[18] + muff_lp * (sig - filt[18])
            filt[18] = ml
            sig = ml

        # ---- induction (intake-mouth) signal: same monopole radiation ----
        # NB: this path MUST be de-hashed like the exhaust, else the 1D duct's
        # numerical HF hash leaks through as an "electric buzz" that grows with
        # rpm. A 2-pole lowpass (SVF) removes it and softens the intake tone.
        if Ni > 0:
            mdot_in_now = mdot_in_acc * fa_in[Ni] / nsub_in
            p_rin = (mdot_in_now - filt[6]) * (sr * 0.0795775)
            filt[6] = mdot_in_now
            filt[23] += arad_in * (p_rin - filt[23])
            raw_in = filt[23]
            hp_in = raw_in - filt[22] + 0.995 * filt[7]
            filt[22] = raw_in
            filt[7] = hp_in
            xin = hp_in * outgain
            lpi = filt[8]; bpi = filt[9]
            hpi = xin - lpi - d_main * bpi
            bpi += f_in * hpi
            lpi += f_in * bpi
            filt[8] = lpi; filt[9] = bpi
            sig += ingain * lpi

        # ---- static soft-clip (crest control without flat-top OR pumping) ----
        # The blowdown spikes have a high crest factor. A feedback limiter would
        # duck the gain on every pulse and pump (the sound "cutting off" between
        # the firing bangs of a slow diesel); a plain tanh flat-tops the peak into
        # a hard clip. The algebraic sigmoid x/sqrt(1+x^2) is a STATELESS curve:
        # linear at low level (idle detail kept), it bends the loud peaks over
        # smoothly and never flat-tops -- no clip, no pump.
        if sat_on:
            x = 1.1 * sig
            out_audio[s] = x / np.sqrt(1.0 + x * x)
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
    st[S_OMEGA_W] = omega_w
    st[S_VY] = vy
    st[S_YAW] = yaw
    st[S_HEADING] = heading
    P[P_MAP] = map_acc / n_samples    # report manifold pressure for the UI
    P[P_QWALL] = qwall_acc            # heat into the metal this block (thermal model)
    P[P_BOOSTOUT] = boost_out         # report actual boost (Pa, gauge)
    P[P_KNOCK] = knock_acc            # knock intensity this block (ECU reads it)

    return torque_acc / n_samples
