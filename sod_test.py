"""Standalone validation of the new MUSCL-Hancock + HLLC quasi-1D solver.
Sod shock tube vs the analytic Riemann solution (constant area). The waves do
not reach the domain ends by t=0.2, so the boundary conditions are irrelevant.
"""
import numpy as np
from engine_sim.core import gas_step_q

G = 1.4


def analytic_sod(x, t):
    """Exact Sod solution at positions x (diaphragm at 0.5), time t. Returns
    (rho, u, p). Standard left (1,0,1) / right (0.125,0,0.1) state."""
    rL, uL, pL = 1.0, 0.0, 1.0
    rR, uR, pR = 0.125, 0.0, 0.1
    aL = np.sqrt(G * pL / rL)
    aR = np.sqrt(G * pR / rR)
    # solve star pressure (Newton on the two-shock/rarefaction functions)
    def f(p, rk, pk, ak):
        if p > pk:  # shock
            A = 2.0 / ((G + 1.0) * rk); B = (G - 1.0) / (G + 1.0) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        else:        # rarefaction
            return 2.0 * ak / (G - 1.0) * ((p / pk) ** ((G - 1.0) / (2 * G)) - 1.0)
    pstar = 0.5 * (pL + pR)
    for _ in range(80):
        fl = f(pstar, rL, pL, aL); fr = f(pstar, rR, pR, aR)
        d = 1e-6
        dfl = (f(pstar + d, rL, pL, aL) - fl) / d
        dfr = (f(pstar + d, rR, pR, aR) - fr) / d
        g_ = fl + fr + (uR - uL)
        pstar = max(1e-8, pstar - g_ / (dfl + dfr))
    ustar = 0.5 * (uL + uR) + 0.5 * (f(pstar, rR, pR, aR) - f(pstar, rL, pL, aL))

    out_r = np.empty_like(x); out_u = np.empty_like(x); out_p = np.empty_like(x)
    for i, xi in enumerate(x):
        s = (xi - 0.5) / t
        # left of contact
        if s < ustar:
            if pstar > pL:  # left shock (not in Sod, but general)
                rstar = rL * ((pstar / pL + (G - 1) / (G + 1)) /
                              ((G - 1) / (G + 1) * pstar / pL + 1))
                ss = uL - aL * np.sqrt((G + 1) / (2 * G) * pstar / pL + (G - 1) / (2 * G))
                r, u, p = (rL, uL, pL) if s < ss else (rstar, ustar, pstar)
            else:           # left rarefaction (Sod case)
                astar = aL * (pstar / pL) ** ((G - 1) / (2 * G))
                sH = uL - aL; sT = ustar - astar
                if s < sH:
                    r, u, p = rL, uL, pL
                elif s > sT:
                    r = rL * (pstar / pL) ** (1 / G); u, p = ustar, pstar
                else:
                    u = 2 / (G + 1) * (aL + (G - 1) / 2 * uL + s)
                    a = 2 / (G + 1) * (aL + (G - 1) / 2 * (uL - s))
                    r = rL * (a / aL) ** (2 / (G - 1)); p = pL * (a / aL) ** (2 * G / (G - 1))
        else:               # right of contact -> right shock (Sod)
            rstar = rR * ((pstar / pR + (G - 1) / (G + 1)) /
                          ((G - 1) / (G + 1) * pstar / pR + 1))
            ss = uR + aR * np.sqrt((G + 1) / (2 * G) * pstar / pR + (G - 1) / (2 * G))
            r, u, p = (rstar, ustar, pstar) if s < ss else (rR, uR, pR)
        out_r[i], out_u[i], out_p[i] = r, u, p
    return out_r, out_u, out_p


def run(N=200, tau=0.2):
    # Euler is scale-invariant: run at engine pressure scale (alpha=1e5 Pa) so the
    # solver's 1.0 Pa vacuum floor is negligible, then compare against the
    # nondimensional analytic solution. Dimensional time t maps to nondim time
    # tau = t*sqrt(alpha); pressures scale by alpha, velocities by sqrt(alpha).
    alpha = 1.0e5
    sa = np.sqrt(alpha)
    t_end = tau / sa
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    rho = np.where(x < 0.5, 1.0, 0.125).astype(float)
    p = np.where(x < 0.5, 1.0, 0.1) * alpha
    R = 287.0                          # physical R so T stays under the 2800 K cap
    T_open = (0.1 * alpha) / (0.125 * R)
    mom = np.zeros(N)
    Ene = p / (G - 1.0) + 0.5 * mom * mom / rho
    area = np.ones(N)
    aface = np.ones(N + 1)
    wk = np.zeros((15, N))
    src_m = np.zeros(N); src_E = np.zeros(N)

    t = 0.0
    while t < t_end:
        u = mom / rho
        pp = (G - 1.0) * (Ene - 0.5 * rho * u * u)
        a = np.sqrt(G * np.maximum(pp, 1e-9) / rho)
        amax = np.max(np.abs(u) + a)
        dt = 0.4 * dx / amax
        if t + dt > t_end:
            dt = t_end - t
        gas_step_q(rho, mom, Ene, N, dx, dt, G, R, 0.1 * alpha, T_open,
                   0.1 * alpha, T_open, 0.0, np.zeros(N), src_m, src_E,
                   area, aface, wk, np.zeros(2))
        t += dt

    u = mom / rho / sa                                # back to nondim velocity
    pp = (G - 1.0) * (Ene - 0.5 * rho * (mom / rho) ** 2) / alpha   # nondim p
    ar, au, ap = analytic_sod(x, tau)
    # exclude the 4 cells nearest each end (BC ghosts) from the error norm
    sl = slice(4, N - 4)
    err_r = np.sqrt(np.mean((rho[sl] - ar[sl]) ** 2))
    err_u = np.sqrt(np.mean((u[sl] - au[sl]) ** 2))
    err_p = np.sqrt(np.mean((pp[sl] - ap[sl]) ** 2))
    print(f"N={N}: L2 err  rho={err_r:.4f}  u={err_u:.4f}  p={err_p:.4f}")
    print(f"  star pressure  sim plateau~{np.median(pp[(x>0.55)&(x<0.65)]):.4f} "
          f"vs analytic {np.median(ap[(x>0.55)&(x<0.65)]):.4f}")
    print(f"  contact rho jump sim {rho[int(0.6*N)]:.4f} / {rho[int(0.7*N)]:.4f}")
    finite = np.all(np.isfinite(rho)) and np.all(np.isfinite(pp))
    print(f"  finite: {finite}")
    return err_r, err_u, err_p


if __name__ == "__main__":
    run(100); print()
    run(200); print()
    run(400)
