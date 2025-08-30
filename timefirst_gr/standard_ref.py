
import numpy as np
import math

def thomas(a, b, c, d):
    n = len(b)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for i in range(1, n):
        m = ac[i] / bc[i-1]
        bc[i] = bc[i] - m * cc[i-1]
        dc[i] = dc[i] - m * dc[i-1]
    x = np.zeros(n, dtype=float)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n-2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i+1]) / bc[i]
    return x

class StandardEllipticSolver:
    """
    Reference "standard" stepper: at each dt, solve nonlinear BVP G_tt(Φ)=8πGT_tt/c^4
    with Newton + tridiagonal linear solves. This is intentionally heavier per step
    to illustrate the computational benefit of the lapse-first explicit update.
    """
    def __init__(self, r_min=2.2, r_max=80.0, nr=1000, G=1.0, c=1.0):
        self.G, self.c = G, c
        self.r = np.linspace(r_min, r_max, nr)
        self.dr = self.r[1] - self.r[0]
        self.nr = nr
        self.t = 0.0
        self.Phi = np.zeros_like(self.r)
        self.set_static_schwarzschild(M=1.0)
        self.T_tt = lambda t, r: np.zeros_like(r)

    def set_static_schwarzschild(self, M):
        A = 1.0 - 2.0 * self.G * M / (self.c**2 * self.r)
        A = np.clip(A, 1e-12, None)
        self.Phi = 0.5 * np.log(A)

    def set_T_tt(self, T_tt_callable):
        self.T_tt = T_tt_callable

    def A(self, Phi=None):
        if Phi is None: Phi = self.Phi
        return np.clip(np.exp(2.0*Phi), 1e-12, 1e12)

    def _F_and_J(self, Phi, src):
        r = self.r; dr = self.dr; A = self.A(Phi); n = len(Phi)
        F = np.zeros(n, dtype=float)
        a = np.zeros(n); b = np.zeros(n); c = np.zeros(n)

        # Inner BC: Neumann dΦ/dr = 0 -> (Φ1 - Φ0)/dr = 0
        F[0] = (Phi[1] - Phi[0]) / dr
        b[0] = -1.0/dr; c[0] = 1.0/dr

        # Interior rows
        for i in range(1, n-1):
            ri = r[i]; Ai = A[i]
            dPhi_dr_i = (Phi[i+1] - Phi[i-1])/(2.0*dr)
            Gtt_i = (Ai/(ri**2)) * (-2.0*ri*Ai*dPhi_dr_i - Ai + 1.0)
            F[i] = Gtt_i - src[i]

            # Approx Jacobian (diag via finite-diff, neighbors via dΦ/dr chain rule)
            eps = 1e-6
            Ai_eps = math.exp(2.0*(Phi[i]+eps))
            Gtt_eps = (Ai_eps/(ri**2)) * (-2.0*ri*Ai_eps*dPhi_dr_i - Ai_eps + 1.0)
            dG_dPhi_i = (Gtt_eps - Gtt_i)/eps
            dG_d_dPhi = -2.0*Ai*Ai/ri
            a[i] = dG_d_dPhi * (-1.0/(2.0*dr))
            b[i] = dG_dPhi_i
            c[i] = dG_d_dPhi * (+1.0/(2.0*dr))

        # Outer BC: Dirichlet Φ(Rmax)=0
        F[-1] = Phi[-1] - 0.0
        b[-1] = 1.0

        return F, a, b, c

    def solve_step(self, t, T_tt, max_newton=12, tol=1e-6):
        r = self.r
        src = 8.0*np.pi*self.G*T_tt(t, r)/(self.c**4)
        Phi = self.Phi.copy()
        for _ in range(max_newton):
            F, a, b, c = self._F_and_J(Phi, src)
            res = np.linalg.norm(F, ord=np.inf)
            if res < tol: break
            dPhi = thomas(a, b, c, -F)
            # Armijo backtracking
            lam = 1.0
            for _ in range(10):
                trial = Phi + lam*dPhi
                F_trial, *_ = self._F_and_J(trial, src)
                if np.linalg.norm(F_trial, ord=np.inf) < res:
                    Phi = trial; break
                lam *= 0.5
        self.Phi = Phi

    def run(self, t_end, dt, T_tt):
        steps = int(np.ceil((t_end - self.t)/dt))
        for _ in range(steps):
            self.solve_step(self.t, T_tt)
            self.t += dt
    
    def mass_function(self):
        A = self.A()
        return (self.c**2 * self.r / (2.0 * self.G)) * (1.0 - A)
    
    def constraints_residuals(self):
        """Add constraint residuals for compatibility with validation."""
        r = self.r
        A = self.A()
        Phi = self.Phi
        dr = self.dr
        dPhi_dr = np.gradient(Phi, r, edge_order=2)
        G_rr = 2.0 * dPhi_dr / r + 1.0 / (r**2) - 1.0 / (A * r**2)
        G_tt = (A / r**2) * (-2.0 * r * A * dPhi_dr - A + 1.0)
        
        # For standard solver, we don't store matter functions, so return diagnostic values
        # This is mainly used for validation comparisons
        T_tt_current = self.T_tt(self.t, r) if hasattr(self, 'T_tt') else np.zeros_like(r)
        src_rr = np.zeros_like(r)  # Standard solver doesn't use T_rr
        src_tt = 8.0 * np.pi * self.G * T_tt_current / (self.c**4)
        
        return (G_rr - src_rr, G_tt - src_tt)
