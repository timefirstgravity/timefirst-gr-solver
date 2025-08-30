
import numpy as np

class TimeFirstGRSolver:
    """
    1+1D spherical GR solver in the lapse-first gauge:
        ds^2 = -e^{2Φ(t,r)} dt^2 + e^{-2Φ(t,r)} dr^2 + r^2 dΩ^2
    Evolution (mixed component):
        ∂_t Φ = -4π G r T_tr / c^4
    Constraints (diagnostics only):
        G_rr = 2 ∂_r Φ / r + 1/r^2 - 1/(A r^2)
        G_tt = (A/r^2)(-2 r A ∂_r Φ - A + 1)
        with A = e^{2Φ}.
    Matter is provided via callables T_tr(t, r), T_tt(t, r), T_rr(t, r).
    """
    def __init__(self, r_min=2.2, r_max=100.0, nr=2000, G=1.0, c=1.0, 
                 enforce_boundaries=True):
        self.G = G
        self.c = c
        self.r = np.linspace(r_min, r_max, nr)
        self.dr = self.r[1] - self.r[0]
        self.nr = nr
        self.t = 0.0
        self.M0 = 0.0
        self.Phi = np.zeros_like(self.r)  # A = e^{2Φ} = 1 initially
        self.matter = None
        self.enforce_boundaries = enforce_boundaries

    # ---- Matter models --------------------------------------------------
    def set_matter_model(self, matter_obj):
        """
        Set matter using a UnifiedMatter object for consistent physics.
        """
        def T_tr(t, r): return matter_obj.T_tr(t, r, c=self.c)
        def T_tt(t, r): return matter_obj.T_tt(t, r, c=self.c)  
        def T_rr(t, r): return matter_obj.T_rr(t, r, c=self.c)
        self.matter = (T_tr, T_tt, T_rr)
        
    def set_vacuum(self):
        def T_tr(t, r): return np.zeros_like(r)
        def T_tt(t, r): return np.zeros_like(r)
        def T_rr(t, r): return np.zeros_like(r)
        self.matter = (T_tr, T_tt, T_rr)

    def set_null_dust(self, L_of_t, r0=0.0, direction="ingoing"):
        """
        Vaidya-like null dust (legacy method - prefer set_matter_model with VaidyaLikeNull):
          T_tr(t,r) = ± L(t) / (4π r^2 c^2), r >= r0; sign by direction.
        T_tt and T_rr set to zero here (diagnostics will report mismatch).
        """
        sgn = -1.0 if direction == "ingoing" else +1.0
        def T_tr(t, r):
            val = np.zeros_like(r)
            mask = r >= r0
            val[mask] = sgn * L_of_t(t) / (4.0 * np.pi * r[mask]**2 * (self.c**2))
            return val
        def T_tt(t, r): return np.zeros_like(r)
        def T_rr(t, r): return np.zeros_like(r)
        self.matter = (T_tr, T_tt, T_rr)

    # ---- Initialization --------------------------------------------------
    def set_static_schwarzschild(self, M):
        self.M0 = M
        A = 1.0 - 2.0 * self.G * M / (self.c**2 * self.r)
        A = np.clip(A, 1e-12, None)  # avoid horizon blow-ups
        self.Phi = 0.5 * np.log(A)

    # ---- Evolution -------------------------------------------------------
    def step(self, dt):
        if self.matter is None:
            raise RuntimeError("Matter not set. Use set_vacuum() or set_matter_model(...).")
        T_tr, T_tt, T_rr = self.matter
        r = self.r
        dPhi_dt = -4.0 * np.pi * self.G * r * T_tr(self.t, r) / (self.c**4)
        
        # Apply evolution
        self.Phi = self.Phi + dt * dPhi_dt
        
        # Enforce boundary conditions if requested (matching standard solver)
        if self.enforce_boundaries:
            # Inner BC: Neumann dΦ/dr = 0 (regularity at center)
            # Use extrapolation from interior points to maintain smoothness
            self.Phi[0] = self.Phi[1] - (self.Phi[2] - self.Phi[1])
            
            # Outer BC: Asymptotic flatness - more gradual approach
            # Instead of hard Φ=0, use exponential decay to asymptotic value
            r_outer = self.r[-10:]  # Last 10 points
            Phi_outer = self.Phi[-10:]
            # Fit exponential decay: Φ ~ A exp(-r/L) 
            try:
                # Simple linear fit in log space for last few points
                mask = Phi_outer > 1e-8
                if np.sum(mask) >= 3:
                    r_fit = r_outer[mask]
                    log_Phi_fit = np.log(np.abs(Phi_outer[mask]))
                    # Linear regression: log|Φ| = log|A| - r/L
                    coeffs = np.polyfit(r_fit, log_Phi_fit, 1)
                    decay_rate = -coeffs[0]
                    # Extrapolate to boundary
                    self.Phi[-1] = np.exp(coeffs[1] - decay_rate * self.r[-1])
                    if Phi_outer[-1] < 0:
                        self.Phi[-1] = -self.Phi[-1]
            except:
                # Fallback: simple exponential decay
                self.Phi[-1] = self.Phi[-2] * 0.9
            
        self.t += dt
        return dPhi_dt

    def run(self, t_end, dt, progress=False):
        n_steps = int(np.ceil((t_end - self.t) / dt))
        for k in range(n_steps):
            self.step(dt)
        return self.t

    # ---- Diagnostics & Observables --------------------------------------
    def A(self):
        # mildly clip to keep numerics sane in derived quantities
        return np.clip(np.exp(2.0 * self.Phi), 1e-12, 1e12)

    def mass_function(self):
        A = self.A()
        return (self.c**2 * self.r / (2.0 * self.G)) * (1.0 - A)

    def constraints_residuals(self):
        r = self.r
        A = self.A()
        Phi = self.Phi
        dPhi_dr = np.gradient(Phi, r, edge_order=2)
        G_rr = 2.0 * dPhi_dr / r + 1.0 / (r**2) - 1.0 / (A * r**2)
        G_tt = (A / r**2) * (-2.0 * r * A * dPhi_dr - A + 1.0)
        if self.matter is None:
            T_tr, T_tt, T_rr = (lambda *_:0.0,)*3
        else:
            T_tr, T_tt, T_rr = self.matter
        src_rr = 8.0 * np.pi * self.G * T_rr(self.t, r) / (self.c**4)
        src_tt = 8.0 * np.pi * self.G * T_tt(self.t, r) / (self.c**4)
        return (G_rr - src_rr, G_tt - src_tt)

    def proper_time_increment(self, dt_coord, r_idx):
        return float(np.exp(self.Phi[r_idx]) * dt_coord)

    def redshift_factor(self, i_emit, i_obs):
        A = self.A()
        return float(np.sqrt(A[i_obs] / A[i_emit]))

    def light_travel_time(self, i_start, i_end):
        A = self.A()
        if i_end >= i_start:
            r_path = self.r[i_start:i_end+1]
            A_path = A[i_start:i_end+1]
        else:
            r_path = self.r[i_end:i_start+1][::-1]
            A_path = A[i_end:i_start+1][::-1]
        return float(np.trapz(1.0 / A_path, r_path))

    def snapshot(self):
        return dict(t=float(self.t), r=self.r.copy(), Phi=self.Phi.copy(),
                    A=self.A(), M=self.mass_function())
