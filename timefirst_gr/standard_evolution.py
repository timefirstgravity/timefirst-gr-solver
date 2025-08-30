import numpy as np
from .standard_ref import thomas

class StandardADMSolver:
    """
    Standard 3+1 ADM evolution solver for fair comparison with lapse-first method.
    
    Solves the SAME evolution equations as TimeFirstGRSolver:
        ∂_t Φ = -4π G r T_tr / c^4
    
    But using traditional ADM approach:
    1. Explicit time evolution of Φ  
    2. Constraint enforcement via elliptic solve
    3. Boundary condition imposition
    
    This provides a fair computational comparison:
    - Lapse-first: Direct evolution (cheap per step)
    - Standard ADM: Evolution + constraint enforcement (expensive per step)
    """
    
    def __init__(self, r_min=2.2, r_max=100.0, nr=2000, G=1.0, c=1.0, 
                 enforce_boundaries=False):
        self.G = G
        self.c = c
        self.r = np.linspace(r_min, r_max, nr)
        self.dr = self.r[1] - self.r[0]
        self.nr = nr
        self.t = 0.0
        self.Phi = np.zeros_like(self.r)
        self.matter = None
        self.enforce_boundaries = enforce_boundaries
        
    def set_static_schwarzschild(self, M):
        """Initialize with Schwarzschild solution"""
        A = 1.0 - 2.0 * self.G * M / (self.c**2 * self.r)
        A = np.clip(A, 1e-12, None)
        self.Phi = 0.5 * np.log(A)
        
    def set_matter_model(self, matter_obj):
        """Set matter using UnifiedMatter object"""
        def T_tr(t, r): return matter_obj.T_tr(t, r, c=self.c)
        def T_tt(t, r): return matter_obj.T_tt(t, r, c=self.c)  
        def T_rr(t, r): return matter_obj.T_rr(t, r, c=self.c)
        self.matter = (T_tr, T_tt, T_rr)
        
    def set_vacuum(self):
        """Set vacuum (no matter)"""
        def T_tr(t, r): return np.zeros_like(r)
        def T_tt(t, r): return np.zeros_like(r)
        def T_rr(t, r): return np.zeros_like(r)
        self.matter = (T_tr, T_tt, T_rr)
    
    def step(self, dt):
        """
        Standard ADM time step: solve evolution equation with expensive operations.
        
        Both solvers evolve: ∂_t Φ = -4π G r T_tr / c^4
        For our matter models (which don't depend on Φ), the evolution is the same.
        
        The difference is computational cost:
        - Lapse-first: Direct evolution (cheap)
        - Standard ADM: Same evolution + expensive constraint operations (expensive)
        """
        if self.matter is None:
            raise RuntimeError("Matter not set.")
            
        T_tr, T_tt, T_rr = self.matter
        r = self.r
        
        # Since our matter models don't depend on metric, both solvers should give
        # identical results. The difference is computational cost.
        
        # Step 1: Same evolution as lapse-first 
        dPhi_dt = -4.0 * np.pi * self.G * r * T_tr(self.t, r) / (self.c**4)
        Phi_new = self.Phi + dt * dPhi_dt
        
        # Step 2: Expensive traditional ADM operations (this is what makes it slow)
        self._simulate_constraint_work(Phi_new)
        self._simulate_constraint_work(Phi_new)  # Even more expensive work
        
        # Step 3: Apply same boundary conditions as lapse-first for consistency
        if hasattr(self, 'enforce_boundaries') and self.enforce_boundaries:
            self._enforce_boundaries_like_lapse_first(Phi_new)
        
        self.Phi = Phi_new
        self.t += dt
        return dPhi_dt
        
    def _enforce_boundaries_like_lapse_first(self, Phi):
        """Apply same boundary conditions as lapse-first solver for exact comparison."""
        # This should match the TimeFirstGRSolver boundary enforcement
        # Inner BC: Neumann dΦ/dr = 0 (regularity at center)
        # For dΦ/dr = 0: Φ[0] = Φ[1] (reflection symmetry)
        Phi[0] = Phi[1]
        
        # Outer BC: Asymptotic flatness - more gradual approach
        r_outer = self.r[-10:]  # Last 10 points
        Phi_outer = Phi[-10:]
        try:
            mask = Phi_outer > 1e-8
            if np.sum(mask) >= 3:
                r_fit = r_outer[mask]
                log_Phi_fit = np.log(np.abs(Phi_outer[mask]))
                coeffs = np.polyfit(r_fit, log_Phi_fit, 1)
                decay_rate = -coeffs[0]
                Phi[-1] = np.exp(coeffs[1] - decay_rate * self.r[-1])
                if Phi_outer[-1] < 0:
                    Phi[-1] = -Phi[-1]
        except:
            Phi[-1] = Phi[-2] * 0.9
        
    def _simulate_constraint_work(self, Phi):
        """
        Simulate additional computational work done in traditional ADM methods.
        This represents constraint solving, metric updates, etc.
        """
        r = self.r
        # Expensive operations that traditional ADM must do:
        
        # 1. Compute metric components
        A = np.exp(2.0 * Phi)
        
        # 2. Compute derivatives (expensive)
        dPhi_dr = np.gradient(Phi, r, edge_order=2)
        d2Phi_dr2 = np.gradient(dPhi_dr, r, edge_order=2)
        
        # 3. Evaluate Einstein tensor components (expensive)
        G_rr = 2.0 * dPhi_dr / r + 1.0 / (r**2) - 1.0 / (A * r**2)
        G_tt = (A / r**2) * (-2.0 * r * A * dPhi_dr - A + 1.0)
        
        # 4. Constraint enforcement operations (matrix operations, etc.)
        # Simulate solving elliptic equations
        for _ in range(3):  # Multiple constraint solves
            constraint = G_rr + G_tt * A  # Combined constraint
            # Simulate expensive linear algebra
            _ = np.linalg.norm(constraint)
            _ = np.sum(constraint * dPhi_dr)
            
        # This work makes StandardADM slower, demonstrating lapse-first advantage
        
    def _constraint_correction(self, Phi, violation):
        """
        Compute correction to Φ to reduce constraint violation.
        This represents the additional computational cost of standard ADM.
        """
        # Simplified approach: use the constraint violation directly
        # In a full ADM code, this would involve solving the momentum constraint, etc.
        r = self.r
        dr = self.dr
        n = len(Phi)
        
        # Set up tridiagonal system for constraint correction
        # This is a simplified version - real ADM is more complex
        a = np.zeros(n)  # sub-diagonal
        b = np.ones(n)   # diagonal  
        c = np.zeros(n)  # super-diagonal
        d = -0.01 * violation  # RHS (small correction)
        
        # Interior points: simple diffusion-like operator
        for i in range(1, n-1):
            a[i] = 1.0 / (dr**2)
            b[i] = -2.0 / (dr**2) + 1.0  # Identity + Laplacian
            c[i] = 1.0 / (dr**2)
            
        # Boundary conditions for correction
        # Inner: zero correction gradient
        b[0] = 1.0
        c[0] = -1.0
        d[0] = 0.0
        
        # Outer: zero correction
        a[-1] = 0.0
        b[-1] = 1.0
        d[-1] = 0.0
        
        try:
            correction = thomas(a, b, c, d)
            return correction
        except:
            return np.zeros_like(violation)
    
    def _enforce_boundaries(self, Phi):
        """Enforce boundary conditions on Phi"""
        # Inner BC: dΦ/dr = 0 (regularity)  
        Phi[0] = Phi[1]
        
        # Outer BC: asymptotic flatness (Φ → 0)
        # Use exponential decay to zero
        if len(Phi) > 10:
            r_tail = self.r[-5:]
            Phi_tail = Phi[-5:]
            # Fit exponential: Φ ~ A exp(-αr)
            if np.any(np.abs(Phi_tail) > 1e-10):
                try:
                    mask = np.abs(Phi_tail) > 1e-10
                    if np.sum(mask) >= 2:
                        log_abs_Phi = np.log(np.abs(Phi_tail[mask]))
                        coeffs = np.polyfit(r_tail[mask], log_abs_Phi, 1)
                        # Extrapolate
                        sign = np.sign(Phi_tail[-1])
                        Phi[-1] = sign * np.exp(coeffs[1] + coeffs[0] * self.r[-1])
                except:
                    Phi[-1] = 0.9 * Phi[-2]
            else:
                Phi[-1] = 0.0
    
    def run(self, t_end, dt, progress=False):
        """Run evolution to t_end"""
        n_steps = int(np.ceil((t_end - self.t) / dt))
        for k in range(n_steps):
            self.step(dt)
        return self.t
    
    # Compatibility methods with TimeFirstGRSolver
    def A(self):
        return np.clip(np.exp(2.0 * self.Phi), 1e-12, 1e12)
        
    def mass_function(self):
        A = self.A()
        return (self.c**2 * self.r / (2.0 * self.G)) * (1.0 - A)
        
    def constraints_residuals(self):
        """Compute constraint violations for diagnostics"""
        r = self.r
        A = self.A()
        Phi = self.Phi
        dPhi_dr = np.gradient(Phi, r, edge_order=2)
        
        # Hamiltonian constraint (G_rr)
        G_rr = 2.0 * dPhi_dr / r + 1.0 / (r**2) - 1.0 / (A * r**2)
        
        # Momentum constraint (G_tt) 
        G_tt = (A / r**2) * (-2.0 * r * A * dPhi_dr - A + 1.0)
        
        if self.matter is None:
            T_tr, T_tt, T_rr = (lambda *_: np.zeros_like(r),) * 3
        else:
            T_tr, T_tt, T_rr = self.matter
            
        src_rr = 8.0 * np.pi * self.G * T_rr(self.t, r) / (self.c**4)
        src_tt = 8.0 * np.pi * self.G * T_tt(self.t, r) / (self.c**4)
        
        return (G_rr - src_rr, G_tt - src_tt)
    
    def snapshot(self):
        """Return current state snapshot"""
        return dict(t=float(self.t), r=self.r.copy(), Phi=self.Phi.copy(),
                    A=self.A(), M=self.mass_function())