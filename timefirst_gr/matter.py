import numpy as np

class UnifiedMatter:
    """
    Unified matter model that provides consistent stress-energy components
    for both lapse-first and standard GR solvers.
    
    The key insight is that both approaches should solve the same physics,
    just using different mathematical formulations:
    - Lapse-first: explicit evolution via ∂_t Φ = -4π G r T_tr / c^4  
    - Standard: implicit constraint G_tt = 8π G T_tt / c^4
    
    For consistency, we need T_tr and T_tt to represent the same physical
    energy-momentum distribution.
    """
    
    def __init__(self, energy_density_func, radial_velocity_func=None):
        """
        Initialize with energy density and optional radial velocity.
        
        Parameters:
        -----------
        energy_density_func : callable(t, r) -> array
            Energy density ρ(t,r) in the matter rest frame
        radial_velocity_func : callable(t, r) -> array, optional
            Radial velocity v_r(t,r). If None, assumes matter at rest.
        """
        self.rho = energy_density_func
        self.v_r = radial_velocity_func if radial_velocity_func else lambda t, r: np.zeros_like(r)
    
    def T_tt(self, t, r, c=1.0):
        """
        T_tt component: energy density in coordinate time.
        For perfect fluid (dust): T_tt = ρ γ² c²
        """
        rho = self.rho(t, r)
        v_r = self.v_r(t, r)
        gamma_sq = 1.0 / (1.0 - (v_r/c)**2)
        return rho * gamma_sq * c**2
    
    def T_rr(self, t, r, c=1.0):
        """
        T_rr component: radial pressure/stress.
        """
        rho = self.rho(t, r)
        v_r = self.v_r(t, r)
        gamma_sq = 1.0 / (1.0 - (v_r/c)**2)
        return rho * gamma_sq * v_r**2
    
    def T_tr(self, t, r, c=1.0):
        """
        T_tr component: energy flux in radial direction.
        For perfect fluid (dust): T_tr = ρ γ² c v_r
        """
        rho = self.rho(t, r)
        v_r = self.v_r(t, r)
        gamma_sq = 1.0 / (1.0 - (v_r/c)**2)
        return rho * gamma_sq * c * v_r

class GaussianPulse(UnifiedMatter):
    """
    Gaussian energy pulse - useful for testing and benchmarking.
    """
    
    def __init__(self, amplitude=1.0, r_center=10.0, r_width=2.0, 
                 t_center=0.5, t_width=0.2, velocity=0.0):
        """
        Parameters:
        -----------
        amplitude : float
            Peak energy density
        r_center, r_width : float
            Radial Gaussian center and width
        t_center, t_width : float  
            Temporal Gaussian center and width
        velocity : float
            Radial velocity (positive = outward)
        """
        self.amplitude = amplitude
        self.r_center = r_center
        self.r_width = r_width
        self.t_center = t_center
        self.t_width = t_width
        self.velocity = velocity
        
        def rho_func(t, r):
            spatial = np.exp(-0.5 * ((r - self.r_center) / self.r_width)**2)
            temporal = np.exp(-0.5 * ((t - self.t_center) / self.t_width)**2)
            return self.amplitude * spatial * temporal / (np.sqrt(2*np.pi) * self.t_width)
        
        def v_r_func(t, r):
            return self.velocity * np.ones_like(r)
            
        super().__init__(rho_func, v_r_func)

class VaidyaLikeNull(UnifiedMatter):
    """
    Null dust similar to Vaidya solution - pure radial energy flux.
    This provides a bridge between the old null_dust in solver.py
    and the new unified approach.
    """
    
    def __init__(self, luminosity_func, r_min=0.0, direction="ingoing"):
        """
        Parameters:
        -----------
        luminosity_func : callable(t) -> float
            L(t) luminosity function
        r_min : float
            Minimum radius where flux is non-zero
        direction : str
            "ingoing" or "outgoing"
        """
        self.L_func = luminosity_func
        self.r_min = r_min
        self.sign = +1.0 if direction == "ingoing" else -1.0
        
        # For null dust: T_tt = T_rr = 0, only T_tr ≠ 0
        def rho_func(t, r):
            return np.zeros_like(r)
            
        def v_r_func(t, r):
            # Null motion: |v_r| → c, but we handle this in T_tr directly
            return np.zeros_like(r)
        
        super().__init__(rho_func, v_r_func)
    
    def T_tt(self, t, r, c=1.0):
        """Null dust has T_tt = 0"""
        return np.zeros_like(r)
    
    def T_rr(self, t, r, c=1.0):
        """Null dust has T_rr = 0"""
        return np.zeros_like(r)
        
    def T_tr(self, t, r, c=1.0):
        """
        Pure flux: T_tr = ± L(t)/(4π r² c²)
        
        For ingoing EF null dust: T_vv > 0, and T_tr = T_vv/A > 0 in diagonal gauge.
        For outgoing: T_vv < 0, T_tr < 0.
        """
        val = np.zeros_like(r)
        mask = r >= self.r_min
        val[mask] = self.sign * self.L_func(t) / (4.0 * np.pi * r[mask]**2 * c**2)
        return val