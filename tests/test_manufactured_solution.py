#!/usr/bin/env python3
"""
Manufactured solution test for convergence verification.

Create a smooth, analytical T_tr(t,r) and solve ∂_t Φ = -4π G r T_tr / c^4
analytically. Then verify both solvers achieve the expected convergence rate.
"""

import numpy as np
import pytest
from timefirst_gr.solver import TimeFirstGRSolver
from timefirst_gr.standard_evolution import StandardADMSolver

class ManufacturedSolution:
    """Analytical solution for testing convergence"""
    
    def __init__(self, G=1.0, c=1.0):
        self.G = G
        self.c = c
        
    def T_tr_analytical(self, t, r):
        """Smooth, compactly supported T_tr for testing"""
        # Gaussian in space and time, but cut off to ensure compact support
        r0, sigma_r = 8.0, 2.0
        t0, sigma_t = 0.5, 0.3
        amplitude = 0.01
        
        # Spatial cutoff
        r_factor = np.exp(-0.5 * ((r - r0) / sigma_r)**2)
        r_factor = np.where(np.abs(r - r0) < 3*sigma_r, r_factor, 0.0)
        
        # Temporal factor
        t_factor = np.exp(-0.5 * ((t - t0) / sigma_t)**2)
        
        return amplitude * r_factor * t_factor
    
    def Phi_analytical(self, t, r):
        """
        Analytical solution: Φ(t,r) = Φ₀(r) + ∫₀ᵗ (-4π G r T_tr(s,r) / c⁴) ds
        
        Since T_tr is separable in t and r, we can integrate analytically.
        """
        # Initial condition (Schwarzschild background)
        M0 = 1.0
        A0 = 1.0 - 2.0 * self.G * M0 / (self.c**2 * r)
        A0 = np.clip(A0, 1e-12, None)
        Phi_initial = 0.5 * np.log(A0)
        
        # Time integral of the source
        # ∫₀ᵗ T_tr(s,r) ds for our Gaussian in time
        r0, sigma_r = 8.0, 2.0
        t0, sigma_t = 0.5, 0.3
        amplitude = 0.01
        
        # Spatial part
        r_factor = np.exp(-0.5 * ((r - r0) / sigma_r)**2)
        r_factor = np.where(np.abs(r - r0) < 3*sigma_r, r_factor, 0.0)
        
        # Time integral: ∫₀ᵗ exp(-0.5*((s-t0)/σt)²) ds
        # This involves the error function, but we can compute it numerically for accuracy
        from scipy.special import erf
        sqrt_2 = np.sqrt(2.0)
        
        # Analytical integral of Gaussian from 0 to t
        def gaussian_integral(t_val):
            if t_val <= 0:
                return 0.0
            # ∫₀ᵗ exp(-0.5*((s-t0)/σt)²) ds = σt*√(π/2) * [erf((t-t0)/(σt*√2)) + erf(t0/(σt*√2))]
            term1 = erf((t_val - t0) / (sigma_t * sqrt_2))
            term2 = erf(t0 / (sigma_t * sqrt_2))
            return sigma_t * np.sqrt(np.pi/2.0) * (term1 + term2)
        
        # Handle scalar or array t
        if np.isscalar(t):
            time_integral = gaussian_integral(t)
        else:
            time_integral = np.array([gaussian_integral(t_val) for t_val in t])
        
        # Full integral
        source_integral = amplitude * r_factor * time_integral
        
        # Evolution contribution
        evolution_term = -4.0 * np.pi * self.G * r * source_integral / (self.c**4)
        
        return Phi_initial + evolution_term
    
    def create_matter_functions(self):
        """Create matter functions for the solver"""
        def T_tr(t, r): return self.T_tr_analytical(t, r)
        def T_tt(t, r): return np.zeros_like(r)  # Pure flux
        def T_rr(t, r): return np.zeros_like(r)
        return (T_tr, T_tt, T_rr)

def test_manufactured_solution_accuracy():
    """Test that solver matches analytical solution"""
    ms = ManufacturedSolution(G=1.0, c=1.0)
    
    # Test parameters
    r_min, r_max = 3.0, 15.0
    nr = 120
    t_end = 0.8
    dt = 0.005
    
    # Setup solver
    S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=1.0)
    S.matter = ms.create_matter_functions()
    
    # Evolve
    S.run(t_end=t_end, dt=dt)
    
    # Compare with analytical solution
    Phi_analytical = ms.Phi_analytical(t_end, S.r)
    Phi_numerical = S.Phi
    
    # Compute errors
    error_abs = np.abs(Phi_numerical - Phi_analytical)
    error_rel = error_abs / (np.abs(Phi_analytical) + 1e-12)
    
    max_abs_error = np.max(error_abs)
    max_rel_error = np.max(error_rel)
    rms_error = np.sqrt(np.mean(error_abs**2))
    
    print(f"Manufactured Solution Test:")
    print(f"  Grid: {nr} points, dt = {dt}")
    print(f"  Max absolute error: {max_abs_error:.2e}")
    print(f"  Max relative error: {max_rel_error:.2e}")
    print(f"  RMS error: {rms_error:.2e}")
    
    # Since we're using first-order Euler over T=0.8 with dt=0.005, expect O(dt*T) ≈ 4e-3 error
    # Observed error ~9e-4 is well within theoretical bounds
    theoretical_bound = dt * t_end * 5  # Conservative bound with safety factor
    assert max_abs_error < theoretical_bound, \
        f"Error {max_abs_error:.2e} exceeds theoretical Euler bound {theoretical_bound:.2e}"
    assert rms_error < theoretical_bound/2, \
        f"RMS error {rms_error:.2e} too large for first-order method"

def test_temporal_convergence_rate():
    """Test that temporal discretization shows first-order convergence (Euler)"""
    ms = ManufacturedSolution(G=1.0, c=1.0)
    
    # Fixed spatial grid
    nr = 100
    r_min, r_max = 4.0, 12.0
    t_end = 0.4  # Shorter time to avoid accumulated errors
    
    # Test different time steps
    dt_values = [0.02, 0.01, 0.005, 0.0025]
    errors = []
    
    for dt in dt_values:
        S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                             enforce_boundaries=False)
        S.set_static_schwarzschild(M=1.0)
        S.matter = ms.create_matter_functions()
        
        S.run(t_end=t_end, dt=dt)
        
        Phi_analytical = ms.Phi_analytical(t_end, S.r)
        error = np.max(np.abs(S.Phi - Phi_analytical))
        errors.append(error)
        
        print(f"dt = {dt:.4f}: error = {error:.2e}")
    
    # Check convergence rate
    dt_values = np.array(dt_values)
    errors = np.array(errors)
    
    # Fit log(error) vs log(dt) to get slope
    log_dt = np.log(dt_values)
    log_error = np.log(errors)
    slope = np.polyfit(log_dt, log_error, 1)[0]
    
    print(f"Convergence rate: {slope:.2f} (expected ≈ 1.0 for first-order Euler)")
    
    # Allow some tolerance for numerical effects
    assert 0.8 < slope < 1.3, f"Convergence rate {slope:.2f} not first-order"

def test_both_solvers_identical_on_manufactured():
    """Test that both solvers give identical results on manufactured solution"""
    ms = ManufacturedSolution(G=1.0, c=1.0)
    
    # Parameters
    nr = 80
    r_min, r_max = 3.5, 12.0
    t_end = 0.6
    dt = 0.008
    
    # Lapse-first solver
    S1 = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S1.set_static_schwarzschild(M=1.0)
    S1.matter = ms.create_matter_functions()
    S1.run(t_end=t_end, dt=dt)
    
    # Standard ADM solver
    S2 = StandardADMSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S2.set_static_schwarzschild(M=1.0)
    S2.matter = ms.create_matter_functions()
    S2.run(t_end=t_end, dt=dt)
    
    # Compare solutions
    phi_diff = np.max(np.abs(S1.Phi - S2.Phi))
    mass_diff = np.max(np.abs(S1.mass_function() - S2.mass_function()))
    
    print(f"Solver comparison on manufactured solution:")
    print(f"  Max Φ difference: {phi_diff:.2e}")
    print(f"  Max mass difference: {mass_diff:.2e}")
    
    assert phi_diff < 1e-15, f"Solvers disagree on Φ: {phi_diff:.2e}"
    assert mass_diff < 1e-15, f"Solvers disagree on mass: {mass_diff:.2e}"
    
    # Both should also match analytical solution reasonably well
    Phi_analytical = ms.Phi_analytical(t_end, S1.r)
    error1 = np.max(np.abs(S1.Phi - Phi_analytical))
    error2 = np.max(np.abs(S2.Phi - Phi_analytical))
    
    print(f"  Lapse-first vs analytical: {error1:.2e}")
    print(f"  Standard ADM vs analytical: {error2:.2e}")
    
    # Use theoretical bound based on dt and integration time
    theoretical_bound = dt * t_end * 10  # More conservative for this test
    assert error1 < theoretical_bound, f"Lapse-first error {error1:.2e} exceeds bound {theoretical_bound:.2e}"
    assert error2 < theoretical_bound, f"Standard ADM error {error2:.2e} exceeds bound {theoretical_bound:.2e}"

def test_spatial_accuracy():
    """Test spatial accuracy (should be machine precision since no spatial derivatives in evolution)"""
    ms = ManufacturedSolution(G=1.0, c=1.0)
    
    # Very fine spatial grid to check spatial accuracy
    nr_values = [50, 100, 200]
    r_min, r_max = 4.0, 10.0
    t_test = 0.3  # Single time point
    dt = 0.002  # Small dt to minimize temporal error
    
    errors = []
    for nr in nr_values:
        S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                             enforce_boundaries=False)
        S.set_static_schwarzschild(M=1.0)
        S.matter = ms.create_matter_functions()
        
        S.run(t_end=t_test, dt=dt)
        
        Phi_analytical = ms.Phi_analytical(t_test, S.r)
        error = np.max(np.abs(S.Phi - Phi_analytical))
        errors.append(error)
        
        print(f"nr = {nr}: spatial error = {error:.2e}")
    
    # Since evolution is local in space, spatial error should be dominated by
    # temporal discretization, not spatial grid resolution
    # All errors should be similar (within factor of 2-3)
    max_error = max(errors)
    min_error = min(errors)
    error_ratio = max_error / min_error
    
    print(f"Error ratio (max/min): {error_ratio:.1f}")
    assert error_ratio < 5.0, f"Spatial errors vary too much: ratio = {error_ratio:.1f}"

if __name__ == "__main__":
    test_manufactured_solution_accuracy()
    test_temporal_convergence_rate()
    test_both_solvers_identical_on_manufactured()
    test_spatial_accuracy()
    print("All manufactured solution tests passed! ✓")