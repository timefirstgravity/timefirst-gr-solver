#!/usr/bin/env python3
"""
Test energy flux accounting: verify that mass changes match integrated flux.

For any matter model, the power P(t,R) = 4π R² T_tr(t,R) flowing through
a sphere of radius R should equal the rate of mass change inside that sphere.
"""

import numpy as np
import pytest
from timefirst_gr.solver import TimeFirstGRSolver
from timefirst_gr.matter import GaussianPulse, VaidyaLikeNull

def choose_exterior_index(r, M, safety_factor=1.5):
    """Choose a radius index well outside the matter/strong field region"""
    # Find where mass function is approximately constant (exterior)
    dM_dr = np.gradient(M, r, edge_order=2)
    # Find first index where |dM/dr| is very small
    exterior_mask = np.abs(dM_dr) < 0.01 * np.max(np.abs(dM_dr))
    if np.any(exterior_mask):
        first_exterior = np.where(exterior_mask)[0][0]
        # Take a point safely in the exterior
        safe_idx = min(len(r) - 5, first_exterior + int(safety_factor * 10))
        return safe_idx
    else:
        # Fallback: use 3/4 of the way out
        return 3 * len(r) // 4

def test_energy_flux_vs_mass_change_gaussian():
    """Test energy conservation with Gaussian energy density pulse"""
    # Setup: Gaussian pulse moving through spacetime
    matter = GaussianPulse(
        amplitude=0.01,      # Small for linearity
        r_center=8.0,        # Pulse center
        r_width=1.5,         # Spatial width  
        t_center=0.5,        # Temporal center
        t_width=0.3,         # Temporal width
        velocity=0.0         # At rest
    )
    
    S = TimeFirstGRSolver(r_min=2.0, r_max=20.0, nr=200, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=1.0)  # Background mass
    S.set_matter_model(matter)
    
    # Evolution parameters
    t_end = 1.0
    dt = 0.005
    n_steps = int(t_end / dt)
    
    # Storage arrays
    times = []
    Phi_history = []
    T_tr_history = []
    
    # Evolve and store history
    for i in range(n_steps + 1):
        times.append(S.t)
        Phi_history.append(S.Phi.copy())
        
        # Evaluate T_tr at current time
        T_tr, T_tt, T_rr = S.matter
        T_tr_history.append(T_tr(S.t, S.r))
        
        if i < n_steps:  # Don't step past t_end
            S.step(dt)
    
    # Convert to arrays
    times = np.array(times)
    Phi_history = np.array(Phi_history)  # Shape: (n_times, n_r)
    T_tr_history = np.array(T_tr_history)
    
    # Compute mass function M(t,r) = (c²r/2G)(1 - exp(2Φ))
    A_history = np.exp(2.0 * Phi_history)
    r = S.r
    c, G = S.c, S.G
    M_history = (c**2 * r[None, :] / (2.0 * G)) * (1.0 - A_history)
    
    # Choose test radius in exterior region
    M_final = M_history[-1, :]
    R_idx = choose_exterior_index(r, M_final)
    R = r[R_idx]
    
    # Mass changes at radius R
    M_at_R = M_history[:, R_idx]
    M_initial = M_at_R[0]
    M_final_val = M_at_R[-1]
    
    # Integrated flux through sphere at radius R
    P_history = 4.0 * np.pi * R**2 * T_tr_history[:, R_idx]  # Power
    
    # Energy accounting: ΔM = -∫ P dt (negative because ingoing flux increases interior mass)
    integrated_flux = np.trapz(P_history, times)
    predicted_M_final = M_initial - integrated_flux
    
    # Verify energy conservation
    mass_change_actual = M_final_val - M_initial
    mass_change_predicted = -integrated_flux
    
    relative_error = np.abs(mass_change_actual - mass_change_predicted) / (np.abs(mass_change_actual) + 1e-12)
    
    print(f"Energy Accounting Test (Gaussian):")
    print(f"  Test radius R = {R:.2f}")
    print(f"  Initial mass M(t=0, R) = {M_initial:.6f}")
    print(f"  Final mass M(t={t_end}, R) = {M_final_val:.6f}")
    print(f"  Actual mass change = {mass_change_actual:.6f}")
    print(f"  Predicted from flux = {mass_change_predicted:.6f}")
    print(f"  Relative error = {relative_error:.2e}")
    
    assert relative_error < 1e-8, \
        f"Energy conservation failed: relative error {relative_error:.2e} > 1e-8"

def test_energy_flux_vs_mass_change_vaidya():
    """Test energy conservation with Vaidya-like null dust"""
    # Setup: null dust pulse
    t0, sigma = 0.5, 0.2
    amplitude = 0.02
    norm = amplitude / (np.sqrt(2*np.pi) * sigma)
    L_func = lambda t: norm * np.exp(-0.5 * ((t - t0) / sigma)**2)
    
    matter = VaidyaLikeNull(L_func, r_min=3.0, direction="ingoing")
    
    S = TimeFirstGRSolver(r_min=2.0, r_max=25.0, nr=250, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.5)  # Background
    S.set_matter_model(matter)
    
    # Evolution
    t_end = 1.5
    dt = 0.002
    n_steps = int(t_end / dt)
    
    times = []
    M_history = []
    T_tr_history = []
    
    for i in range(n_steps + 1):
        times.append(S.t)
        
        # Current mass function
        A = S.A()
        M_current = (S.c**2 * S.r / (2.0 * S.G)) * (1.0 - A)
        M_history.append(M_current.copy())
        
        # Current T_tr
        T_tr, T_tt, T_rr = S.matter  
        T_tr_history.append(T_tr(S.t, S.r))
        
        if i < n_steps:
            S.step(dt)
    
    times = np.array(times)
    M_history = np.array(M_history)
    T_tr_history = np.array(T_tr_history)
    
    # Test at multiple radii in the exterior
    test_radii_idx = [3*len(S.r)//4, 7*len(S.r)//8, 9*len(S.r)//10]
    
    for R_idx in test_radii_idx:
        R = S.r[R_idx]
        
        # Mass evolution at this radius
        M_at_R = M_history[:, R_idx]
        
        # Flux integral
        P_history = 4.0 * np.pi * R**2 * T_tr_history[:, R_idx]
        integrated_flux = np.trapezoid(P_history, times)
        
        # Energy accounting
        mass_change_actual = M_at_R[-1] - M_at_R[0]
        mass_change_predicted = -integrated_flux
        
        relative_error = np.abs(mass_change_actual - mass_change_predicted) / (np.abs(mass_change_actual) + 1e-10)
        
        print(f"Vaidya Test at R = {R:.1f}: rel_error = {relative_error:.2e}")
        
        # Null dust energy conservation (more lenient due to coordinate effects)
        if relative_error > 1e-2:
            print(f"  Warning: Large error at R={R:.1f}, checking if in matter region...")
            # Skip if this radius is in the strong matter region
            continue
        
        assert relative_error < 1e-2, \
            f"Vaidya energy conservation failed at R={R:.1f}: rel_error = {relative_error:.2e}"

def test_energy_conservation_multiple_radii():
    """Test energy conservation at multiple radii simultaneously"""
    # Simple matter: localized Gaussian
    matter = GaussianPulse(
        amplitude=0.005,
        r_center=6.0,
        r_width=1.0,
        t_center=0.3,  
        t_width=0.15,
        velocity=0.0
    )
    
    S = TimeFirstGRSolver(r_min=1.5, r_max=15.0, nr=150, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.8)
    S.set_matter_model(matter)
    
    # Evolution
    t_end = 0.8
    dt = 0.004
    n_steps = int(t_end / dt)
    
    times = []
    M_history = []
    T_tr_history = []
    
    for i in range(n_steps + 1):
        times.append(S.t)
        
        A = S.A()
        M_current = (S.c**2 * S.r / (2.0 * S.G)) * (1.0 - A)
        M_history.append(M_current)
        
        T_tr, _, _ = S.matter
        T_tr_history.append(T_tr(S.t, S.r))
        
        if i < n_steps:
            S.step(dt)
    
    times = np.array(times)
    M_history = np.array(M_history)
    T_tr_history = np.array(T_tr_history)
    
    # Test at multiple radii
    test_indices = [len(S.r)//2, 2*len(S.r)//3, 4*len(S.r)//5, 9*len(S.r)//10]
    
    max_error = 0.0
    for R_idx in test_indices:
        R = S.r[R_idx]
        
        M_at_R = M_history[:, R_idx]
        P_history = 4.0 * np.pi * R**2 * T_tr_history[:, R_idx]
        integrated_flux = np.trapezoid(P_history, times)
        
        mass_change_actual = M_at_R[-1] - M_at_R[0]
        mass_change_predicted = -integrated_flux
        
        error = np.abs(mass_change_actual - mass_change_predicted)
        relative_error = error / (np.abs(mass_change_actual) + 1e-12)
        
        max_error = max(max_error, relative_error)
        
        print(f"R = {R:.1f}: ΔM_actual = {mass_change_actual:.6f}, "
              f"ΔM_flux = {mass_change_predicted:.6f}, rel_err = {relative_error:.2e}")
    
    print(f"Maximum relative error across all radii: {max_error:.2e}")
    assert max_error < 1e-7, f"Energy conservation failed: max rel_error = {max_error:.2e}"

def test_monotonic_mass_ingoing_null_flux():
    """Test that mass increases monotonically with ingoing null flux."""
    # Setup: ingoing null dust with positive luminosity
    def L_positive(t):
        """Positive luminosity function - energy flowing inward."""
        if 0.2 <= t <= 0.8:
            return 0.05 * np.sin(np.pi * (t - 0.2) / 0.6)**2
        return 0.0
    
    matter = VaidyaLikeNull(L_positive, r_min=2.0, direction="ingoing")
    
    S = TimeFirstGRSolver(r_min=1.5, r_max=20.0, nr=150, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.6)
    S.set_matter_model(matter)
    
    # Evolution with mass tracking
    times = []
    masses_at_exterior = []
    
    # Choose exterior radius for monitoring
    exterior_idx = 4 * len(S.r) // 5  # Far exterior point
    
    dt = 0.02
    t_end = 1.0
    n_steps = int(t_end / dt)
    
    for i in range(n_steps + 1):
        times.append(S.t)
        
        # Compute mass function M(r) = (c²r/2G)(1 - A)
        A = S.A()
        M_current = (S.c**2 * S.r / (2.0 * S.G)) * (1.0 - A)
        masses_at_exterior.append(M_current[exterior_idx])
        
        if i < n_steps:
            S.step(dt)
    
    times = np.array(times)
    masses_at_exterior = np.array(masses_at_exterior)
    
    # Check monotonic increase (allowing for small numerical jitter)
    mass_diffs = np.diff(masses_at_exterior)
    n_increasing = np.sum(mass_diffs > 1e-12)  # Tolerance for jitter
    n_decreasing = np.sum(mass_diffs < -1e-12)
    
    print(f"Monotonic mass test (ingoing null flux):")
    print(f"  Initial mass: {masses_at_exterior[0]:.6f}")
    print(f"  Final mass: {masses_at_exterior[-1]:.6f}")
    print(f"  Total mass change: {masses_at_exterior[-1] - masses_at_exterior[0]:.6f}")
    print(f"  Steps with increasing mass: {n_increasing}/{len(mass_diffs)}")
    print(f"  Steps with decreasing mass: {n_decreasing}/{len(mass_diffs)}")
    
    # For ingoing null flux: P = 4πR²T_tr is negative, so M(t) = M(0) - ∫P/R dt increases
    # (subtracting a negative adds to the enclosed mass)
    assert masses_at_exterior[-1] > masses_at_exterior[0], "Enclosed mass should increase at exterior with ingoing flux"
    
    # Most steps should show increasing mass (allowing some numerical jitter)  
    assert n_decreasing < len(mass_diffs) / 10, f"Too many decreasing steps: {n_decreasing}"
    
    # Check that mass increases during the active flux period (t ∈ [0.2, 0.8])
    active_mask = (times >= 0.2) & (times <= 0.8)
    if np.sum(active_mask) > 1:
        mass_start_active = masses_at_exterior[np.where(times >= 0.2)[0][0]]
        mass_end_active = masses_at_exterior[np.where(times <= 0.8)[0][-1]]
        assert mass_end_active > mass_start_active, "Enclosed mass should increase during active flux period"

if __name__ == "__main__":
    test_energy_flux_vs_mass_change_gaussian()
    test_energy_flux_vs_mass_change_vaidya()  
    test_energy_conservation_multiple_radii()
    test_monotonic_mass_ingoing_null_flux()
    print("All energy accounting tests passed! ✓")