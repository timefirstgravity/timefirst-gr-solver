#!/usr/bin/env python3
"""
Test flux law units, sign, and consistency across SI ↔ geometric units.
This validates the core evolution equation: ∂_t Φ = -4π G r T_tr / c^4
"""

import numpy as np
import pytest
from timefirst_gr.solver import TimeFirstGRSolver

def test_flux_sign_and_units_SI():
    """Test flux law with SI units: ∂_t Φ = -4π G r T_tr / c^4"""
    # SI units
    G_SI = 6.67430e-11  # m³/(kg⋅s²)
    c_SI = 2.99792458e8  # m/s
    
    # Test parameters
    r = 10.0  # meters
    T_tr = 1.0  # kg/(m⋅s³) - energy flux density
    
    # Expected result
    expected_dPhi_dt = -4.0 * np.pi * G_SI * r * T_tr / (c_SI**4)
    
    # Create solver and test evolution
    S = TimeFirstGRSolver(r_min=5.0, r_max=15.0, nr=11, G=G_SI, c=c_SI, 
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.0)  # Start with flat space
    
    # Set constant T_tr
    def T_tr_const(t, r_arr): return T_tr * np.ones_like(r_arr)
    def T_tt_zero(t, r_arr): return np.zeros_like(r_arr) 
    def T_rr_zero(t, r_arr): return np.zeros_like(r_arr)
    S.matter = (T_tr_const, T_tt_zero, T_rr_zero)
    
    # Compute one evolution step
    dt = 0.001
    dPhi_dt_actual = S.step(dt)
    
    # Find the r=10m grid point
    r_idx = np.argmin(np.abs(S.r - r))
    actual_at_r = dPhi_dt_actual[r_idx]
    
    # Verify sign and magnitude
    assert np.isclose(actual_at_r, expected_dPhi_dt, rtol=1e-12), \
        f"Expected {expected_dPhi_dt:.2e}, got {actual_at_r:.2e}"
    
    # Verify negative sign (ingoing flux should decrease Φ)
    assert actual_at_r < 0, "Positive T_tr should give negative dΦ/dt"

def test_flux_geometric_units():
    """Test flux law with geometric units (G=c=1): ∂_t Φ = -4π r T_tr"""
    # Geometric units
    G = 1.0
    c = 1.0
    
    # Test parameters  
    r = 5.0
    T_tr = 0.1
    
    expected_dPhi_dt = -4.0 * np.pi * r * T_tr
    
    S = TimeFirstGRSolver(r_min=2.0, r_max=8.0, nr=13, G=G, c=c,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.0)
    
    def T_tr_const(t, r_arr): return T_tr * np.ones_like(r_arr)
    def T_zero(t, r_arr): return np.zeros_like(r_arr)
    S.matter = (T_tr_const, T_zero, T_zero)
    
    dt = 0.01
    dPhi_dt_actual = S.step(dt)
    
    r_idx = np.argmin(np.abs(S.r - r))
    actual_at_r = dPhi_dt_actual[r_idx]
    
    assert np.isclose(actual_at_r, expected_dPhi_dt, rtol=1e-12), \
        f"Geometric units: expected {expected_dPhi_dt:.6f}, got {actual_at_r:.6f}"

def test_unit_conversion_consistency():
    """Test that SI and geometric units give equivalent dimensionless results"""
    # Physical scenario: energy pulse
    r_test = 10.0  # meters (SI) or geometric length
    T_tr_SI = 1e-6  # kg/(m⋅s³)
    
    # SI version
    G_SI = 6.67430e-11
    c_SI = 2.99792458e8
    S_SI = TimeFirstGRSolver(r_min=5.0, r_max=15.0, nr=21, G=G_SI, c=c_SI,
                            enforce_boundaries=False)
    S_SI.set_static_schwarzschild(M=0.0)
    
    def T_tr_SI_func(t, r): return T_tr_SI * np.ones_like(r)
    def T_zero(t, r): return np.zeros_like(r)
    S_SI.matter = (T_tr_SI_func, T_zero, T_zero)
    
    # Geometric version (scaled appropriately)
    c_geom = 1.0
    G_geom = 1.0
    # Convert T_tr to geometric units: [T_tr_geom] = [T_tr_SI] * G/c³
    T_tr_geom = T_tr_SI * G_SI / (c_SI**3)
    
    S_geom = TimeFirstGRSolver(r_min=5.0, r_max=15.0, nr=21, G=G_geom, c=c_geom,
                              enforce_boundaries=False)  
    S_geom.set_static_schwarzschild(M=0.0)
    
    def T_tr_geom_func(t, r): return T_tr_geom * np.ones_like(r)
    S_geom.matter = (T_tr_geom_func, T_zero, T_zero)
    
    # Compare evolution rates (should be identical when properly scaled)
    dt = 0.001
    dPhi_dt_SI = S_SI.step(dt)
    dPhi_dt_geom = S_geom.step(dt)
    
    # Find test radius
    r_idx_SI = np.argmin(np.abs(S_SI.r - r_test))
    r_idx_geom = np.argmin(np.abs(S_geom.r - r_test))
    
    # Scale SI result to geometric units: multiply by c²/G to get dimensionless rate
    rate_SI_scaled = dPhi_dt_SI[r_idx_SI] * (c_SI**2) / G_SI
    rate_geom = dPhi_dt_geom[r_idx_geom]
    
    assert np.isclose(rate_SI_scaled, rate_geom, rtol=1e-10), \
        f"Unit conversion failed: SI scaled = {rate_SI_scaled:.2e}, geom = {rate_geom:.2e}"

def test_static_vacuum_preservation():
    """Test that vacuum (T_tr=0) preserves static Schwarzschild exactly"""
    M = 1.0
    S = TimeFirstGRSolver(r_min=3.0, r_max=20.0, nr=100, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=M)
    S.set_vacuum()
    
    # Store initial state
    Phi_initial = S.Phi.copy()
    
    # Evolve for many steps  
    dt = 0.01
    n_steps = 1000
    for _ in range(n_steps):
        dPhi_dt = S.step(dt)
        # Each step should give zero evolution
        assert np.max(np.abs(dPhi_dt)) < 1e-15, \
            f"Vacuum should give zero evolution, got max |dΦ/dt| = {np.max(np.abs(dPhi_dt)):.2e}"
    
    # Final state should be identical to initial
    Phi_final = S.Phi
    max_drift = np.max(np.abs(Phi_final - Phi_initial))
    assert max_drift < 1e-14, \
        f"Schwarzschild vacuum drifted by {max_drift:.2e} over {n_steps} steps"

def test_origin_regularity():
    """Test that r*T_tr factor handles r=0 correctly (should vanish)"""
    S = TimeFirstGRSolver(r_min=0.0, r_max=5.0, nr=51, G=1.0, c=1.0,
                         enforce_boundaries=False)  
    S.set_static_schwarzschild(M=0.0)
    
    # Set uniform T_tr (finite at origin)
    T_tr_val = 1.0
    def T_tr_uniform(t, r): return T_tr_val * np.ones_like(r)
    def T_zero(t, r): return np.zeros_like(r)
    S.matter = (T_tr_uniform, T_zero, T_zero)
    
    dt = 0.001
    dPhi_dt = S.step(dt)
    
    # At r=0, evolution should be zero despite finite T_tr
    assert np.abs(dPhi_dt[0]) < 1e-15, \
        f"At r=0, dΦ/dt should be zero, got {dPhi_dt[0]:.2e}"
    
    # Nearby points should have small but finite evolution
    assert np.abs(dPhi_dt[1]) > 1e-10, \
        f"Near r=0, dΦ/dt should be finite, got {dPhi_dt[1]:.2e}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])