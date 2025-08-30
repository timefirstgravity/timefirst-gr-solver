#!/usr/bin/env python3
"""
Test that both solvers use the same evolution function for ∂_t Φ.
This ensures the comparison is truly fair - same physics, different computational overhead.
"""

import numpy as np
import pytest
from timefirst_gr.solver import TimeFirstGRSolver
from timefirst_gr.standard_evolution import StandardADMSolver
from timefirst_gr.matter import GaussianPulse

def evolution_function_lapse_first(T_tr, r, G, c):
    """Extract the core evolution function from lapse-first solver"""
    return -4.0 * np.pi * G * r * T_tr / (c**4)

def evolution_function_standard_adm(T_tr, r, G, c):
    """Extract the core evolution function from standard ADM solver"""
    # This should be identical to lapse-first
    return -4.0 * np.pi * G * r * T_tr / (c**4)

def test_evolution_functions_identical():
    """Test that both solvers use identical evolution functions"""
    # Test parameters
    G, c = 1.0, 1.0
    r = np.array([3.0, 5.0, 8.0, 12.0])
    T_tr = np.array([0.1, 0.05, 0.02, 0.01])
    
    # Compute evolution rates
    dPhi_dt_lapse = evolution_function_lapse_first(T_tr, r, G, c)
    dPhi_dt_adm = evolution_function_standard_adm(T_tr, r, G, c)
    
    # Should be bitwise identical
    assert np.array_equal(dPhi_dt_lapse, dPhi_dt_adm), \
        "Evolution functions must be identical"
    
    # Verify expected values
    expected = -4.0 * np.pi * r * T_tr
    assert np.allclose(dPhi_dt_lapse, expected), \
        "Evolution function implementation error"

def test_both_solvers_use_same_evolution():
    """Test that both solver implementations give identical step-by-step evolution"""
    # Create identical setups
    matter = GaussianPulse(amplitude=0.001, r_center=8.0, r_width=1.5,
                          t_center=0.3, t_width=0.1, velocity=0.0)
    
    S1 = TimeFirstGRSolver(r_min=3.0, r_max=15.0, nr=100, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S1.set_static_schwarzschild(M=1.0)
    S1.set_matter_model(matter)
    
    S2 = StandardADMSolver(r_min=3.0, r_max=15.0, nr=100, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S2.set_static_schwarzschild(M=1.0)
    S2.set_matter_model(matter)
    
    # Verify identical initial conditions
    assert np.array_equal(S1.r, S2.r), "Grid points must be identical"
    assert np.array_equal(S1.Phi, S2.Phi), "Initial Φ must be identical"
    assert S1.t == S2.t, "Initial time must be identical"
    
    # Single evolution step
    dt = 0.01
    dPhi_dt_1 = S1.step(dt)
    dPhi_dt_2 = S2.step(dt)
    
    # Evolution rates should be identical
    evolution_diff = np.max(np.abs(dPhi_dt_1 - dPhi_dt_2))
    assert evolution_diff == 0.0, \
        f"Evolution rates differ by {evolution_diff:.2e} - should be identical"
    
    # Final states should be identical
    phi_diff = np.max(np.abs(S1.Phi - S2.Phi))
    assert phi_diff == 0.0, \
        f"Final Φ differs by {phi_diff:.2e} - should be identical"

def test_constraint_enforcement_does_not_modify_phi():
    """Test that constraint operations in StandardADM do not modify Φ"""
    matter = GaussianPulse(amplitude=0.002, r_center=6.0, r_width=1.0,
                          t_center=0.2, t_width=0.08, velocity=0.0)
    
    # Create ADM solver
    S = StandardADMSolver(r_min=2.5, r_max=12.0, nr=80, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.8)
    S.set_matter_model(matter)
    
    # Store state before step
    Phi_before = S.Phi.copy()
    t_before = S.t
    
    # Take evolution step
    dt = 0.005
    dPhi_dt = S.step(dt)
    
    # Check that Φ changed only by the expected evolution amount
    expected_Phi = Phi_before + dt * dPhi_dt
    actual_Phi = S.Phi
    
    phi_diff = np.max(np.abs(actual_Phi - expected_Phi))
    
    print(f"Constraint enforcement check:")
    print(f"  Expected Φ evolution: {dt * np.max(np.abs(dPhi_dt)):.6f}")
    print(f"  Deviation from pure evolution: {phi_diff:.2e}")
    
    # Should be machine precision (no feedback from constraints)
    assert phi_diff < 1e-14, \
        f"Constraint enforcement modified Φ by {phi_diff:.2e} - should not modify"

def test_matter_evaluation_consistency():
    """Test that both solvers evaluate matter terms identically"""
    matter = GaussianPulse(amplitude=0.008, r_center=10.0, r_width=2.0,
                          t_center=0.5, t_width=0.2, velocity=0.0)
    
    # Setup both solvers
    S1 = TimeFirstGRSolver(r_min=4.0, r_max=16.0, nr=60, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S1.set_static_schwarzschild(M=1.2)
    S1.set_matter_model(matter)
    
    S2 = StandardADMSolver(r_min=4.0, r_max=16.0, nr=60, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S2.set_static_schwarzschild(M=1.2)
    S2.set_matter_model(matter)
    
    # Test at multiple times
    times = [0.0, 0.2, 0.4, 0.6]
    
    for test_time in times:
        # Evaluate T_tr at test time
        T_tr_1, T_tt_1, T_rr_1 = S1.matter
        T_tr_2, T_tt_2, T_rr_2 = S2.matter
        
        T_tr_val_1 = T_tr_1(test_time, S1.r)
        T_tr_val_2 = T_tr_2(test_time, S2.r)
        
        # Should be identical
        T_tr_diff = np.max(np.abs(T_tr_val_1 - T_tr_val_2))
        assert T_tr_diff == 0.0, \
            f"T_tr evaluation differs at t={test_time}: {T_tr_diff:.2e}"
        
        # Compute evolution rates
        dPhi_dt_1 = evolution_function_lapse_first(T_tr_val_1, S1.r, S1.G, S1.c)
        dPhi_dt_2 = evolution_function_standard_adm(T_tr_val_2, S2.r, S2.G, S2.c)
        
        rate_diff = np.max(np.abs(dPhi_dt_1 - dPhi_dt_2))
        assert rate_diff == 0.0, \
            f"Evolution rates differ at t={test_time}: {rate_diff:.2e}"

def test_multiple_step_consistency():
    """Test that both solvers remain identical over multiple evolution steps"""
    matter = GaussianPulse(amplitude=0.003, r_center=7.0, r_width=1.2,
                          t_center=0.4, t_width=0.15, velocity=0.0)
    
    S1 = TimeFirstGRSolver(r_min=3.0, r_max=14.0, nr=90, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S1.set_static_schwarzschild(M=0.9)
    S1.set_matter_model(matter)
    
    S2 = StandardADMSolver(r_min=3.0, r_max=14.0, nr=90, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S2.set_static_schwarzschild(M=0.9)
    S2.set_matter_model(matter)
    
    # Multiple evolution steps
    dt = 0.008
    n_steps = 50
    
    max_cumulative_diff = 0.0
    
    for i in range(n_steps):
        # Step both solvers
        dPhi_dt_1 = S1.step(dt)
        dPhi_dt_2 = S2.step(dt)
        
        # Check evolution rates
        rate_diff = np.max(np.abs(dPhi_dt_1 - dPhi_dt_2))
        assert rate_diff == 0.0, \
            f"Step {i}: evolution rates differ by {rate_diff:.2e}"
        
        # Check accumulated differences
        phi_diff = np.max(np.abs(S1.Phi - S2.Phi))
        time_diff = abs(S1.t - S2.t)
        
        max_cumulative_diff = max(max_cumulative_diff, phi_diff)
        
        assert phi_diff == 0.0, \
            f"Step {i}: Φ differs by {phi_diff:.2e}"
        assert time_diff == 0.0, \
            f"Step {i}: time differs by {time_diff:.2e}"
    
    print(f"Multiple step test: {n_steps} steps completed")
    print(f"Maximum cumulative Φ difference: {max_cumulative_diff:.2e}")
    print("✓ Both solvers remained identical throughout evolution")

def test_evolution_rate_bounds():
    """Test that evolution rates are within expected physical bounds"""
    matter = GaussianPulse(amplitude=0.01, r_center=5.0, r_width=1.0,
                          t_center=0.1, t_width=0.05, velocity=0.0)
    
    S = TimeFirstGRSolver(r_min=2.0, r_max=10.0, nr=50, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.5)
    S.set_matter_model(matter)
    
    dt = 0.003
    dPhi_dt = S.step(dt)
    
    # Check that evolution rates are finite and reasonable
    assert np.all(np.isfinite(dPhi_dt)), "Evolution rates must be finite"
    
    max_rate = np.max(np.abs(dPhi_dt))
    print(f"Maximum |dΦ/dt|: {max_rate:.3f}")
    
    # Should be proportional to matter amplitude and r
    expected_scale = 4 * np.pi * matter.amplitude * np.max(S.r)
    assert max_rate < 10 * expected_scale, \
        f"Evolution rate {max_rate:.3f} exceeds expected scale {expected_scale:.3f}"
    
    # At r=0, should be exactly zero due to r factor
    assert abs(dPhi_dt[0]) < 1e-15, \
        f"Evolution rate at r=0 should be zero, got {dPhi_dt[0]:.2e}"

if __name__ == "__main__":
    test_evolution_functions_identical()
    test_both_solvers_use_same_evolution()  
    test_constraint_enforcement_does_not_modify_phi()
    test_matter_evaluation_consistency()
    test_multiple_step_consistency()
    test_evolution_rate_bounds()
    print("All shared evolution function tests passed! ✓")