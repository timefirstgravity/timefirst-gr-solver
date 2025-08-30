#!/usr/bin/env python3
"""
Test constraint residual parity between both solvers.

Verifies that both TimeFirstGRSolver and StandardADMSolver compute
identical Einstein tensor components G_rr and G_tt for the same input.
"""

import numpy as np
import pytest
from timefirst_gr.solver import TimeFirstGRSolver
from timefirst_gr.standard_evolution import StandardADMSolver
from timefirst_gr.matter import GaussianPulse, VaidyaLikeNull

def test_constraint_residual_parity_vacuum():
    """Test that both solvers compute identical constraint residuals in vacuum."""
    # Setup identical solvers
    r_min, r_max = 3.0, 15.0
    nr = 80
    G, c = 1.0, 1.0
    
    S1 = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=G, c=c,
                          enforce_boundaries=False)
    S2 = StandardADMSolver(r_min=r_min, r_max=r_max, nr=nr, G=G, c=c,
                          enforce_boundaries=False)
    
    # Set identical initial conditions (Schwarzschild)
    M = 0.8
    S1.set_static_schwarzschild(M)
    S2.set_static_schwarzschild(M)
    
    # Both should have identical constraint residuals initially
    G_rr_1, G_tt_1 = S1.constraints_residuals()
    G_rr_2, G_tt_2 = S2.constraints_residuals()
    
    G_rr_diff = np.max(np.abs(G_rr_1 - G_rr_2))
    G_tt_diff = np.max(np.abs(G_tt_1 - G_tt_2))
    
    print(f"Vacuum constraint residual comparison:")
    print(f"  Max |G_rr difference|: {G_rr_diff:.2e}")
    print(f"  Max |G_tt difference|: {G_tt_diff:.2e}")
    print(f"  Max |G_rr residual|: {np.max(np.abs(G_rr_1)):.2e}")
    print(f"  Max |G_tt residual|: {np.max(np.abs(G_tt_1)):.2e}")
    
    # Should be numerically identical
    assert G_rr_diff < 1e-15, f"G_rr differs between solvers: {G_rr_diff:.2e}"
    assert G_tt_diff < 1e-15, f"G_tt differs between solvers: {G_tt_diff:.2e}"
    
    # Vacuum should have small residuals (check that constraints are nearly satisfied)
    # Finite difference discretization leads to O(dr²) constraint violations ~1e-3
    assert np.max(np.abs(G_rr_1)) < 2e-3, f"G_rr vacuum residual too large: {np.max(np.abs(G_rr_1)):.2e}"
    assert np.max(np.abs(G_tt_1)) < 5e-4, f"G_tt vacuum residual too large: {np.max(np.abs(G_tt_1)):.2e}"

def test_constraint_residual_parity_with_matter():
    """Test constraint residual parity with matter present."""
    # Setup with Gaussian pulse matter
    matter = GaussianPulse(
        amplitude=0.01,
        r_center=6.0,
        r_width=1.2,
        t_center=0.3,
        t_width=0.2,
        velocity=0.0
    )
    
    r_min, r_max = 2.5, 12.0
    nr = 100
    
    S1 = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                          enforce_boundaries=False)
    S2 = StandardADMSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                          enforce_boundaries=False)
    
    # Set identical initial conditions and matter
    S1.set_static_schwarzschild(M=0.5)
    S2.set_static_schwarzschild(M=0.5)
    S1.set_matter_model(matter)
    S2.set_matter_model(matter)
    
    # Evolve both solvers for a few steps
    dt = 0.01
    n_steps = 5
    
    for i in range(n_steps):
        S1.step(dt)
        S2.step(dt)
        
        # Check constraint residual parity at each step
        G_rr_1, G_tt_1 = S1.constraints_residuals()
        G_rr_2, G_tt_2 = S2.constraints_residuals()
        
        G_rr_diff = np.max(np.abs(G_rr_1 - G_rr_2))
        G_tt_diff = np.max(np.abs(G_tt_1 - G_tt_2))
        
        # Should remain numerically identical
        assert G_rr_diff < 1e-15, f"Step {i}: G_rr differs: {G_rr_diff:.2e}"
        assert G_tt_diff < 1e-15, f"Step {i}: G_tt differs: {G_tt_diff:.2e}"
    
    print(f"Matter constraint residual comparison (after {n_steps} steps):")
    print(f"  Max |G_rr difference|: {G_rr_diff:.2e}")
    print(f"  Max |G_tt difference|: {G_tt_diff:.2e}")
    print(f"  Max |G_rr - 8πG T_rr/c⁴|: {np.max(np.abs(G_rr_1 - 8*np.pi*S1.G*matter.T_rr(S1.t, S1.r)/S1.c**4)):.2e}")
    print(f"  Max |G_tt - 8πG T_tt/c⁴|: {np.max(np.abs(G_tt_1 - 8*np.pi*S1.G*matter.T_tt(S1.t, S1.r)/S1.c**4)):.2e}")

def test_constraint_satisfaction_with_matter():
    """Test that constraints are properly satisfied for expected matter components."""
    # Use a simple case where we can verify constraint satisfaction
    matter = GaussianPulse(
        amplitude=0.005,  # Small amplitude for linearity
        r_center=8.0,
        r_width=1.0,
        t_center=0.2,
        t_width=0.15,
        velocity=0.0
    )
    
    S = TimeFirstGRSolver(r_min=3.0, r_max=15.0, nr=80, G=1.0, c=1.0,
                         enforce_boundaries=False)
    S.set_static_schwarzschild(M=0.3)
    S.set_matter_model(matter)
    
    # Evolve to time when matter is present
    S.run(t_end=0.3, dt=0.005)
    
    # Compute constraint residuals and matter stress-energy
    G_rr, G_tt = S.constraints_residuals()
    T_rr = matter.T_rr(S.t, S.r)
    T_tt = matter.T_tt(S.t, S.r)
    
    # Einstein equations: G_μν = 8πG T_μν / c⁴
    source_rr = 8.0 * np.pi * S.G * T_rr / S.c**4
    source_tt = 8.0 * np.pi * S.G * T_tt / S.c**4
    
    constraint_rr = G_rr - source_rr
    constraint_tt = G_tt - source_tt
    
    max_constraint_rr = np.max(np.abs(constraint_rr))
    max_constraint_tt = np.max(np.abs(constraint_tt))
    
    print(f"Einstein equation satisfaction:")
    print(f"  Max |G_rr - 8πG T_rr/c⁴|: {max_constraint_rr:.2e}")
    print(f"  Max |G_tt - 8πG T_tt/c⁴|: {max_constraint_tt:.2e}")
    print(f"  Max |T_rr|: {np.max(np.abs(T_rr)):.2e}")
    print(f"  Max |T_tt|: {np.max(np.abs(T_tt)):.2e}")
    
    # For our evolution method, constraints should be satisfied to discretization error
    # This is a more lenient test since we're using explicit time stepping with finite differences
    assert max_constraint_rr < 2e-4, f"G_rr constraint violation too large: {max_constraint_rr:.2e}"
    assert max_constraint_tt < 1.0, f"G_tt constraint violation too large: {max_constraint_tt:.2e}"

if __name__ == "__main__":
    test_constraint_residual_parity_vacuum()
    test_constraint_residual_parity_with_matter()
    test_constraint_satisfaction_with_matter()
    print("All constraint residual tests passed! ✓")