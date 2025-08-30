#!/usr/bin/env python3
"""
Debug the source of Vaidya cross-gauge validation error.
"""

import numpy as np
import matplotlib.pyplot as plt
from timefirst_gr.solver import TimeFirstGRSolver
from tests.test_vaidya_cross_gauge import VaidyaReference, linear_mass_ramp

def debug_vaidya_error():
    """Analyze sources of error in Vaidya validation."""
    
    mass_func = linear_mass_ramp()
    vaidya_ref = VaidyaReference(mass_func, G=1.0, c=1.0)
    
    # Setup solver
    r_min, r_max = 3.0, 20.0
    nr = 100
    S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                         enforce_boundaries=False)
    
    print("=== DEBUGGING VAIDYA CROSS-GAUGE ERROR ===")
    print(f"Linear mass ramp: m(v) = 0.5 + 0.1*max(0, v-1.0)")
    print(f"Grid: {nr} points, r ∈ [{r_min:.1f}, {r_max:.1f}]")
    
    # 1. Check initial conditions
    print("\n1. INITIAL CONDITIONS CHECK:")
    t_initial = 0.0
    v_initial = vaidya_ref.coordinate_transformation(t_initial, S.r)
    print(f"   t_initial = {t_initial}")
    print(f"   v_initial range: [{np.min(v_initial):.2f}, {np.max(v_initial):.2f}]")
    print(f"   Mass at v_initial: m(v) = {mass_func(v_initial[0]):.3f} to {mass_func(v_initial[-1]):.3f}")
    
    # Set initial conditions
    Phi_initial = vaidya_ref.initial_phi_from_vaidya(t_initial, S.r)
    S.Phi = Phi_initial.copy()
    
    # Compare initial A with analytical
    A_initial_numerical = S.A()
    A_initial_analytical = vaidya_ref.A_vaidya(v_initial, S.r)
    initial_error = np.max(np.abs(A_initial_numerical - A_initial_analytical))
    print(f"   Initial A error: {initial_error:.2e}")
    
    # 2. Check coordinate transformation consistency
    print("\n2. COORDINATE TRANSFORMATION CHECK:")
    t_test = 0.5
    v_test = vaidya_ref.coordinate_transformation(t_test, S.r)
    print(f"   At t = {t_test}:")
    print(f"   v range: [{np.min(v_test):.2f}, {np.max(v_test):.2f}]")
    print(f"   Coordinate transform converged in iteration check...")
    
    # Check if fixed point iteration actually converged
    v_guess = t_test + S.r
    for i in range(10):
        r_star = vaidya_ref._tortoise_coordinate_analytical(v_guess, S.r)
        v_new = t_test + r_star
        max_change = np.max(np.abs(v_new - v_guess))
        print(f"     Iteration {i}: max change = {max_change:.2e}")
        if max_change < 1e-12:
            print(f"     Converged after {i+1} iterations")
            break
        v_guess = v_new
    
    # 3. Evolution and error analysis
    print("\n3. EVOLUTION AND ERROR ANALYSIS:")
    
    def T_tr_vaidya(t, r):
        v_current = vaidya_ref.coordinate_transformation(t, r)
        return vaidya_ref.T_tr_from_T_vv(v_current, r)
    
    def T_tt_zero(t, r): return np.zeros_like(r)
    def T_rr_zero(t, r): return np.zeros_like(r)
    
    S.matter = (T_tr_vaidya, T_tt_zero, T_rr_zero)
    
    # Track evolution at several time points
    t_end = 1.0
    dt = 0.002
    n_check_points = 5
    
    check_times = np.linspace(0, t_end, n_check_points)
    errors = []
    
    for i, t_check in enumerate(check_times):
        if i > 0:  # Skip initial (t=0)
            # Run to checkpoint
            S.run(t_end=t_check, dt=dt)
        
        # Compute error at this time
        A_numerical = S.A()
        v_final = vaidya_ref.coordinate_transformation(S.t, S.r)
        A_analytical = vaidya_ref.A_vaidya(v_final, S.r)
        
        abs_error = np.abs(A_numerical - A_analytical)
        rel_error = abs_error / (np.abs(A_analytical) + 1e-12)
        
        max_rel_error = np.max(rel_error)
        rms_error = np.sqrt(np.mean(abs_error**2))
        
        errors.append((S.t, max_rel_error, rms_error))
        print(f"   t = {S.t:.3f}: max_rel_error = {max_rel_error:.3e}, rms = {rms_error:.3e}")
        
        # Check where the largest errors occur
        worst_idx = np.argmax(rel_error)
        print(f"     Worst error at r = {S.r[worst_idx]:.2f}, v = {v_final[worst_idx]:.2f}")
    
    # 4. Analyze error distribution
    print("\n4. ERROR DISTRIBUTION ANALYSIS:")
    
    # Final state analysis
    A_numerical = S.A()
    v_final = vaidya_ref.coordinate_transformation(S.t, S.r)
    A_analytical = vaidya_ref.A_vaidya(v_final, S.r)
    
    rel_error = np.abs(A_numerical - A_analytical) / (np.abs(A_analytical) + 1e-12)
    
    # Where are the largest errors?
    large_error_mask = rel_error > 0.01  # Errors > 1%
    if np.any(large_error_mask):
        print(f"   Regions with >1% error:")
        r_bad = S.r[large_error_mask]
        v_bad = v_final[large_error_mask]
        error_bad = rel_error[large_error_mask]
        for j in range(min(5, len(r_bad))):
            print(f"     r = {r_bad[j]:.2f}, v = {v_bad[j]:.2f}, error = {error_bad[j]:.3e}")
    
    # Check if errors correlate with mass function behavior
    print("\n5. MASS FUNCTION ANALYSIS:")
    v_range = v_final
    m_values = np.array([mass_func(v) for v in v_range])
    dm_dv = np.gradient(m_values, v_range)
    
    print(f"   v range: [{np.min(v_range):.2f}, {np.max(v_range):.2f}]")
    print(f"   m(v) range: [{np.min(m_values):.3f}, {np.max(m_values):.3f}]")
    print(f"   dm/dv range: [{np.min(dm_dv):.4f}, {np.max(dm_dv):.4f}]")
    
    # The mass ramp starts at v=1.0, so let's see what fraction of points have v>1
    active_flux_mask = v_range > 1.0
    print(f"   Points with v > 1.0 (active flux): {np.sum(active_flux_mask)}/{len(v_range)}")
    
    if np.sum(active_flux_mask) > 0:
        error_in_active = rel_error[active_flux_mask]
        error_in_inactive = rel_error[~active_flux_mask]
        print(f"   Max error in active region: {np.max(error_in_active):.3e}")
        print(f"   Max error in inactive region: {np.max(error_in_inactive):.3e}")

if __name__ == "__main__":
    debug_vaidya_error()