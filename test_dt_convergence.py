#!/usr/bin/env python3
"""
Test dt convergence to confirm temporal discretization is the error source.
If temporal error dominates, halving dt should roughly halve the final error.
"""

import numpy as np
from timefirst_gr.solver import TimeFirstGRSolver
from tests.test_vaidya_cross_gauge import VaidyaReference, linear_mass_ramp

def test_dt_halving():
    """Test if halving dt roughly halves the error (confirming O(dt) global error)."""
    
    mass_func = linear_mass_ramp()
    vaidya_ref = VaidyaReference(mass_func, G=1.0, c=1.0)
    
    # Fixed parameters
    r_min, r_max = 3.0, 20.0
    nr = 100
    t_end = 1.0
    
    # Test different dt values
    dt_values = [0.004, 0.002, 0.001]  # Each halving the previous
    errors = []
    
    print("=== DT HALVING TEST ===")
    print("Testing if error ∝ dt (global O(dt) discretization)")
    print(f"Fixed: r ∈ [{r_min}, {r_max}], nr={nr}, t_end={t_end}")
    print()
    
    for dt in dt_values:
        # Setup fresh solver
        S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                             enforce_boundaries=False)
        
        # Set initial conditions
        t_initial = 0.0
        Phi_initial = vaidya_ref.initial_phi_from_vaidya(t_initial, S.r)
        S.Phi = Phi_initial.copy()
        
        # Set up matter source
        def T_tr_vaidya(t, r):
            v_current = vaidya_ref.coordinate_transformation(t, r)
            return vaidya_ref.T_tr_from_T_vv(v_current, r)
        
        def T_tt_zero(t, r): return np.zeros_like(r)
        def T_rr_zero(t, r): return np.zeros_like(r)
        
        S.matter = (T_tr_vaidya, T_tt_zero, T_rr_zero)
        
        # Evolve
        S.run(t_end=t_end, dt=dt)
        
        # Compare with analytical solution
        A_numerical = S.A()
        v_final = vaidya_ref.coordinate_transformation(t_end, S.r)
        A_analytical = vaidya_ref.A_vaidya(v_final, S.r)
        
        # Compute error
        abs_error = np.abs(A_numerical - A_analytical)
        rel_error = abs_error / (np.abs(A_analytical) + 1e-12)
        max_rel_error = np.max(rel_error)
        
        errors.append(max_rel_error)
        n_steps = int(t_end / dt)
        
        print(f"dt = {dt:.3f} ({n_steps:4d} steps): max_rel_error = {max_rel_error:.4f}")
    
    print()
    print("Convergence analysis:")
    
    # Check convergence ratios
    for i in range(1, len(dt_values)):
        dt_ratio = dt_values[i-1] / dt_values[i]  # Should be 2.0
        error_ratio = errors[i-1] / errors[i]
        expected_ratio = dt_ratio  # For O(dt) global error
        
        print(f"  dt {dt_values[i-1]:.3f} → {dt_values[i]:.3f}: error ratio = {error_ratio:.2f} (expected ≈ {expected_ratio:.1f} for O(dt))")
    
    # Overall scaling check
    if len(errors) >= 2:
        # Fit error = C * dt^p
        log_dt = np.log(dt_values)
        log_error = np.log(errors)
        slope = np.polyfit(log_dt, log_error, 1)[0]
        
        print(f"\nError scaling: error ∝ dt^{slope:.2f}")
        print(f"Expected: slope ≈ 1.0 for first-order temporal discretization")
        
        if 0.8 < slope < 1.2:
            print("✓ Confirms O(dt) temporal discretization dominates error")
        else:
            print("⚠ Error scaling suggests other factors may dominate")
    
    return dt_values, errors

if __name__ == "__main__":
    test_dt_halving()