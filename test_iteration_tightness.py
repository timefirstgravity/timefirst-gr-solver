#!/usr/bin/env python3
"""
Test fixed-point iteration tightness to see if v-solve contributes to error.
"""

import numpy as np
from timefirst_gr.solver import TimeFirstGRSolver
from tests.test_vaidya_cross_gauge import VaidyaReference, linear_mass_ramp

def test_iteration_tightness():
    """Test if tightening v-solve iterations affects the error."""
    
    mass_func = linear_mass_ramp()
    
    # Test different iteration settings
    configs = [
        {"max_iter": 5, "tol": 1e-10, "name": "Original (5 iter, 1e-10)"},
        {"max_iter": 8, "tol": 1e-12, "name": "Current (8 iter, 1e-12)"},
        {"max_iter": 15, "tol": 1e-14, "name": "Very tight (15 iter, 1e-14)"},
        {"max_iter": 25, "tol": 1e-16, "name": "Extreme (25 iter, 1e-16)"},
    ]
    
    print("=== ITERATION TIGHTNESS TEST ===")
    print("Testing if v-solve precision affects cross-gauge error")
    print()
    
    # Fixed parameters
    r_min, r_max = 3.0, 20.0
    nr = 100
    dt = 0.002
    t_end = 1.0
    
    errors = []
    
    for config in configs:
        print(f"Testing {config['name']}...")
        
        # Create custom VaidyaReference with modified iteration parameters
        class CustomVaidyaReference(VaidyaReference):
            def _coordinate_transformation_analytical(self, t, r):
                """Modified analytical coordinate transformation."""
                v_guess = t + r
                
                for iteration in range(config["max_iter"]):
                    r_star = self._tortoise_coordinate_analytical(v_guess, r)
                    v_new = t + r_star
                    if np.allclose(v_new, v_guess, rtol=config["tol"]):
                        break
                    v_guess = v_new
                
                return v_new
        
        vaidya_ref = CustomVaidyaReference(mass_func, G=1.0, c=1.0)
        
        # Setup solver
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
        rms_error = np.sqrt(np.mean(abs_error**2))
        
        errors.append(max_rel_error)
        print(f"  Max relative error: {max_rel_error:.4f}")
        print(f"  RMS error: {rms_error:.4f}")
        print()
    
    print("Summary:")
    for i, config in enumerate(configs):
        print(f"  {config['name']}: {errors[i]:.4f}")
    
    # Check if iterations matter
    error_range = max(errors) - min(errors)
    print(f"\nError range across all iterations: {error_range:.5f}")
    
    if error_range < 1e-4:
        print("✓ Iteration precision has negligible effect - error dominated by other factors")
    else:
        print("⚠ Iteration precision affects error significantly")

if __name__ == "__main__":
    test_iteration_tightness()