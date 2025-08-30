#!/usr/bin/env python3
"""
Compare different initial condition approaches for Vaidya validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from timefirst_gr.solver import TimeFirstGRSolver
from tests.test_vaidya_cross_gauge import VaidyaReference, linear_mass_ramp

def compare_initial_conditions():
    """Compare different ways of setting initial conditions."""
    
    mass_func = linear_mass_ramp()
    vaidya_ref = VaidyaReference(mass_func, G=1.0, c=1.0)
    
    # Setup parameters
    r_min, r_max = 3.0, 20.0
    nr = 100
    
    print("=== INITIAL CONDITIONS COMPARISON ===")
    print(f"Linear mass ramp: m(v) = 0.5 + 0.1*max(0, v-1.0)")
    print(f"Grid: r ∈ [{r_min}, {r_max}], nr = {nr}")
    print()
    
    # Create grid
    r = np.linspace(r_min, r_max, nr)
    
    # Method 1: Direct Schwarzschild at m(0)
    m0 = mass_func(0.0)  # = 0.5
    A_schw = 1.0 - 2.0 * m0 / r
    A_schw = np.clip(A_schw, 1e-12, None)
    
    # Method 2: Vaidya pullback (what we were doing before)
    t_initial = 0.0
    v_pullback = vaidya_ref.coordinate_transformation(t_initial, r)
    A_pullback = vaidya_ref.A_vaidya(v_pullback, r)
    
    # Method 3: What the analytical Vaidya solution expects at t=1
    t_test = 1.0
    dt = 0.002
    v_final_expected = vaidya_ref.coordinate_transformation(t_test, r)
    A_final_expected = vaidya_ref.A_vaidya(v_final_expected, r)
    
    print("1. INITIAL CONDITIONS ANALYSIS:")
    print(f"   Method 1 (Schwarzschild m(0)): A range [{np.min(A_schw):.3f}, {np.max(A_schw):.3f}]")
    print(f"   Method 2 (Vaidya pullback): A range [{np.min(A_pullback):.3f}, {np.max(A_pullback):.3f}]")
    print(f"   |A_pullback - A_schw| max: {np.max(np.abs(A_pullback - A_schw)):.3f}")
    print()
    print(f"   v_pullback range: [{np.min(v_pullback):.2f}, {np.max(v_pullback):.2f}]")
    print(f"   Corresponding m(v) range: [{mass_func(np.min(v_pullback)):.3f}, {mass_func(np.max(v_pullback)):.3f}]")
    print()
    
    # Test both approaches
    approaches = [
        ("Schwarzschild m(0)", "schwarzschild"),
        ("Vaidya pullback", "pullback")
    ]
    
    for name, method in approaches:
        print(f"2. TESTING {name.upper()}:")
        
        S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                             enforce_boundaries=False)
        
        if method == "schwarzschild":
            S.set_static_schwarzschild(M=m0)
        else:  # pullback
            Phi_initial = vaidya_ref.initial_phi_from_vaidya(0.0, S.r)
            S.Phi = Phi_initial.copy()
        
        # Set up matter
        def T_tr_vaidya(t, r):
            v_current = vaidya_ref.coordinate_transformation(t, r)
            return vaidya_ref.T_tr_from_T_vv(v_current, r)
        
        def T_zero(t, r): return np.zeros_like(r)
        S.matter = (T_tr_vaidya, T_zero, T_zero)
        
        # Check initial T_tr
        T_tr_initial = T_tr_vaidya(0.0, S.r)
        print(f"   Initial T_tr range: [{np.min(T_tr_initial):.2e}, {np.max(T_tr_initial):.2e}]")
        
        # Short evolution
        S.run(t_end=t_test, dt=dt)
        
        # Compare with expected
        A_numerical = S.A()
        error = np.abs(A_numerical - A_final_expected) / (np.abs(A_final_expected) + 1e-12)
        max_error = np.max(error)
        
        print(f"   Final max rel error: {max_error:.3e}")
        print(f"   Error location: r = {S.r[np.argmax(error)]:.2f}")
        print()
    
    # 3. Check what happens if we don't evolve - just compare initial states
    print("3. STATIC COMPARISON (no evolution):")
    
    # What should A be at different v values?
    v_test_values = [0.0, 2.0, 4.0, 6.0]
    
    for v_test in v_test_values:
        A_at_v = vaidya_ref.A_vaidya(np.full_like(r, v_test), r)
        diff_from_schw = np.max(np.abs(A_at_v - A_schw))
        diff_from_pullback = np.max(np.abs(A_at_v - A_pullback))
        
        print(f"   A_vaidya(v={v_test:.1f}) vs A_schw: max diff = {diff_from_schw:.3f}")
        print(f"   A_vaidya(v={v_test:.1f}) vs A_pullback: max diff = {diff_from_pullback:.3f}")
        print(f"   m(v={v_test:.1f}) = {mass_func(v_test):.3f}")
    
    print()
    print("4. COORDINATE TRANSFORMATION CHECK:")
    # What v values do we get at t=1?
    v_at_t1 = vaidya_ref.coordinate_transformation(1.0, r)
    print(f"   v(t=1, r) range: [{np.min(v_at_t1):.2f}, {np.max(v_at_t1):.2f}]")
    print(f"   Should compare A_numerical(t=1) with A_vaidya(v(t=1,r), r)")
    print(f"   This uses m(v) ranging from {mass_func(np.min(v_at_t1)):.3f} to {mass_func(np.max(v_at_t1)):.3f}")

if __name__ == "__main__":
    compare_initial_conditions()