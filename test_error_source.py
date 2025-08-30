#!/usr/bin/env python3
"""
Investigate the true source of the ~2% Vaidya error.
Since it's not dt or iteration precision, what is it?
"""

import numpy as np
from timefirst_gr.solver import TimeFirstGRSolver
from tests.test_vaidya_cross_gauge import VaidyaReference, linear_mass_ramp

def test_error_source():
    """Deep dive into what's causing the constant 2% error."""
    
    mass_func = linear_mass_ramp()
    vaidya_ref = VaidyaReference(mass_func, G=1.0, c=1.0)
    
    # Test different aspects
    print("=== ERROR SOURCE INVESTIGATION ===")
    print("Error is independent of dt and iteration precision.")
    print("Investigating other potential sources...")
    print()
    
    # 1. Grid resolution effect
    print("1. GRID RESOLUTION TEST:")
    nr_values = [50, 100, 200]
    r_min, r_max = 3.0, 20.0
    dt = 0.002
    t_end = 1.0
    
    for nr in nr_values:
        S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                             enforce_boundaries=False)
        
        t_initial = 0.0
        Phi_initial = vaidya_ref.initial_phi_from_vaidya(t_initial, S.r)
        S.Phi = Phi_initial.copy()
        
        def T_tr_vaidya(t, r):
            v_current = vaidya_ref.coordinate_transformation(t, r)
            return vaidya_ref.T_tr_from_T_vv(v_current, r)
        
        def T_zero(t, r): return np.zeros_like(r)
        S.matter = (T_tr_vaidya, T_zero, T_zero)
        
        S.run(t_end=t_end, dt=dt)
        
        A_numerical = S.A()
        v_final = vaidya_ref.coordinate_transformation(t_end, S.r)
        A_analytical = vaidya_ref.A_vaidya(v_final, S.r)
        
        rel_error = np.abs(A_numerical - A_analytical) / (np.abs(A_analytical) + 1e-12)
        max_rel_error = np.max(rel_error)
        
        print(f"   nr = {nr:3d}: max_rel_error = {max_rel_error:.4f}")
    
    # 2. Different time ranges
    print("\n2. TIME RANGE EFFECT:")
    t_end_values = [0.2, 0.5, 1.0, 1.5]
    nr = 100
    
    for t_end in t_end_values:
        S = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=nr, G=1.0, c=1.0,
                             enforce_boundaries=False)
        
        t_initial = 0.0
        Phi_initial = vaidya_ref.initial_phi_from_vaidya(t_initial, S.r)
        S.Phi = Phi_initial.copy()
        
        def T_tr_vaidya(t, r):
            v_current = vaidya_ref.coordinate_transformation(t, r)
            return vaidya_ref.T_tr_from_T_vv(v_current, r)
        
        def T_zero(t, r): return np.zeros_like(r)
        S.matter = (T_tr_vaidya, T_zero, T_zero)
        
        S.run(t_end=t_end, dt=dt)
        
        A_numerical = S.A()
        v_final = vaidya_ref.coordinate_transformation(t_end, S.r)
        A_analytical = vaidya_ref.A_vaidya(v_final, S.r)
        
        rel_error = np.abs(A_numerical - A_analytical) / (np.abs(A_analytical) + 1e-12)
        max_rel_error = np.max(rel_error)
        
        print(f"   t_end = {t_end:.1f}: max_rel_error = {max_rel_error:.4f}")
    
    # 3. Test exact vs approximate initial conditions
    print("\n3. INITIAL CONDITIONS TEST:")
    
    # Method A: Current approach (transform t=0 to v, then back)
    S_current = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=100, G=1.0, c=1.0,
                                 enforce_boundaries=False)
    Phi_current = vaidya_ref.initial_phi_from_vaidya(0.0, S_current.r)
    S_current.Phi = Phi_current.copy()
    
    # Method B: Direct Schwarzschild at m(0)
    S_direct = TimeFirstGRSolver(r_min=r_min, r_max=r_max, nr=100, G=1.0, c=1.0,
                                enforce_boundaries=False)
    S_direct.set_static_schwarzschild(M=mass_func(0.0))
    
    # Compare initial A values
    A_current_init = S_current.A()
    A_direct_init = S_direct.A()
    init_diff = np.max(np.abs(A_current_init - A_direct_init))
    print(f"   Initial A difference (transform vs direct): {init_diff:.2e}")
    
    # 4. Test if the issue is in the analytical Vaidya formula itself
    print("\n4. VAIDYA FORMULA VALIDATION:")
    
    # At t=1, compare different ways of computing v
    t_test = 1.0
    r_test = S_current.r
    
    # Method 1: Our iterative coordinate transformation
    v_iterative = vaidya_ref.coordinate_transformation(t_test, r_test)
    
    # Method 2: Simple approximation v ≈ t + r
    v_simple = t_test + r_test
    
    # Method 3: More precise approximation accounting for mass
    # For Vaidya: r* ≈ r + 2m log|r-2m| (using average mass)
    m_avg = mass_func(t_test + np.mean(r_test))  # Rough estimate
    r_star_approx = r_test + 2*m_avg * np.log(np.abs(r_test - 2*m_avg))
    v_approx = t_test + r_star_approx
    
    # Compare A values using different v calculations
    A_iterative = vaidya_ref.A_vaidya(v_iterative, r_test)
    A_simple = vaidya_ref.A_vaidya(v_simple, r_test) 
    A_approx = vaidya_ref.A_vaidya(v_approx, r_test)
    
    diff_simple = np.max(np.abs(A_iterative - A_simple))
    diff_approx = np.max(np.abs(A_iterative - A_approx))
    
    print(f"   v range (iterative): [{np.min(v_iterative):.2f}, {np.max(v_iterative):.2f}]")
    print(f"   v range (simple t+r): [{np.min(v_simple):.2f}, {np.max(v_simple):.2f}]")
    print(f"   v range (approx): [{np.min(v_approx):.2f}, {np.max(v_approx):.2f}]")
    print(f"   A difference (iterative vs simple): {diff_simple:.4f}")
    print(f"   A difference (iterative vs approx): {diff_approx:.4f}")
    
    # 5. Check consistency of Vaidya stress-energy
    print("\n5. STRESS-ENERGY CONSISTENCY:")
    
    # At t=0.5, compute T_tr and check if it's reasonable
    t_mid = 0.5
    v_mid = vaidya_ref.coordinate_transformation(t_mid, r_test)
    T_tr_computed = vaidya_ref.T_tr_from_T_vv(v_mid, r_test)
    T_vv_original = vaidya_ref.T_vv_eddington_finkelstein(v_mid, r_test)
    A_mid = vaidya_ref.A_vaidya(v_mid, r_test)
    
    print(f"   At t = {t_mid}:")
    print(f"   T_vv range: [{np.min(T_vv_original):.2e}, {np.max(T_vv_original):.2e}]")
    print(f"   T_tr range: [{np.min(T_tr_computed):.2e}, {np.max(T_tr_computed):.2e}]")
    print(f"   A range: [{np.min(A_mid):.3f}, {np.max(A_mid):.3f}]")
    print(f"   T_tr/T_vv ratio: {np.mean(T_tr_computed/T_vv_original):.3f} (should equal 1/A_avg = {1/np.mean(A_mid):.3f})")

if __name__ == "__main__":
    test_error_source()