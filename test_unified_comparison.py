#!/usr/bin/env python3
"""
Test script to verify the unified matter comparison works correctly
and that both solvers produce the same physics.
"""

import numpy as np
from timefirst_gr import TimeFirstGRSolver, StandardEllipticSolver
from timefirst_gr import GaussianPulse, VaidyaLikeNull
from timefirst_gr import compare_physics_validation, print_validation_summary

def test_vacuum_comparison():
    """Test that both solvers give identical results for vacuum."""
    print("Testing vacuum comparison...")
    
    # Both solvers with identical setup
    S = TimeFirstGRSolver(r_min=3.0, r_max=50.0, nr=200, enforce_boundaries=True)
    S.set_static_schwarzschild(M=1.0)
    S.set_vacuum()
    
    Std = StandardEllipticSolver(r_min=3.0, r_max=50.0, nr=200)
    Std.set_static_schwarzschild(M=1.0)
    Std.set_T_tt(lambda t, r: np.zeros_like(r))
    
    # Short evolution
    dt = 0.01
    t_end = 0.1
    S.run(t_end, dt)
    Std.run(t_end, dt, lambda t, r: np.zeros_like(r))
    
    # Physics validation
    validation = compare_physics_validation(S, Std, rtol=1e-10, atol=1e-12)
    print_validation_summary(validation)
    
    return validation['physics_agreement']

def test_gaussian_matter_comparison():
    """Test unified Gaussian matter model."""
    print("\nTesting Gaussian matter comparison...")
    
    # Create matter model (smaller amplitude to avoid numerical issues)
    matter = GaussianPulse(amplitude=0.001, r_center=10.0, r_width=2.0,
                          t_center=0.05, t_width=0.02, velocity=0.0)
    
    # Lapse-first solver
    S = TimeFirstGRSolver(r_min=3.0, r_max=30.0, nr=300, enforce_boundaries=True)
    S.set_static_schwarzschild(M=1.0)
    S.set_matter_model(matter)
    
    # Standard solver with same matter
    Std = StandardEllipticSolver(r_min=3.0, r_max=30.0, nr=300)
    Std.set_static_schwarzschild(M=1.0)
    
    # Evolution
    dt = 0.005
    t_end = 0.1
    S.run(t_end, dt)
    Std.run(t_end, dt, lambda t, r: matter.T_tt(t, r, c=1.0))
    
    # Physics validation
    validation = compare_physics_validation(S, Std, rtol=1e-2, atol=1e-6)
    print_validation_summary(validation, verbose=True)
    
    return validation['physics_agreement']

def test_vaidya_comparison():
    """Test null dust (Vaidya-like) matter."""
    print("\nTesting Vaidya null dust comparison...")
    
    # Null dust luminosity
    L = lambda t: 0.1 * np.exp(-0.5 * ((t - 0.05) / 0.02)**2)
    matter = VaidyaLikeNull(L, r_min=5.0, direction="ingoing")
    
    # Lapse-first solver
    S = TimeFirstGRSolver(r_min=3.0, r_max=40.0, nr=400, enforce_boundaries=True)
    S.set_static_schwarzschild(M=1.0)  
    S.set_matter_model(matter)
    
    # Standard solver (T_tt = 0 for null dust)
    Std = StandardEllipticSolver(r_min=3.0, r_max=40.0, nr=400)
    Std.set_static_schwarzschild(M=1.0)
    
    # Evolution
    dt = 0.002
    t_end = 0.1
    S.run(t_end, dt)
    Std.run(t_end, dt, lambda t, r: matter.T_tt(t, r, c=1.0))  # Should be zero
    
    # For null dust, we expect some differences since it's a more challenging case
    validation = compare_physics_validation(S, Std, rtol=1e-1, atol=1e-4)
    print_validation_summary(validation)
    
    return validation['physics_agreement']

def test_boundary_conditions():
    """Test that boundary conditions are properly enforced."""
    print("\nTesting boundary condition enforcement...")
    
    matter = GaussianPulse(amplitude=0.02, r_center=8.0, r_width=1.5,
                          t_center=0.03, t_width=0.01, velocity=0.0)
    
    # With boundary enforcement
    S_bc = TimeFirstGRSolver(r_min=2.5, r_max=25.0, nr=200, enforce_boundaries=True)
    S_bc.set_static_schwarzschild(M=1.0)
    S_bc.set_matter_model(matter)
    
    # Without boundary enforcement  
    S_no_bc = TimeFirstGRSolver(r_min=2.5, r_max=25.0, nr=200, enforce_boundaries=False)
    S_no_bc.set_static_schwarzschild(M=1.0)
    S_no_bc.set_matter_model(matter)
    
    # Evolution
    dt = 0.005
    t_end = 0.05
    S_bc.run(t_end, dt)
    S_no_bc.run(t_end, dt)
    
    # Check boundary conditions
    inner_bc_enforced = abs(S_bc.Phi[0] - S_bc.Phi[1]) < 1e-14  # Neumann BC
    outer_bc_enforced = abs(S_bc.Phi[-1]) < 1e-14              # Dirichlet BC
    
    inner_bc_free = abs(S_no_bc.Phi[0] - S_no_bc.Phi[1]) > 1e-10
    outer_bc_free = abs(S_no_bc.Phi[-1]) > 1e-10
    
    print(f"With BC enforcement:")
    print(f"  Inner BC (dΦ/dr=0): {inner_bc_enforced} (Φ[0]-Φ[1] = {S_bc.Phi[0]-S_bc.Phi[1]:.2e})")
    print(f"  Outer BC (Φ=0): {outer_bc_enforced} (Φ[-1] = {S_bc.Phi[-1]:.2e})")
    
    print(f"Without BC enforcement:")  
    print(f"  Inner different: {inner_bc_free} (Φ[0]-Φ[1] = {S_no_bc.Phi[0]-S_no_bc.Phi[1]:.2e})")
    print(f"  Outer different: {outer_bc_free} (Φ[-1] = {S_no_bc.Phi[-1]:.2e})")
    
    return inner_bc_enforced and outer_bc_enforced

if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED MATTER MODEL VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Vacuum comparison", test_vacuum_comparison),
        ("Gaussian matter", test_gaussian_matter_comparison), 
        ("Vaidya null dust", test_vaidya_comparison),
        ("Boundary conditions", test_boundary_conditions),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"\n{name}: {status}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"\n{name}: ✗ ERROR - {e}")
        print("-" * 60)
    
    # Summary
    print(f"\n{'TEST SUMMARY':=^60}")
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✓" if result else "✗"
        if error:
            print(f"{status} {name}: ERROR - {error}")
        else:
            print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    if passed == total:
        print("🎉 All tests passed! The unified comparison is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")