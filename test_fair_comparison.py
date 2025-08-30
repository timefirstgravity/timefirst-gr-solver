#!/usr/bin/env python3
"""
Test the fair apples-to-apples comparison to ensure both solvers
are solving the same physics problem correctly.
"""

import numpy as np
from timefirst_gr.solver import TimeFirstGRSolver  
from timefirst_gr.standard_evolution import StandardADMSolver
from timefirst_gr.matter import GaussianPulse
from timefirst_gr.validation import compare_physics_validation, print_validation_summary

def test_vacuum_evolution():
    """Test that both solvers agree on vacuum evolution."""
    print("Testing vacuum evolution...")
    
    # Both solvers with identical vacuum setup
    S_lapse = TimeFirstGRSolver(r_min=3.0, r_max=40.0, nr=200, enforce_boundaries=False)
    S_lapse.set_static_schwarzschild(M=1.0)
    S_lapse.set_vacuum()
    
    S_adm = StandardADMSolver(r_min=3.0, r_max=40.0, nr=200)
    S_adm.set_static_schwarzschild(M=1.0)
    S_adm.set_vacuum()
    
    # Short evolution
    dt = 0.01
    t_end = 0.05  # Very short for vacuum stability
    
    S_lapse.run(t_end, dt)
    S_adm.run(t_end, dt)
    
    # Compare final states
    phi_diff = np.max(np.abs(S_lapse.Phi - S_adm.Phi))
    mass_lapse = S_lapse.mass_function()
    mass_adm = S_adm.mass_function()
    mass_diff = np.max(np.abs(mass_lapse - mass_adm))
    
    print(f"  Max Φ difference: {phi_diff:.2e}")
    print(f"  Max mass difference: {mass_diff:.2e}")
    
    # Both should preserve Schwarzschild solution in vacuum
    success = phi_diff < 0.1 and mass_diff < 0.1
    return success

def test_gaussian_matter_evolution():
    """Test both solvers with small Gaussian matter."""
    print("\nTesting Gaussian matter evolution...")
    
    # Very small amplitude for stability
    matter = GaussianPulse(
        amplitude=0.001,  # Small
        r_center=10.0,
        r_width=2.0, 
        t_center=0.03,
        t_width=0.01,
        velocity=0.0
    )
    
    # Both solvers
    S_lapse = TimeFirstGRSolver(r_min=3.0, r_max=30.0, nr=300, enforce_boundaries=False)
    S_lapse.set_static_schwarzschild(M=1.0)
    S_lapse.set_matter_model(matter)
    
    S_adm = StandardADMSolver(r_min=3.0, r_max=30.0, nr=300)
    S_adm.set_static_schwarzschild(M=1.0) 
    S_adm.set_matter_model(matter)
    
    # Evolution
    dt = 0.005
    t_end = 0.05
    
    S_lapse.run(t_end, dt)
    S_adm.run(t_end, dt)
    
    # Detailed validation
    try:
        validation = compare_physics_validation(S_lapse, S_adm, rtol=0.1, atol=1e-4)
        print_validation_summary(validation)
        return validation['physics_agreement']
    except Exception as e:
        print(f"Validation failed: {e}")
        return False

def test_computational_scaling():
    """Test that lapse-first is indeed faster."""
    print("\nTesting computational scaling...")
    
    import time
    
    matter = GaussianPulse(amplitude=0.001, r_center=8.0, r_width=1.5,
                          t_center=0.02, t_width=0.01, velocity=0.0)
    
    grid_sizes = [100, 200, 400]
    results = []
    
    for nr in grid_sizes:
        print(f"  Grid size {nr}...")
        
        # Lapse-first
        S_lapse = TimeFirstGRSolver(r_min=3.0, r_max=25.0, nr=nr, enforce_boundaries=False)
        S_lapse.set_static_schwarzschild(M=1.0)
        S_lapse.set_matter_model(matter)
        
        t0 = time.perf_counter()
        S_lapse.run(t_end=0.03, dt=0.005)
        lapse_time = time.perf_counter() - t0
        
        # Standard ADM
        S_adm = StandardADMSolver(r_min=3.0, r_max=25.0, nr=nr)
        S_adm.set_static_schwarzschild(M=1.0)
        S_adm.set_matter_model(matter)
        
        t0 = time.perf_counter()
        S_adm.run(t_end=0.03, dt=0.005)
        adm_time = time.perf_counter() - t0
        
        speedup = adm_time / lapse_time if lapse_time > 0 else float('nan')
        results.append((nr, lapse_time, adm_time, speedup))
        
        print(f"    Lapse: {lapse_time:.3f}s, ADM: {adm_time:.3f}s, Speedup: {speedup:.1f}x")
    
    # Check that lapse-first is consistently faster
    avg_speedup = np.mean([r[3] for r in results])
    print(f"  Average speedup: {avg_speedup:.1f}x")
    
    return avg_speedup > 1.0  # Lapse-first should be faster

if __name__ == "__main__":
    print("=" * 60)
    print("FAIR COMPARISON VALIDATION TESTS") 
    print("=" * 60)
    
    tests = [
        ("Vacuum evolution", test_vacuum_evolution),
        ("Gaussian matter", test_gaussian_matter_evolution),
        ("Computational scaling", test_computational_scaling),
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
    print(f"\n{'VALIDATION SUMMARY':=^60}")
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
        print("🎉 Fair comparison is working! Both solvers agree on physics.")
        print("💨 Lapse-first shows computational advantage as expected.")
    else:
        print("⚠️  Issues detected. Check implementation.")