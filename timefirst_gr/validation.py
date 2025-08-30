import numpy as np

def compare_physics_validation(solver_lapse, solver_standard, rtol=1e-3, atol=1e-6):
    """
    Enhanced physics validation comparing lapse-first and standard solvers.
    
    Returns a dictionary with detailed comparison metrics to ensure
    both solvers are computing the same physics.
    """
    results = {}
    
    # Ensure both solvers are at same time
    if abs(solver_lapse.t - solver_standard.t) > 1e-12:
        raise ValueError(f"Solvers at different times: {solver_lapse.t} vs {solver_standard.t}")
    
    # Get common grid (interpolate if needed)
    r_lapse = solver_lapse.r
    r_std = solver_standard.r
    Phi_lapse = solver_lapse.Phi
    Phi_std = solver_standard.Phi
    
    # Interpolate to common grid if different resolutions
    if len(r_lapse) != len(r_std) or not np.allclose(r_lapse, r_std):
        r_common = r_lapse if len(r_lapse) <= len(r_std) else r_std
        Phi_lapse_interp = np.interp(r_common, r_lapse, Phi_lapse)
        Phi_std_interp = np.interp(r_common, r_std, Phi_std)
    else:
        r_common = r_lapse
        Phi_lapse_interp = Phi_lapse
        Phi_std_interp = Phi_std
    
    # 1. Compare metric function Φ(r)
    phi_diff = Phi_lapse_interp - Phi_std_interp
    results['phi_max_abs_diff'] = np.max(np.abs(phi_diff))
    results['phi_rms_diff'] = np.sqrt(np.mean(phi_diff**2))
    results['phi_relative_error'] = np.max(np.abs(phi_diff) / (np.abs(Phi_std_interp) + atol))
    
    # 2. Compare metric coefficient A = exp(2Φ)
    A_lapse = np.exp(2.0 * Phi_lapse_interp)
    A_std = np.exp(2.0 * Phi_std_interp) 
    A_diff = A_lapse - A_std
    results['A_max_abs_diff'] = np.max(np.abs(A_diff))
    results['A_relative_error'] = np.max(np.abs(A_diff) / (A_std + atol))
    
    # 3. Compare mass function M(r) = (c²r/2G)(1 - A)
    c = solver_lapse.c
    G = solver_lapse.G
    M_lapse = (c**2 * r_common / (2.0 * G)) * (1.0 - A_lapse)
    M_std = (c**2 * r_common / (2.0 * G)) * (1.0 - A_std)
    M_diff = M_lapse - M_std
    results['mass_max_abs_diff'] = np.max(np.abs(M_diff))
    results['mass_relative_error'] = np.max(np.abs(M_diff) / (np.abs(M_std) + atol))
    
    # 4. Compare constraint violations
    res_rr_lapse, res_tt_lapse = solver_lapse.constraints_residuals()
    res_rr_std, res_tt_std = solver_standard.constraints_residuals()
    
    # Interpolate constraint residuals to common grid
    if len(r_lapse) != len(r_common):
        res_rr_lapse = np.interp(r_common, r_lapse, res_rr_lapse)
        res_tt_lapse = np.interp(r_common, r_lapse, res_tt_lapse)
    if len(r_std) != len(r_common):
        res_rr_std = np.interp(r_common, r_std, res_rr_std)
        res_tt_std = np.interp(r_common, r_std, res_tt_std)
    
    results['constraint_rr_lapse_max'] = np.max(np.abs(res_rr_lapse))
    results['constraint_rr_std_max'] = np.max(np.abs(res_rr_std))
    results['constraint_tt_lapse_max'] = np.max(np.abs(res_tt_lapse))  
    results['constraint_tt_std_max'] = np.max(np.abs(res_tt_std))
    
    # 5. Physics agreement check
    results['physics_agreement'] = (
        results['phi_relative_error'] < rtol and
        results['A_relative_error'] < rtol and 
        results['mass_relative_error'] < rtol
    )
    
    # 6. Sample point comparisons at specific radii
    n_points = len(r_common)
    sample_indices = [n_points//8, n_points//4, n_points//2, 3*n_points//4]
    results['sample_comparisons'] = []
    
    for i in sample_indices:
        if i < len(r_common):
            sample = {
                'r': float(r_common[i]),
                'phi_lapse': float(Phi_lapse_interp[i]),
                'phi_std': float(Phi_std_interp[i]),
                'phi_diff': float(phi_diff[i]),
                'A_lapse': float(A_lapse[i]),
                'A_std': float(A_std[i]),
                'mass_lapse': float(M_lapse[i]),
                'mass_std': float(M_std[i])
            }
            results['sample_comparisons'].append(sample)
    
    return results

def print_validation_summary(validation_results, verbose=False):
    """Print a summary of the physics validation results."""
    results = validation_results
    
    print("\n=== Physics Validation Summary ===")
    print(f"Φ(r) agreement:")
    print(f"  Max absolute difference: {results['phi_max_abs_diff']:.2e}")
    print(f"  RMS difference: {results['phi_rms_diff']:.2e}")
    print(f"  Max relative error: {results['phi_relative_error']:.2e}")
    
    print(f"\nMetric coefficient A = exp(2Φ):")
    print(f"  Max absolute difference: {results['A_max_abs_diff']:.2e}")
    print(f"  Max relative error: {results['A_relative_error']:.2e}")
    
    print(f"\nMass function M(r):")
    print(f"  Max absolute difference: {results['mass_max_abs_diff']:.2e}")
    print(f"  Max relative error: {results['mass_relative_error']:.2e}")
    
    print(f"\nConstraint violations:")
    print(f"  G_rr residual (lapse-first): {results['constraint_rr_lapse_max']:.2e}")
    print(f"  G_rr residual (standard): {results['constraint_rr_std_max']:.2e}")
    print(f"  G_tt residual (lapse-first): {results['constraint_tt_lapse_max']:.2e}")
    print(f"  G_tt residual (standard): {results['constraint_tt_std_max']:.2e}")
    
    agreement_status = "✓ PASS" if results['physics_agreement'] else "✗ FAIL"
    print(f"\nOverall physics agreement: {agreement_status}")
    
    if verbose and 'sample_comparisons' in results:
        print(f"\nSample point comparisons:")
        for sample in results['sample_comparisons']:
            print(f"  r = {sample['r']:.2f}:")
            print(f"    Φ: {sample['phi_lapse']:.6f} (lapse) vs {sample['phi_std']:.6f} (std), diff = {sample['phi_diff']:.2e}")
            print(f"    A: {sample['A_lapse']:.6f} (lapse) vs {sample['A_std']:.6f} (std)")
            print(f"    M: {sample['mass_lapse']:.6f} (lapse) vs {sample['mass_std']:.6f} (std)")