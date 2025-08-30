#!/usr/bin/env python3
"""
Fair apples-to-apples comparison between lapse-first and standard ADM evolution.

Both solvers now solve the SAME evolution problem:
    ∂_t Φ = -4π G r T_tr / c^4

The difference is computational strategy:
- Lapse-first: Direct explicit evolution (cheap per step)
- Standard ADM: Evolution + constraint enforcement (expensive per step)

This demonstrates the computational benefit of lapse-first formulation.
"""

import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

from ..solver import TimeFirstGRSolver
from ..standard_evolution import StandardADMSolver
from ..matter import GaussianPulse, VaidyaLikeNull
from ..validation import compare_physics_validation, print_validation_summary

def fair_benchmark_case(nr=1000, t_end=1.0, dt=0.01, matter_scale=0.01,
                       r_pulse=12.0, sigma_r=2.0, t0=0.5, sigma_t=0.2, 
                       matter_type="gaussian", validate_physics=True):
    """
    Fair comparison: both solvers evolving identical Einstein equations.
    
    Key difference:
    - Lapse-first: ∂_t Φ = -4π G r T_tr / c^4 (direct evolution)
    - Standard ADM: Same evolution + constraint enforcement each step
    
    MATHEMATICAL GUARANTEE: Both solvers call the same function to compute
    ∂_t Φ from T_tr. The Standard ADM path additionally solves constraint
    equations and records residuals, but does NOT feed constraint solutions
    back into Φ. Hence physics is identical and differences are purely
    computational overhead.
    """
    
    # Create unified matter model
    if matter_type == "gaussian":
        matter = GaussianPulse(
            amplitude=matter_scale,
            r_center=r_pulse, 
            r_width=sigma_r,
            t_center=t0,
            t_width=sigma_t,
            velocity=0.0  # At rest for stability
        )
    elif matter_type == "vaidya":
        norm = matter_scale/(np.sqrt(2*np.pi)*sigma_t)
        L = lambda t: norm*np.exp(-0.5*((t - t0)/sigma_t)**2)
        matter = VaidyaLikeNull(L, r_min=5.0, direction="ingoing")
    else:
        raise ValueError(f"Unknown matter_type: {matter_type}")

    # Lapse-first solver: Direct evolution (efficient)
    S_lapse = TimeFirstGRSolver(r_min=2.2, r_max=80.0, nr=nr, G=1.0, c=1.0, 
                               enforce_boundaries=False)  # Let it evolve freely
    S_lapse.set_static_schwarzschild(M=1.0)
    S_lapse.set_matter_model(matter)

    t0_lapse = time.perf_counter()
    S_lapse.run(t_end=t_end, dt=dt, progress=False)
    lapse_time = time.perf_counter() - t0_lapse

    # Standard ADM solver: Same evolution + constraint enforcement (expensive)
    S_adm = StandardADMSolver(r_min=2.2, r_max=80.0, nr=nr, G=1.0, c=1.0, 
                             enforce_boundaries=False)  # Match lapse-first
    S_adm.set_static_schwarzschild(M=1.0)
    S_adm.set_matter_model(matter)

    t0_adm = time.perf_counter()
    S_adm.run(t_end=t_end, dt=dt, progress=False)
    adm_time = time.perf_counter() - t0_adm

    steps = int(np.ceil(t_end/dt))
    i_near = nr//8
    i_mid = nr//2
    i_far = 3*nr//4
    
    # Basic results
    results = {
        "N_r": nr, 
        "steps": steps, 
        "dt": dt, 
        "t_end": t_end,
        "matter_type": matter_type,
        "lapse_first_time_s": lapse_time, 
        "standard_adm_time_s": adm_time,
        "speedup_x": (adm_time / lapse_time) if lapse_time > 0 else float("nan"),
        
        # Compare final Φ at multiple points
        "phi_near_lapse": float(S_lapse.Phi[i_near]),
        "phi_near_adm": float(S_adm.Phi[i_near]),
        "phi_mid_lapse": float(S_lapse.Phi[i_mid]),  
        "phi_mid_adm": float(S_adm.Phi[i_mid]),
        "phi_far_lapse": float(S_lapse.Phi[i_far]),
        "phi_far_adm": float(S_adm.Phi[i_far]),
        
        # Compare mass functions
        "mass_near_lapse": float(S_lapse.mass_function()[i_near]),
        "mass_near_adm": float(S_adm.mass_function()[i_near]),
    }
    
    # Physics validation
    if validate_physics:
        try:
            validation = compare_physics_validation(S_lapse, S_adm, rtol=0.1, atol=1e-4)
            results.update({
                'physics_agreement': validation['physics_agreement'],
                'phi_max_diff': validation['phi_max_abs_diff'],
                'phi_relative_error': validation['phi_relative_error'],
                'mass_relative_error': validation['mass_relative_error'],
                'constraint_rr_lapse': validation['constraint_rr_lapse_max'],
                'constraint_rr_adm': validation['constraint_rr_std_max'],
            })
            
            # Store validation details for first case
            if nr == 1000:  # Store details for typical case
                results['validation_details'] = validation
                
        except Exception as e:
            print(f"Warning: Physics validation failed: {e}")
            results.update({
                'physics_agreement': False,
                'phi_max_diff': float('nan'),
                'phi_relative_error': float('nan'), 
                'mass_relative_error': float('nan'),
                'constraint_rr_lapse': float('nan'),
                'constraint_rr_adm': float('nan'),
            })
    
    return results

def main(argv=None):
    ap = argparse.ArgumentParser(description="Fair apples-to-apples benchmark: lapse-first vs standard ADM")
    ap.add_argument("--nr", type=int, nargs="+", default=[400, 800, 1600, 3200], 
                    help="Grid sizes")
    ap.add_argument("--t-end", type=float, default=0.2, 
                    help="Evolution time (shorter for stability)")
    ap.add_argument("--dt", type=float, default=0.005, 
                    help="Time step")
    ap.add_argument("--matter-type", type=str, default="gaussian", 
                    choices=["gaussian", "vaidya"],
                    help="Matter model type")
    ap.add_argument("--matter-scale", type=float, default=0.005,
                    help="Matter amplitude (small for stability)")
    ap.add_argument("--validate", action="store_true", default=True,
                    help="Enable physics validation")
    ap.add_argument("--out-dir", type=str, default="./out")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("FAIR APPLES-TO-APPLES COMPARISON")
    print("Lapse-First vs Standard ADM Evolution")
    print("=" * 70)
    print(f"Matter type: {args.matter_type}")
    print(f"Evolution time: {args.t_end}")
    print(f"Time step: {args.dt}")
    print(f"Matter scale: {args.matter_scale}")
    print()
    
    rows = []
    validation_shown = False
    
    for i, n in enumerate(args.nr):
        print(f"Running benchmark {i+1}/{len(args.nr)}: N_r={n}...")
        
        result = fair_benchmark_case(
            nr=n, 
            t_end=args.t_end, 
            dt=args.dt, 
            matter_scale=args.matter_scale,
            matter_type=args.matter_type, 
            validate_physics=args.validate
        )
        rows.append(result)
        
        # Show speedup immediately
        speedup = result['speedup_x']
        print(f"  Speedup: {speedup:.1f}x  (Lapse: {result['lapse_first_time_s']:.3f}s, ADM: {result['standard_adm_time_s']:.3f}s)")
        
        # Show validation for first case
        if i == 0 and args.validate and not validation_shown:
            if result['physics_agreement']:
                print(f"  ✓ Physics validation passed (φ error: {result['phi_relative_error']:.2e})")
            else:
                print(f"  ⚠ Physics differences detected (φ error: {result['phi_relative_error']:.2e})")
                if 'validation_details' in result:
                    print("  Detailed validation:")
                    print_validation_summary(result['validation_details'])
            validation_shown = True
        print()
    
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"fair_benchmark_{args.matter_type}.csv"
    df.to_csv(csv_path, index=False)

    # Results summary
    print("=" * 70) 
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    display_cols = ['N_r', 'speedup_x', 'lapse_first_time_s', 'standard_adm_time_s']
    if args.validate:
        display_cols.extend(['physics_agreement', 'phi_relative_error'])
    
    print(df[display_cols].to_string(index=False, float_format='%.3f'))
    
    # Key insights
    avg_speedup = df['speedup_x'].mean()
    max_speedup = df['speedup_x'].max()
    print(f"\n🎯 KEY RESULTS:")
    print(f"   Average speedup: {avg_speedup:.1f}x")
    print(f"   Maximum speedup: {max_speedup:.1f}x") 
    
    if args.validate:
        agreement_rate = df['physics_agreement'].mean() * 100
        print(f"   Physics agreement: {agreement_rate:.0f}% of cases")
    
    print(f"\n📁 Detailed results: {csv_path}")
    
    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Runtime comparison
        ax1.loglog(df['N_r'], df['lapse_first_time_s'], 'o-', label='Lapse-first', linewidth=2)
        ax1.loglog(df['N_r'], df['standard_adm_time_s'], 's-', label='Standard ADM', linewidth=2)
        ax1.set_xlabel('Grid size N_r')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.set_title('Computational Cost Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Speedup 
        ax2.semilogx(df['N_r'], df['speedup_x'], 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('Grid size N_r')
        ax2.set_ylabel('Speedup (×)')  
        ax2.set_title('Lapse-First Computational Advantage')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plot_path = out_dir / f"fair_comparison_{args.matter_type}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plots saved: {plot_path}")
        
    except Exception as e:
        print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    main()