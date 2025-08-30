
import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

from ..solver import TimeFirstGRSolver
from ..standard_ref import StandardEllipticSolver
from ..matter import GaussianPulse, VaidyaLikeNull
from ..validation import compare_physics_validation, print_validation_summary

def benchmark_case(nr=1000, t_end=1.0, dt=0.005, matter_scale=0.03,
                   r_pulse=12.0, sigma_r=2.0, t0=0.5, sigma_t=0.2, 
                   matter_type="gaussian", validate_physics=True):
    """
    Improved benchmark using unified matter models for fair comparison.
    
    Parameters:
    -----------
    matter_type : str
        "gaussian" for energy density pulse, "vaidya" for null dust
    validate_physics : bool  
        Whether to perform detailed physics validation
    """
    
    # Create unified matter model
    if matter_type == "gaussian":
        # Gaussian energy density pulse - same physics for both solvers
        matter = GaussianPulse(
            amplitude=matter_scale,
            r_center=r_pulse, 
            r_width=sigma_r,
            t_center=t0,
            t_width=sigma_t,
            velocity=0.0  # Matter at rest
        )
    elif matter_type == "vaidya":
        # Null dust - ensures identical T_tr and T_tt=T_rr=0 for both
        norm = matter_scale/(np.sqrt(2*np.pi)*sigma_t)
        L = lambda t: norm*np.exp(-0.5*((t - t0)/sigma_t)**2)
        matter = VaidyaLikeNull(L, r_min=5.0, direction="ingoing")
    else:
        raise ValueError(f"Unknown matter_type: {matter_type}")

    # Lapse-first solver (with boundary enforcement for fair comparison)
    S = TimeFirstGRSolver(r_min=2.2, r_max=80.0, nr=nr, G=1.0, c=1.0, 
                         enforce_boundaries=True)
    S.set_static_schwarzschild(M=1.0)
    S.set_matter_model(matter)

    t0_lapse = time.perf_counter()
    S.run(t_end=t_end, dt=dt, progress=False)
    lapse_time = time.perf_counter() - t0_lapse

    # Standard elliptic solver  
    Std = StandardEllipticSolver(r_min=2.2, r_max=80.0, nr=nr, G=1.0, c=1.0)
    Std.set_static_schwarzschild(M=1.0)
    
    # Use same matter model for consistency
    def T_tt_unified(t, rarr):
        return matter.T_tt(t, rarr, c=1.0)

    t0_std = time.perf_counter()
    Std.run(t_end=t_end, dt=dt, T_tt=T_tt_unified)
    std_time = time.perf_counter() - t0_std

    steps = int(np.ceil(t_end/dt))
    i_near = nr//16
    
    # Basic results
    results = {
        "N_r": nr, "steps": steps, "dt": dt, "t_end": t_end,
        "matter_type": matter_type,
        "lapse_first_time_s": lapse_time, "standard_time_s": std_time,
        "speedup_x": (std_time / lapse_time) if lapse_time > 0 else float("nan"),
        "phi_near_lapse": float(S.Phi[i_near]),
        "phi_near_standard": float(Std.Phi[i_near]),
        "A_near_lapse": float(np.exp(2*S.Phi[i_near])),
        "A_near_standard": float(np.exp(2*Std.Phi[i_near])),
    }
    
    # Physics validation
    if validate_physics:
        try:
            validation = compare_physics_validation(S, Std)
            results.update({
                'physics_agreement': validation['physics_agreement'],
                'phi_max_diff': validation['phi_max_abs_diff'],
                'phi_relative_error': validation['phi_relative_error'],
                'mass_relative_error': validation['mass_relative_error'],
                'constraint_rr_lapse': validation['constraint_rr_lapse_max'],
                'constraint_tt_std': validation['constraint_tt_std_max'],
            })
        except Exception as e:
            print(f"Warning: Physics validation failed: {e}")
            results.update({
                'physics_agreement': False,
                'phi_max_diff': float('nan'),
                'phi_relative_error': float('nan'),
                'mass_relative_error': float('nan'),
                'constraint_rr_lapse': float('nan'),
                'constraint_tt_std': float('nan'),
            })
    
    return results

def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark lapse-first vs standard GR stepper")
    ap.add_argument("--nr", type=int, nargs="+", default=[400, 800, 1600], help="Grid sizes")
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--matter-type", type=str, default="gaussian", 
                    choices=["gaussian", "vaidya"], 
                    help="Matter model type")
    ap.add_argument("--validate", action="store_true", default=True,
                    help="Enable physics validation")
    ap.add_argument("--out-dir", type=str, default="./out")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n=== timefirst-gr Benchmark ({args.matter_type} matter) ===")
    rows = []
    for i, n in enumerate(args.nr):
        print(f"Running benchmark {i+1}/{len(args.nr)}: N_r={n}...")
        result = benchmark_case(nr=n, t_end=args.t_end, dt=args.dt, 
                               matter_type=args.matter_type, 
                               validate_physics=args.validate)
        rows.append(result)
        
        # Show validation for first case
        if i == 0 and args.validate and result['physics_agreement']:
            print("✓ Physics validation passed for first case")
        elif i == 0 and args.validate:
            print("⚠ Physics validation concerns - see detailed results")
    
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"benchmark_results_{args.matter_type}.csv"
    df.to_csv(csv_path, index=False)

    # Console table with key metrics
    display_cols = ['N_r', 'speedup_x', 'lapse_first_time_s', 'standard_time_s']
    if args.validate:
        display_cols.extend(['physics_agreement', 'phi_relative_error'])
    
    print(f"\nResults summary:")
    print(df[display_cols].to_string(index=False, float_format='%.3f'))
    print(f"\nSaved detailed CSV -> {csv_path}")

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["N_r"], df["lapse_first_time_s"], marker="o", label="Lapse-first")
        plt.plot(df["N_r"], df["standard_time_s"], marker="o", label="Standard")
        plt.xlabel("N_r"); plt.ylabel("Wall time (s)"); plt.title("Runtime vs Grid Size"); plt.legend()
        fig1 = out_dir / "runtime_vs_grid.png"; plt.savefig(fig1, dpi=160, bbox_inches="tight")

        plt.figure()
        plt.plot(df["N_r"], df["speedup_x"], marker="o")
        plt.xlabel("N_r"); plt.ylabel("Speedup (×)"); plt.title("Speedup: Standard / Lapse-first")
        fig2 = out_dir / "speedup_vs_grid.png"; plt.savefig(fig2, dpi=160, bbox_inches="tight")
        print(f"Saved plots -> {fig1}, {fig2}")
    except Exception as e:
        print("Plotting skipped:", e)

if __name__ == "__main__":
    main()
