
import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

from ..solver import TimeFirstGRSolver
from ..standard_ref import StandardEllipticSolver

def benchmark_case(nr=1000, t_end=1.0, dt=0.005, matter_scale=0.03,
                   r_pulse=12.0, sigma_r=2.0, t0=0.5, sigma_t=0.2):
    # Lapse-first (flux-driven)
    S = TimeFirstGRSolver(r_min=2.2, r_max=80.0, nr=nr, G=1.0, c=1.0)
    S.set_static_schwarzschild(M=1.0)
    norm = matter_scale/(np.sqrt(2*np.pi)*sigma_t)
    L = lambda t: norm*np.exp(-0.5*((t - t0)/sigma_t)**2)
    S.set_null_dust(L, r0=5.0, direction="ingoing")

    t0_lapse = time.perf_counter()
    S.run(t_end=t_end, dt=dt, progress=False)
    lapse_time = time.perf_counter() - t0_lapse

    # Standard (elliptic per step)
    Std = StandardEllipticSolver(r_min=2.2, r_max=80.0, nr=nr, G=1.0, c=1.0)
    r = Std.r
    def T_tt(t, rarr):
        spatial = np.exp(-0.5*((rarr - r_pulse)/sigma_r)**2)
        temporal = np.exp(-0.5*((t - t0)/sigma_t)**2)
        E0 = matter_scale/(np.sqrt(2*np.pi)*sigma_t)
        return E0 * spatial * temporal

    t0_std = time.perf_counter()
    Std.run(t_end=t_end, dt=dt, T_tt=T_tt)
    std_time = time.perf_counter() - t0_std

    steps = int(np.ceil(t_end/dt))
    i_near = nr//16
    return {
        "N_r": nr, "steps": steps, "dt": dt, "t_end": t_end,
        "lapse_first_time_s": lapse_time, "standard_time_s": std_time,
        "speedup_x": (std_time / lapse_time) if lapse_time > 0 else float("nan"),
        "ephi_near_lapse": float(np.exp(S.Phi[i_near])),
        "ephi_near_standard": float(np.exp(Std.Phi[i_near])),
    }

def main(argv=None):
    ap = argparse.ArgumentParser(description="Benchmark lapse-first vs standard GR stepper")
    ap.add_argument("--nr", type=int, nargs="+", default=[400, 800, 1600], help="Grid sizes")
    ap.add_argument("--t-end", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--out-dir", type=str, default="./out")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = [benchmark_case(nr=n, t_end=args.t_end, dt=args.dt) for n in args.nr]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    # Simple console table
    print("\n=== timefirst-gr Benchmark ===")
    print(df.to_string(index=False))
    print(f"\nSaved CSV -> {csv_path}")

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
