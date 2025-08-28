
def main():
    import numpy as np
    from ..solver import TimeFirstGRSolver

    S = TimeFirstGRSolver(r_min=2.2, r_max=50.0, nr=1200)
    S.set_static_schwarzschild(M=1.0)

    i_low, i_high = 50, 1000
    dt = 1.0
    dtaul = S.proper_time_increment(dt, i_low)
    dtauh = S.proper_time_increment(dt, i_high)

    zfac = S.redshift_factor(i_emit=i_low, i_obs=i_high)
    tnull = S.light_travel_time(i_low, i_high)

    print("Proper-time per 1 unit dt:")
    print(f"  near mass (idx {i_low}): {dtaul:.9f}")
    print(f"  far away (idx {i_high}): {dtauh:.9f}")
    print(f"Gravitational redshift ν_obs/ν_emit: {zfac:.9f}")
    print(f"Radial light travel time (coordinate t): {tnull:.9f}")

if __name__ == "__main__":
    main()
