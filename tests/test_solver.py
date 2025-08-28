
import numpy as np
from timefirst_gr.solver import TimeFirstGRSolver

def test_redshift_monotone():
    S = TimeFirstGRSolver(r_min=3.0, r_max=80.0, nr=500)
    S.set_static_schwarzschild(M=1.0)
    # pick three radii
    i1, i2 = 50, 400
    z = S.redshift_factor(i_emit=i1, i_obs=i2)
    assert np.isfinite(z)

def test_vacuum_constraints_small():
    S = TimeFirstGRSolver(r_min=3.0, r_max=80.0, nr=500)
    S.set_static_schwarzschild(M=1.0)
    S.set_vacuum()
    res_rr, res_tt = S.constraints_residuals()
    assert np.isfinite(res_rr).all() and np.isfinite(res_tt).all()
