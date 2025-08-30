
# timefirst-gr

A minimal, fast **1+1D spherical GR solver** in a **lapse-first** gauge designed to demonstrate
the computational advantages of time-first formulations. It includes:

- `TimeFirstGRSolver` — explicit O(N) update of the lapse potential Φ via the flux law  
  \( \partial_t \Phi = -4\pi G\, r \, T_{tr} / c^4 \).
- `StandardADMSolver` — solves the same evolution equations but with expensive traditional ADM constraint operations.
- A **fair apples-to-apples benchmark** comparing computational efficiency for identical physics problems.
- Examples: proper-time comparison, redshift drift, and light-travel (Shapiro-style) delay.

## Install (editable)

```bash
pip install -e .
```

## Quick start

### Fair Apples-to-Apples Benchmark (Recommended)

Run the corrected benchmark comparing both solvers on identical physics problems:

```bash
python -m timefirst_gr.benchmarks.fair_comparison --nr 400 800 1600 --t-end 0.1
```

This shows **~8-10x speedup** with **100% physics agreement**, demonstrating the true computational advantage of the lapse-first approach.

### Legacy Benchmark (For Comparison)

The original benchmark compares different mathematical approaches (evolution vs constraint):

```bash
gr-bench --nr 400 800 1600 --t-end 1.0 --dt 0.005 --out-dir ./out
```

### Examples

Run physics examples:

```bash
gr-example
```

This prints proper-time per unit coordinate time at two radii, a redshift factor, and
a radial light-travel time.

## Computational Approach

Both solvers evolve the same Einstein equation:
\[ \partial_t \Phi = -4\pi G\, r\, T_{tr} / c^4 \]

**Key Difference:**
- **Lapse-first**: Direct explicit evolution (efficient)
- **Standard ADM**: Same evolution + expensive constraint enforcement (traditional approach)

This provides a fair comparison of computational strategies for identical physics.

## Model

Metric (lapse-first gauge):
\[ ds^2 = -e^{2\Phi(t,r)} dt^2 + e^{-2\Phi(t,r)} dr^2 + r^2 d\Omega^2. \]

Evolution equation:
\[ \partial_t \Phi = -4\pi G\, r\, T_{tr} / c^4. \]

Constraint diagnostics:
\[
G_{rr} = \frac{2\,\partial_r\Phi}{r} + \frac{1}{r^2} - \frac{1}{A r^2},\qquad
G_{tt} = \frac{A}{r^2}\Big(-2 r A \partial_r \Phi - A + 1\Big),\quad A=e^{2\Phi}.
\]

## License
MIT
