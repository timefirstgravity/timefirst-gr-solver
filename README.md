
# timefirst-gr

A minimal, fast **1+1D spherical GR solver** in a **lapse-first** gauge designed to make the
computational benefits immediately measurable. It includes:

- `TimeFirstGRSolver` — explicit O(N) update of the lapse potential Φ via the flux law  
  \( \partial_t \Phi = -4\pi G\, r \, T_{tr} / c^4 \).
- A reference **standard** solver that performs a **nonlinear elliptic solve** per time step using
  the \(G_{tt}\) constraint (Newton + tridiagonal), representative of traditional pipelines.
- A **benchmark CLI** that prints wall-clock time for both methods and reports **speedup (×)**.
- Examples: proper-time comparison, redshift drift, and light-travel (Shapiro-style) delay.

## Install (editable)

```bash
pip install -e .
```

## Quick start

Run the benchmark on a few grid sizes:

```bash
gr-bench --nr 400 800 1600 --t-end 1.0 --dt 0.005 --out-dir ./out
```

You will get a CSV and PNG plots showing wall time and speedup Standard/Lapse-first.

Or run the example:

```bash
gr-example
```

which prints proper-time per unit coordinate time at two radii, a redshift factor, and
a radial light-travel time.

## Model

Metric (time-first gauge):
\[ ds^2 = -e^{2\Phi(t,r)} dt^2 + e^{-2\Phi(t,r)} dr^2 + r^2 d\Omega^2. \]

Evolution (mixed Einstein equation):
\[ \partial_t \Phi = -4\pi G\, r\, T_{tr} / c^4. \]

Radial constraints (diagnostics):
\[
G_{rr} = \frac{2\,\partial_r\Phi}{r} + \frac{1}{r^2} - \frac{1}{A r^2},\qquad
G_{tt} = \frac{A}{r^2}\Big(-2 r A \partial_r \Phi - A + 1\Big),\quad A=e^{2\Phi}.
\]

## License
MIT
