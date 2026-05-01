# Figure 4 — Block-Krylov RHS scaling: residual traces and per-RHS speedup

This document records the exact computation conditions behind
`fig4_rhs_scaling.{png,pdf}` and the physical / numerical
interpretation of the trends visible in the figure.

## 1. Purpose

Demonstrate the block-Krylov benefit on a single fixed problem
(sphere × material × $r_{\rm ve}$) by showing **two complementary
views** stacked vertically:

- **Row 1 (top): residual traces** $\|r_k\|_F / \|b\|_F$ vs iteration
  index $k$ for $L \in \{1, 16, 128\}$. Demonstrates *why* a larger
  block size $L$ speeds up the per-RHS cost: a richer Krylov subspace
  reaches any fixed residual threshold in fewer iterations.
- **Row 2 (bottom): per-RHS speedup** $t(L=1)/t(L)$ where $t(L)$ is the
  wall time per RHS to reach $\|r\|/\|b\|<10^{-3}$ (the relaxed Au
  gate also used by fig 2 / fig 3). The ideal amortisation line
  $y = L$ is overlaid as a reference.

The two-row layout is essential because the bottom-row speedup is a
single number per $(L, \text{slot})$ that does not by itself convey
the mechanism (fewer iterations to a given residual). The top row
makes the mechanism explicit so the reader can connect "$L=128$
speedup of $X\times$" with "$L=128$ reaches $10^{-3}$ in $Y$ iterations
vs $L=1$ needs $Z$ iterations".

## 2. Layout

**Single figure file**: `fig4_rhs_scaling.{png,pdf}`, **2 × 3 panels**
(`figsize=(11, 7.5)`, `gridspec_kw={'wspace': 0.10, 'hspace': 0.30}`,
square axis boxes via `ax.set_box_aspect(1.0)`, `sharey='row'` so
y-tick labels appear only on the leftmost column of each row).

Columns ordered by $|m_p|$:

- **Col 1**: n15 ($m_p = 1.5+0.01i$).
- **Col 2**: n20 ($m_p = 2.0+0.0i$).
- **Col 3**: Au ($m_p = 0.18+3.48i$).

### 2.1 Per-panel slot

Both rows of a column share a single $(\text{solver}, r_{\rm ve})$
slot for coherence:

| material | $(\text{solver}, r_{\rm ve})$ | top-row trace overlay | bottom-row speedup curves |
| --- | --- | --- | --- |
| n15 | (VIEM, 0.40 μm) | $L \in \{1, 16, 128\}$ | DDA + VIEM at $r_{\rm ve}=0.40$ |
| n20 | (VIEM, 0.40 μm) | $L \in \{1, 16, 128\}$ | DDA + VIEM at $r_{\rm ve}=0.40$ |
| Au  | (VIEM, 0.10 μm) | $L \in \{1, 16, 128\}$ | VIEM at $r_{\rm ve}=0.10$ only |

Per-panel choice rationale:

- For n15 / n20 the largest production size $r_{\rm ve}=0.40$ is
  chosen because the iteration count there is high enough that the
  block-Krylov amortisation is visible across the full $L$ range;
  smaller sizes converge in $\le 30$ iterations and the curves flatten.
- For Au the available production grid is $\{0.05, 0.10, 0.20\}$ μm.
  $r_{\rm ve}=0.10$ is chosen as the panel slot because (i) it is the
  largest size where the residual trace at $L=1$ both stalls in a
  visually meaningful way and is well above the $10^{-3}$ threshold
  early in the trace, making the L-dependent gain visible; (ii) the
  $r_{\rm ve}=0.05$ size is too small for block-Krylov amortisation to
  be visible (low iter count); (iii) DDA is excluded because every Au
  case stalls at MAXITER on every $L$ with residual above
  $10^{-3}$ for several L.

## 3. Top row — residual traces

### 3.1 Y-axis

$\|r_k\|_F / \|b\|_F$ on log scale, fixed range $10^{-6} \dots 10^{0}$.
Two horizontal dotted reference lines are drawn at $10^{-3}$
(`PHASE5_RESIDUAL_THRESHOLD`, the residual threshold used to define
the bottom-row speedup) and $10^{-5}$ (the production tolerance
`SOLVER_TOL`).

### 3.2 X-axis

Iteration index $k = 1 \dots 200$ (`MAXITER`), linear scale, fixed
range $0 \dots 200$ across all three panels.

### 3.3 Encoding

Three traces per panel for $L \in \{1, 16, 128\}$, distinguished by
linestyle (colour kept in the same gray family for a neutral palette):

| L | linestyle | colour |
| ---: | --- | --- |
| 1   | dotted (`:`)  | `'0.55'` (light gray) |
| 16  | dashed (`--`) | `'0.30'` (mid gray) |
| 128 | solid (`-`)   | `k` (black) |

Single legend on the leftmost panel.

### 3.4 Source

`target/rhs_scaling/gmres/residual_history` from
`sphere_<mat>.hdf5` (DDA: `dda_results/paper/`; VIEM:
`viem_results/paper/`). The HDF5 layout differs between the two
sides — `_plot_io.load_rhs_scaling` normalises both to
`(N_L, N_{r_v}, \text{MAXITER})` so the plotting code is
side-agnostic.

## 4. Bottom row — per-RHS speedup at $\|r\|/\|b\|<10^{-3}$

### 4.1 Definition

The starting point is the raw per-RHS wall time at block size $L$ as
recorded in the HDF5 sweep:

$$
t_L^{\rm raw} \;\equiv\; \frac{T_{\rm block}(L)}{L}
\;=\; \texttt{t\_end2end\_per\_orient\_s}.
$$

In words, $t_L^{\rm raw}$ is the **block size $L$ end-to-end wall
time per RHS** — the total observed wall time of the $L$-RHS block
solve (setup + iterations + post-processing, with iterations
dominant) divided by $L$. The "per_orient" suffix in the HDF5 field
name is a historical label tying each RHS to one orientation in the
production sweep; the quantity itself is exactly the wall time per
RHS at block size $L$.

The fig-4 bottom-row speedup uses a **threshold-corrected variant**
of this quantity, evaluated at a fixed residual $10^{-3}$ rather than
at solver termination:

$$
\text{speedup}(L) \;=\; \frac{t_{L=1}^{(10^{-3})}}{t_L^{(10^{-3})}},
\qquad
t_L^{(10^{-3})} \;=\; t_L^{\rm raw}
   \cdot \frac{k_L^{(10^{-3})}}{k_L^{\rm done}},
$$

where

- $T_{\rm block}(L)$ is the measured wall time of the $L$-RHS block
  solve.
- $k_L^{\rm done}$ is the iteration count at which the solve actually
  terminated (either at `SOLVER_TOL = 1e-5` or `MAXITER = 200`).
- $k_L^{(10^{-3})}$ is the iteration count at which
  $\|r_k\|/\|b\| = 10^{-3}$, log-linearly interpolated from
  `residual_history`.

Equivalently, $t_L^{(10^{-3})}$ is the wall time per RHS that the
solver would have needed to reach the $10^{-3}$ residual threshold,
under the (empirically <2 %) assumption that per-iteration block cost
is constant within a single solve.

**Summary**:

| symbol | meaning | source |
| --- | --- | --- |
| $t_L^{\rm raw}$ | block size $L$ end-to-end wall time per RHS, **at solver termination** ($10^{-5}$ or MAXITER, whichever first) | HDF5 `t_end2end_per_orient_s` |
| $t_L^{(10^{-3})}$ | same per-RHS wall time, but rescaled to "to reach $\|r\|/\|b\|<10^{-3}$" | $t_L^{\rm raw}\cdot k_L^{(10^{-3})}/k_L^{\rm done}$ |
| speedup($L$) | block-Krylov amortisation at common residual | $t_{L=1}^{(10^{-3})}/t_L^{(10^{-3})}$ |

### 4.2 Why $t_L^{(10^{-3})}$ and not the raw $t_L^{\rm raw}$

A naive speedup $t_{L=1}^{\rm raw}/t_L^{\rm raw}$ — which would be the
ratio of "block size $L$ end-to-end wall time per RHS" at production
termination — conflates two unrelated effects:

1. The block-Krylov benefit (fewer iterations to a given residual
   for larger $L$) — what we actually want to measure.
2. The arbitrary stopping condition (`SOLVER_TOL` vs `MAXITER`).
   Different $L$ at the same slot can land at very different residual
   levels — e.g. for sphere × Au at $r_{\rm ve}=0.10$:
   - $L=1$ stalls at MAXITER=200 with residual $6.0\times 10^{-4}$.
   - $L=128$ converges in 58 iterations with residual $3.6\times 10^{-6}$.
   The ratio of those two raw wall times credits $L=128$ for
   over-converging past $10^{-3}$ down to $10^{-6}$ — work the user
   never asked for.

Restricting the comparison to a common residual threshold $10^{-3}$
removes that confound. The threshold value matches
`PHASE5_RESIDUAL_THRESHOLD = 1e-3`, the same relaxed Au gate used in
fig 2 (VIEM $\ell_c$ convergence) and fig 3 (production scatter), so
the cross-references stay consistent across the paper.

**For users who want the raw end-to-end speedup** (e.g. for queue
planning where the production termination is what they actually
pay), the same `residual_history` and `t_end2end_per_orient_s`
fields suffice — divide $t_{L=1}^{\rm raw}$ by $t_L^{\rm raw}$
directly. The figure plots the threshold-corrected version because
it is the cleaner physical statement of "how much does block-Krylov
help".

### 4.3 Y-axis

Speedup factor $t(L=1)/t(L)$ on log scale, lower bound fixed at $0.5$
(sub-$1\times$ would mean $L$ hurts vs single-RHS — flagged
visually), upper bound from the data ceiling, log-rounded.

### 4.4 X-axis

$L \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ on log scale, with explicit
major-tick labels at every $L$ value (`FixedLocator` +
`ScalarFormatter`). Range fixed at
$L_{\min}\!\times\!0.85 \dots L_{\max}\!\times\!1.15$ across all three
bottom panels so the Au column (which only has $L \ge 32$ data after
the `solver_err` filter on raw points) shares the visual $L=1\dots128$
footprint with the n15 / n20 columns.

### 4.5 Encoding (grayscale)

Colour adds no information once the panel legend is in place, so the
figure stays in grayscale. The marker / linestyle / fill family
carries solver + $r_{\rm ve}$ identity:

| panel(s)  | curve                            | marker  | linestyle | colour |
| ---       | ---                              | ---     | ---       | ---    |
| n15 / n20 | DDA  $r_{\rm ve}=0.4$ μm         | ● filled circle | solid `-`  | `k` (black)    |
| n15 / n20 | VIEM $r_{\rm ve}=0.4$ μm         | □ open  square  | dashed `--`| `k` (black)    |
| Au        | VIEM $r_{\rm ve}=0.10$ μm        | □ open  square  | dashed `--`| `k` (black)    |
| all       | ideal speedup $y = L$            | (line)          | dotted `:` | `'0.55'` (gray)|

The dotted ideal $y=L$ line is the perfect-amortisation reference;
gap to the data marks the practical block-Krylov efficiency.

## 5. RHS-scaling sweep configuration

### 5.1 Particle slots

- **Shape**: sphere only (other shapes are not in the RHS-scaling
  sweep).
- **Materials**: $m_p \in \{1.5+0.01i,\ 2.0+0.0i,\ 0.17525+3.483i\}$
  (n15 / n20 / Au).
- **$r_{\rm ve}$**: $\{0.05, 0.10, 0.20, 0.40\}$ μm for n15 / n20;
  $\{0.05, 0.10, 0.20\}$ μm for Au (no $r_{\rm ve}=0.40$).
- **Wavelength**: $\lambda_0 = 0.638\,\mu$m, host $m_m = 1$.

The full grid is in the HDF5; the figure plots one slot per panel
(see §2.1).

### 5.2 Block sizes

$L \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ — 8 values per
$r_{\rm ve}$.

### 5.3 Orientations

Deterministic uniform-on-SO(3) Euler sequence with fixed RNG seed
(`numpy.random.default_rng(12345)`). Larger-$L$ blocks are nested
($L=1 \subset L=2 \subset L=4 \subset \cdots$) so the same
orientations recur across all $L$ values.

This is **different from fig 3's spheroid-mode grid**
($\alpha=\gamma=0$, $N_\beta=5$) — this figure deliberately samples
random orientations to measure generic block-Krylov scaling, not the
production sweep's $\alpha=\gamma=0$ spheroid block.

### 5.4 Solver

- **Method**: block-GMRES, **unpreconditioned on both sides**
  (`bl_krylov.bl_gmres_mvp_fft` on the DDA side,
  `BlockVIEM.block_gmres` with `:aim_gmres` on the VIEM side).
  Block-Hessenberg least-squares minimisation by incremental
  block-Givens QR (same as fig 1 / fig 2).
- **Tolerance**: $\|R\|_F / \|B\|_F < 10^{-5}$.
- **Maximum iterations**: 200.
- **Residual gate** (Au plasmonic): `PHASE5_RESIDUAL_THRESHOLD` =
  $10^{-3}$. The bottom-row speedup is defined via this threshold
  (see §4.1). Top-row residual traces show the threshold as a
  horizontal dotted line.
- **Discretisations**:
  - **DDA**: point-dipole basis, FFT-accelerated MVP (Goodman
    algorithm).
  - **VIEM**: volume integral on tetrahedral SWG basis,
    AIM-accelerated MVP.

## 6. What the figure shows

### 6.1 Top row (residual traces)

- **n15 r_ve=0.40 μm**: all three $L$ converge below $10^{-5}$ in
  $\le 55$ iterations; the L=1 trace is the slowest but still
  well-behaved.
- **n20 r_ve=0.40 μm**: $L=1$ very nearly stalls (still descending
  but cap'd at MAXITER=200 with residual just below $10^{-5}$);
  $L=128$ converges in $\sim 30$ iterations. The dramatic L-dependence
  here is the cleanest illustration of the block-Krylov benefit.
- **Au r_ve=0.10 μm**: $L=1$ and $L=16$ both stall at MAXITER=200
  with residuals between $10^{-3}$ and $10^{-4}$, slowly grinding
  past the relaxed gate; $L=128$ descends past $10^{-5}$ in 58
  iterations. The trace makes the qualitative argument that the
  Au-gate-pass requires a large enough $L$ to fully resolve the
  near-degenerate plasmonic eigenspace, not just more iterations.

### 6.2 Bottom row (speedup at $10^{-3}$)

- **n15 / n20**: monotonic speedup with $L$, climbing toward but
  remaining below the ideal $y=L$ line. DDA reaches a slightly higher
  speedup than VIEM at the same $L$ because DDA's per-iter MVP is
  cheaper (point-dipole FFT vs SWG AIM-FFT) so the orthog overhead is
  a larger fraction of the block solve.
- **Au**: similar speedup magnitude as n15 / n20 once the unified
  $10^{-3}$ definition is used. The earlier "raw end-to-end" speedup
  (deprecated in favour of the present definition) understated the
  Au amortisation because it credited $L=128$ for over-converging
  past the gate.

The headline message is unchanged: block-Krylov pays off across the
full material range, with no inversion at large $L$, so there is no
penalty for choosing $L$ aggressively up to $128$ — but the bottom
row now reads as a fair speedup metric tied to a single, paper-wide
residual threshold.

### 6.3 Deviation from the ideal $y=L$ line

The dominant cause of the gap between the measured speedup and the
ideal $y=L$ reference is **algorithmic, not hardware**. Decomposing
the per-iteration cost of block-GMRES,

$$
t_{\rm iter}(L) \;\approx\;
\underbrace{a\,L\,N\log N}_{\text{block-FFT MVP (linear)}}
\;+\;
\underbrace{b\,L^2\,k\,N}_{\text{block-Arnoldi orthog }(L^2)}
\;+\;
\underbrace{c\,L^3\,k}_{\text{block-Givens (small)}}
$$

where $k$ is the running Krylov dimension and $N$ the DOF count, the
per-RHS cost contains a term

$$
\frac{t_{\rm iter}(L)}{L} \;\supset\; b\,L\,k\,N
$$

that **grows linearly with $L$** even after dividing by $L$, while
the block-Krylov benefit only saves iterations sub-linearly
($k(L) \sim k(1)/\sqrt{L}$ heuristically, slower for plasmonic Au).
Concretely, going from $L=1$ to $L=128$ multiplies the orthog cost
per RHS by $\sim 128 \cdot k_{128}/k_{1}$. Even with $k_{128}/k_1
\approx 1/3$ (block-Krylov early termination), this is a
$\sim\!40\!\times$ orthog overhead, which is the principal mechanism
that compresses the measured speedup from the ideal $128$ to the
$30$–$50\times$ range observed.

Hardware effects are **secondary** on the present platform
(i9-14900K, 24 phys / 32 logical cores; bottom-row data taken with
FFTW threads $= 8$):

| effect | bearing on speedup |
| --- | --- |
| $L >$ CPU cores (e.g. $L=128$) | not directly — block-Krylov parallelism comes from FFT / BLAS thread pools, not from $L$ itself; $L$ is the algebraic block size, not the parallelism degree. |
| Block-vector data exceeding L3 cache (~30 MB) at large $L$ | shifts the MVP into a memory-bandwidth-bound regime; small additional flat factor in $t_{\rm iter}$. |
| FFTW plan-cache lock contention at large $L$ + threads | observed and discussed in `memo20260430-2.md §C5`; small additional factor on top of memory bandwidth. |

In short, **the slope of the speedup curve below $y=L$ is set by the
block-Arnoldi $L \cdot k \cdot N$ orthog cost** — a property of the
block-Krylov algorithm itself, not of the test hardware. Hardware
saturation contributes additional flat factors but does not change
the slope materially within the measured $L$ range.

## 7. Output

- `dda_results/paper/figures/fig4_rhs_scaling.{png,pdf}`
  (PNG dpi = 150, PDF vector).
- Generated by **Phase 5** of `plot_paper_results.ipynb`
  (cell id `fig6-cost`; the cell id is a historical name and is not
  updated to `fig4-cost` to preserve commit-history blame continuity).
- Top-row residual traces consume
  `target/rhs_scaling/gmres/residual_history` from the per-material
  HDF5 (DDA: `dda_results/paper/sphere_<mat>.hdf5`; VIEM:
  `~/Julia/block-VIEM.jl/viem_results/paper/sphere_<mat>.hdf5`).
- Bottom-row speedup uses both the same `residual_history` (to find
  $k_L^{(10^{-3})}$) and `t_end2end_per_orient_s` × `iters` (for
  per-iter wall-time scaling).
