# Figure 4 — Block-Krylov RHS scaling: per-RHS cost vs block size $L$

This document records the exact computation conditions behind
`fig4_rhs_scaling.{png,pdf}` and the physical / numerical
interpretation of the trends visible in the figure.

## 1. Purpose

For a fixed particle (sphere × material × $r_{\rm ve}$) measure how the
**per-RHS solving cost** [s] depends on the block size $L$ (number of
right-hand sides solved together by block-GMRES). The per-RHS cost is

$$
t_{\rm per\text{-}RHS}(L) \;=\; \frac{T_{\rm block}(L)}{L}
\;=\; t_{\rm end2end\_per\_orient\_s},
$$

i.e. the total block solving time for the $L$ RHSs divided by $L$. A
flat curve means no amortisation; a downward trend with $L$ is the
block-Krylov benefit.

Plotting this single derived quantity (instead of the prior 2-row layout
with iterations + time per orientation) is the central editorial choice
of this figure: the iteration count and the wall time are essentially
two views of the same `(iter × per-iter-cost)` quantity, and the
per-RHS cost is the most paper-relevant scalar — it is the units in
which the downstream Bayesian retrieval consumes solver throughput.

## 2. Layout

**Single figure file**: `fig4_rhs_scaling.{png,pdf}`, **1 × 3 panels**
with square axis boxes (`ax.set_box_aspect(1.0)`, `figsize=(11, 4.2)`,
`gridspec_kw={'wspace': 0.10}` to tighten the column gap).

Columns ordered by $|m_p|$:

- **Col 1**: n15 ($m_p = 1.5+0.01i$) — DDA + VIEM, $r_{\rm ve} = 0.40$ μm.
- **Col 2**: n20 ($m_p = 2.0+0.0i$) — DDA + VIEM, $r_{\rm ve} = 0.40$ μm.
- **Col 3**: Au ($m_p = 0.18+3.48i$) — VIEM only,
  $r_{\rm ve} \in \{0.05,\,0.10\}$ μm. Au has no $r_{\rm ve}=0.40$ case
  in the production grid.

Per-panel choice rationale:

- For n15 / n20 the largest production size $r_{\rm ve}=0.40$ is
  chosen because the iteration count there is high enough that the
  block-Krylov amortisation is visible across the full $L$ range.
  Smaller sizes converge in $\le 30$ iterations and the curves are
  essentially flat.
- For Au the available production grid is $\{0.05,\,0.10,\,0.20\}$.
  $r_{\rm ve}=0.20$ is dropped to keep the panel uncluttered; the
  remaining two sizes give the size-dependence direction without
  overplotting. DDA is excluded because every Au case stalls at
  MAXITER on every $L$ with residual above PHASE5_RESIDUAL_THRESHOLD
  ($10^{-3}$, the relaxed gate also used by fig 2 / fig 3 wherever
  Au plasmonic stalls appear).

Per-panel inset legend (`ax.legend(loc='upper right', frameon=False)`)
because each panel displays a different combination of (solver,
$r_{\rm ve}$) — there is no single colour key across the figure.

## 3. Y-axis

Solving time per RHS [s], **log y axis**, **fixed lower bound at
$10^{-2}$ s** (sub-$10\,$ms values cluster on the per-iteration noise
floor and would otherwise dominate the visible y-range, hiding the
iter-count-driven structure of the slower curves). Upper bound from
the data ceiling, log-rounded.

X-axis: $L \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ on log scale, with
explicit major-tick labels at the $L$ values (`FixedLocator` +
`ScalarFormatter`).

## 4. Encoding (grayscale)

Colour adds no information once the panel legend is in place, so the
figure is rendered in grayscale. The marker / linestyle / fill family
carries solver + $r_{\rm ve}$ identity:

| panel(s)  | curve                            | marker  | linestyle | colour |
| ---       | ---                              | ---     | ---       | ---    |
| n15 / n20 | DDA  $r_{\rm ve}=0.4$ μm         | ● filled circle | solid `-`  | `k` (black)    |
| n15 / n20 | VIEM $r_{\rm ve}=0.4$ μm         | □ open  square  | dashed `--`| `k` (black)    |
| Au        | VIEM $r_{\rm ve}=0.05$ μm        | ○ open  circle  | dashed `--`| `0.55` (gray)  |
| Au        | VIEM $r_{\rm ve}=0.10$ μm        | □ open  square  | dashed `--`| `k` (black)    |

The gray shade for Au $r_{\rm ve}=0.05$ separates it from the
$r_{\rm ve}=0.10$ curve at a glance without needing colour.

## 5. RHS-scaling sweep configuration

### 5.1 Particle slots

- **Shape**: sphere only (other shapes are not in the RHS-scaling
  sweep).
- **Materials**: $m_p \in \{1.5+0.01i,\ 2.0+0.0i,\ 0.17525+3.483i\}$
  (n15 / n20 / Au).
- **$r_{\rm ve}$**: $\{0.05, 0.10, 0.20, 0.40\}$ μm for n15 / n20;
  $\{0.05, 0.10, 0.20\}$ μm for Au (no $r_{\rm ve}=0.40$).
- **Wavelength**: $\lambda_0 = 0.638\,\mu$m, host $m_m = 1$.

The full grid is in the HDF5; the figure plots a chosen subset
(see §2).

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
- **Tolerance**: $\|R\|_F / \|B\|_F < 10^{-5}$.
- **Maximum iterations**: 200.
- **Residual gate** (Au plasmonic): `PHASE5_RESIDUAL_THRESHOLD` =
  $10^{-3}$. Curves on Au panels are plotted only at $L$ values where
  the solver either reached the production tolerance or has residual
  $\le 10^{-3}$. This matches the analogous gates in fig 2 (VIEM
  $\ell_c$ convergence) and fig 3 (production scatter).
- **Discretisations**:
  - **DDA**: point-dipole basis, FFT-accelerated MVP (Goodman
    algorithm).
  - **VIEM**: volume integral on tetrahedral SWG basis,
    AIM-accelerated MVP.

## 6. What the figure shows

- **n15 / n20 (real-index)**: per-RHS cost drops monotonically with
  $L$, by roughly a decade between $L=1$ and $L=128$. DDA curves sit
  about a decade below VIEM at every $L$, reflecting the
  point-dipole MVP being cheaper than the SWG-basis MVP.
- **Au (plasmonic)**: a noticeably steeper drop with $L$ — block-
  Krylov amortisation is more pronounced when the iteration count is
  the dominant cost driver. The two $r_{\rm ve}$ sizes track each
  other, with the smaller $r_{\rm ve}=0.05$ curve a fixed factor
  below $r_{\rm ve}=0.10$ (smaller mesh = cheaper per iteration).

The headline message is that block-Krylov pays off across the full
material range — the per-RHS cost flattens but does not invert at
large $L$, so there is no penalty for choosing $L$ aggressively up
to the largest tested value of $128$.

## 7. Output

- `dda_results/paper/figures/fig4_rhs_scaling.{png,pdf}`
  (PNG dpi = 150, PDF vector).
- Generated by **Phase 5** of `plot_paper_results.ipynb`
  (cell id `fig6-cost`; the cell id is a historical name and is not
  updated to `fig4-cost` to preserve commit-history blame continuity).
