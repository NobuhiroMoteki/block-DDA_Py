# Figure 1 — dpl / lc convergence study (sphere at $r_{\rm ve}=0.1\,\mu$m, single orientation)

This document records the exact computation conditions behind
`fig1_dpl_lc_convergence.{png,pdf}` in `dda_results/paper/figures/`.

## 1. Purpose

Quantify how the DDA and VIEM solvers' relative error against the **Mie
exact reference** (sphere only) shrinks as the spatial discretization is
refined. The x-axis is the number of volume elements ($N_{\rm occ}$ for
DDA, $n_{\rm tet}$ for VIEM); the y-axis is $|X - X^{\rm Mie}| /
|X^{\rm Mie}|$ for four observables. Used to certify that both solvers
converge to Mie at the expected polynomial-in-h rates and to bracket the
discretization error of the production sweep (fig 3).

## 2. Layout

- **One figure file**: `fig1_dpl_lc_convergence.{png,pdf}`.
- 1 × 3 panels = 3 materials (n15, n20, Au).
- Within each panel: 4 observables × 2 solvers = up to 8 lines (Au panel: only VIEM lines; n20 panel omits $Q_{\rm abs}$ since $m_p$ is purely real).
- `sharex=True, sharey=True`: identical axis range across panels.

## 3. Convergence-study configuration

### 3.1 Particle slot

- Shape: **sphere only**. Other shapes' convergence files (`convergence_oblate_*.hdf5`, `convergence_gre_*.hdf5`) exist but lack a Mie reference and are not plotted in fig 1.
- Materials: $m_p \in \{1.5+0.01i,\ 2.0+0.0i,\ 0.17525+3.483i\}$ (n15 / n20 / Au).
- $r_{\rm ve} = 0.10\,\mu$m **fixed** — single representative size per CLAUDE.md §4.
- Wavelength: $\lambda_0 = 0.638\,\mu$m, host $m_m = 1$.

### 3.2 Orientation

**$L = 1$, single orientation = ZYZ identity** $(\alpha, \beta, \gamma) = (0, 0, 0)$.

[`scripts/run_dpl_convergence.py:51`](../../scripts/run_dpl_convergence.py):
```python
SINGLE_ORIENT = np.array([[0.0, 0.0, 0.0]])  # ZYZ identity
```

[`viem_results/paper/run_lc_convergence.jl:47`](../../../../Julia/block-VIEM.jl/viem_results/paper/run_lc_convergence.jl):
```julia
const SINGLE_ORIENT = (0.0, 0.0, 0.0)        # ZYZ, identity rotation
```

This means the incidence direction (lab $+z$) coincides with the body
$+z$ axis. For sphere this is ostensibly orientation-independent, but
the **discretized cubic lattice on DDA side** and the **unstructured
tetrahedral mesh on VIEM side** both break exact rotational symmetry,
so the choice of orientation does affect convergence. In particular the
Au plasmonic case has the cubic lattice's $C_4$ axes coincide with the
incidence polarization axes, near-degenerating the LSP modes — see
§5 for the consequence.

### 3.3 Discretization sweep parameters

| Side | Parameter | Values | Mechanism |
| --- | --- | --- | --- |
| DDA  | `dpl` (dipoles per wavelength) | $\{10, 14, 17, 24, 34\}$ | explicit, fixed list |
| VIEM | `lc_factor` (× `adaptive_lc`)  | $\{1.5, 1.0, 0.7, 0.5, 0.35\}$ | factor × auto base |

[`scripts/run_dpl_convergence.py:50`](../../scripts/run_dpl_convergence.py):
```python
DPL_LIST = [10, 14, 17, 24, 34]
```

[`viem_results/paper/run_lc_convergence.jl:44-46`](../../../../Julia/block-VIEM.jl/viem_results/paper/run_lc_convergence.jl):
```julia
const LC_FACTORS = haskey(ENV, "LC_FACTORS") ?
                    parse.(Float64, split(ENV["LC_FACTORS"], ',')) :
                    [1.5, 1.0, 0.7, 0.5, 0.35]
```

Coarse → fine (left → right).

#### DDA "fixed dpl" (this study only)

The dpl convergence study uses **explicit dpl values** rather than the
`auto` (`dpl_for_slot`, fig 3) or `default` (`dpl=17`, fig 2) mechanisms
— it is its own third mode of choosing the lattice, sweeping
deliberately to map out the convergence curve. Each dpl gives lattice
spacing
$$
d \;=\; \frac{\lambda_0}{|m_p|_{\max}\cdot \text{dpl}}.
$$
The list is centered on dpl = 17 (= the production-sweep DDA "default" /
near the typical "auto" output), with two coarser and two finer values
spanning roughly an order of magnitude in $N_{\rm occ}$.

#### VIEM "lc factor × adaptive_lc"

The base `adaptive_lc(p; wl_0, m_p_max, N_pw=10)` is the same function
used in fig 3 / fig 2 production (see [`fig3_description.md §3.4`](fig3_description.md)):
$$
\text{lc}_{\rm base} \;=\; \min\!\Bigl(
    \tfrac{\lambda_0}{|m_p|_{\max}\cdot 10},\;
    \tfrac{c}{3},\;
    [\tfrac{0.3\,c}{3}\ \text{if } \beta_{\rm gre}>0]
\Bigr).
$$
The convergence sweep multiplies this base by `lc_factor`:
$$
\text{lc}_{\rm sweep} \;=\; \text{lc}_{\rm factor}\cdot \text{lc}_{\rm base}.
$$
`lc_factor = 1.0` reproduces the production mesh; `< 1.0` refines it,
`> 1.0` coarsens.

### 3.4 Solver

- **Method**: block-GMRES, unpreconditioned (same as fig 3 and fig 2).
  - DDA: `bl_krylov.bl_gmres_mvp_fft`
  - VIEM: `BlockVIEM.block_gmres` (`:aim_gmres` variant)
- **Tolerance**: $\|R\|_F / \|B\|_F < 10^{-5}$.
- **Maximum iterations**: 200 (raised from 100 to 200 in v0.7.6 for plasmonic Au headroom — see comment in `scripts/run_dpl_convergence.py:48`).

### 3.5 Mie reference

For each (material, $r_{\rm ve}$), `analytical_scattering_theories.homogeneous_sphere.mie_compute_q_and_s`
gives $(Q_{\rm sca}, Q_{\rm abs}, Q_{\rm ext}, S_{\rm fw}, S_{\rm bk})$
for the homogeneous sphere. These scalars are stored under
`/target/reference/{Q_ext_mie, Q_abs_mie, Q_sca_mie, S_fw_mie, S_bk_mie}`
in each `convergence_sphere_<mat>.hdf5`.

Since $r_{\rm ve}=0.1\,\mu$m is fixed and sphere is rotation-invariant
(at the continuum level), one Mie scalar per (material, observable) is
sufficient.

### 3.6 Observables on the y-axis

Per `CONV_OBS` in the Phase 4 cell of `plot_paper_results.ipynb`:

| key | latex | kind | n20 |
| --- | --- | --- | --- |
| `Q_ext`      | $Q_{\rm ext}$ | real | shown |
| `Q_abs`      | $Q_{\rm abs}$ | real | **omitted** ($m_p$ purely real → $Q_{\rm abs}$ not meaningful) |
| `S_fw_theta` | $\|S_{\rm fw}^{\theta}\|$ | mag | shown |
| `S_bk`       | $\|S_{\rm bk}\|$ | mag | shown |

Relative error: $\varepsilon = |X - X^{\rm Mie}| / |X^{\rm Mie}|$.

## 4. Per-panel display

- **DDA**: filled circle, solid line, color by observable (C0 = $Q_{\rm ext}$, C1 = $Q_{\rm abs}$, C2 = $|S_{\rm fw}^{\theta}|$, C3 = $|S_{\rm bk}|$).
- **VIEM**: open square, dashed line, same color per observable.
- Horizontal guide lines at $\varepsilon = 10^{-2}$ and $10^{-3}$ (gray dotted).
- **Production sweep marker** (added 2026-04-29): a thin gray vertical band per panel, centered on the geometric mean of (DDA $N_{\rm occ}^{\rm prod}$, VIEM $n_{\rm tet}^{\rm prod}$) at $r_{\rm ve}=0.1$, with fixed log-decade half-width = 0.075 dec so visual width is consistent across panels. Both DDA (`dpl_for_slot` auto) and VIEM (`adaptive_lc`) chose the production lattice automatically per slot.
- Inside-pointing mirror ticks on all four sides (added 2026-04-29).
- Both axes log-scaled.

## 5. Au stagnation (axis-aligned plus cubic lattice)

For sphere × Au, **all 5 dpl values stall at MAXITER=200** with
relative residual $\approx 10^{-3}$:

| dpl | $N_{\rm occ}$ | iters | converged | $\|R\|_F/\|B\|_F$ at exit |
| --- | --- | --- | --- | --- |
| 10  |   696  | 200 | 0 | $1.0\times 10^{-3}$ |
| 14  |  1869  | 200 | 0 | $2.6\times 10^{-3}$ |
| 17  |  3378  | 200 | 0 | $2.2\times 10^{-3}$ |
| 24  |  9454  | 200 | 0 | $\sim 10^{-3}$ |
| 34  | 26866  | 200 | 0 | $\sim 10^{-3}$ |

→ All observables are NaN → **the Au panel of fig 1 has no DDA points**.

VIEM converges at every lc factor at this orientation, so the Au panel
shows only VIEM curves.

The cause is *not* a method-vs-method issue but the combination of:
1. Au's LSP near-resonance pushing the operator spectrum to near-marginal eigenvalues;
2. Single-orientation $L = 1$ (no block-Krylov subspace amortization);
3. **Axis-aligned ZYZ-identity orientation** — incidence along $+z$ coincides with the cubic lattice's $\hat{z}$ principal axis. The lattice's $C_4$ rotational symmetry combines with the LSP modes' polarization structure to produce near-degenerate operator eigenvalues.

A non-axis-aligned orientation (e.g. the first uniform-on-SO(3) draw used by `run_rhs_scaling.py`) breaks the symmetry and would let DDA converge at most or all dpl values for Au; this is *not* part of the current saved data and is left for future runs (see [`fig2_description.md` §9](fig2_description.md) for the practical implications).

## 6. Convergence rate and scaling laws

The relative error vs lattice resolution on log-log axes (fig 1 itself)
exposes a clear **asymmetry between DDA and VIEM** in the empirical
power-law slope $s$ of $\varepsilon \propto N_{\rm vol}^{\,s}$:

| Solver | Material | Observable | Slope $s$ | $N$ range | $\varepsilon$ range |
| --- | --- | --- | ---: | --- | --- |
| DDA  | n15 | $Q_{\rm ext}$           | $-0.24$ | $57$ … $2\,175$    | $1.0\%$ – $2.5\%$ |
| DDA  | n15 | $Q_{\rm abs}$           | $-0.15$ | $57$ … $2\,175$    | $0.9\%$ – $1.8\%$ |
| DDA  | n15 | $|S_{\rm fw}^{\theta}|$ | $-0.18$ | $57$ … $2\,175$    | $0.4\%$ – $0.7\%$ |
| DDA  | n15 | $|S_{\rm bk}|$          | $-0.19$ | $57$ … $2\,175$    | $0.4\%$ – $1.1\%$ |
| DDA  | n20 | $Q_{\rm ext}$           | $-0.27$ | $127$ … $5\,081$   | $2.3\%$ – $6.3\%$ |
| DDA  | n20 | $|S_{\rm fw}^{\theta}|$ | $-0.13$ | $127$ … $5\,081$   | $0.3\%$ – $0.7\%$ |
| DDA  | n20 | $|S_{\rm bk}|$          | $-0.28$ | $127$ … $5\,081$   | $2.9\%$ – $8.0\%$ |
| VIEM | n15 | $Q_{\rm ext}$           | $-0.62$ | $243$ … $12\,349$  | $0.08\%$ – $0.85\%$ |
| VIEM | n15 | $Q_{\rm abs}$           | $-0.64$ | $243$ … $12\,349$  | $0.14\%$ – $1.7\%$ |
| VIEM | n15 | $|S_{\rm fw}^{\theta}|$ | $-0.63$ | $243$ … $12\,349$  | $0.10\%$ – $1.1\%$ |
| VIEM | n15 | $|S_{\rm bk}|$          | $-0.70$ | $243$ … $12\,349$  | $0.07\%$ – $1.1\%$ |
| VIEM | n20 | $Q_{\rm ext}$           | $-0.69$ | $246$ … $14\,962$  | $0.20\%$ – $3.4\%$ |
| VIEM | n20 | $|S_{\rm fw}^{\theta}|$ | $-0.69$ | $246$ … $14\,962$  | $0.17\%$ – $2.8\%$ |
| VIEM | n20 | $|S_{\rm bk}|$          | $-0.66$ | $246$ … $14\,962$  | $0.04\%$ – $0.74\%$ |

(Au is excluded — every dpl / lc value stalled on the DDA side, and the VIEM convergence study likewise reaches the LSP-stress regime, so the slopes are not informative of asymptotic behaviour.)

### 6.1 VIEM — universal $\varepsilon \propto n_{\rm tet}^{-2/3}$

All four observables for n15 and n20 collapse onto **slopes $-0.62$ to $-0.70$**, tightly bracketing the theoretical value $-2/3 \approx -0.667$. Within the convergence sweep this is one universal power law independent of observable and material.

**Theoretical basis.** The SWG basis used by `BlockVIEM` represents the volume current $\bm{J} = \chi(\bm{r})\,\bm{E}(\bm{r})$ as a piecewise-linear field on tetrahedra (constant divergence per tet). For a Galerkin discretization of an integral-operator equation with this basis, classical convergence theory (Schaubert, Wilton, Glisson 1984; Markkanen, Ylä-Oijala, Sihvola 2014) gives an $L^2$-error bound
$$
\|\bm{J} - \bm{J}_h\|_{L^2}\;\le\;C\,h^{p+1}
$$
for a basis of polynomial order $p$. SWG has $p = 1$, so $\|\bm{J} - \bm{J}_h\|_{L^2} = O(h^{2})$. Far-field observables (cross sections, scattering amplitudes) are computed as **bounded linear functionals** of $\bm{J}$, so they inherit the same rate (and sometimes one extra order via post-processing identities, but for the data range tested $h^2$ is what we see). Substituting $h^3 \propto V/n_{\rm tet}$:
$$
\varepsilon \;=\; |X^h - X^{\rm Mie}|/|X^{\rm Mie}|\;\propto\;h^2\;\propto\;n_{\rm tet}^{-2/3}.
$$
The cluster of empirical slopes near $-2/3$ confirms this is the active regime over $n_{\rm tet}\in[2.4\times 10^2,\ 1.5\times 10^4]$.

### 6.2 DDA — slopes $\sim -0.2$, much shallower than $-2/3$

DDA slopes scatter between $-0.13$ and $-0.28$ across observables, **none reach the asymptotic Galerkin-like $-2/3$**. Visually fig 1's DDA points fall onto curves that flatten at the right (large $N_{\rm occ}$) toward an error floor of about $0.5$–$3\%$, depending on observable.

**Theoretical basis.** Two error sources contribute to DDA cross-section accuracy:

1. **Polarizability formula** (CM, LDR, FCD, …) introduces a fixed-order correction $\sim (k d)^2 |m_p|^2$ that scales as $d^2 \propto N_{\rm occ}^{-2/3}$. For low-loss dielectrics this would give the same $-2/3$ slope as VIEM in the asymptotic limit.
2. **Shape error**: the cubic lattice "stair-steps" the smooth particle surface. The volume mismatch between the dipole-occupied lattice and the true sphere is $O(d/L)$ where $L$ is the particle size; integrated cross-section error is $O(d) \propto N_{\rm occ}^{-1/3}$ (Yurkin & Hoekstra 2007 §6).

For typical paper-sized particles ($x \lesssim 5$, $|m_p|$ moderate), the shape error dominates the polarizability-formula correction over the dpl range we test, giving an asymptotic prediction of slope $-1/3$. The observed slopes ($-0.13$ to $-0.28$) are *even shallower* than $-1/3$, indicating that within dpl $\in [10, 34]$ neither term is in its clean asymptotic regime: a constant offset from polarizability sits on top of a slowly-decreasing shape error, and the curves bend below true power-law on log-log axes.

In short, **DDA does not exhibit a universal $h^p$ scaling on this convergence sweep** — the leading-order error is "shape-floor-limited", not Galerkin-limited. To reach $\varepsilon \lesssim 0.5\%$ on a sphere DDA needs much finer lattices than VIEM does (or shape-fitted lattice generators, which the production code does not implement).

### 6.3 Practical implication

For the paper-production sweep at $r_{\rm ve}=0.1$ (fig 3 / fig 1 production lattice marked by the gray band):

- VIEM error sits on the universal $h^2$ curve and is predictable from $n_{\rm tet}$ alone — at $n_{\rm tet}\!\approx\!700$ (band center for n20), $\varepsilon \approx 1\%$ for cross sections.
- DDA error is dominated by the shape-step contribution at $N_{\rm occ}\!\approx\!700$, giving $\varepsilon \approx 2$–$3\%$ — about $2\!-\!3\times$ worse than VIEM at the same lattice count, even though the two solvers solve the same physics.

This $\sim 2$–$3\times$ DDA-vs-VIEM accuracy gap at the same $N_{\rm vol}$ is a generic feature of cubic-lattice DDA on smooth particles, not a fault of the production sweep. It is the trade-off accepted in exchange for the cubic-lattice FFT acceleration that makes DDA fast.

## 7. Source code

- DDA dpl-convergence sweep: [`scripts/run_dpl_convergence.py`](../../scripts/run_dpl_convergence.py)
- VIEM lc-convergence sweep: `~/Julia/block-VIEM.jl/viem_results/paper/run_lc_convergence.jl`
- Plotting: Phase 4 cell of [`plot_paper_results.ipynb`](plot_paper_results.ipynb)
- Output HDF5: `dda_results/paper/convergence_sphere_<mat>.hdf5` for DDA columns; `~/Julia/block-VIEM.jl/viem_results/paper/convergence_sphere_<mat>.hdf5` for VIEM columns. Loaded together by `_plot_io.load_convergence`.

## 7. Cross-references

- [`fig3_description.md`](fig3_description.md) — production sweep (auto-dpl / adaptive-lc); the band on each fig 1 panel marks where the production lattice sits on this convergence curve.
- [`fig2_description.md`](fig2_description.md) — RHS-scaling (default-dpl on DDA / adaptive-lc on VIEM); same lattice mechanism on VIEM as fig 1's `lc_factor=1.0` point.
- [`fig2_description.md` §9](fig2_description.md) — discusses the production-vs-RHS-scaling Au discrepancy; the lattice-resolution sensitivity exposed here in fig 1 (Au DDA non-converged at every dpl with axis-aligned orientation) is the same physics that makes Au touchy in fig 3.
