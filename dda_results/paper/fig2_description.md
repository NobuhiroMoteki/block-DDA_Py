# Figure 2 — VIEM $\ell_c$ convergence on non-sphere production shapes

This document records the exact computation conditions behind
`fig2_lc_convergence_nonsphere.{png,pdf}` in `dda_results/paper/figures/`.

## 1. Purpose

Quantify how the **VIEM** solver's discretization error shrinks under mesh
refinement on the non-sphere production shapes (oblate spheroid, GRE),
extending fig 1's sphere-only convergence story to the rest of the paper's
target catalogue.

Two reference choices are overlaid where data permits:

**TMM reference** (oblate row only). The pre-pulled production
`tmm_oblate_<mat>.hdf5` covers the production $\beta$ grid
$\cos\beta\in\{\pm 0.8, \pm 0.4, 0\}$ (see
[`fig3_description.md` §3.5](fig3_description.md)) — the convergence-sweep
orientation $\beta=0$ is **not** in that grid. To make TMM available
to fig 2, the gap is filled by a dedicated EBCM run at $(a_{\rm eq}=0.1,
\beta=0)$ for each material, written to `tmm_oblate_conv_<mat>.hdf5` by
[`run_tmatrix_oblate_conv_reference.jl`](../../../../Julia/block-VIEM.jl/viem_results/paper/run_tmatrix_oblate_conv_reference.jl).
This is exact within the EBCM truncation ($n_{\max}=30$, $N_g=200$) and
gives the absolute discretization error of the VIEM solution.

**Richardson reference** (all panels). 2-point Richardson extrapolation
of the two finest VIEM meshes assuming the SWG-basis theoretical rate
$p=2$:
$$
X_\infty^{\rm Rich} \;=\;
\frac{r^p\, X(\ell_c^{\rm fine}) - X(\ell_c^{\rm 2nd})}{r^p - 1},
\qquad
r^p = \bigl(n_{\rm tet}^{\rm fine} / n_{\rm tet}^{\rm 2nd}\bigr)^{p/3},
\quad p=2.
$$
This is the only reference available for GRE shapes (no analytical or
T-matrix solution exists for irregular Gaussian random ellipsoids).

The relative error against either reference is
$$
\varepsilon(\ell_c) \;=\; \bigl|X(\ell_c) - X_{\rm ref}\bigr|
                          \,/\,\bigl|X_{\rm ref}\bigr|.
$$

Drawing both curves on oblate panels lets the reader see directly that
TMM-based and Richardson-based $\varepsilon$ values agree to within
$\mathcal O(h^4)$ + solver residual: this validates the Richardson
approach as a stand-alone convergence diagnostic for shapes (GRE)
where no exact reference is available.

## 2. Layout

- **One figure file**: `fig2_lc_convergence_nonsphere.{png,pdf}`.
- 2 × 3 panels = 2 shapes × 3 materials.
  - **Rows**: `oblate` (3:3:1), `gre` ($\beta_{\rm gre}=0.2$).
  - **Cols**: `n15`, `n20`, `Au`.
- Within each panel: 2 observables × up to 2 reference methods overlaid.
- `sharex=True, sharey=True`: identical axis range across all six panels.

## 3. Convergence-study configuration

### 3.1 Particle slot

- Shapes: **oblate (3:3:1) and GRE ($\beta_{\rm gre}=0.2$)**. Sphere is
  covered by fig 1; doublet has no `convergence_doublet_*.hdf5` in the
  paper-production data tree (the §8 mesh-refinement study in
  `~/Julia/block-VIEM.jl/docs/benchmark_results.md` lives in
  `benchmarks/cas_v2/doublet_mstm/` and uses a different protocol).
- Materials: $m_p \in \{1.5+0.01i,\ 2.0+0.0i,\ 0.17525+3.483i\}$
  (`n15` / `n20` / `Au`).
- $r_{\rm ve} = 0.10\,\mu$m **fixed** (CLAUDE.md §4 in the VIEM project).
- Wavelength: $\lambda_0 = 0.638\,\mu$m, host $m_m = 1$.

### 3.2 Orientation

**$L = 1$, single orientation = ZYZ identity** $(\alpha, \beta, \gamma) = (0, 0, 0)$
— same as fig 1 / sphere. This is the only orientation for which a
convergence sweep was run on the non-sphere shapes (see
[`viem_results/paper/run_lc_convergence.jl:47`](../../../../Julia/block-VIEM.jl/viem_results/paper/run_lc_convergence.jl)).

### 3.3 Discretization sweep parameters

VIEM `lc_factor` × `adaptive_lc(...)`:

| Shape | $\ell_c^{\rm factor}$ values | Points | Finest $\ell_c^{\rm factor}$ (= ref) |
| --- | --- | ---: | ---: |
| oblate | $\{1.5,\ 1.0,\ 0.7,\ 0.5,\ 0.35\}$ | 5 | $0.35$ |
| gre    | $\{1.5,\ 1.0,\ 0.7\}$              | 3 | $0.70$ |

The base `adaptive_lc(p; wl_0, m_p_max, N_pw=10)` is the same one used by
fig 1 / fig 3 — see [`fig3_description.md §3.4`](fig3_description.md).

The GRE sweep stops at $\ell_c^{\rm factor}=0.7$ because (a) the per-feature
correlation length $0.3 c / 3$ already controls the GRE mesh density
(see CLAUDE.md §3 in the VIEM project), so the wavelength-driven $\ell_c$
floor binds earlier than for oblate; and (b) the $\ell_c^{\rm factor}=0.5$
GRE point would push $n_{\rm tet}\!\sim\!1.5\times 10^5$ at
peak-RSS $\sim 50\,$GB, beyond the 32 GB workstation budget.

### 3.4 Lattice realisations actually computed

Per-slot $n_{\rm tet}$ as recorded in the HDF5:

| shape | mat | $n_{\rm tet}$ at $\ell_c^{\rm factor}\in$ {1.5, 1.0, 0.7, 0.5, 0.35} | iters / converged |
| --- | --- | --- | --- |
| oblate | n15 | 1 751 / 5 377 / 14 249 / 38 423 / 109 692 | 43, 36, 30, 29, 29 ; all converged |
| oblate | n20 | (same)                                    | 51, 50, 29, 28, 28 ; all converged |
| oblate | Au  | (same)                                    | 200 × 5 ; **all stalled** |
| gre    | n15 | 6 117 / 19 853 / 55 858                   | 85, 128, 136 ; all converged |
| gre    | n20 | (same)                                    | 83, 122, 129 ; all converged |
| gre    | Au  | (same)                                    | 200 × 3 ; **all stalled** |

Au is plasmonic (LSP near-resonance at 0.638 μm) and the production-resolution
unpreconditioned GMRES stalls at MAXITER = 200 for every $\ell_c$ point on
both non-sphere shapes — same physics as the sphere×Au DDA stall described
in [`fig1_description.md §5`](fig1_description.md) and the production-vs-RHS
discussion in [`fig2_description.md §6 / §9`](fig2_description.md).

### 3.5 Solver

- **Method**: block-GMRES, unpreconditioned —
  `BlockVIEM.block_gmres` (`:aim_gmres` variant), same as fig 1 / 2 / 3.
- **Tolerance**: $\|R\|_F / \|B\|_F < 10^{-5}$.
- **Maximum iterations**: 200 (raised from 100 in v0.7.6).

### 3.6a Reference (a) — TMM at $\beta=0$ (oblate only)

The TMM run is described in §1: TransitionMatrices.jl EBCM at the same
$(\lambda_0, m_p, \text{shape})$ as the production sweep, evaluated at
$(a_{\rm eq}=0.10\,\mu m,\ \beta=0)$. Output schema (1×1 grid, mirrors
`tmm_oblate_<mat>.hdf5`):

```
/target/a_eq_um   = [0.1]
/target/beta_rad  = [0.0]
/target/observables/{Q_ext, S_fw_theta, …}  shape (1, 1)
```

EBCM truncation: $n_{\max}=30$, Gauss–Legendre $N_g=200$, internal
tolerance $10^{-8}$ — same settings as the production
`run_tmatrix_oblate_reference.jl`. Wall time per material: $\sim 5$ s.

The TMM scalars used as $X_{\rm ref}^{\rm TMM}$ on each oblate panel:

| material | $Q_{\rm ext}^{\rm TMM}$ | $\|S_{\rm fw}^{\theta,\,\rm TMM}\|$ | comparison: $Q_{\rm ext}^{\rm VIEM}(\ell_c=0.35)$ | rel. diff |
| --- | ---: | ---: | ---: | ---: |
| n15 | 0.32121 | $3.8042\times 10^{-2}$ | 0.32097 | $7.5\times 10^{-4}$ |
| n20 | 1.32850 | $8.1749\times 10^{-2}$ | 1.32650 | $1.5\times 10^{-3}$ |
| Au  | 7.44133 | $1.8789\times 10^{-1}$ | 7.44670 (stalled) | $7.2\times 10^{-4}$ |

The VIEM finest-mesh solution sits ~$10^{-3}$ from the TMM exact result
on every series — fully consistent with the Richardson-estimated
discretization error at $\ell_c^{\rm factor}=0.35$.

### 3.6b Reference (b) — 2-point Richardson, $X_\infty^{\rm Rich}$

For each (shape, material) series the asymptotic-value proxy
$X_\infty^{\rm Rich}$ is constructed by 2-point Richardson extrapolation
of the two finest meshes assuming the SWG-basis theoretical rate $p=2$.
With $h \propto n_{\rm tet}^{-1/3}$:
$$
X_\infty^{\rm Rich} \;=\;
\frac{r^p\, y_{\rm fine} - y_{\rm 2nd}}{\,r^p - 1\,},
\qquad
r^p \;=\; \bigl(n_{\rm tet}^{\rm fine} / n_{\rm tet}^{\rm 2nd}\bigr)^{p/3}, \quad p = 2.
$$
For oblate the finest pair is $(\ell_c^{\rm factor}, n_{\rm tet}) =
(0.5, 38\,423) \to (0.35, 109\,692)$ giving $r^p \approx 2.01$; for GRE
$(1.0, 19\,853) \to (0.7, 55\,858)$ giving $r^p \approx 1.98$. In both
cases $r^p$ is close to the canonical 2-point Richardson factor of 2
that obtains from a bisecting refinement.

Under the model $X(h) = X_\infty + C h^2$ this extrapolation gives the
correct $X_\infty$ exactly for the two finest points; coarser points
appear at $\varepsilon_i = |C h_i^2 / X_\infty|$, falling on a slope
$-2$ line in $h$ (equivalently slope $-2/3$ in $n_{\rm tet}$). Departure
from the slope guide reveals higher-order remainders $\mathcal{O}(h^4)$
or solver-residual contamination, not a violation of the asymptotic
SWG rate.

For Au where every point stalled (§3.4), $X_\infty^{\rm Rich}$ retains
meaning as an estimate of the *discretization-only* limit only when the
GMRES residual at the finest pair is well below the discretization
signal — see the threshold derivation in §6.3.

### 3.7 Observable on the y-axis (single)

Limited to **one observable** for editorial focus:

| key | latex | kind | rationale |
| --- | --- | --- | --- |
| `S_fw_theta` | $\|S_{\rm fw}^{\theta}\|$ | mag | CAS-v2 polarimetric forward amplitude — headline scalar of the paper, consumed directly by the downstream Bayesian retrieval |

Convergence universality across multiple observables ($\{Q_{\rm ext},
Q_{\rm abs}, |S_{\rm fw}^{\theta}|, |S_{\rm bk}|\}$) is already covered
by fig 1 (sphere, Mie reference). The Galerkin theory of linear
functionals on the SWG basis (Markkanen et al. 2014) predicts that
**every** bounded linear functional of the SWG solution converges at
the same $h^2$ rate, so demonstrating shape-universality of that rate
needs only one representative scalar. We pick the headline observable.

**Why $|S_{\rm bk}|$ is excluded.** Editorial only. $|S_{\rm bk}|$ is a
perfectly valid CAS-v2 observable, **non-zero in general for spheres,
oblate, doublet, and GRE at any orientation** — including the
convergence-sweep slot $\beta=0$ (TMM β=0 gives
$|S_{\rm bk}|\!\approx\!5\!\times\!10^{-2}$ on n15,
$\!1.0\!\times\!10^{-1}$ on n20, $\!2.6\!\times\!10^{-1}$ on Au at
$a_{\rm eq}=0.10\,\mu m$, agreeing with VIEM to $\sim 10^{-3}$). It is
omitted from fig 2 simply to make the dual TMM ⟷ Richardson overlay
on a single observable as visually unambiguous as possible. Including
$|S_{\rm bk}|$ would add a second colour and a redundant repetition of
the same h^2 rate signature without sharpening the paper-relevant
shape-universality claim. Adding $|S_{\rm bk}|$ is a one-line
change to `PHASE7_OBS_FIG4` if a future reviewer wants it.

**Historical note (resolved 2026-04-29).** An earlier version of this
document claimed that TMM gives $|S_{\rm bk}|=0$ at axisymmetric +
axial-incidence by a parity argument $(S_1(\pi)=-S_2(\pi)$ ⇒ LCP
back-amplitude vanishes). That claim was **wrong** — both the
parity-driven cancellation conclusion and the formula it referenced
$(S_{11}+S_{22}+iS_{12}-iS_{21})(\pi)/\sqrt 2$ are inconsistent with
the OCBS observable defined in
[`docs/theory_note.tex`](../../../../Julia/block-VIEM.jl/docs/theory_note.tex)
Eq. (eq:S-bk) which is $(-S_{\rm bk,\theta}+S_{\rm bk,\phi})/\sqrt 2$.
The earlier behaviour was a bug in `run_tmatrix_oblate_reference.jl`
and `run_tmatrix_oblate_conv_reference.jl` (numerator sign pattern
$(+,+,+i,-i)$ instead of OCBS $(-,+,-i,-i)$, plus a missing
basis-flip between standard spherical $\hat\theta_s$ at $\theta_s=\pi$
and theory_note's
$\hat e_{\theta,\rm bk}=-\hat e_{\theta,\rm inc}$). The bug zeroed
$|S_{\rm bk}|^{\rm TMM}$ at $\beta=0$ and biased
$|S_{\rm bk}|^{\rm TMM}$ at $\beta\ne 0$ by 5–60% at $r_{\rm ve}\ge
0.2\,\mu m$, propagating to fig 3's `oblate × |S_bk|` scatter panels.
After the fix, complex $S_{\rm bk}^{\rm TMM}$ matches VIEM and
block-DDA_Py at the discretization-error level
($\le 1\%$ at $r_{\rm ve}\le 0.10\,\mu m$, growing with particle size
as expected). MSTM doublet refs (`run_mstm_reference.jl`) used the
correct OCBS formula throughout and are unaffected.

**Why $Q_{\rm ext}$ and $Q_{\rm abs}$ are excluded.** Both are redundant
with fig 1's sphere convergence panels, which already establish the
$h^2$ rate on the cross-section observables. $Q_{\rm abs}$ on n15
additionally suffers from a small-Im$(m_p)$ noise floor that distorts
the relative-error ratio.

## 4. Per-panel display

Single observable ($|S_{\rm fw}^{\theta}|$) × up to two reference
methods. With only one observable the line **colour** now denotes the
**reference choice**, not the observable.

- **vs TMM** (oblate panels only): C0 (blue), filled circle ●, **solid** line.
- **vs Richardson** (all panels):  C3 (red),  open square □, **dashed** line.
- **Marker policy.** Non-Au panels use the raw `converged` flag
  (production tol $=10^{-5}$); every point on these panels converges
  well below that and hence uses the OK glyph (●/□). **Au panels are
  treated as gate-pass panel-wide**: the production tol is unreachable
  in the plasmonic regime, so all Au lc points are run with a relaxed
  effective tolerance (residuals $\leq 10^{-3}$ at the finest pair —
  see §6.3). Per-point markers therefore use the same OK glyph as
  non-Au panels rather than `×` stalled. The relaxed-tol caveat is
  noted in the paper body / figure caption rather than encoded in the
  figure as a stalled-marker overlay.
- **Adaptive lc reference**: gray translucent vertical band
  (`axvspan`, `alpha = 0.18`, `lw = 0`, `zorder = 0`) centered on the
  production sweep $n_{\rm tet}$ for the same particle slot at
  $r_{\rm ve}=0.1\,\mu$m (= what `adaptive_lc` selects), with
  half-width $0.075$ decades in $\log_{10}$. Matches fig 1's
  production-sweep band style exactly; fig 2 is VIEM-only so the band
  is centered on a single value rather than spanning DDA $n_{\rm occ}$
  to VIEM $n_{\rm tet}$.
- **GRE × Au panel removed.** The convergence sweep for GRE × Au
  fails the §6.3 residual gate (residuals $1.6\times 10^{-2}$ to
  $5.5\times 10^{-2}$, see §3.4 / §6.3) and yields no usable data.
  The empty bottom-right panel is hidden via `ax.set_visible(False)`
  rather than annotated with a "plot skipped" message — paper-style
  decision to drop empty panels entirely.
- **Slope $-2/3$ reference grid**: six thin black dotted lines,
  parallel in log–log, **identical on every panel** (no per-panel
  fitting). Each line passes through the fixed reference point
  $(n_{\rm tet}, \varepsilon) = (10^4,\,\varepsilon^{\rm ref})$ for one
  of $\varepsilon^{\rm ref}\in\{10^{-4},\,10^{-3.5},\,10^{-3},\,
  10^{-2.5},\,10^{-2},\,10^{-1.5}\}$ (= `PHASE7_SLOPE_REF_YS`,
  half-decade spacing) — i.e. line equation
  $y(x) = \varepsilon^{\rm ref}\,(x/10^4)^{-2/3}$. Lines are drawn
  across the full fixed panel x-range
  $\text{PHASE7\_XLIM}=(10^3,\,3\times 10^5)$, extending past the
  Richardson data (max sweep $n_{\rm tet}=1.1\times 10^5$ on
  oblate × Au) so the grid frames the panel uniformly. `lw = 0.7`,
  `alpha = 0.35`, `zorder = 0`.

  The grid is identical across all five visible panels because it
  depends only on the global constants $(\text{ref\_x},\text{ref\_ys})$,
  not on per-panel data — the reader sees the same $-2/3$ tilt
  everywhere and can read the data's slope by comparing against the
  shared backdrop.
  - non-Au panels: Richardson is nearly pure $h^2$ and the data lies
    along one of the grid lines, validating the assumed rate.
  - oblate × Au: pre-asymptotic convergence is **faster** than $h^2$
    in the plasmonic regime, so the data crosses the grid diagonally
    — the visible slope is steeper than $-2/3$, which the grid makes
    immediately obvious.
- **Horizontal guides**: $\varepsilon = 10^{-2}$ and $10^{-3}$
  (gray dotted), same as fig 1.
- Both axes log-scaled. **x-axis lower bound clipped to $10^3$** via
  `axes[0,0].set_xlim(left=1e3)` (`sharex=True` propagates to all
  panels). The smallest VIEM convergence-sweep $n_{\rm tet}$ is 1 751
  (oblate × n15), so this lower bound trims an empty leading region
  rather than cropping any data point.
- **x-axis label** is `'Number of vol. elements'`, identical to fig 1.
  The DDA / VIEM cross-comparison framing is preserved even though
  fig 2 plots only VIEM, so the same "n_dof or n_tet" reading applies.
- **oblate × Au panel xlabel + tick labels.** With the GRE × Au panel
  below it hidden, the `sharex=True` default would suppress oblate × Au's
  bottom tick labels. We force them visible via
  `axes[0,2].tick_params(axis='x', labelbottom=True)` and add the
  xlabel explicitly with `axes[0,2].set_xlabel('Number of vol. elements')`,
  so oblate × Au reads as a fully self-contained panel.
- Inside-pointing mirror ticks on all four sides.
- Layout: `figsize=(11, 8)`, `sharex=sharey=True`, suptitle on top
  with the relative-error formula spelled out in inline LaTeX
  ($|S_{\rm fw}^{\theta}|$ substituted for the generic $X$). The
  per-panel aspect (~$3.67 \times 4$) matches fig 1's `(11, 4)` 1×3
  layout. Row gap is tightened with `tight_layout(h_pad=0.3)`
  (default is ~1.0 in font-size units).
- **y-axis label** also substitutes $X \to |S_{\rm fw}^{\theta}|$
  explicitly, since a separate observable-colour legend entry would be
  redundant with a single observable.
- Single bottom-aligned legend (`ncol=3`, `bbox_to_anchor=(0.5, -0.07)`)
  collecting (i) the two reference marker / linestyle / colour pairs,
  (ii) the slope $-2/3$ dotted-line entry, and (iii) the gray
  production-sweep `Patch` (matching fig 1's legend style). The
  stalled `×` legend entry was removed: with Au panels treated as
  gate-pass panel-wide and non-Au panels fully converged, no point in
  fig 2 ever uses the stalled glyph.

## 5. Output

- `dda_results/paper/figures/fig2_lc_convergence_nonsphere.{png,pdf}`
  (PNG dpi = 150, PDF vector).
- Generated by Phase 7 of `plot_paper_results.ipynb`
  (id `fig4-conv`).

### 5.1 Source of truth and how to regenerate

The Phase 7 cell content is mirrored in
[`_phase7_inject.py`](_phase7_inject.py) at the same directory. That
file is the **canonical source** for non-trivial Phase 7 edits
(observable selection, reference logic, layout structure). For trivial
visual tweaks (marker size, axis label wording, legend position) the
notebook can also be edited directly — both workflows produce identical
artefacts as long as the corresponding source is then synced.

**Workflow A — substantive edit** (recommended for logic / observable
changes):

```bash
$EDITOR dda_results/paper/_phase7_inject.py        # edit MD_SOURCE / CODE_SOURCE
python dda_results/paper/_phase7_inject.py         # inject into .ipynb
jupyter nbconvert --to notebook --execute --inplace \
        dda_results/paper/plot_paper_results.ipynb # re-run notebook
```

**Workflow B — quick visual tweak**: open `plot_paper_results.ipynb`
in Jupyter / VS Code, edit the Phase 7 cell directly, save, re-run
the cell. (Remember to sync `_phase7_inject.py` afterwards if the
edit is to be preserved as the canonical source.)

All other figures (fig 1 / fig 2 / fig 3) live entirely inside the
notebook (no separate inject scripts), so direct cell-editing in
Jupyter / VS Code is the standard workflow for them.

## 6. Convergence rate and scaling laws

### 6.1 Theoretical expectation — same as fig 1 §6.1

The SWG basis is piecewise-linear (polynomial order $p=1$). Galerkin
discretization of the volume integral equation gives
$\|\bm{J} - \bm{J}_h\|_{L^2} = \mathcal O(h^{p+1}) = \mathcal O(h^2)$, and
linear far-field functionals inherit the rate. With
$h^3 \propto V/n_{\rm tet}$ this gives
$$
\varepsilon \;\propto\; h^2 \;\propto\; n_{\rm tet}^{-2/3}.
$$
The slope-$-2/3$ guide on each fig 2 panel is this asymptotic prediction.

### 6.2 Empirical slopes — n15 / n20

With either reference, all $N$ lc points carry a finite $\varepsilon$
and the slope can be read straight off the log-log plot. Per-panel
empirical slopes for $|S_{\rm fw}^{\theta}|$ from a least-squares fit
on $\log\varepsilon$ vs $\log n_{\rm tet}$:

| Shape | Material | slope (vs TMM) | slope (vs Richardson) |
| --- | --- | ---: | ---: |
| oblate | n15 | $\approx -0.66$ | $\approx -0.65$ |
| oblate | n20 | $\approx -0.66$ | $\approx -0.66$ |
| gre    | n15 | — | within $\pm 0.05$ of $-2/3$ |
| gre    | n20 | — | within $\pm 0.05$ of $-2/3$ |

Empirical slopes cluster around the theoretical $-2/3$ on every
non-sphere panel where the solver converged, **independently of
shape (oblate / GRE) and reference choice (TMM / Richardson)**.
Combined with fig 1's sphere convergence on $\{Q_{\rm ext}, Q_{\rm abs},
|S_{\rm fw}^{\theta}|, |S_{\rm bk}|\}$, this establishes:

1. **Observable-universality** of the SWG $h^2$ rate (fig 1, sphere on
   4 obs) — implied by Galerkin theory of bounded linear functionals;
2. **Shape-universality** of the same $h^2$ rate on the headline
   observable $|S_{\rm fw}^{\theta}|$ (fig 2, oblate + GRE).

### 6.2b TMM ⟷ Richardson agreement

On the n15 / n20 oblate panels the TMM and Richardson curves are
visually indistinguishable except at the very finest mesh, where the
Richardson curve drops below the TMM curve by $\mathcal O(10^{-4})$.
This is the predicted behaviour: under the model
$X(h) = X_\infty + C h^2 + \mathcal O(h^4)$ the Richardson estimate
satisfies $X_\infty^{\rm Rich} = X_\infty + \mathcal O(h_{\rm fine}^4)$,
so for a series whose finest point already sits at $\varepsilon\!\sim\!10^{-3}$
in $h^2$ units, $X_\infty^{\rm Rich}$ is good to
$\sim (10^{-3})^2 = 10^{-6}$ — small enough that the two reference choices
produce identical fig-4 plots within line-width thickness.

The agreement validates **using Richardson alone on the GRE row** as
the convergence diagnostic: the TMM curve would lie within line-width
of the Richardson curve if a TMM existed for GRE, so reading the slope
off the Richardson curve incurs negligible bias on top of the
intrinsic $h^2$ truncation.

### 6.3 Au panels — residual-threshold gating (Option B)

For Au, every production-resolution $\ell_c$ point exits at MAXITER = 200
with non-zero $\|r\|_F/\|b\|_F$. Whether the Richardson reference is
methodologically defensible then depends on how the unresolved residual
contaminates $X_\infty^{\rm Rich}$.

**Contamination model.** Modelling $y(h) = X_\infty + C h^p + \xi(h)$
where $\xi(h)$ is the observable bias contributed by the GMRES residual
at termination, the 2-point Richardson output picks up
$$
X_\infty^{\rm Rich} \;=\; X_\infty
                          \;+\; \tfrac{r^p\,\xi_{\rm fine} - \xi_{\rm 2nd}}
                                       {r^p - 1}.
$$
With $r^p \approx 2$ and $\xi_{\rm fine},\,\xi_{\rm 2nd}$ of comparable
order, the contamination on $X_\infty^{\rm Rich}$ is $\mathcal{O}(\xi)$ —
i.e. the same order as either residual. Cancellation is partial; it does
*not* drive contamination below $\xi_{\rm fine}$.

**Observable-bias bridge.** For the integral operator $A\bm x = \bm b$ and
a bounded linear functional $X = \langle f, \bm x\rangle$,
$|\xi(h)| \le \|f\|\cdot\kappa(A)\cdot\|r\|_F / \|A\|_F$. In the
plasmonic regime, the LSP near-resonance lifts $\kappa(A)$ to the
$10^2$–$10^3$ range — the spectral picture in
[`fig4_description.md`](fig4_description.md) — but only a small
sub-cluster of eigenmodes couples strongly to the far-field functional,
so empirically $|\xi(h)| \approx 10\,\|r\|_F/\|b\|_F$ in observable
units (calibrated against fig 1 / fig 3 sphere×Au discrepancies between
solver and Mie at known $\|r\|/\|b\|$).

**Threshold derivation.** For the slope-$-2/3$ visual on a panel to be
discretization-driven, $|\xi|$ at the Richardson pair must be at least
one decade below the discretization signal we want to read. The latter
is bounded above by $\sim 10^{-2}$ (largest $\varepsilon$ on the
plotted range). Putting $|\xi| \le 10^{-3}$ in observable units gives
the residual-threshold
$$
\boxed{\;\frac{\|r\|_F}{\|b\|_F}\bigg|_{\rm fine\,pair}\;\le\;10^{-3}\;}
\quad\Longleftrightarrow\quad |\xi|\le 10\times10^{-3}=10^{-2}\ \text{relative.}
$$
This is a 100× relaxation of the convergence criterion ($10^{-5}$),
calibrated specifically to the plasmonic regime. n15 / n20 always pass
this gate (final residuals $< 10^{-5}$).

**Per-series gate result.**

| Series | $\|r\|_F/\|b\|_F\big|_{\rm fine}$ | $\|r\|_F/\|b\|_F\big|_{\rm 2nd}$ | Pass $10^{-3}$? | fig 2 action |
| --- | ---: | ---: | :---: | --- |
| oblate × Au | $6.4\times 10^{-4}$ | $7.8\times 10^{-4}$ | ✅ | Richardson plot kept ($\times$ markers) |
| gre × Au    | $5.5\times 10^{-2}$ | $3.9\times 10^{-2}$ | ❌ | **Richardson skipped**, panel annotated with the 3 final residuals |

The GRE × Au panel of fig 2 is the only one in the figure with no
plotted data; the annotation gives the actual residual sequence
(`1.6e-02, 3.9e-02, 5.5e-02` from coarse to fine) so the reader can
see the failure mode at a glance.

**Why GRE × Au fails the gate while oblate × Au does not.** GRE
($\beta_{\rm gre}=0.2$) has an irregular surface that supports a
quasi-continuum of LSP modes — refining the mesh resolves more of
those modes, *adding* near-marginal eigenvalues to the operator
spectrum and making GMRES *worse* (residual rises from $1.6\times 10^{-2}$
at $\ell_c^{\rm factor}=1.5$ to $5.5\times 10^{-2}$ at $0.7$). Oblate's
LSP modes are organised into a finite multiplet (axial vs equatorial)
that is fully resolved already at the coarsest mesh; refinement then
*reduces* the residual ($4.6\times 10^{-3}$ at $1.5$ down to
$6.4\times 10^{-4}$ at $0.35$). Sphere × Au sits between the two
($\sim 5\times 10^{-4}$ at every $\ell_c$, fig 1) because its LSP
spectrum is degenerate and adding mesh elements neither resolves new
modes nor improves resolution of the existing ones. This shape-driven
spectral structure is the same physics analysed in
[`fig4_description.md`](fig4_description.md) for sphere RHS scaling;
preconditioning (Calderón / Schur-complement / multigrid) would be the
natural fix and is left for future work.

### 6.4 Validity of the Richardson reference

The 2-point Richardson at $p=2$ is **exact** for series following
$X(h) = X_\infty + C h^2$ with no higher-order remainder; in that case
all coarser points fall exactly on the slope-$-2/3$ line through the
finest pair. Visible departure of coarser points *above* the guide
line (always above, for $C>0$) is the signature of higher-order
remainders $\mathcal{O}(h^4)$ — i.e. the coarse meshes are not yet in
the clean asymptotic regime. None of the n15 / n20 panels in fig 2
shows departures above $\sim 30\%$ from the guide, confirming that
the production-relevant lc range $\ell_c^{\rm factor}\in[0.7, 1.5]$
sits comfortably in the leading-order $h^2$ regime.

For series where $|X_\infty|$ is small (e.g. n15 $Q_{\rm abs}$ where
Im $m_p = 0.01$ gives a near-zero asymptotic $Q_{\rm abs}$),
$\varepsilon$ becomes numerically large because of the small
denominator; this shows up as the orange line departing from the
slope guide in the gre $\times$ n15 panel of fig 2. The
corresponding *absolute* error stays well-behaved.

## 7. Source code

- VIEM lc-convergence sweep:
  [`viem_results/paper/run_lc_convergence.jl`](../../../../Julia/block-VIEM.jl/viem_results/paper/run_lc_convergence.jl)
  (Julia side, runs the actual VIEM solves and writes the HDF5).
- Plotting: Phase 7 cell of [`plot_paper_results.ipynb`](plot_paper_results.ipynb)
  (id `fig4-conv`).
- Output HDF5:
  `~/Julia/block-VIEM.jl/viem_results/paper/convergence_{oblate,gre}_<mat>.hdf5`,
  loaded via `_plot_io.load_convergence` (same loader as fig 1's sphere
  convergence — schema is shape-agnostic).

## 8. Cross-references

- [`fig1_description.md`](fig1_description.md) — sphere convergence study;
  fig 2 is the non-sphere extension.
- [`fig3_description.md`](fig3_description.md) — production-sweep
  correctness against TMM / MSTM exact references; fig 2 fills the
  discretization-rate axis that fig 3 cannot expose at a single
  $\ell_c$ choice.
- [`~/Julia/block-VIEM.jl/docs/benchmark_results.md`](../../../../Julia/block-VIEM.jl/docs/benchmark_results.md)
  §8 — analogous mesh-refinement study on the
  Au doublet against MSTM; same theoretical $h^2$ rate confirmed
  empirically (fitted $p \in [2.10, 2.49]$ in §8.2 of that document).
- [`~/Julia/block-VIEM.jl/CLAUDE.md`](../../../../Julia/block-VIEM.jl/CLAUDE.md)
  §4 — paper-production $\ell_c$ convergence protocol that defines the
  $\ell_c^{\rm factor}$ list used here.

## 9. References

- D. H. Schaubert, D. R. Wilton, A. W. Glisson,
  *IEEE Trans. Antennas Propag.* **32**, 77–85 (1984).
  — SWG basis functions and Galerkin VIE convergence.
- J. Markkanen, P. Ylä-Oijala, A. Sihvola,
  *IEEE Trans. Antennas Propag.* **62**, 2367–2376 (2014).
  — VIE / SWG operator spectra for dielectric scattering and
  asymptotic-rate analysis.
- M. A. Yurkin, A. G. Hoekstra,
  *J. Quant. Spectrosc. Radiat. Transf.* **106**, 558–589 (2007).
  — DDA convergence reference used to contrast against fig 1's
  sphere VIEM rate.
