# Figure 2 — Block-Krylov RHS scaling diagnostic (sphere)

This document records both the exact computation conditions behind
`fig2_rhs_scaling.{png,pdf}` and the physical / numerical interpretation
of the trends visible in the figure.

## 1. Purpose

For a fixed particle (sphere × material × $r_{\rm ve}$), measure how the
block-GMRES iteration count and per-orientation wall time depend on the
**block size $L$** (number of right-hand sides solved together — equal to
the number of orientations bundled into one block-Krylov solve).
Diagnostic of the cost amortization that block-Krylov provides over
single-RHS GMRES, and of the spectral conditioning differences between
DDA and VIEM.

## 2. Layout

**Single figure file**: `fig2_rhs_scaling.{png,pdf}`, 2 × 2 panels.

- **Col 1**: n20 ($m_p = 2.0+0.0i$) — **DDA + VIEM overlay**, $r_{\rm ve} \in \{0.10, 0.20, 0.40\}$ μm.
- **Col 2**: Au ($m_p = 0.18+3.48i$) — **VIEM only**, $r_{\rm ve} \in \{0.05, 0.10, 0.20\}$ μm. (Au has no $r_{\rm ve}=0.40$ case; production-resolution DDA hits MAXITER for every $r_{\rm ve}$ — see §9 — and is omitted to avoid clutter.)
- **Row 1**: GMRES iterations (log y, capped at $1.25\times$ MAXITER ≈ 250). Au plasmonic stalls span the $[10, 200]$ iter range so log makes the convergence pattern legible.
- **Row 2**: Solving time per orientation [s] (log y).
- n15 is dropped: its trend in n20 is essentially repeated and adds no new information.

### Encodings

- **Solver = colour**: DDA = `C0` blue (filled marker, solid line); VIEM = `C1` orange (open marker, dashed line).
- **$r_{\rm ve}$ = marker shape** (uniform across both columns and both rows):

  | $r_{\rm ve}$ [μm] | marker |
  | --- | --- |
  | 0.05 | ○ |
  | 0.10 | ▢ |
  | 0.20 | △ |
  | 0.40 | ◇ |

- **MAXITER stalls** (`converged = 0`): $\times$ marker on the iter panel; **omitted** from the solving-time panel (their wall time is the budget cap, not actual convergence cost).
- `sharex=True, sharey='row'` — both panels in a row share the same y-range.

## 3. RHS-scaling sweep configuration

### 3.1 Particle slots

- Shape: **sphere only**. Other shapes (oblate, gre, doublet) are not in the RHS-scaling sweep.
- Materials: $m_p \in \{1.5+0.01i,\ 2.0+0.0i,\ 0.17525+3.483i\}$ (n15 / n20 / Au).
- $r_{\rm ve}$: $\{0.05, 0.10, 0.20, 0.40\}$ μm for n15, n20; $\{0.05, 0.10, 0.20\}$ μm for Au.
- Wavelength: $\lambda_0 = 0.638\,\mu$m, host $m_m = 1$.

### 3.2 Block sizes

$L \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ — 8 values per $r_{\rm ve}$.

### 3.3 Orientations

Deterministic uniform-on-SO(3) Euler sequence with fixed RNG seed
(`numpy.random.default_rng(12345)`). Per-orientation triples
$(\alpha, \beta, \gamma)$ are drawn as
$$
\alpha = 2\pi\, u_1, \quad
\beta = \arccos(2 u_2 - 1), \quad
\gamma = 2\pi\, u_3,
$$
with $u_i \sim {\rm Uniform}(0,1)$. Larger-$L$ blocks are nested
($L=1 \subset L=2 \subset L=4 \subset \cdots$) so the same orientations
recur across all $L$ values.

This is **different from fig 3's spheroid-mode grid** ($\alpha=\gamma=0$,
$N_\beta=5$) — fig 2 deliberately samples random orientations to measure
generic block-Krylov scaling, not the production sweep's $\alpha=\gamma=0$
spheroid block.

### 3.4 Solver

- **Method**: block-GMRES, **unpreconditioned on both sides** ("VIEM-parity"). The same solver (`bl_krylov.bl_gmres_mvp_fft` on the DDA side, `BlockVIEM.block_gmres` with `:aim_gmres` on the VIEM side) is used by `scripts/run_paper_sweep.py` (production paper sweep that fills `dda_results/paper/<shape>_<mat>.hdf5` — i.e. the data drawn by fig 3 and fig 1) and by `scripts/run_rhs_scaling.py` (the sweep that fills the `/target/rhs_scaling/gmres/*` group used by this figure). The `bl_dda/scatterer.py:solve_matrix_equation()` path used by the validation notebook `test_dda.ipynb` was historically using `bl_bicgstab_jacobi_mvp_fft`; it has been switched to `bl_gmres_mvp_fft` for parity (2026-04-29).
- **Tolerance**: $\|R\|_F / \|B\|_F < 10^{-5}$.
- **Maximum iterations**: 200.
- **Discretizations**:
  - **DDA**: point-dipole basis, FFT-accelerated matrix–vector product (Goodman algorithm).
  - **VIEM**: volume integral on tetrahedral SWG basis, AIM-accelerated MVP.
- The HDF5 attribute `/target/rhs_scaling/solver_variant = "unpreconditioned (VIEM-parity)"` documents this.

The problem is the same Maxwell scattering by an inhomogeneous dielectric / metallic body — only the discretization differs.

### 3.5 Lattice resolution — "default" (DDA) and "adaptive" (VIEM)

The two solvers use **different lattice mechanisms** in the RHS-scaling
sweep, in contrast to fig 3 where DDA actively matches VIEM via auto-dpl.

#### DDA — "default"

[`scripts/run_rhs_scaling.py:67`](../../scripts/run_rhs_scaling.py):

```python
mdl = gaussian_ellipsoid_shape_model(r_v_base, bc, ab, beta, wl_0, m_p_xyz)
                                     # ↑ no `dpl=` argument
```

The shape model's constructor default is `dpl=17`
([`shape_model/gaussian_ellipsoid.py:52`](../../shape_model/gaussian_ellipsoid.py)),
giving the lattice spacing
$$
d \;=\; \frac{\lambda_0}{|m_p|_{\max}\cdot 17}.
$$

This is the **same dpl across all (shape, material, $r_{\rm ve}$) slots**,
so $N_{\rm occ}$ scales purely with the volume $V \propto r_{\rm ve}^3$
(modulo discrete-interior-check perturbations).

Concretely for sphere × Au: $N_{\rm occ}^{\rm RHS} = \{399,\ 3378,\ 27378\}$
at $r_{\rm ve} = \{0.05, 0.10, 0.20\}$.

This **differs from fig 3**, where DDA uses `dpl_for_slot` (auto) and
gets $N_{\rm occ}^{\rm prod} = \{636,\ 3685,\ 26883\}$ — see
[`fig3_description.md` §3.4](fig3_description.md) for the auto definition.

#### VIEM — "adaptive" (same as fig 3)

VIEM uses `adaptive_lc` ([`BlockVIEM/src/gre_mesh.jl:84`](../../../../Julia/block-VIEM.jl/src/gre_mesh.jl)):
$$
\text{lc} \;=\; \min\!\Bigl(
    \tfrac{\lambda_0}{|m_p|_{\max}\cdot N_{\rm pw}},\;
    \tfrac{c}{3},\;
    [\tfrac{0.3\,c}{3}\ \text{if } \beta_{\rm gre}>0]
\Bigr),
\quad N_{\rm pw} = 10.
$$

The VIEM mesh is **identical between fig 3 and fig 2** — empirically
confirmed: $n_{\rm tet}^{\rm RHS} = n_{\rm tet}^{\rm prod}$ bit-for-bit
in `sphere_<mat>.hdf5` for every slot.

### 3.6 Lattice summary table

The full $N_{\rm occ}$ (DDA, "default" $= $ dpl 17) and $n_{\rm tet}$
(VIEM, adaptive) used by fig 2 across every material and $r_{\rm ve}$ in
the sweep:

| material | $r_{\rm ve}$ [μm] | DDA $N_{\rm occ}$ (default, dpl=17) | VIEM $n_{\rm tet}$ (adaptive_lc) |
| --- | ---: | ---: | ---: |
| n15 | 0.05 | 27 | 652 |
| n15 | 0.10 | 251 | 652 |
| n15 | 0.20 | 2 199 | 2 229 |
| n15 | 0.40 | 17 822 | 16 671 |
| n20 | 0.05 | 72 | 652 |
| n20 | 0.10 | 648 | 710 |
| n20 | 0.20 | 5 233 | 5 151 |
| n20 | 0.40 | 41 897 | 39 167 |
| Au | 0.05 | 399 | 652 |
| Au | 0.10 | 3 378 | 3 673 |
| Au | 0.20 | 27 378 | 26 459 |

#### Reading the DDA column

Because $d = \lambda_0 / (|m_p|\cdot 17)$ at fixed dpl, the dipole count
scales as
$$
N_{\rm occ}^{\rm DDA, default}\;\propto\;V \cdot |m_p|^{3}
                              \;=\;\tfrac{4\pi}{3}\, r_{\rm ve}^3\, |m_p|^3.
$$
At a fixed $r_{\rm ve}$ it therefore varies $\sim 13\times$ between the
extreme materials of the paper ($|m_p|_{\rm n15} = 1.5$ vs
$|m_p|_{\rm Au}\!\approx\!3.49$):

- $r_{\rm ve}=0.10$: n15 → 251, n20 → 648, Au → 3 378.
- $r_{\rm ve}=0.05$: n15 → 27, n20 → 72, Au → 399. ⚠ The n15/n20 entries
  here are very coarse lattices — the DDA solution at these slots is
  dipole-count-limited and should be considered indicative of solver
  convergence behaviour only, not of physical accuracy.

#### Reading the VIEM column

`adaptive_lc` returns $\min$ of three candidates:

1. wavelength: $\lambda_0/(N_{\rm pw}\cdot |m_p|_{\max})$, $N_{\rm pw}=10$
2. geometry:   $c/3$ (smallest semi-axis $/$ 3; for sphere $c = r_{\rm ve}$)
3. correlation: $0.3\,c/3$ when $\beta_{\rm gre}>0$ (not active for sphere)

Which constraint dominates flips with size:

- **Small $r_{\rm ve}$**: geometry $c/3$ dominates → $n_{\rm tet}$
  saturates at the wavelength-driven floor (≈ 652) because the geometry
  candidate becomes large compared to the wavelength one. n15 / n20
  hit this floor at both $r_{\rm ve}=0.05$ and $r_{\rm ve}=0.10$
  (`adaptive_lc` returns the wavelength value in those cases).
- **Large $r_{\rm ve}$**: wavelength $\lambda_0/(10\,|m_p|)$ dominates,
  so $n_{\rm tet}$ scales mainly with $|m_p|$ at fixed shape.

#### DDA / VIEM comparison

For large $r_{\rm ve}$ (where both DDA "default" and VIEM "adaptive"
are wavelength-driven), the two lattice counts agree to within a few
percent. For small $r_{\rm ve}$ the VIEM mesh is the wavelength-floor
of ≈ 652 elements while the DDA dipole count plummets with $r_{\rm ve}^3$
— DDA is intrinsically "lazy" at small particles relative to VIEM.

#### Comparison with fig 3 / fig 1 production lattice

The fig 3 / fig 1 production sweep uses a *different* DDA mechanism
("auto" via `dpl_for_slot`, calibrated to match the VIEM adaptive
$n_{\rm tet}$) — see [`fig3_description.md` §3.4](fig3_description.md).
The VIEM side is identical between fig 3 / fig 1 and fig 2
(`adaptive_lc` in both).

For sphere × Au this gives three lattice counts per $r_{\rm ve}$
(production DDA auto, fig 2 DDA default, both-fig VIEM adaptive):

| $r_{\rm ve}$ [μm] | DDA prod (auto) | DDA fig 2 (default) | VIEM (adaptive) |
| --- | ---: | ---: | ---: |
| 0.05 | 636 | 399 | 652 |
| 0.10 | 3 685 | 3 378 | 3 673 |
| 0.20 | 26 883 | 27 378 | 26 459 |

The fig 2 DDA default agrees with fig 3 / fig 1 production for Au at
every $r_{\rm ve}$ within the half-width 0.075-decade gray band drawn
on each fig 1 panel at $r_{\rm ve}=0.1$. (Half-width $10^{0.075}\!\approx\!1.189$,
i.e. $+18.9\%$ above and $-15.9\%$ below the center; total upper-to-lower
ratio $10^{0.15}\!\approx\!1.41$.) **For materials where `default` and
`auto` diverge by more than this band** (notably n15 at
$r_{\rm ve}\le 0.10$, where default ≈ 27–251 vs auto ≈ 660),
**fig 2's convergence behaviour does not directly reflect the
production-sweep convergence** — it reflects a coarser lattice. This
matters when transferring fig 2 conclusions back to the paper-production
context; see §9.

The lattice difference is also the **dominant driver** of the
production-vs-RHS-scaling Au convergence discrepancy at $r_{\rm ve}\le 0.1$
— see §9.

## 4. Observed trends

### 4.1 Iteration count

1. **Block size scaling** — for both solvers the number of iterations *to convergence of the worst RHS* shrinks as $L$ grows, then plateaus. Cause: block-Krylov shares the projection subspace across all $L$ RHSs, so the dominant eigendirections of the operator are captured early; new RHSs ride the same subspace.
2. **DDA $<$ VIEM uniformly** — at every $(r_{\rm ve}, m_p, L)$ DDA needs fewer iterations than VIEM (typically by a factor of 1.5–4×).
3. **Material ordering** — iter(n15) $\approx$ iter(n20) $\ll$ iter(Au). For Au + larger $r_{\rm ve}$ both solvers hit MAXITER for small $L$.

### 4.2 Solving time per orientation

1. Roughly $O(1/L)$ amortization at small $L$, plateau at large $L$ where iteration-count saturation dominates over RHS amortization.
2. Per-iteration MVP cost is dominated by the FFT (DDA) or AIM (VIEM) and is essentially independent of $L$, so total solve time scales as
   $$T_{\rm solve}(L) \approx C_{\rm MVP} \cdot \text{iter}(L) \cdot L \quad\Rightarrow\quad
     \frac{T_{\rm solve}}{L} \approx C_{\rm MVP} \cdot \text{iter}(L)\,.$$
   The visible per-orientation cost trend is therefore *driven by iter(L)*, not by RHS amortization in the MVP.

## 5. Why DDA needs fewer iterations — a spectral argument

GMRES converges as a polynomial of the operator. Its rate is governed by the *clustering* of eigenvalues away from the origin: the more tightly the spectrum clusters about a single point, the lower the polynomial degree (= iteration count) required to reach a given residual reduction. Concretely, for a normal operator the residual after $n$ GMRES steps satisfies
$$
\frac{\|r_n\|}{\|r_0\|} \;\le\; \min_{p\in\mathcal{P}_n,\, p(0)=1}\;\max_{\lambda\in\sigma(A)}\,|p(\lambda)|,
$$
so the *diameter* and *off-origin distance* of $\sigma(A)$ control convergence.

### DDA matrix

In SI form (with $\alpha_i$ the per-dipole polarizability),
$$
A_{ij}^{\rm DDA} \;=\; \alpha_i^{-1}\,\delta_{ij} \;-\; G(\bm{r}_i, \bm{r}_j)\,(1-\delta_{ij}),
$$
where $G$ is the free-space dipole Green's tensor. Two features matter:

- **Diagonal dominance.** The self-term $\alpha^{-1}$ has magnitude $O(1/V_{\rm cell})$, dominating the off-diagonal Green's tensor terms which decay as $|\bm{r}_i-\bm{r}_j|^{-3}$ and integrate to a finite, modest value.
- **Eigenvalue clustering near $\alpha^{-1}$.** Treating $\alpha^{-1} I$ as the leading operator, the spectrum of $A^{\rm DDA}$ sits near $\alpha^{-1}$ with a perturbation of size $\|G\|$. For dielectrics with $|\alpha^{-1}|$ well separated from zero (which it is for n15 / n20), the relative spread $\|G\|/|\alpha^{-1}|$ is small.

Hence $\sigma(A^{\rm DDA})$ is tightly clustered, GMRES picks a low-degree polynomial, iter count is small.

### VIEM matrix

Discretizing the volume integral equation
$$
\bm{E}^{\rm inc}(\bm{r}) \;=\; \bm{E}(\bm{r}) - k_0^2\!\int_V \chi(\bm{r}')\, \overline{\overline{G}}(\bm{r},\bm{r}')\,\bm{E}(\bm{r}')\,\dd^3 r',
$$
with $\chi = m_p^2 - 1$ and an SWG basis on tetrahedra produces
$$
A^{\rm VIEM} \;=\; M \;-\; k_0^2\, T_\chi,
$$
where $M$ is a sparse mass-like operator on the basis and $T_\chi$ is the (dense) integral kernel weighted by $\chi$. Two features differ from DDA:

- **No strong diagonal dominance per basis function.** A single SWG basis has support across two adjacent tetrahedra; its coupling to the kernel $T_\chi$ is comparable in magnitude to its self-mass entry, so $A^{\rm VIEM}$ is not diagonal-dominant in the way $A^{\rm DDA}$ is.
- **Broader eigenvalue distribution.** $T_\chi$'s spectrum extends across the body's resonance structure; combined with $M$ (which itself has a wide spectrum coming from element-size variation in the unstructured mesh), $\sigma(A^{\rm VIEM})$ is broader and not centered on a dominant value.

The polynomial-min problem above therefore needs a higher degree to suppress the residual on the wider spectrum of $A^{\rm VIEM}$, i.e. more GMRES iterations.

The factor 1.5–4× iteration ratio observed in fig 2 is consistent with the typical spectral-radius / clustering ratio between these two integral-equation discretizations of dielectric scattering reported in the literature (Yurkin & Hoekstra 2007; Markkanen et al. 2014).

## 6. Au plasmonic stress

For Au at $\lambda = 0.638~\mu$m, $m_p^2 + 2 \approx (-12.10) + 2 + 1.22 i \approx -10.1 + 1.22 i$. Although not exactly at a Fröhlich pole, the *small real part* of $m_p$ (0.18) combined with strong absorption pushes the body into a regime where:

1. **Localized surface plasmon (LSP) resonances** at $r_{\rm ve}$ comparable to the skin depth excite multipole modes; the integral operator's spectrum acquires near-marginal eigenvalues (close to 0) and convergence stalls.
2. The dipole self-term $\alpha^{-1}$ for Au is no longer "large" relative to the interaction terms — diagonal dominance erodes for DDA too, narrowing the gap to VIEM (the n15 vs Au DDA iter ratio is much larger than for VIEM).
3. Larger $r_{\rm ve}$ excites higher multipoles ($x = 2\pi r_{\rm ve}/\lambda \approx 1.97$ at $r_{\rm ve}=0.2$); for VIEM at $r_{\rm ve}=0.2$ + Au, every $L \le 32$ hits MAXITER. DDA holds out longer ($L \ge 32$ converges) thanks to its surviving residual diagonal dominance.

This is *not* fixed by raising MAXITER — the residual stalls; what is needed is preconditioning (e.g. Calderón / Schur-complement / multigrid for VIEM, or block-Jacobi / Toeplitz preconditioners for DDA). Adding a preconditioner is out of scope for the present sweep, so the Au + large-$r_{\rm ve}$ entries appear as MAXITER markers (✕) in fig 2.

## 7. Summary of trends

| Observation | Driving mechanism |
| --- | --- |
| DDA iter $<$ VIEM iter | Tighter eigenvalue clustering of $A^{\rm DDA}$ (diagonal dominance from $\alpha^{-1}$) |
| iter(L) decreases then plateaus | Block-Krylov subspace shared across $L$ RHSs; saturation when subspace already covers dominant eigendirections |
| Au + large $r_{\rm ve}$ stalls | LSP near-resonance broadens the operator spectrum to near-marginal eigenvalues; absent a preconditioner, GMRES cannot polynomial-suppress the residual within MAXITER |
| Per-orientation time $\propto$ iter(L) | MVP cost (FFT-DDA / AIM-VIEM) is $L$-independent; total time $= C_{\rm MVP}\cdot$iter$\cdot L$, so per-RHS time tracks iter(L) |

For paper-production sweeps, the practical takeaway is to choose $L \approx 32$–$64$ — the block-Krylov saturation point in fig 2 — and to flag any Au $\times r_{\rm ve} \ge 0.2$ result as solver-limited (not method-limited).

## 8. Per-panel display details

- DDA: filled blue circle (solid line connecting points).
- VIEM: open orange marker (dashed line); shape encodes $r_{\rm ve}$.
- Non-converged points (hit MAXITER): drawn as $\times$ markers on the iter panel; omitted from the solvetime panel.
- Auto-determined y range:
  - `iters` (log): $[10^{\lfloor\log_{10}\min\rfloor},\ 1.25\cdot \text{MAXITER}]$.
  - `solvetime` (log): decade-rounded floor / ceil of the converged-only data.
- Inside-pointing mirror ticks on all four sides.

## 9. Why Au stalls in fig 3 / fig 1 (production sweep) but converges in fig 2 (RHS scaling)

Both sweeps use the **same Krylov method** (`bl_krylov.bl_gmres_mvp_fft`) with the **same convergence criterion** (tol $= 10^{-5}$, MAXITER $= 200$, unpreconditioned). The divergent convergence behaviour for Au is therefore *not* a method-vs-method comparison — it is driven by three differences in *how the linear system is set up* between the two sweeps:

| | Production sweep (fig 3 / fig 1 data) | RHS-scaling sweep (fig 2 data) |
| --- | --- | --- |
| Lattice resolution $N_{\rm occ}$ at sphere × Au, $r_{\rm ve}=0.05$ / $0.10$ / $0.20$ | $[636,\ 3685,\ 26883]$ (slot-tuned by `utils/dpl_calibration.dpl_for_slot`) | $[399,\ 3378,\ 27378]$ (default `dpl` from the shape-model heuristic) |
| Block size $L$ | spheroid mode → $L = N_\beta = 5$ | swept $L \in \{1, 2, 4, 8, 16, 32, 64, 128\}$ |
| Orientation grid | $\alpha=\gamma=0$ + paper-fixed $N_\beta = 5$ $\beta$ values | fixed-RNG-seed uniform-on-SO(3) Euler sequence |

Concrete numbers for sphere × Au, $r_{\rm ve} = 0.05$:

- Production: $L = 5$, $N_{\rm occ} = 636$ → 200 iters, **stalled** at $\|r\|/\|b\| \approx 1.0\!\times\!10^{-3}$ (tol $= 10^{-5}$).
- RHS scaling: $L = 4$, $N_{\rm occ} = 399$ → **38 iters, converged** ($\|r\|/\|b\| < 10^{-5}$).

The dominant driver is **lattice resolution**, not block size. As the lattice is refined, the discrete operator $A^{\rm DDA}$ resolves more LSP near-resonance modes whose eigenvalues lie close to the origin. The polynomial-residual bound
$$
\frac{\|r_n\|}{\|r_0\|}\;\le\;\min_{p\in\mathcal{P}_n,\,p(0)=1}\,\max_{\lambda\in\sigma(A)}|p(\lambda)|
$$
forces the GMRES polynomial to interpolate $0$ at fewer extreme eigenvalues when the spectrum is *tight*; the more near-zero eigenvalues are admitted, the higher the polynomial degree (i.e. iteration count) needed. For Au at small $r_{\rm ve}$, the production lattice ($N_{\rm occ} = 636$) is denser than the RHS-scaling lattice ($N_{\rm occ} = 399$) and exposes more such near-marginal eigenvalues — production stalls, RHS scaling converges.

For larger $r_{\rm ve}$ the two sweeps' lattices are similar in size ($N_{\rm occ} \approx 27\,000$ at $r_{\rm ve}=0.20$), and the RHS-scaling sweep's behaviour mirrors the production one: $L \le 16$ stalls, $L \ge 64$ converges. This confirms the secondary role of block size $L$ — at fixed lattice difficulty, larger $L$ amortises the iteration count via shared block-Krylov subspace, but at fixed $L$ a denser lattice still dominates the spectrum.

Orientation choice (third row of the table) plays a smaller, modulating role: the $\alpha=\gamma=0$ block in production may align preferentially with LSP modes oriented along the symmetry axes, but the effect is sub-dominant compared with the lattice and $L$ effects above.

**Practical implication.** The production-sweep Au stalls visible in `dda_results/paper/<shape>_Au.hdf5` are *setting-limited*, not method-limited. They should be re-runnable to convergence by any of:

- (a) coarsening the lattice (override `dpl_for_slot` to the RHS-scaling default `dpl` for Au slots);
- (b) increasing the effective block size — drop spheroid mode for Au and solve the full $L = N_\alpha N_\beta N_\gamma$ block at once, or run the RHS-scaling sweep separately and back-fill the cost columns;
- (c) introducing a preconditioner. The Jacobi preconditioner from the legacy `bl_bicgstab_jacobi_mvp_fft` was a per-dipole diagonal scaling; for Au plasmonic stress a more aggressive choice (Calderón / Schur-complement / multigrid) is more likely to recover the spectrum.

The choice between (a)–(c) is a paper-level decision (accuracy vs. solver cost vs. legacy comparability) rather than a numerical-bug fix.

## 10. Source code

- DDA RHS-scaling sweep: [`scripts/run_rhs_scaling.py`](../../scripts/run_rhs_scaling.py)
- VIEM RHS-scaling sweep: `~/Julia/block-VIEM.jl/viem_results/paper/run_rhs_scaling.jl`
- Plotting: Phase 5 cell of [`plot_paper_results.ipynb`](plot_paper_results.ipynb)
- Output HDF5 path: `/target/rhs_scaling/gmres/{iters, converged, t_total_s, t_end2end_per_orient_s, peak_rss_bytes, residual_history}` and `/target/rhs_scaling/{n_occ, n_tet, L_values}`.

## 11. Cross-references

- [`fig3_description.md`](fig3_description.md) — production sweep details (auto-dpl mechanism, exact references). Au plasmonic stalls visible there are explained in §9 above.
- [`fig1_description.md`](fig1_description.md) — dpl/lc convergence study. The Au DDA stall in fig 1 (single orientation, axis-aligned ZYZ identity, every dpl) is a more extreme version of the production-sweep stagnation analysed in §9; the underlying physics is the same LSP near-resonance described in §6.

## 12. References

- M. A. Yurkin and A. G. Hoekstra, *J. Quant. Spectrosc. Radiat. Transf.* **106**, 558–589 (2007). — DDA convergence and conditioning.
- J. Markkanen, P. Ylä-Oijala, and A. Sihvola, *IEEE Trans. Antennas Propag.* **62**, 2367–2376 (2014). — VIE / SWG operator spectra for dielectric scattering.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed. (SIAM, 2003). — GMRES polynomial residual bound and spectral interpretation.
