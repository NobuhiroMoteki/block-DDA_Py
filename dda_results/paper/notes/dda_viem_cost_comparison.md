# DDA vs VIEM — Production-sweep cost comparison

This document compares the wall-time and peak-RSS cost of `block-DDA_Py`
and `block-VIEM.jl` on the v0.7.7 paper-production sweep. The two
solvers are operated in their **practical** discretisation modes:

- **DDA**: `dpl_for_slot` auto-calibrated to the VIEM `n_tet` of each slot
  (production "auto" mechanism — same lattice as in fig 1's gray band).
  Solver: `bl_krylov.bl_gmres_mvp_fft`, unpreconditioned, tol = 10⁻⁵,
  MAXITER = 200.
- **VIEM**: `adaptive_lc` with $N_{\rm pw} = 10$ wavelength-sampling and
  $c/3$ geometry-floor.  Solver: `BlockVIEM.block_gmres` (`:aim_gmres`
  variant), same tol / MAXITER.

The solvers differ only in basis (point dipoles on a cubic lattice
vs. SWG basis on a tetrahedral mesh) and in MVP acceleration (FFT vs.
AIM); both target the same Maxwell scattering problem under identical
host wavelength, polarisation, and orientation grid. The comparison
below isolates the cost difference per physical observable, **at
matched volume-element count**.

Schema details and source-code pointers: see
[fig1_description.md](../fig1_description.md) §3 (lc / dpl mechanisms),
[fig2_description.md](../fig2_description.md) §3 (RHS-scaling lattice
table), and [cost_estimates.md](../cost_estimates.md) (per-file totals).

## 1. Notation: volume-element counts vs. unknown counts

Each solver uses **two distinct counts** that look interchangeable but
play different roles in the cost analysis. Misreading them causes a
factor 2 – 9 confusion in any back-of-envelope estimate.

### 1.1 DDA — `N_cuboid` vs `N_occ`

| symbol | meaning |
| --- | --- |
| `N_cuboid` | total cells in the bounding-box cubic lattice that contains the particle (**including vacuum cells inside the bbox but outside the particle**). Sets the FFT size — Toeplitz embedding requires the full bbox. |
| `N_occ` | "occupied" cells — those whose centre lies inside the particle. Each occupied cell carries one Cartesian-vector dipole, so the **physical unknown count is $3\,N_{\rm occ}$**. The matrix–vector workload per dipole scales as $N_{\rm occ}$. |

Relation: $N_{\rm occ} \le N_{\rm cuboid}$. The ratio $N_{\rm cuboid}
/ N_{\rm occ}$ is the bbox-dilution factor:

| shape family | $N_{\rm cuboid}/N_{\rm occ}$ |
| --- | ---: |
| sphere / oblate / doublet | ~2.2 |
| GRE (sparse, non-convex) | ~8 – 9 |

DDA's FFT working buffers scale with $N_{\rm cuboid}\cdot L$ rather
than $N_{\rm occ}\cdot L$ — this is why GRE shapes spend disproportionate
memory per unit physical volume (§6).

### 1.2 VIEM — `N_tet` vs `N_dof`

| symbol | meaning |
| --- | --- |
| `N_tet` | tetrahedra in the unstructured mesh of the particle interior (VIEM does not discretise the host vacuum). Sets mesh-build / geometry-loop cost; the natural "discretisation resolution" indicator. |
| `N_dof` | SWG (Schaubert-Wilton-Glisson) basis-function count = **VIEM unknown count**. Each SWG basis lives on a tet face (vector flow across a face shared by two tetrahedra), with an additional contribution from boundary faces under the production setting `include_boundary_faces=true`. |

Relation: in a closed tetrahedral mesh, internal faces number
$\approx 2\,N_{\rm tet}$ (each tet has 4 faces; each internal face is
shared by two tets). With boundary faces included,
$$
N_{\rm dof}\;\approx\;2.0 - 2.3\;\times\;N_{\rm tet}.
$$
Empirically: sphere n20 at $a_{\rm ve}=0.40$ µm gives
$N_{\rm dof}/N_{\rm tet} = 80\,810/39\,167 \approx 2.06$;
sphere n15 at $a_{\rm ve}=0.05$ µm gives $1\,465/652 \approx 2.25$.

### 1.3 Which count is "the" volume-element count?

When this document says "volume-element count" or "matched $N_{\rm
vol}$", the pairing is

$$
N_{\rm occ}\;\;\leftrightarrow\;\;N_{\rm tet}
$$

— both count *interior particle elements directly*, ignoring vacuum.
This is the pairing that `dpl_for_slot` (DDA's auto-dpl) calibrates
against — see [fig1_description.md §3.4](../fig1_description.md).

Three different pairings are needed for different cost questions:

| cost question | DDA | VIEM |
| --- | --- | --- |
| physical-volume resolution | $N_{\rm occ}$ | $N_{\rm tet}$ |
| linear-system unknowns | $3\,N_{\rm occ}$ | $N_{\rm dof} \approx 2\,N_{\rm tet}$ |
| memory-scaling unit (FFT pads / dense block) | $N_{\rm cuboid}$ | $N_{\rm dof}$ |

At matched $N_{\rm occ} = N_{\rm tet}$ the linear-system sizes are
similar ($3\,N_{\rm occ}$ vs $\sim 2\,N_{\rm tet}$ → DDA has ~1.5×
more unknowns per matched element), but the **memory-scaling units
are structurally different** — $N_{\rm cuboid}$ vs $N_{\rm dof}$ —
which is why §6's memory model uses different counts on each side.

## 2. Volume-element counts are matched within a few percent

The "auto" DDA mechanism deliberately solves
$$
\text{dpl}_{\rm DDA}\;:\;N_{\rm occ}^{\rm DDA}\;\approx\;n_{\rm tet}^{\rm VIEM}\cdot 1.0
$$
to within $\pm 5\%$ for axially symmetric shapes (sphere, oblate,
doublet) and within $\pm 35\%$ for GRE (DDA tends to over-resolve
because of cubic-lattice mass-discretisation overhead near the
boundary). The realised ratio $r = N_{\rm occ}^{\rm DDA}/n_{\rm tet}^{\rm VIEM}$
per slot:

| shape   | n15 (4 slots)   | n20 (4 slots)   | Au (3 slots)    |
| ---     | ---             | ---             | ---             |
| sphere  | 0.98 – 1.05     | 0.98 – 1.03     | 0.98 – 1.02     |
| oblate  | 0.96 – 1.06     | 0.96 – 1.04     | 0.96 – 1.02     |
| gre     | 1.23 – 1.38     | 1.23 – 1.35     | 1.23 – 1.33     |
| doublet | 0.98 – 1.00     | 0.98 – 1.00     | 0.98 – 1.00     |

For axially symmetric shapes "matched volume elements" is essentially
exact. For GRE, $N_{\rm occ}^{\rm DDA}$ runs ~1.3× larger than
$n_{\rm tet}^{\rm VIEM}$; this should be kept in mind when interpreting
the GRE rows below — a fully apples-to-apples GRE comparison would
need a tighter `dpl_for_slot` tolerance, but the present setting is the
production paper choice and is reported as such.

## 3. Wall time at matched $N_{\rm vol}$ — converged slots

Slots where **both** solvers reached tol $= 10^{-5}$.  Au is excluded
here (handled separately in §4) because both solvers stall on Au.

Time is end-to-end `t_total_s` per slot (build + setup + solve +
observables). $L$ is the effective block size used by the spheroid /
GRE dispatch (sphere / oblate / doublet → $L = N_\beta = 5$; GRE →
$L = N_\alpha N_\beta N_\gamma = 100$).

| shape   | mat | $a_{\rm ve}$ [µm] | $L$ | $N_{\rm occ}$ | $n_{\rm tet}$ | iter D / V | $t_{\rm DDA}$ [s] | $t_{\rm VIEM}$ [s] | $t_{\rm V}/t_{\rm D}$ |
| ---     | --- | ---:              | ---:| ---:          | ---:          | ---        | ---:              | ---:               | ---:                  |
| sphere  | n15 | 0.05              | 5   |    636        |    652        | 7 / 24     | 1.7               | 10.4               | 6.2                   |
| sphere  | n15 | 0.10              | 5   |    660        |    652        | 8 / 24     | 1.7               | 5.4                | 3.1                   |
| sphere  | n15 | 0.20              | 5   |  2 313        |  2 229        | 11 / 27    | 2.7               | 10.2               | 3.8                   |
| sphere  | n15 | 0.40              | 5   | 17 425        | 16 671        | 22 / 39    | 9.2               | 102.9              | 11.1                  |
| sphere  | n20 | 0.05              | 5   |    636        |    652        | 11 / 24    | 1.8               | 9.9                | 5.7                   |
| sphere  | n20 | 0.10              | 5   |    696        |    710        | 13 / 24    | 1.8               | 5.3                | 2.9                   |
| sphere  | n20 | 0.20              | 5   |  5 329        |  5 151        | 22 / 34    | 4.3               | 22.0               | 5.1                   |
| sphere  | n20 | 0.40              | 5   | 40 515        | 39 167        | 70 / 88    | 68.0              | 187.9              | 2.8                   |
| oblate  | n15 | 0.40              | 5   | 18 049        | 17 046        | 22 / 39    | 9.9               | 107.6              | 10.8                  |
| oblate  | n20 | 0.40              | 5   | 40 818        | 39 107        | 66 / 84    | 63.5              | 185.4              | 2.9                   |
| gre     | n15 | 0.40              | 100 | 27 438        | 19 853        | 11 / 51    | 236.0             | 517.6              | 2.2                   |
| gre     | n20 | 0.20              | 100 | 26 614        | 19 853        | 15 / 51    | 327.8             | 522.3              | 1.6                   |
| gre     | n20 | 0.40              | 100 | 52 775        | 39 167        | 20 / 61    | 888.2             | 5 189.8            | 5.8                   |
| doublet | n15 | 0.40              | 5   | 17 593        | 17 612        | 22 / 39    | 7.7               | 101.8              | 13.2                  |
| doublet | n20 | 0.40              | 5   | 41 313        | 41 441        | 62 / 78    | 60.1              | 301.5              | 5.0                   |

(Smaller $a_{\rm ve}$ rows for oblate / gre / doublet follow the same
trend as sphere and are omitted for brevity; full table in
[notebook Section 6](../plot_paper_results.ipynb) cost summary cell.)

**Pattern.**  The VIEM-to-DDA wall-time ratio across all 32 converged
slots is $t_{\rm V}/t_{\rm D} \approx 2 - 30$, with the spread driven
by:

1. *Iteration count* — DDA always converges in fewer GMRES steps than
   VIEM (typically by a factor 1.5 – 4×), see fig 2 §5 for the spectral
   reason (DDA's $\alpha^{-1}$ self-term yields tighter eigenvalue
   clustering than VIEM's SWG / mass operator).
2. *Per-iteration MVP cost* — DDA's FFT on the cubic lattice is faster
   per element than VIEM's AIM on the unstructured tet mesh in the
   $N_{\rm occ} \lesssim 5\,\!000$ regime (fixed plan / projection
   overhead amortised differently), even when $N_{\rm occ} \approx
   n_{\rm tet}$.
3. *Setup overhead* — VIEM build + AIM-projection assembly is a
   one-time $O(n_{\rm tet})$ cost that dominates at small particles.
   At $a_{\rm ve} = 0.05$ µm this gives the largest $t_{\rm V}/t_{\rm D}$
   ratios (factor ~30 for doublet n15).

The advantage shrinks as $N_{\rm vol}$ grows (factor ~2.8 for
sphere n20 at $a_{\rm ve} = 0.40$ µm) because DDA's FFT cost scales as
$O(N_{\rm cuboid} \log N_{\rm cuboid})$ while VIEM's AIM scales similarly
but with smaller pre-factor for the dense near-field block.

## 4. Wall time on Au — both solvers stall, but cost asymmetry remains

Every Au slot hits MAXITER = 200 on at least one side; $t_{\rm total}$
therefore reflects the budget cap rather than convergence cost.
Even so, the comparison is informative because the per-iteration
MVP cost differs between solvers and shapes:

| shape   | $a_{\rm ve}$ [µm] | $L$  | $N_{\rm occ}$ | $n_{\rm tet}$ | iter D / V | $t_{\rm DDA}$ [s] | $t_{\rm VIEM}$ [s] | $t_{\rm V}/t_{\rm D}$ |
| ---     | ---:              | ---: | ---:          | ---:          | ---        | ---:              | ---:               | ---:                  |
| sphere  | 0.05              | 5    |    636        |    652        | 200 / 176  | 7.6               | 26.2               | 3.4                   |
| sphere  | 0.10              | 5    |  3 685        |  3 673        | 200 / 200  | 28.2              | 42.7               | 1.5                   |
| sphere  | 0.20              | 5    | 26 883        | 26 459        | 200 / 200  | 212.2             | 201.9              | 0.95                  |
| oblate  | 0.20              | 5    | 26 728        | 26 160        | 200 / 200  | 204.5             | 196.1              | 0.96                  |
| doublet | 0.20              | 5    | 27 505        | 27 573        | 200 / 200  | 216.9             | 713.4              | 3.3                   |
| **gre** | 0.05              | 100  | 24 416        | 19 853        | 200 / 200  | **9 607**         | 2 245              | **0.23**              |
| **gre** | 0.10              | 100  | 25 578        | 19 853        | 200 / 200  | **9 980**         | 2 257              | **0.23**              |
| **gre** | 0.20              | 100  | 35 095        | 26 459        | 200 / 200  | **12 681**        | 2 474              | **0.20**              |

Two regimes appear:

- **Spheroid mode ($L = 5$)**: cost ratio remains close to 1 at large
  $N_{\rm vol}$ (sphere / oblate / doublet at $a_{\rm ve} = 0.20$ µm).
  DDA's per-iteration FFT is slightly faster than VIEM's AIM at this
  $N_{\rm occ}$ but block-Krylov costs equalise.
- **GRE mode ($L = 100$)**: DDA is **4 – 5× slower** than VIEM. Per-iter
  cost in DDA scales as $O(N_{\rm cuboid} \cdot L \cdot \log N_{\rm cuboid})$
  through the L-replicated FFT. With $N_{\rm cuboid} = 220\,990$ and
  $L = 100$ the FFT block dominates total wall time; AIM with its
  separate near-field block scales weaker in $L$.

Au is reported via VIEM in the paper (planned mitigation — see
[cost_estimates.md §v0.7.6 findings](../cost_estimates.md) item 2).

## 5. Peak RSS at matched $N_{\rm vol}$

Per-slot peak resident set size, all converged + Au slots:

| shape   | mat | $a_{\rm ve}$ [µm] | $L$ | $N_{\rm cub}$ (DDA) | $N_{\rm dof}$ (VIEM) | RSS DDA [GB] | RSS VIEM [GB] | RSS V/D |
| ---     | --- | ---:              | ---:| ---:                | ---:                 | ---:         | ---:          | ---:    |
| sphere  | n20 | 0.05              | 5   |   2 197             |   1 465              | 0.40         | 1.22          | 3.1     |
| sphere  | n20 | 0.10              | 5   |   2 744             |   1 581              | 0.41         | 1.27          | 3.1     |
| sphere  | n20 | 0.20              | 5   |  13 824             |  10 925              | 0.43         | 2.28          | 5.3     |
| sphere  | n20 | 0.40              | 5   |  91 125             |  80 810              | 1.53         | 9.69          | 6.3     |
| oblate  | n20 | 0.40              | 5   |  91 287             |  81 312              | 1.47         | 9.13          | 6.2     |
| doublet | n20 | 0.40              | 5   |  98 568             |  86 045              | 1.38         | 9.89          | 7.2     |
| **gre** | n15 | 0.40              | 100 | 220 990             |  41 285              | **27.92**    | 14.19         | **0.51**|
| **gre** | n20 | 0.40              | 100 | 423 096             |  80 810              | **55.48**    | 20.25         | **0.37**|
| **gre** | Au  | 0.20              | 100 | 291 600             |  54 806              | **69.63**    | 28.64         | **0.41**|

A clear **crossover** appears between $L = 5$ and $L = 100$:

- For $L = 5$ (axially symmetric): $\text{RSS}_{\rm V}/\text{RSS}_{\rm D}
  \approx 3 - 8$.  VIEM holds a dense per-tet AIM-projection block plus
  a near-field tetra-tetra interaction matrix that together dwarf
  DDA's FFT-only working set.
- For $L = 100$ (GRE): $\text{RSS}_{\rm V}/\text{RSS}_{\rm D}
  \approx 0.4 - 0.5$.  DDA's FFT working buffers replicate per RHS
  column, so peak RSS scales linearly in $L$; VIEM keeps a single
  AIM-projection matrix and only the Krylov subspace scales in $L$.

The transition happens at $L \approx 70 - 80$ for typical paper-
production $N_{\rm vol}$ — above this the per-RHS replication of FFT
pads catches up with VIEM's single dense block. The exact crossover
depends on the DDA bbox-to-occupancy ratio $N_{\rm cuboid}/N_{\rm
occ}$ (~2.2 for sphere / oblate / doublet but ~8 – 9 for the sparse
GRE shapes — see the $N_{\rm cub}$ column above), so GRE crosses
earlier (effective $L \sim 50$) than convex shapes.

## 6. Empirical memory model

Fitting the production-sweep data to a two-term model RSS $= A + B
\cdot N_{\rm vol} \cdot L$:

### DDA

$$
\text{RSS}_{\rm DDA}\;\approx\;A_{\rm D} + B_{\rm D}\cdot N_{\rm cuboid}\cdot L,
\qquad
A_{\rm D}\approx 0.4\ \text{GB},\quad
B_{\rm D}\approx 1.2 - 3.4\ \text{KB}/(\text{cuboid}\cdot\text{RHS}).
$$

The wide $B_{\rm D}$ range reflects that the FFT working arrays scale
with the *bounding box* $N_{\rm cuboid}$ rather than the *occupancy*
$N_{\rm occ}$, and small / non-convex shapes have $N_{\rm cuboid}
\gg N_{\rm occ}$. For sphere-like shapes $N_{\rm cuboid}/N_{\rm occ}
\approx 2.2$; for GRE this ratio is closer to 9 (see [GRE rows above]).

### VIEM

$$
\text{RSS}_{\rm VIEM}\;\approx\;A_{\rm V} + B_{\rm V}\cdot N_{\rm dof} + C_{\rm V}\cdot N_{\rm dof}\cdot L,
\qquad
A_{\rm V}\approx 1\ \text{GB},
$$
with $B_{\rm V} \approx 100\ \text{KB}/\text{DOF}$ (dense near-field +
mass + AIM projection — independent of $L$) and $C_{\rm V} \approx
1.4\ \text{KB}/(\text{DOF}\cdot\text{RHS})$ (Krylov subspace, scales
in $L$).  The $B_{\rm V}$ term dominates at small / moderate $L$;
$C_{\rm V}\cdot L$ reaches parity with $B_{\rm V}$ only at $L \approx
B_{\rm V}/C_{\rm V} \approx 73$ — i.e. VIEM is essentially $L$-flat in
peak RSS up to $L \sim$ a few tens.

### Why VIEM's AIM cubic grid is *not* memory-dominant

VIEM also runs an FFT on a Cartesian cubic grid (the AIM grid), but
unlike DDA's $N_{\rm cuboid}\cdot L$ scaling this contribution is
absorbed inside the $L$-independent $B_{\rm V}$ term. This subsection
explains why, since the AIM grid is naively *larger* than DDA's
$N_{\rm cuboid}$.

**AIM grid size.** The production setting
([run_lc_convergence.jl:42-43](../../../../../Julia/block-VIEM.jl/viem_results/paper/run_lc_convergence.jl)):

```julia
const AIM_PITCH_RATIO = 0.5
const AIM_PADDING     = 4
pitch = AIM_PITCH_RATIO * h_bar    # h_bar = mean tet edge length
```

gives AIM grid spacing $\Delta_{\rm AIM} = 0.5\,\bar{h}$, which is
**half** of DDA's matched-discretisation lattice spacing $d \approx
\bar{h}$.  Per unit volume the AIM grid therefore has $2^3 = 8\times$
more cells than DDA's $N_{\rm cuboid}$.  For sphere n20 at $a_{\rm
ve}=0.40$ µm, $\bar{h}\approx 0.032$ µm so pitch $\approx 0.016$ µm,
the bbox $\approx 0.93$ µm gives

$$
N_{\rm AIM\_grid}\;\approx\;\bigl(0.93/0.016\bigr)^{3}\;\approx\;1.95\times 10^{5},
$$

which is ~2× larger than the DDA $N_{\rm cuboid}=91\,125$ on the same
slot.  Naively one might expect VIEM's FFT memory to dominate.

**Why it doesn't dominate, point 1: the kernel FFT is precomputed and
shared.** VIEM precomputes the dyadic Green tensor on the AIM grid
once (6 unique components by symmetry, padded to power-of-2 for FFT):

$$
\widehat{G}(\bm{k})\;:\;\text{shape}\;\approx\;\bigl(6,\ N_{\rm AIM\_grid}^{\rm padded}\bigr)\;\text{complex128}.
$$

This is a one-off $O(N_{\rm AIM\_grid})$ buffer that does not scale in
$L$ or in the GMRES iteration count.  For sphere n20 at
$a_{\rm ve}=0.40$ µm this is ~140 MB, i.e. ~1.8 KB/DOF — small relative
to the 100 KB/DOF of $B_{\rm V}$.

**Why it doesn't dominate, point 2: AIM MVP working buffers are not
$L$-replicated.** VIEM's `:aim_gmres` applies the operator to the
block $X_{\rm block}\in\mathbb{C}^{N_{\rm dof}\times L}$ by **reusing
the AIM grid buffers across columns** (project SWG → grid, FFT, kernel
multiply, IFFT, project grid → SWG, repeated per column or in small
batches).  The peak grid-side working set is $O(N_{\rm AIM\_grid})$,
**not** $O(N_{\rm AIM\_grid}\cdot L)$.

DDA's `mvp_fft`, in contrast, applies a **batched FFT across all $L$
columns simultaneously** — array shape $(3,\ N_{\rm cuboid}^{\rm padded},\ L)$
in complex128 — to amortise FFTW plan overhead and exploit thread-level
parallelism. This is faster per MVP but makes the FFT pad memory scale
linearly in $L$, producing the $B_{\rm D}\,N_{\rm cuboid}\,L$ term
above.

**Breakdown of VIEM's $B_{\rm V}\approx 100$ KB/DOF.** Approximate
attribution from the production data and the AIM-GMRES code path:

| component | KB/DOF |
| --- | ---: |
| dense near-field block (SWG self + nearest neighbours, $\sim 50$–100 couplings × complex128) | ~80 |
| AIM projection matrix (sparse, $(\text{poly\_order}+1)^3 \times \text{stencil}^3$ nonzeros / basis) | ~10 |
| AIM kernel FFT (precomputed dyadic Green tensor on the cubic grid, shared) | ~2 |
| mass matrix, mesh / basis structures, Julia GC overhead | ~8 |

The AIM-grid FFT thus accounts for only a few percent of $B_{\rm V}$;
the dense near-field block dominates.

**Structural summary.** The DDA-vs-VIEM memory asymmetry is therefore
not about discretisation-unit counts but about **MVP batching strategy**:

| | DDA | VIEM (AIM) |
| --- | --- | --- |
| cubic-grid cell count | $N_{\rm cuboid}$ | $N_{\rm AIM\_grid}\approx 2\,N_{\rm cuboid}$ |
| precomputed kernel FFT (shared, $L$-flat) | $\sim N_{\rm cuboid}$ | $\sim N_{\rm AIM\_grid}$ |
| MVP working buffer | $\propto N_{\rm cuboid}\cdot L$ (batched) | $\propto N_{\rm AIM\_grid}$ ($L$-flat) |
| dominant $L$-independent term | FFT pad | dense near-field |
| dominant $L$-dependent term | FFT pad | Krylov subspace |

VIEM's wider cubic grid is more than offset by its non-batched MVP,
which is why VIEM holds a roughly $L$-flat memory footprint up to
$L \approx 70$–$80$ while DDA grows linearly in $L$ from the start.

### Why VIEM should *not* adopt DDA's batched-FFT strategy

The asymmetry above prompts a natural question: should VIEM not
adopt the same batched-FFT MVP as DDA to chase its wall-time edge?
The answer is **no** — five structural constraints push VIEM the
opposite way.

**(1) AIM-MVP is multi-stage; FFT is one ingredient of four.**

| stage | DDA | VIEM (AIM) |
| --- | --- | --- |
| (i) project SWG → AIM grid | — | sparse matvec, $O(N_{\rm dof}\cdot \text{stencil}^3)$ |
| (ii) 3D FFT (forward + kernel multiply + inverse) | **the entire MVP**, $O(N_{\rm cub}\log N_{\rm cub})$ | $O(N_{\rm AIM}\log N_{\rm AIM})$ |
| (iii) project AIM grid → SWG | — | sparse matvec |
| (iv) dense near-field correction | — | dense block, $O(N_{\rm dof}^{\rm near})$ |

DDA's MVP **is** the FFT, so batching across $L$ amortises the entire
MVP cost. VIEM's AIM-MVP has four stages and the FFT accounts for
only ~30 – 50% of the operation count; batching it alone captures at
most that fraction of wall-time savings while paying the full per-$L$
memory penalty.

**(2) The AIM grid is 8× finer per volume than DDA's lattice.**

With $\Delta_{\rm AIM} = 0.5\,\bar{h}$ (production setting) versus
DDA's $d \approx \bar{h}$, the AIM grid has $2^3 = 8\times$ more cells
per unit volume. A hypothetical batched-FFT VIEM would carry an FFT
pad scaling as
$$
\text{(batched VIEM)}\;\propto\;8\,N_{\rm cuboid}\cdot L,
$$
which at $L = 100$ on the heaviest paper slot (gre n20 $a_{\rm ve}=0.40$
µm) balloons to ~100 – 200 GB **for the FFT working set alone**. Combined
with VIEM's existing dense near-field this exceeds the 1 TiB host's
~700 GB usable budget — i.e. the slot becomes unsolvable on the same
hardware.

**(3) It would defeat VIEM's role in the solver portfolio.**

Per §8, the production-sweep solver choice rests on a clear redundancy:
DDA covers $L \le 32$ (small-to-medium block); VIEM covers $L \ge 100$
(full SO(3) GRE grid / stiff materials). A batched-FFT VIEM would OOM
at $L = 100$ alongside DDA, collapsing the redundancy and leaving
**no solver** capable of running the paper's GRE × Au case. VIEM's
$L$-flat memory is therefore not a dispensable design preference but
the **load-bearing property** of the two-solver design.

**(4) Block-Krylov's payoff is iteration count, not per-MVP cost.**

fig 2 §4 shows that for both solvers
$$
\text{iter}(L)\;\downarrow\;\text{with}\;L,\qquad
t_{\rm MVP}(L)\;\approx\;L\cdot t_{\rm MVP}(1),
$$
so per-MVP cost scales linearly in $L$ regardless of batching strategy.
The marginal wall-time gain from batched FFT is at most the FFT-plan
amortisation factor (~10 – 20%), an order of magnitude smaller than
the iteration-count drop that block-Krylov already provides without
batching. VIEM cannot recoup the memory blow-up via wall-time.

**(5) Projection stages are memory-bandwidth-bound.**

The SWG ↔ AIM-grid projections (stages (i) and (iii) above) are
sparse matvecs that are bandwidth-limited rather than compute-limited.
Julia's `SparseMatrixCSC * Matrix` walks the CSC structure column by
column with cache-friendly access; batching across $L$ would not
unlock additional arithmetic intensity but would inflate the active
working set. The end-to-end MVP is therefore not bottlenecked by FFT
plan overhead even at large $L$.

**Net effect.** The DDA / VIEM design asymmetry is not arbitrary. DDA
chose batched FFT because (a) MVP is *only* FFT, (b) the cubic
lattice is the coarsest reasonable grid, and (c) DDA's target regime
is small-to-medium $L$ where memory-per-RHS is acceptable. VIEM chose
column-wise / small-batch MVP because (a) MVP is multi-stage with
sub-linear FFT share, (b) the AIM grid is structurally finer, and
(c) VIEM's target regime is large-$L$ / stiff-material problems where
$L$-flat memory is non-negotiable. The two strategies are matched to
different cost regimes; transplanting either onto the other solver
would degrade rather than improve its role.

### Implication for solver choice

The peak-RSS crossover at $L \approx 70 - 80$ defines the regime
where the choice between DDA and VIEM is **memory-bounded** rather
than purely time-bounded:

- **Single-orientation or small-block ($L \le 16$)** problems:
  DDA wins on both wall time *and* memory.
- **Large-block GRE ($L = 100$, full $N_\alpha N_\beta N_\gamma$
  grid)**: VIEM wins on memory by a factor 2 – 3, and (because of
  Au plasmonic non-convergence in DDA at the production lattice) by a
  factor 4 – 5 on wall time. This is why the paper's GRE × Au results
  are reported via VIEM.

## 7. Aggregate sweep totals

Summed over **all** slots (converged + stalled), all shapes × all
materials × all $a_{\rm ve}$:

| solver | $\sum t_{\rm total}$ [min] | peak RSS observed [GB] |
| ---    | ---:                       | ---:                   |
| DDA    | 596.8                      | 69.63                  |
| VIEM   | 305.4                      | 28.64                  |

(Both peaks were observed at the same heaviest slot: GRE × Au ×
$a_{\rm ve} = 0.20$ µm, $L = 100$.)

*Caveat.* DDA's totals are inflated by its 10 stalled Au slots
(MAXITER = 200 × full block-MVP cost on the heavy GRE × Au geometry
contributes ~540 min by itself — see
[cost_estimates.md §v0.7.6 findings](../cost_estimates.md) item 3).
**Restricted to converged-only slots** the comparison reverses: DDA
sums to **45.7 min** vs VIEM's **166.9 min**, i.e. DDA is **~3.6×
faster** end-to-end on the converged subset. The raw aggregate above
mainly reflects the asymmetric Au-stall penalty.

## 8. Discussion

1. **DDA is faster *per converged slot* by 2 – 30 ×** in the practical
   matched-lattice regime, driven by ~2× fewer GMRES iterations
   (spectral clustering near $\alpha^{-1}$) and lower per-MVP cost on
   the cubic lattice.
2. **VIEM is more memory-frugal at large $L$**: its dense near-field +
   AIM-projection block is $L$-independent, while DDA's FFT pads
   replicate per RHS column. The crossover is at $L \approx 70 - 80$
   for convex shapes and at $L \approx 50$ for sparse / non-convex GRE
   (where DDA's bbox-to-occupancy ratio is ~8 – 9).
3. **Both solvers stall on Au under unpreconditioned GMRES** at the
   production lattice: DDA more deterministically (every Au slot at
   the auto-dpl resolution), VIEM at all but the smallest sphere × Au
   slot. The paper reports Au results via VIEM only because (a) VIEM's
   DOF-per-tet is lower for the same lattice budget so its near-marginal
   eigenvalue density is slightly less, and (b) VIEM's MVP at $L=100$
   is much cheaper, so MAXITER × MVP is acceptable while DDA's is not.
4. **Auto-dpl is the right matching mechanism for sphere / oblate /
   doublet** ($r$ within $\pm 5\%$). For GRE the calibration is
   intentionally loose ($r$ up to 1.38) to keep $N_{\rm occ}$ above a
   shape-fidelity floor; tightening it would not change the cost
   conclusions because GRE is dominated by the $L = 100$ and shape-
   error-floor effects, not by the lattice mismatch itself.
5. **For future paper-production planning**, the cost rule of thumb
   from this comparison is:

   - $L \le 32$: DDA is uniformly preferable on both wall time and
     peak RSS.
   - $L \approx 32 - 64$: DDA wall time still wins; memory roughly
     comparable.
   - $L \ge 100$ (full SO(3) GRE grid): VIEM preferable on memory by
     ~2 – 3× and (for plasmonic / stiff materials) on wall time too —
     this is the regime where the paper's GRE × Au results are
     reported via VIEM.

   The crossover is structural, not implementation-specific: the same
   pattern will reappear with any FFT-on-cubic-lattice solver (DDA
   variants) versus AIM-on-tet (VIEM variants) at comparable algorithmic
   maturity.
