# Figure 2 — Lab-frame surface meshes of the three non-spherical paper targets

This document records the exact rendering conditions behind
`fig2_target_geometries.pdf` in `dda_results/paper/figures/`.

## 1. Purpose

Show, in a single look, the lab-frame surface geometry of the three
non-spherical particle targets used in the convergence-test
(Sec. 4.2) and the solver-vs-exact-reference comparison (Sec. 4.3) of
the paper:

- (a) **oblate spheroid** with semi-axial ratio $a:b:c = 3:3:1$
- (b) **sphere doublet** with monomer radius $R = r_{\rm ve}/2^{1/3}$
  and axis-direction surface gap $0.1\,R$
- (c) **Gaussian random sphere** (GRE on a spherical $a=b=c$ base
  ellipsoid) with $\beta_{\rm gre} = 0.2$ and correlation length
  $\ell = 0.3\,c$

All three panels share a common axis range and are rendered at the
convergence-test orientation $(\alpha,\beta,\gamma) = (0,0,0)$ and
volume-equivalent radius $r_{\rm ve} = 0.10\,\mu$m.

The figure is the visual anchor for the paper's
laboratory-frame / particle-frame discussion (Sec. 2.3): the black
$+\hat{\bm{z}}$ arrow on each panel is the fixed CAS-v2 incident
plane-wave propagation direction in the laboratory frame.

## 2. Output file

- **Single PDF**: `dda_results/paper/figures/fig2_target_geometries.pdf`,
  3 panels arranged in a 1 × 3 layout.

## 3. Generator

Rendered by the Julia / CairoMakie script
`/home/moteki/Julia/block-VIEM.jl/viz/visualize_paper_targets.jl`,
which reuses `block-VIEM.jl`'s mesh-generation pipeline:

- **Oblate** and **GRS**: meshed by `gre_mesh(GREParams(...))`.
- **Doublet**: built by `make_linear_chain(2, R; gap = 0.1·R)` and
  meshed by `mesh_sphere_aggregate(neck_ratio = 0)` (no sintering).

Mesh element size is intentionally coarser than the production-sweep
discretisation so the wireframe reads cleanly at print resolution
(see the caption note "visualisation mesh, coarser than the production
mesh").

The script writes the PDF directly to the path passed on the command
line:

```bash
julia --project=viz viz/visualize_paper_targets.jl \
    /home/moteki/Python_in_WSL/block-DDA_Py/dda_results/paper/figures/fig2_target_geometries.pdf
```

## 4. Visual conventions

- **Surface**: opaque triangulated surface, fill colour
  `:lightsteelblue`, with the boundary triangulation drawn as a
  hidden-line-removed black wireframe so the discretisation is
  visible (matching the visualisations in `block-VIEM.jl/viz/figs/`).
- **Axis arrows**: drawn from the origin in the lab frame.
  Black arrow = $+\hat{\bm{z}}$ (incident propagation direction);
  red = $+\hat{\bm{x}}$; blue = $+\hat{\bm{y}}$.
  +x and +y arrows use a 3D cone tip rendered via `_cone_mesh` (cone
  aspect ~6:1).  +z is currently rendered as a plain line segment
  without an arrowhead — the arrowhead is added by hand to the saved
  PDF (the foreshortening of a vertical 3D cone under camera
  elevation $\pi/8$ produces a visually unsatisfying tip in CairoMakie,
  so manual annotation is preferred for the paper figure).
- **Panel layout**: equal-width panels with a small empty column
  between adjacent panels (`Fixed` column widths) so the rightmost
  axis tick labels of one panel do not collide with the leftmost
  labels of the next.
- **Per-panel labels**: bold "(a)", "(b)", "(c)" only; full shape
  identifications (axial ratio, gap, $\beta_{\rm gre}$, $\ell$) are
  given in the figure caption in the paper.
- **Per-panel view angles**: oblate and GRS use
  `(azimuth, elevation) = (1.275π, π/8)` (default Makie 3D view);
  the doublet uses `(π/2 + π/16, π/9)` (broadside view) so the
  axis-direction surface gap between the two monomers is visible.

## 5. Cross-references

- Caption text: `manuscript/sections/04_results.tex` (around the
  `\label{fig:target-geometries}` block).
- Theory link: Sec. 2.3 ("From laboratory frame to particle frame")
  introduces the canonical-orientation / inverse-rotated-incident
  formulation that the figure visualises.
- Application link: Sec. 1 (Introduction) lists the environmental
  aerosol species each shape archetype represents (mineral dust,
  black-carbon aggregates, surface-rough irregular particles).
