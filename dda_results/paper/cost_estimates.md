# Paper sweep cost estimates (CLAUDE.md §5)

Pre-run estimates captured by `scripts/estimate_cost.py` before each
production slot launch. Records worst-case shape × (wl × m_p) per file.

Columns:
- **peak_RSS**: actual peak RSS across all shape slots (from `/target/cost/peak_rss_bytes`)
- **t_sweep**: actual total wall time (sum of `/target/cost/t_total_s`)
- **converged**: number of slots that reached tol=1e-5 / total slots

Solver: `bl_gmres_mvp_fft` (unpreconditioned, VIEM parity), tol=1e-5, maxiter=200.

---

## v0.7.5 reference run (n317 era, 2026-04-22, 24-core / 249 GiB host, maxiter=100)

Solver: `bl_gmres_mvp_fft` (unpreconditioned, VIEM parity), tol=1e-5, maxiter=100.

| file | L_solve | peak_RSS | t_sweep | converged | non-conv slots |
|------|---------|----------|---------|-----------|----------------|
| sphere_n15    | 5 (spheroid) |  0.47 GB |   0.5 min | 4/4 ✓ | — |
| oblate_n15    | 5 (spheroid) |  0.54 GB |   0.4 min | 4/4 ✓ | — |
| doublet_n15   | 5 (spheroid) |  0.49 GB |   0.1 min | 4/4 ✓ | — |
| gre_n15       | 100 (GRE)    | 19.81 GB |   7.1 min | 4/4 ✓ | — |
| sphere_n317   | 5 (spheroid) |  6.45 GB |  15.7 min | 3/4   | r_v=0.40 (err=3.6e-3) |
| oblate_n317   | 5 (spheroid) |  6.55 GB |  13.5 min | 3/4   | r_v=0.40 (err=5.2e-3) |
| doublet_n317  | 5 (spheroid) |  6.47 GB |  10.9 min | 3/4   | r_v=0.40 (err=3.4e-3) |
| gre_n317      | 100 (GRE)    | **215.1 GB** | **195.5 min** | 4/4 ✓ | — (iter=55 at r_v=0.4) |
| sphere_Au     | 5 (spheroid) |  1.20 GB |   7.1 min | 0/3   | all 3 a_eq (err ~1–2e-2) |
| oblate_Au     | 5 (spheroid) |  1.27 GB |   7.0 min | 0/3   | all 3 a_eq (err ~1–3e-2) |
| doublet_Au    | 5 (spheroid) |  1.27 GB |   6.5 min | 0/3   | all 3 a_eq (err ~1–2e-2) |
| gre_Au        | 100 (GRE)    | 45.26 GB | **706.2 min** | 1/3 | r_v=0.10, 0.20 (err ~1–2e-2); a_eq=0.05 ✓ (iter=16) |

---

## v0.7.6 production run (n20 replaces n317, 2026-04-24/25, 32-core / 1 TiB host, maxiter=200)

Solver: `bl_gmres_mvp_fft` with incremental block-Givens QR, auto-dpl (per-slot
VIEM n_tet matching, 1.5× tolerance). Paper-production scope — final data.

| file | L_solve | peak_RSS | t_sweep | converged | non-conv slots |
|------|---------|----------|---------|-----------|----------------|
| doublet_n15   | 5 (spheroid)  |  0.59 GB |   0.2 min | 4/4 ✓ | — |
| sphere_n15    | 5 (spheroid)  |  0.54 GB |   0.3 min | 4/4 ✓ | — |
| oblate_n15    | 5 (spheroid)  |  0.61 GB |   0.3 min | 4/4 ✓ | — |
| doublet_n20   | 5 (spheroid)  |  1.38 GB |   1.1 min | 4/4 ✓ | — |
| sphere_n20    | 5 (spheroid)  |  1.53 GB |   1.3 min | 4/4 ✓ | — |
| oblate_n20    | 5 (spheroid)  |  1.47 GB |   1.3 min | 4/4 ✓ | — |
| doublet_Au    | 5 (spheroid)  |  1.85 GB |   4.3 min | 0/3   | all 3 a_eq (err ~3–4e-3) |
| sphere_Au     | 5 (spheroid)  |  1.85 GB |   4.1 min | 0/3   | all 3 a_eq (err ~1–3e-3) |
| oblate_Au     | 5 (spheroid)  |  1.86 GB |   4.8 min | 0/3   | all 3 a_eq (err ~2–5e-3) |
| gre_n15       | 100 (GRE)     | 27.92 GB |  12.7 min | 4/4 ✓ | — (iters 8–11) |
| gre_n20       | 100 (GRE)     | 55.48 GB |  28.7 min | 4/4 ✓ | — (iters 12–20) |
| gre_Au        | 100 (GRE)     | 69.63 GB | **537.8 min** | 0/3 | all 3 a_eq (err ~2–5e-3) |

**v0.7.6 totals**: 11.3 hours wall, 69.6 GB peak RSS. 32/42 slots converged
(76%); remaining 10 are all Au slots consistent with §10-7 (DDA not viable
for Au via unpreconditioned GMRES).

### v0.7.6 findings

1. **n317 → n20 pays off for gre**: `gre_n20 × r_v=0.40` peak RSS 55 GB vs
   n317's 215 GB (4× reduction), wall 4.6 min vs ~142 min at the heavy slot.
   Converges in 20 iters (vs n317's 55).

2. **Au remains non-convergent across the board** — including
   `gre_Au × a_eq=0.05` which v0.7.5 reported as converging in 16 iter.
   The regression is attributable to auto-dpl: v0.7.5 used fixed dpl=17 giving
   a coarse lattice for a_eq=0.05 (tiny N_occ), while v0.7.6 auto-dpl picks
   dpl=61.46 to match VIEM `n_tet`, yielding N_occ=24416 and a stiffness
   regime where unpreconditioned GMRES stagnates. Behavior is physically
   correct (denser lattice = better discretization); Au is reported via VIEM
   as planned. No action needed.

3. **gre_Au dominates wall time**: 538 min / 11.3 h total = 79% of the sweep.
   All 3 slots stagnate at iter=200 with err ~2–5e-3. Each slot walls
   ~2h40m–3h31m (grows with N_occ).

4. **Au spheroid slots also non-convergent at every size** — consistent
   with v0.7.5 observations. DDA not viable for Au under this solver
   configuration; reporting via VIEM-side results.

---

## §11.4 dpl convergence study (2026-04-25, runs as 03:00 JST)

`scripts/run_dpl_convergence.py <shape> <material>`, dpl ∈ [10, 14, 17, 24, 34]
at a_eq=0.1 μm, ZYZ-identity orientation (L=1). Output:
`dda_results/paper/convergence_{shape}_{material}.hdf5`.

| shape  | material | converged (dpl) | t_sweep | notes |
|--------|----------|-----------------|---------|-------|
| sphere | n15 | 5/5 ✓ | <10 s | iters=8 flat across dpl |
| sphere | n20 | 5/5 ✓ | <15 s | iters=13 flat |
| sphere | Au  | 0/5   | ~2.3 min | all stagnate (iter=200) |
| oblate | n15 | 5/5 ✓ | <15 s | iters=8 flat |
| oblate | n20 | 5/5 ✓ | <15 s | iters=9 flat |
| oblate | Au  | 0/5   | ~1.8 min | all stagnate |
| gre    | n15 | 5/5 ✓ | <15 s | iters=8 flat |
| gre    | n20 | 5/5 ✓ | <15 s | iters=13–14 flat |
| gre    | Au  | 0/5   | ~7.0 min | all stagnate; dpl=34 largest at 291600 N_cuboid / 34140 N_occ |

---

## §11.5 RHS-scaling diagnostic (2026-04-25, runs as 03:05–06:23 JST)

`scripts/run_rhs_scaling.py <sphere_sweep.hdf5>`, L ∈ [1,2,4,8,16,32,64,128],
GMRES only. Output stored at `/target/rhs_scaling/gmres/` in the sweep file.

| file | t_sweep | notes |
|------|---------|-------|
| sphere_n15 | ~4 min | all slots converged; **L=128 skipped at a_eq=0.05** (N_occ=27 → 3·N_occ=81 < 128, degenerate — fixed in script to skip rather than crash) |
| sphere_n20 | ~20 min | all slots converged; iters scale as ~log(1/L) with block |
| sphere_Au  | ~175 min | a_eq=0.05 L=1,2 stagnate; larger L (≥16) converge; a_eq=0.10 all converge; **a_eq=0.20 L=32 took 49 min (iters=199), L=128 took 54 min (iters=105)** — block-Krylov headroom helps for Au |

**§11.5 script fix**: added `L > 3·N_occ` guard in
`scripts/run_rhs_scaling.py` per-slot loop (line ~237). Previously
crashed with `ValueError: could not broadcast input array from shape
(81,128) into shape (128,128)` at QR because `np.linalg.qr(B, mode='reduced')`
returns rank-deficient R when B has fewer rows than columns.

---

## Summary (v0.7.6, 2026-04-24/25)

- **§11.2 production**: 42 slots, 32 converged, wall 11.3 h, peak RSS 69.6 GB.
- **§11.4 dpl convergence**: 9 (shape × material) combos, 27/45 (dpl, slot) converged (all non-Au), wall < 15 min total.
- **§11.5 RHS-scaling**: 3 sphere materials × 8 L values × 3–4 a_eq, wall ~3.3 h, one skipped L (a_eq=0.05 / L=128).
- **Overall paper-production compute**: ~15 h wall on 32-core / 1 TiB host.
