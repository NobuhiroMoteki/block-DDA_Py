# Paper sweep cost estimates (CLAUDE.md §5)

Pre-run estimates captured by `scripts/estimate_cost.py` before each
production slot launch. Records worst-case shape × (wl × m_p) per file.

Columns:
- **peak_RSS**: actual peak RSS across all shape slots (from `/target/cost/peak_rss_bytes`)
- **t_sweep**: actual total wall time (sum of `/target/cost/t_total_s`)
- **converged**: number of slots that reached tol=1e-5 / total slots

Solver: `bl_gmres_mvp_fft` (unpreconditioned, VIEM parity), tol=1e-5, maxiter=100.
Run on: 2026-04-22, 24-core / 249 GiB host.

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

**Findings:**

1. **r_v=0.40 × n317 × spheroid mode**: GMRES stagnates at iter=100 (err
   3–5×10⁻³) for all three spheroid shapes, but **gre_n317 (L=100) at
   r_v=0.40 converges in 55 iter**. Suggests small-L block stagnation
   (CLAUDE.md §7.6) rather than fundamental ill-conditioning.

2. **gre_n317 × r_v=0.40 memory overshoot**: actual peak RSS 215 GB vs
   estimate 98 GB (2.2x); wall 142 min at this slot alone. Cost model
   (§5) is conservative at small L but optimistic for L=100 with heavy
   index contrast. This is a data point — not a bug.

3. **Au × spheroid mode: non-convergent at every size** — consistent
   with CLAUDE.md §7.3 / §10-7. DDA is not viable for Au via the
   unpreconditioned GMRES / spheroid L=5 path. Reporting via VIEM.

4. **gre_Au × a_eq=0.05 converges** (iter=16, ε=2e-14) — again, L=100
   block-Krylov rescues the small-domain case. a_eq=0.10 and 0.20
   stagnate at iter=100 (wall 5.7–6.1 h each).

5. **Total §10-8 wall time**: ~16.6 hours across all 12 files. Bulk
   concentrated in gre_n317 (3.3 h) and gre_Au (11.8 h).
