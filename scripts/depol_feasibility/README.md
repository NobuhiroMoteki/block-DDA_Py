# Kaolinite forward-depolarization feasibility tests

DDA feasibility studies (2026-07-23) for the strong forward depolarization observed on
liquid Kaolinite by CAS-v2, used by the downstream `PCAS_Bayes_for_liquid_2wls` pipeline.

**Metric.** The forward depolarization ratio (size-scale invariant):

    R_dep = |S_s - S_p| / |S_s + S_p| = |B_fw| / |A_fw|

Observed (large-|S|, 637 nm): **QuartzSand R_dep ~ 0.11** (matches the isotropic spheroid,
validating the DDA/LUT), **Kaolinite R_dep ~ 0.58**.

**Run** (from the block-DDA_Py repo root, using its venv):

    ./.venv/bin/python scripts/depol_feasibility/<script>.py

## Question: what shape/optics reproduces Kaolinite's R_dep ~ 0.58?

| script | what it varies | result: max R_dep |
|--------|----------------|-------------------|
| `test_disk_shape.py`      | flat circular disk vs oblate spheroid, AR & size | ~0.12 (disk ≈ spheroid; edges don't help; DECREASES with size) |
| `test_triaxial.py`        | triaxial ellipsoid a≠b (in-plane elongation) up to 2:1 | ~0.12 (elongation barely changes it) |
| `test_booklet.py`         | stacked booklet (oblate envelope, N solid plates, water gaps) | ~0.10–0.15 (form birefringence of water gaps too weak, δn~0.017) |
| `test_plate_aggregate.py` | rigid aggregates of oblate plates (edge-face / random) | 0.02–0.11 (randomly-oriented plates CANCEL → *less* depol) |
| `dpl_convergence.py`      | dipoles-per-wavelength at AR 6 & 11 | R_dep converged at LUT dpl=17 → the ~0.12 cap is NOT a thin-axis artifact |
| `test_biaxial.py`         | Kaolinite biaxial tensor n_α/n_β/n_γ, scaled ×s | REAL (s=1) → 0.14; s=20 (δn~0.34) → 0.58 |

**Conclusion.** No homogeneous dielectric *shape* exceeds R_dep ≈ 0.18 (the spheroid's max
over the whole LUT is 0.18; disk/triaxial/booklet/aggregate all ≤ 0.15, dpl-converged). The
real biaxial birefringence (δn ~ 0.017) gives only 0.14. **Only a strong effective uniaxial
birefringence reproduces the observed magnitude AND lobe direction**; a flow-weighted
calibration (done on the PCAS side) gives

    δn_eff = n_e - n_o ≈ -0.28   (optic axis = c = platelet normal, optically negative)

This is ~20× kaolinite's textbook bulk birefringence — treated as an empirical per-species
*effective* optical property (an open paradox). It is generated into an anisotropic LUT via
`run_dda_spheroid_sweep.py --delta-n-eff -0.28`.

## Calibrating delta_n_eff (two stages)

1. **`calib_dn_eff_stage1_dda.py`** (this repo, venv) — sweep delta_n on the uniaxial oblate
   spheroid (optic axis = c) at the species' beta and D_ve; save A(theta), B(theta) per
   delta_n to `calib_uniaxial_AB.npz` (uniaxial-along-c is axisymmetric, so the analytic
   phi-expansion holds).
2. **`PCAS_Bayes_for_liquid_2wls/scripts/calibrate_dn_eff.py`** (PCAS side) — flow-weight
   that grid with the species' Phase-1 posterior and DMA-APM beta prior, and find the
   delta_n reproducing the observed R_dep (magnitude) + depol-SNR lobe angle.

For Kaolinite the calibrated value depends on the target subset: all-particle median
R_dep~0.50 -> delta_n_eff ~ -0.26; large-|S| top-20% R_dep~0.66 -> ~ -0.36. The generated
LUT uses -0.28 (a reasonable first value; flow-weighted R_dep~0.53), refinable to ~-0.32.

See `PCAS_Bayes_for_liquid_2wls/docs/theory_note.tex` §3.5 for the full write-up.
