# Changelog

Changes to block-DDA_Py, newest first. Dates are year-month-day.

The first section lists the corrections that change numerical results. Read it if you
have results from an earlier version. The sections after it list all changes by version.
The last section explains the version numbers and tags.

## Corrections that change results

| Fixed in | Affected versions | What was wrong | Results that change | What to do |
|---|---|---|---|---|
| 0.8.2 | all before 0.8.2 | The backward amplitude of the OCBS signal had the factor `1/√2` of the printed Eq. (28) of Moteki and Adachi (2024). The corrected equation, given in the erratum to that paper, has `1/2`. | `S_bk_OCBS` and its Mie reference `S_bk_OCBS_mie`: the output of `run_dda.py` and of the paper scripts, and the files under `dda_results/paper/`. The earlier values are √2 times the corrected ones. The phase and the ratio of the DDA value to the Mie value are unchanged. Cross sections and forward amplitudes are not affected. | Multiply the stored backward amplitudes by `1/√2`. No new calculation is needed. |
| 0.6.0 | 0.1.1 to 0.5.0 | The random orientations drew the Euler angle β uniformly in [0, π]. Orientations that are uniform on the sphere need cos β uniform in [−1, 1]. Orientations with the particle z axis near the propagation direction were drawn too often. | Averages over the stored orientations of nonspherical particles. The values of each single orientation are correct. The spheroid mode of 0.5.0 is not affected. | Recalculate with 0.6.0 or later. Or weight each stored orientation by sin β in the average. The Euler angles are stored in the output since 0.2.1. |

## 0.8.4 (2026-11-20)

Same code as 0.8.3. The documents are brought up to date.

- README: a note states that the published paper used 0.7.8. The memory estimate includes
  the Krylov basis of the block GMRES and counts the four work arrays of the matrix–vector
  product. The earlier formula gave about half of the measured peak memory. Measured solve
  times replace the extrapolated ones. The limitations name the two-sphere shape model.
- This changelog is added. The change history that was in the README is moved here.

## 0.8.3 (2026-11-18)

### Added

- `run_dda.py --shifted-ri`: all isotropic entries of the refractive-index list of a
  (shape, size, wavelength) slot are solved by one shifted block GMRES, for any GRE shape.
  The default mode is unchanged. `run_dda.py --output` selects the output file.
- README: guide to sweeping refractive index, volume, shape and orientation with the two
  sweep drivers and with the API.
- Theory note: comparison with the shifted solver of ADDA (Sec. 5.10).

### Fixed

- Theory note: two entries of the reference list had the title or the authors of another
  paper (Tadano et al. 2009, Gu et al. 2016).
- `dda_results/create_h5py.ipynb`: the text attribute `S_definition` of new sweep files
  still gave the backward amplitude with the factor `1/√2`. No numbers were affected.

## 0.8.2 (2026-11-18)

### Fixed

- `S_bk_OCBS` follows the corrected Eq. (28) of Moteki and Adachi (2024). See the first row
  of the table above. The change is in `compute_OCBS_observable_S_bk`, in
  `compute_observables_shifted` and in the Mie reference `mie_compute_q_and_s`.
  `tests/test_backward_amplitude.py` checks the definition against Mie theory.
- README and theory note: section numbers and version remarks that were out of date.

### Changed

- `test_dda.ipynb` is run again, so its stored outputs show the corrected amplitude.
- The files under `dda_results/paper/` keep the earlier definition. Their schema notes say
  so.

A sweep of anisotropic particles at a constant difference of 1/(ε − 1) between the axes was
added and removed again before this version. The birefringence is not constant along such a
sweep. Section 5.5 of the theory note gives the reason.

## 0.8.1 (2026-11-18)

### Changed

- Shifted block GMRES: the least-squares problem of each refractive index is handled by an
  incremental block-Givens QR (`lsq_method="incremental"`, the default). The dense method of
  0.8.0 remains as `lsq_method="dense"`. The results of the shifted sweep change at
  rounding level only. At 64 orientations the least-squares time drops by a factor of 2.3
  to 6.8.
- Theory note: Sec. 4 describes the block GMRES as the solver. It still described
  Block-BiCGSTAB, which `DiscreteDipoles` has not called since 0.7.8. Benchmarks of the
  shifted solver at 5, 13 and 64 orientations are added.

## 0.8.0 (2026-11-17)

### Added

- `bl_gmres_shifted_mvp_fft`: shifted block GMRES. For a homogeneous isotropic particle it
  solves all refractive indices of a sweep, and all orientations, with one block-Arnoldi
  process on a common lattice.
- Sweep API of `DiscreteDipoles`: `set_interaction_matrix_shifted`,
  `solve_matrix_equation_shifted`, `compute_observables_shifted`. The observables are
  evaluated without assembling the solutions.
- `run_dda_spheroid_sweep.py --shifted-ri`: the refractive-index axis of a slot is solved in
  one process. The default mode is unchanged. The shifted mode uses the lattice of the
  largest refractive index for the whole axis, so its results differ from the default mode
  at the level of the discretization error.
- `tests/`: pytest suite. It compares the shifted path with the existing solver and with
  Mie theory.
- `scripts/run_shift_scaling.py`: benchmark of the shifted solver.
- Theory note: Sec. 5 on the shifted block GMRES.

## Added since 0.7.8 without a version of their own (2026-06-11 to 2026-10-13)

- Spheroid sweep: a `provenance` group with the git commit, a hash of the settings and the
  creation time (2026-07-05).
- Spheroid sweep: presets for the host medium and the grid, `--preset air`, `liquid`
  (2026-07-05), `liquid_prolate` (2026-07-30) and `liquid512` (2026-10-13), and `--output`.
- Spheroid sweep: the size axis is equidistant in log10 and is stored as `log_D_ve_grid`
  in place of `D_ve_grid` (2026-07-06). The default grid changed to D_ve from 0.20 to
  1.20 μm, Re(m_p) from 1.35 to 1.70 and log10(AR) from 0 to 1.55 (2026-07-06 and
  2026-07-21). Files made before and after this change have different axes.
- Spheroid sweep: `--delta-n-eff` for a uniaxial particle with the optic axis along the
  symmetry axis (2026-07-23).
- `scripts/depol_feasibility/`: studies of forward depolarization by kaolinite-like
  particles (2026-07-23).
- The unpublished manuscript draft is no longer tracked in the repository (2026-06-11).

## 0.7.8 (2026-05-13)

The version of the published paper: N. Moteki, J. Quant. Spectrosc. Radiat. Transfer 362,
110046 (2026).

### Changed

- `DiscreteDipoles.solve_matrix_equation` uses the unrestarted block GMRES without
  preconditioning (2026-04-29). It used Block-BiCGSTAB with Jacobi preconditioning before.
  The stopping test now uses the residual of the original system. Results change within the
  solver tolerance. The residual history is kept in `err_history`.
- Figures and plotting scripts of the paper under `dda_results/paper/`.

### Fixed

- `docs/paper_simulation_conditions_dda.md`: sign in the written formula of `S_bk_OCBS`.
  The code was not changed.

## 0.7.7 (2026-04-27)

- Results of the paper-production sweep are stored under `dda_results/paper/`.
- `installation_guide.md` (2026-04-26).
- Fixed in the paper scripts: the key of the gold refractive index in
  `utils/dpl_calibration.py`, the material `n20` in `scripts/run_dpl_convergence.py`, and a
  guard in `scripts/run_rhs_scaling.py` for more right-hand sides than unknowns. The scripts
  stopped with an error before. No wrong results were returned.

## 0.7.6 (2026-04-24)

- `bl_gmres_mvp_fft`: incremental block-Givens QR of the Hessenberg matrix. The two
  unpreconditioned solvers return the residual history.
- Paper sweep: the dipoles per wavelength of each slot are chosen to match the number of
  volume elements of block-VIEM.jl (`utils/dpl_calibration.py`). The benchmark material
  `n20` (m_p = 2.0) replaces `n317` (m_p = 3.17 + 0.16i).
- `docs/paper_simulation_conditions_dda.md`: conditions of the paper sweep.

## 0.7.5 (2026-04-21)

- Block BiCGSTAB and block GMRES without preconditioning (`bl_bicgstab_mvp_fft`,
  `bl_gmres_mvp_fft`).
- Shape model of two equal spheres (`shape_model/two_sphere_cluster.py`).
- Paper sweep: `scripts/run_paper_sweep.py`, `scripts/run_dpl_convergence.py`,
  `scripts/run_rhs_scaling.py`, `scripts/estimate_cost.py` and the file layouts under
  `dda_results/paper/`.
- README: pointer to block-VIEM.jl for particles of high refractive-index contrast
  (2026-04-19).

## 0.7.4 (2026-03-31)

- `read_spheroid_sweep.ipynb` replaces the example script for reading the sweep file.

## 0.7.3 (2026-03-30)

- Spheroid sweep: new grid ranges. One solve attempt per slot (`MAX_TRY = 1`).

## 0.7.2 (2026-03-28)

- Spheroid sweep: several wavelengths in one file. Each (wavelength, medium) pair has its
  own group. The grid axes are shared at the root.

## 0.7.1 (2026-03-28)

- `run_dda_spheroid_sweep.py`: sweep of spheroids over a regular grid of size, refractive
  index and aspect ratio, with the orientations on a reduced domain. Output in HDF5.

## 0.6.0 (2026-03-26)

### Changed

- Orientations are a deterministic grid of Euler angles, `N_alpha_ori × N_beta_ori ×
  N_gamma_ori`, with cos β equally spaced. Random orientations with a total count
  `num_orientations` were used before.
- Spheroid mode: only `N_beta_ori` orientations are solved. The full grid is filled
  analytically for the forward amplitudes, the backward amplitude and the cross sections.

### Fixed

- Random orientations of the general mode are uniform on the sphere (2026-03-26, before
  the grid replaced them). See the second row of the table above.

## 0.5.0 (2026-03-26)

- Spheroid mode: for spheroids (`ab_ratio = 1`, `gre_beta = 0`) the dependence on the
  azimuthal orientation is computed analytically from one solve per polar angle.
  `test_spheroid_symmetry.ipynb` verifies the relations to rounding error.

## 0.4.0 (2026-03-25)

### Changed

- The lattice spacing follows from the dipoles per wavelength inside the particle,
  `d = λ₀ / (max|m_p| × dpl)`, with `dpl = 17` by default. An empirical formula that
  depended on the shape only was used before.
- The lattice is rescaled so that the volume-equivalent radius of the dipole array equals
  `r_v_base` exactly. It was approximate before. `r_v_base` is a required argument of
  `Target`.
- Both changes alter the discretization, so results differ from those of earlier versions
  at the level of the discretization error.
- Theory note (`docs/theory_note.pdf`), added on 2026-03-19: sections on the lattice and on
  the performance of the old and new implementations.

## 0.3.0 (2026-03-09)

### Changed

- Matrix–vector product: the 3D FFT on a doubled grid of Goodman et al. (1991) replaces
  the multilevel block-Toeplitz FFT of Barrowes et al. (2001).
- Block-Krylov solver: all orientations are processed by NumPy and SciPy array operations.
  The `ray` multiprocessing is removed.
- Shape model: NumPy operations replace the `numba` loops.
- Parameter sweep (`run_dda.py`): list of (wavelength, medium) pairs `wl_m_m_pairs` and
  list of particle refractive indices `m_p_xyz_list`, which may be anisotropic. The
  datasets have the shape `(N_pairs, N_mp, N_rv, N_bc, N_ab, N_bt, N_ori)`. They had the
  shape `(N_mm, N_rv, N_bc, N_ab, N_bt, N_ori)` before.
- Dependencies: `ray`, `numba` and `pywin32` are removed. `ipympl` is added.
- Environment: Python 3.13 on Linux or WSL2 with `uv`. Python 3.12 on Windows was used
  before.
- README: section on parallelism and memory.

## 0.2.1 to 0.2.4 (2026-01-22)

- The Euler angles of every orientation are stored in the output (0.2.1).
- README updates (0.2.2 to 0.2.4).

## 0.1.1 (2025-02-10)

- First public version: DDA with block-Krylov solvers for many orientations, FFT-based
  matrix–vector product, Gaussian random ellipsoid shape model, random orientations.

## Version numbers

- Tags exist for 0.1.1, 0.2.1 to 0.2.4, 0.3.0, 0.4.0, 0.5.0, 0.6.0, 0.7.8, 0.8.2, 0.8.3
  and 0.8.4. The other versions are commits whose message names the version.
- GitHub releases exist for 0.1.1, 0.2.4, 0.3.0, 0.4.0, 0.5.0, 0.6.0, 0.7.8, 0.8.3 and
  0.8.4.
- The attribute `block_dda_version` that `run_dda_spheroid_sweep.py` writes stayed at
  `0.7.2` from 0.7.2 until 0.8.0. A sweep file with this value can come from any version in
  that range. From 2026-07-05 the attribute `git_sha` of the group `provenance` identifies
  the commit.
