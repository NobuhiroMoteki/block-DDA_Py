# Exact-reference HDF5 schema (MSTM doublet / T-matrix oblate)

This file is the data contract between the block-DDA_Py paper-figures notebook
([`plot_paper_results.ipynb`](plot_paper_results.ipynb), loaders in
[`_plot_io.py`](_plot_io.py)) and the Julia-side exact-reference producers:

| Shape   | Producer                                           | Filename pattern                |
|---------|----------------------------------------------------|---------------------------------|
| doublet | `~/Julia/MSTMforCAS.jl` (multi-sphere T-matrix)    | `mstm_doublet_<material>.hdf5`  |
| oblate  | `~/Julia/TransitionMatrices.jl` (EBCM single-body) | `tmm_oblate_<material>.hdf5`    |

`<material>` ∈ `{n15, n20, Au}` matching the block-DDA_Py / block-VIEM.jl
production sweeps. Both files live in
`~/Julia/block-VIEM.jl/viem_results/paper/`.

The two producers MUST emit the same layout below; the Python loader uses
the same code path for both. The doublet HDF5 already shipped by
`MSTMforCAS.jl` is the canonical reference — see e.g.
`mstm_doublet_n15.hdf5` for a working example.

## Conventions

- Units: lengths in **μm**, cross-sections in **μm²**, amplitudes in **μm**,
  angles in **rad** (consistent with `units` attribute below).
- Wavelength: `wl_0_um = 0.638` (vacuum), `m_m = 1.0` (vacuum host).
- `β` = polar angle between incidence direction (lab `+z`) and the body
  symmetry axis (oblate: short `c`-axis; doublet: line through the two
  spheres). `α = γ = 0` is implicit (axially symmetric reference).
- Grid: `N_β = 5` β values, equal-area in `cos β` over `(-1, 1)`
  (i.e. `cos β ∈ {±0.8, ±0.4, 0.0}`); same as the block solvers'
  spheroid-mode α=γ=0 slice.
- `N_rv` = number of `a_eq` values (4 for n15/n20: `[0.05, 0.10, 0.20, 0.40]`;
  3 for Au: `[0.05, 0.10, 0.20]`).
- Array layout: `(N_β, N_rv)` (β fastest-varying axis is the *first* index in
  Python; in Julia/HDF5 the file should be stored such that
  `h5read(...)[:, k]` gives all β at fixed `a_eq`). Numpy reads match this
  out-of-the-box because HDF5 uses C order on disk.
- Amplitude conventions for `S_fw_*` / `S_bk` follow MSTM's `S_definition`
  attribute (see below). T-matrix output MUST adopt the same formulas:
  ```
  S(0)_theta = (S2 - i S3) / (-i k)
  S(0)_phi   = (S1 + i S4) / (-i k)
  S_fw_mean  = (S(0)_theta + S(0)_phi) / 2
  S_bk       = (S11 + S22 + i*S12 - i*S21)(180°) / sqrt(2)
  ```

## Common dataset / attribute layout

```
/target  (group)
  attrs:
    shape_kind          string ("doublet" | "oblate")
    scattering_code     string (e.g. "MSTMforCAS.jl ..." / "TransitionMatrices.jl ...")
    units               string ("C:[um^2], S:[um], beta:[rad], a_eq:[um]")
    wl_0_um             float64
    m_m                 float64
    solver_tol          float64
    truncation_order    int64    (T-matrix / MSTM expansion order)
    use_fft             uint8    (0/1; informational only)
    S_definition        string   (formulas above, byte-for-byte)
    geometry            string   (free-text geometry description)
    source_viem_h5      string   (absolute path to the corresponding solver h5)

  /a_eq_um              float64 shape (N_rv,)
  /beta_rad             float64 shape (N_β,)        # sorted ascending or sorted by cos β; either is OK — Python matches by closest cos β
  /m_p                  complex128 shape (3,)       # diagonal elements of m_p tensor

  /observables  (group)
    C_ext, C_abs, C_sca         float64    shape (N_β, N_rv)
    Q_ext, Q_abs, Q_sca         float64    shape (N_β, N_rv)
    S_fw_theta, S_fw_phi,
    S_fw_mean, S_bk             complex128 shape (N_β, N_rv)

  /diagnostics  (group)         # optional but recommended
    n_iterations                int64      shape (N_β, N_rv)   # solver iter count
    converged                   int64      shape (N_β, N_rv)   # 1 = converged
```

`Q_X = C_X / (π · a_eq²)` (size-parameter normalization on the volume-equivalent
sphere). Both `C_X` and `Q_X` are required — the notebook reads both.

## Shape-specific extras

These live as additional datasets at `/target/<key>` (sibling of `a_eq_um`).
The Python loader stores them under `extras` in the returned dict; they are
informational and not required by the figure code, but should be present
for self-describing files.

### doublet (`mstm_doublet_*.hdf5`, already implemented)
- `/target/R_monomer_um` float64 (N_rv,) — monomer sphere radius (= `a_eq / 2^(1/3)`)
- `/target/gap_um`        float64 (N_rv,) — surface-to-surface gap (= `0.1 · R_monomer`)
- `geometry` attr: `"doublet, monomer R = a_eq / 2^(1/3), gap = 0.1 R, axis = +z, β = ∠(incidence, axis)"`

### oblate (`tmm_oblate_*.hdf5`, to be produced)
- `/target/aspect_ratio`   float64 scalar — `c / a` for the oblate spheroid (= `1/3` for the paper geometry, since semi-axes a:b:c = 3:3:1 with `a = b > c`)
- `/target/a_um`           float64 (N_rv,) — equatorial semi-axis ([μm], = `a_eq · (a/c)^(1/3) = a_eq · 3^(1/3)`)
- `/target/c_um`           float64 (N_rv,) — polar      semi-axis ([μm], = `a_eq · (c/a)^(2/3) = a_eq / 3^(2/3)`)
- `geometry` attr: `"oblate spheroid, semi-axes a:b:c = 3:3:1 (a=b>c), symmetry axis = +z, β = ∠(incidence, c-axis)"`

The volume-preservation identity `a · a · c = a_eq³` should be checked against
floating-point round-off in the producer; document any deviation in `geometry`.

## Producer checklist (TransitionMatrices.jl, oblate)

1. Match the dataset paths and dtypes exactly (no `Float32`, no transposed
   axes — HDF5 row-major ⇒ Julia must write `permutedims` if the in-memory
   array is column-major shape `(N_rv, N_β)`).
2. Populate `/target/observables/{C_X, Q_X, S_*}` for every `(β, a_eq)` cell.
3. Populate `/target/diagnostics/{n_iterations, converged}` (set `converged = 1`
   for cells where the EBCM solver met `solver_tol`; `0` otherwise — figure
   code skips non-converged cells).
4. Write all `/target` attributes including the verbatim `S_definition`
   string above. Anything missing is fine for `extras`, but `S_definition`
   is the contract that anchors the amplitude conventions.
5. Sanity check by loading via Python:
   ```python
   from dda_results.paper._plot_io import load_tmm
   d = load_tmm("~/Julia/block-VIEM.jl/viem_results/paper/tmm_oblate_n15.hdf5")
   assert d["beta_rad"].shape == (5,)
   assert d["C_ext"].shape == (5, 4)
   ```
6. Re-run the notebook — Phase 3 oblate panels (column 0) should fill in
   automatically once any one `tmm_oblate_<mat>.hdf5` is present.
