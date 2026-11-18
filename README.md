# block-DDA_Py  [![version](https://img.shields.io/badge/version-v0.8.3-blue)](https://github.com/NobuhiroMoteki/block-DDA_Py/releases)

## 📌 Description
A Python code for the Discrete Dipole Approximation (DDA) using block-Krylov type iterative solvers, with custom features for light-scattering simulations of environmental particles (e.g., mineral dust).

**Version used in the published paper.** The DDA results of Moteki (2026), *J. Quant. Spectrosc. Radiat. Transfer* 362, 110046, were produced with [v0.7.8](https://github.com/NobuhiroMoteki/block-DDA_Py/releases/tag/v0.7.8). Use that release to reproduce the paper. Later versions add the shifted refractive-index sweep (v0.8.0 to v0.8.3) and follow the corrected definition of the backward-scattering amplitude *S*<sub>bak</sub> (from v0.8.2). The corrected *S*<sub>bak</sub> is 1/√2 times that of v0.7.8. All other results are unchanged in the default mode.

### Main Features
1. Supports batch calculations for many particle orientations using a **deterministic uniform Euler angle grid** on SO(3): `N_alpha` × `N_beta` × `N_gamma` divisions, with equal-area polar sampling (`cos(beta)` equally spaced).
2. Supports birefringent particle materials (anisotropic complex refractive index along x, y, z axes).
3. Supports a parametric irregular shape model suitable for mineral dust particles: Gaussian Random Ellipsoid ([GRE](https://doi.org/10.1016/j.jqsrt.2011.02.013)).
4. The lattice spacing is set directly from the user-specified **dpl** (dipoles per wavelength inside the particle): `d = λ₀ / (max|m_p| × dpl)`. The default is `dpl=17`. After discretisation, the lattice is rescaled so that the volume-equivalent radius matches `r_v_base` exactly (**volume-preserving lattice rescaling**; see [docs/theory_note.pdf](docs/theory_note.pdf), Sec. 8.3–8.4).
5. Supports simulation of the key quantities in the Complex Amplitude Sensing version 2 ([CAS-v2](https://doi.org/10.1364/OE.533776)): the polarized complex forward-scattering amplitudes {Re*S*(0°)<sub>*s*</sub>, Im*S*(0°)<sub>*s*</sub>, Re*S*(0°)<sub>*p*</sub>, Im*S*(0°)<sub>*p*</sub>} and the complex backward-scattering amplitude *S*<sub>bak</sub> = (−*S*<sub>11</sub> + *S*<sub>22</sub>)/2 (definition of the erratum to the CAS-v2 paper, from v0.8.2).
6. For reference, block-DDA_Py outputs a comparison of the DDA solution to the Mie solution for a volume-equivalent sphere with axes-average refractive index.
7. Supports parameter sweeps over vacuum wavelength, medium refractive index, particle refractive index (including anisotropic cases), and shape parameters, with results stored in HDF5 format.
8. **Spheroid mode**: For spheroids (`ab_ratio=1`, `beta=0`), the azimuthal orientation average is computed analytically from a single DDA solve per polar angle, reducing the required number of solves from $N_\theta \times N_\phi$ to $N_\theta$ (see [docs/theory_note.pdf](docs/theory_note.pdf), Sec. 9). While the T-matrix method is generally faster for spheroids, it becomes numerically unstable at extreme aspect ratios; DDA with spheroid mode provides a stable alternative in such regimes.
9. **Shifted refractive-index sweep**: For a homogeneous isotropic particle, all refractive indices of a sweep are solved by one shifted block-GMRES process on a common lattice. The cost is close to that of the single hardest refractive index (see [docs/theory_note.pdf](docs/theory_note.pdf), Sec. 5).

### Algorithms
1. block-DDA_Py uses [block-Krylov methods](https://people.math.ethz.ch/~mhg/pub/delhipap.pdf) for iteratively solving the matrix equation, enabling batch computation for many incident beams simultaneously.
2. A batch solution for many incident beams is internally converted to the batch solution for many orientations through rotational transformation.
3. Fast matrix-vector products via the [Goodman 3D FFT algorithm](https://doi.org/10.1364/OL.16.001198): the interaction tensor is stored in a doubled grid of shape `(2Nx, 2Ny, 2Nz, 3, 3)` and convolved using `scipy.fft.fftn/ifftn` with multi-threaded batch FFT.
4. Multi-orientation batch processing is fully vectorized using NumPy/SciPy broadcasting, with no external parallelization library required.

### Limitations

There are some notable limitations of the current block-DDA_Py:

1. Supported shape model is only GRE.
2. Current shape model supports only single-component particles (refractive index is uniform inside each particle).
3. Supported incident beam type is only a plane wave or a set of plane waves from different directions.

I'll remove these limitations upon user requests and application needs.

---

## 🔗 Companion project for high-contrast particles: block-VIEM.jl

For scattering problems where DDA does not converge or provides insufficient accuracy — for example, particles with **high refractive-index contrast** (|m_p/m_m| ≳ 2–3), plasmonic metals (gold, silver, iron oxide aggregates), extreme aspect ratios, or surface roughness features smaller than the DDA lattice spacing — use the companion Julia code [**block-VIEM.jl**](https://github.com/NobuhiroMoteki/block-VIEM.jl).

block-VIEM.jl is a **higher-accuracy successor** that solves the same volume integral equation physics with a tetrahedral-mesh SWG / half-SWG basis, Duffy-transform singular integration, and AIM-FFT-accelerated block-Krylov iteration. The two codes share the same physical input conventions (`wl_0`, `m_m`, `m_p`, including anisotropic `m_p_xyz`), Euler-angle convention (ZYZ intrinsic), GRE shape parametrization, HDF5 output schema, and CAS-v2 observable definitions, so downstream analysis pipelines work with either without modification.

### Summary decision rule

```text
if |m_p / m_m| < 2 and DDA converges:
    use block-DDA_Py         # 100–1000× faster setup for low contrast
else:
    use block-VIEM.jl        # converges where DDA cannot (high contrast, plasmonic)
```

For a detailed benchmark comparison (e.g., a plasmonic gold sphere at m_p = 0.175 + 3.48i where DDA may not converge at all while VIEM reaches sub-1 % accuracy with N = 2134 DOFs), see the "When to use block-VIEM.jl vs block-DDA_Py" section of the [block-VIEM.jl README](https://github.com/NobuhiroMoteki/block-VIEM.jl#when-to-use-block-viemjl-vs-block-dda_py).

---

## 📄 Technical Note

For a detailed description of the theory and algorithms, see [docs/theory_note.pdf](docs/theory_note.pdf).

---

## 🚀 Installation

block-DDA_Py is developed and tested with **Python 3.13 on Linux (WSL2)**.

For step-by-step instructions on building the runtime environment from scratch on either a **Windows machine (WSL2 + Ubuntu)** or a **native Linux (Ubuntu)** machine, see [installation_guide.md](installation_guide.md).

### Quick start (existing Linux/WSL2 environment with `uv` already installed)

```sh
git clone https://github.com/NobuhiroMoteki/block-DDA_Py.git
cd block-DDA_Py
uv venv --python 3.13
uv pip install -r requirements.txt
```

---

## 🔧 Usage

### Visualization of the GRE model particle (Optional)
1. Open `run_gaussian_ellipsoid.ipynb` and edit the following input parameters:
   - Volume-equivalent radius of base ellipsoid: `r_v_base`
   - Ratio of semiradius along y axis (b) to z axis (c): `bc_ratio`
   - Ratio of semiradius along x axis (a) to y axis (b): `ab_ratio`
   - Standard deviation of GRE surface deformation: `beta`
2. Execute the notebook (a 3D plot of the GRE particle will appear).

### Single execution
1. Open `test_dda.ipynb` and edit the following input parameters:
   - GRE shape parameters: `r_v_base`, `bc_ratio`, `ab_ratio`, `beta`
   - Vacuum wavelength: `wl_0` (length unit must be consistent with `r_v_base`)
   - Medium refractive index (real): `m_m`
   - Particle refractive index (complex) for each axis: `m_p_x`, `m_p_y`, `m_p_z`
   - Orientation grid divisions: `N_alpha_ori`, `N_beta_ori`, `N_gamma_ori`
2. Execute `test_dda.ipynb`. DDA results are compared to the Mie reference.

### Sweeping refractive index, volume, shape and orientation

There are three ways to sweep the parameters. Choose by the particle shape and by the size of the sweep.

| | General sweep | Spheroid sweep | Direct use of the API |
|---|---|---|---|
| Entry point | `dda_results/create_h5py.ipynb` + `run_dda.py` | `run_dda_spheroid_sweep.py` | your own script |
| Particle shape | any GRE shape, isotropic or anisotropic | spheroids only | any GRE shape |
| Refractive index | list `m_p_xyz_list`; one solve per entry, or one solve for all isotropic entries with `--shifted-ri` | regular grid `RI_REAL_RANGE`; one solve per value, or one solve for the whole axis with `--shifted-ri` | list of isotropic values, one solve for the whole list |
| Volume | list `r_v_base_list` (volume-equivalent radius) | grid `D_VE_RANGE` (volume-equivalent diameter, log-spaced) | one lattice per size |
| Shape parameters | lists `bc_ratio_list`, `ab_ratio_list`, `gre_beta_list` | grid `LOG10_AR_RANGE` (aspect ratio `b/c`) | one lattice per shape |
| Orientation | Euler-angle grid `N_alpha_ori` × `N_beta_ori` × `N_gamma_ori` | grid in `cos(theta_o)` and `phi_o` | any list of Euler angles |
| Output | HDF5, all observables | HDF5 lookup table, forward amplitudes only | arrays in memory |

How each parameter is handled:

- **Orientation** is never looped. All orientations of one particle are right-hand sides of one block-Krylov solve. For spheroids only the polar angle is solved, and the other two Euler angles are filled in analytically (spheroid mode).
- **Refractive index** needs a new solve per value, except for a homogeneous isotropic particle. There the shifted block GMRES solves all values in one process on a common lattice. It is available as `--shifted-ri` in both sweep drivers and through the API. Birefringent particles are always solved one refractive index at a time.
- **Volume and shape** change the lattice. Every combination is a separate lattice and a separate solve.
- **Wavelength and medium** are given as `(wl_0, m_m)` pairs in both drivers. Each pair is a separate set of solves.

The three ways are described below.

### Parameter sweep (general shapes)
1. Open `dda_results/create_h5py.ipynb` and configure the sweep parameters:
   - List of (vacuum wavelength, medium RI) pairs: `wl_m_m_pairs` — shape `(N_pairs, 2)`
   - List of particle refractive indices (supports anisotropic x/y/z): `m_p_xyz_list` — shape `(N_mp, 3)`
   - List of volume-equivalent radii: `r_v_base_list`
   - List of axis ratios: `bc_ratio_list`, `ab_ratio_list`
   - List of GRE roughness parameters: `gre_beta_list`
   - Orientation grid divisions: `N_alpha_ori`, `N_beta_ori`, `N_gamma_ori`
   - Output filename: `OUTPUT_FILE` in `run_dda.py`
2. Execute `dda_results/create_h5py.ipynb` to generate the HDF5 output file.
3. Execute `run_dda.py`. The sweep loops over all combinations of `wl_m_m_pairs`, `m_p_xyz_list`, `r_v_base_list` and the shape parameters. Each combination is one block solve for all orientations. For spheroids (`ab_ratio=1`, `gre_beta=0`), spheroid mode is activated automatically: only `N_beta` DDA solves are performed, and the full orientation grid is filled analytically.
   **Shifted refractive-index sweep (`--shifted-ri`).** With this flag all isotropic entries of `m_p_xyz_list` (equal `x`, `y` and `z` values) of a (shape, wavelength) slot are solved by one shifted block-GMRES process:

   ```sh
   python run_dda.py --shifted-ri --output dda_results/my_new_sweep.hdf5
   ```

   - The isotropic entries of a slot share one lattice, built for the largest `|m|` among the isotropic entries of the list. The default mode builds a separate lattice for each entry. The results of the two modes therefore differ at the level of the discretization error, except for the entry with the largest `|m|`.
   - Anisotropic entries are solved one at a time on their own lattice, exactly as in the default mode.
   - The time of a slot is close to that of its hardest refractive index alone. The gain grows with the number of isotropic entries.
   - Use the flag for new output files only. A file keeps the mode it was started with, and resuming it in the other mode is refused. The attributes `shifted_ri` and `lattice_policy` of the group `target` mark a file made in this mode.
   - `--output PATH` selects the output file in both modes. Without it the script uses `OUTPUT_FILE`.
4. Use `dda_results/check_h5py.ipynb` to inspect the HDF5 file contents and verify results.
5. Use `plot_dda_results.ipynb` to visualize the parameter-swept DDA results.

### Spheroid parameter sweep (large-scale)

For dedicated spheroid sweeps over a regular grid of physical parameters, use `run_dda_spheroid_sweep.py`.

1. Choose a preset and, if needed, edit the settings section at the top of the script:
   - `--preset` selects the host medium, the wavelengths and the output file: `air` (default; 453, 638 and 834 nm), `liquid` (water; 637 and 773 nm), `liquid512` (water; 512 nm) or `liquid_prolate` (water; prolate spheroids in a narrow refractive-index band). Each preset sets `MEDIUM_CONDITIONS`, the list of `(wavelength [um], n_medium)` pairs. Each pair produces its own HDF5 group.
   - `D_VE_RANGE`, `RI_REAL_RANGE`, `LOG10_AR_RANGE`: each `(min, max, N_grid)`. The volume-equivalent diameter [um] is equidistant in log10. The real refractive index and log10(aspect ratio) are equidistant.
   - `M_IMAG`: imaginary part of the particle refractive index, the same for all grid points (default 0).
   - `N_COS_THETA_O_HALF`, `N_PHI_O`: orientation grid sizes on the reduced domain (`cos(theta_o)` in [0, 1], `phi_o` in [0, pi])
   - `--output PATH` overrides the output file of the preset.
   - `--delta-n-eff DN` makes the spheroid uniaxial with the optic axis along the symmetry axis and the birefringence `n_e - n_o = DN`. The mean index `(2 n_o + n_e)/3` stays on the `RI_real` axis.

2. Execute the script:

   ```sh
   python run_dda_spheroid_sweep.py                    # preset air
   python run_dda_spheroid_sweep.py --preset liquid    # water host
   ```

   The script creates the HDF5 file of the preset (`dda_results/dda_results_spheroid_sweep.h5` for `air`) if it does not exist and runs the sweep. Intermediate results are written per condition, so the run can be interrupted and resumed.

   **Shifted refractive-index sweep (`--shifted-ri`).** For a homogeneous isotropic particle, changing the refractive index only adds a scalar multiple of the identity to the interaction matrix. With this flag all `RI_real` values of a `(wavelength, D_ve, AR)` slot are solved by one shifted block-GMRES process instead of one solve per value:

   ```sh
   python run_dda_spheroid_sweep.py --shifted-ri --output dda_results/my_new_lut.h5
   ```

   - All `RI_real` values of a slot share one lattice, built for the largest `|m|` of the `RI_real` axis. The default mode builds a separate lattice for each value. The results of the two modes therefore differ at the level of the discretization error, except at the largest `RI_real`, where the lattices coincide.
   - Use the flag for new output files only. A file keeps the mode it was created with, and resuming it in the other mode is refused.
   - The flag cannot be combined with a nonzero `--delta-n-eff`. The shifted solve needs the inverse polarizabilities of all axes to change by the same amount, and a sweep at fixed birefringence does not have this property (Sec. 5.5 of `docs/theory_note.pdf`). Birefringent particles are solved one refractive index at a time.
   - The HDF5 layout is unchanged. The root attributes `shifted_ri` and `lattice_policy` and the provenance `config_hash` mark a file made in this mode.
   - Measured gain over the default mode for an oblate spheroid (`b/c = 2`, `D_ve` = 0.6 and 1.0 µm, 13 orientations, `RI_real` from 1.35 to 1.70): the total time drops by a factor of 7.7 to 9.0 with 15 `RI_real` values and by a factor of 15.1 to 17.8 with 30 values. The time of a slot hardly depends on the number of `RI_real` values. The benchmark is `scripts/run_shift_scaling.py`, and the details are in Sec. 5.9 of `docs/theory_note.pdf`.

3. See `read_spheroid_sweep.ipynb` for how to browse the file overview and retrieve results interactively.

**Output HDF5 structure** (`dda_results/dda_results_spheroid_sweep.h5`):

- Shared grid axes (root): `log_D_ve_grid` (log10 of the volume-equivalent diameter in um), `RI_real_grid`, `log_AR_grid`, `cos_theta_o_half_grid`, `phi_o_grid`
- Per-wavelength groups (e.g. `wl_0p453/`, `wl_0p638/`, `wl_0p834/`):
  - Forward scattering amplitudes (5D, float64): `S_fw_theta_re`, `S_fw_theta_im`, `S_fw_phi_re`, `S_fw_phi_im` — shape `(N_Dve, N_RI, N_AR, N_u_half, N_ph)`
  - Convergence flag: `converged` — shape `(N_Dve, N_RI, N_AR)`, bool
  - Group attribute `wl_0` stores the exact wavelength value
- Symmetry operations (cos_theta_o mirror to [-1,1], phi_o extension to [0,2pi]) are handled by the downstream consumer, not by block-DDA_Py

### Refractive-index sweep of any isotropic particle through the API

The shifted block GMRES is not limited to spheroids. For any homogeneous isotropic GRE particle, build the lattice for the largest `|m|` of the sweep and pass the list of refractive indices:

```python
import numpy as np
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

wl_0, m_m = 0.638, 1.0                        # vacuum wavelength [um], medium index
r_v_base, bc_ratio, ab_ratio, beta = 0.2, 1.5, 1.2, 0.1
m_list = np.linspace(1.35, 1.70, 8) + 0j      # refractive indices of the sweep
m_max = m_list[np.argmax(np.abs(m_list))]
m_p_xyz = np.array([m_max, m_max, m_max])     # the lattice is built for the largest |m|

rng = np.random.default_rng(12345)
gre = gaussian_ellipsoid_shape_model(r_v_base, bc_ratio, ab_ratio, beta, wl_0, m_p_xyz)
r_pts, _ = gre.compute_r_points_on_GRE(rng)
_, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
is_in = gre.extract_lattice_address_in_GRE_volume(gre.lattice_lf, gre.distance_factor, lattice_n, dist)
target = Target(gre.name, lattice_n, gre.lattice_lf, grid, is_in, m_p_xyz, r_v_base)

euler = np.array([[0.0, 0.5, 0.0], [1.0, 1.2, 0.3]])   # (L, 3) Euler angles [rad]
dd = DiscreteDipoles(target, IncidentField(wl_0, m_m, euler))
dd.set_interaction_matrix_shifted(m_list)
dd.solve_matrix_equation_shifted()
obs = dd.compute_observables_shifted()        # dict of arrays of shape (N_m, L)
```

`obs` holds `C_ext`, `C_abs`, `S_fw_PCAS_theta`, `S_fw_PCAS_phi` and `S_bk_OCBS` for every refractive index and orientation. `dd.converge_shifted` tells which refractive indices converged within `dd.tol` (default 1e-2) and `dd.itermax` (default 25). Loop over volume and shape in your own script, with one lattice per combination.

---

## ⚡ Performance

For a detailed comparison of computational cost between the old (Barrowes + ray) and new (Goodman + NumPy broadcasting) implementations, see [docs/theory_note.pdf](docs/theory_note.pdf), Sec. 6.

### Multi-core CPU parallelism

block-DDA_Py achieves multi-core parallelism through two complementary mechanisms — no external parallelization library (ray, multiprocessing, etc.) is needed.

| Layer | Mechanism | How to control |
|-------|-----------|----------------|
| **FFT** | `scipy.fft` multi-threaded 3D FFT | `_FFT_WORKERS = max(1, cpu_count − 2)` (auto); override with `DDA_FFT_WORKERS=N python run_dda.py` |
| **Block-Krylov** | All *L* orientations processed simultaneously in each Krylov iteration via NumPy/SciPy broadcasting and BLAS multi-threading | *L* = `N_alpha × N_beta × N_gamma` (general) or `N_beta` (spheroid mode) |

By default, the FFT uses all available CPU cores minus 2 (to keep the system responsive).
To dedicate all cores, set:
```sh
DDA_FFT_WORKERS=$(nproc) python run_dda.py
```

### Memory requirements

The two dominant memory consumers are:

| Array | Shape | Size formula |
|-------|-------|--------------|
| Interaction tensor FFT (static, allocated once) | `(2Nx, 2Ny, 2Nz, 3, 3)` complex128 | `1152 × N_cuboid` bytes |
| Block polarization arrays (peak, during MVP) | `(2Nx, 2Ny, 2Nz, 3, L)` × 2 complex128 | `768 × L × N_cuboid` bytes |

**Peak memory ≈ `(1152 + 768 × L) × N_cuboid` bytes**

where `N_cuboid = Nx × Ny × Nz` is the total number of cuboid grid cells (including vacuum) and `L` is the number of orientations.

The cuboid size scales approximately as:

$$N_\text{cuboid} \approx \left(\frac{2 \times \text{dpl} \times r_v \times \max|m_p|}{\lambda_0}\right)^3$$

The factor $2 \times \text{dpl}$ arises from two steps: (1) the cuboid must span the particle diameter $2r_v$, and (2) the lattice spacing is $d = \lambda_0 / (\max|m_p| \times \text{dpl})$ (default `dpl=17`).

### Practical examples: general mode vs spheroid mode

Conditions: λ₀ = 0.55 μm, max|m_p| = 1.5, dpl = 17, orientation grid *N*<sub>α</sub> = 40, *N*<sub>β</sub> = 20, *N*<sub>γ</sub> = 1 (total 800 orientations).

Both modes output 800 orientations. The difference is the number of DDA solves (*L*):

| Mode | DDA solves (*L*) | Output grid | Description |
|------|------------------|-------------|-------------|
| General  | 800 | 800 orientations | All orientations solved by DDA |
| Spheroid | 20  | 800 orientations | Only *N*<sub>β</sub> = 20 solved; remaining filled analytically |

#### Peak memory

| r_v (μm) | N_cuboid | General mode | Spheroid mode |
|-----------|----------|---------------------|---------------------|
| 0.3       | ~22,000  | ~13 GB              | ~0.4 GB             |
| 0.5       | ~100,000 | ~61 GB              | ~1.6 GB             |
| 1.0       | ~800,000 | ~491 GB             | ~13 GB              |

#### Computation time

Extrapolated from a smaller benchmark (r_v = 0.1 μm, Intel Core i7-13700H, 14 cores) assuming $T \propto L \times N_\text{cuboid} \log N_\text{cuboid}$.

| r_v (μm) | N_cuboid | General mode | Spheroid mode |
|-----------|----------|---------------------|---------------------|
| 0.3       | ~22,000  | ~1 min              | ~2 s                |
| 0.5       | ~100,000 | ~6 min              | ~9 s                |
| 1.0       | ~800,000 | ~58 min             | ~1.4 min            |

Note: actual times for large N_cuboid may be longer due to memory-bandwidth limitations, but the general/spheroid ratio remains valid.

#### Summary

Spheroid mode reduces both peak memory and computation time by a factor of **1/40** (*L*<sub>spheroid</sub> / *L*<sub>general</sub> = 20 / 800).

---

## 🔄 Changes from Previous Version

### Algorithm
| Module | Old | New |
|--------|-----|-----|
| `mvp_fft/mvp_fft.py` | Barrowes (2001) asymmetric multilevel block-Toeplitz FFT | Goodman (1991) 3D FFT on doubled grid `(2Nx,2Ny,2Nz,3,3)` |
| `bl_krylov/bl_krylov.py` | `ray`-based multiprocessing parallelization | NumPy/SciPy broadcasting — fully vectorized, no external parallelism |
| `shape_model/gaussian_ellipsoid.py` | `numba` JIT-compiled nested loops | Pure NumPy `cumsum`-based interior detection |

### Parameter sweep (run_dda.py + dda_results/)
| Item | Old | New |
|------|-----|-----|
| Wavelength / medium RI | Single `wl_0` + list of `m_m` | List of `(wl_0, m_m)` pairs: `wl_m_m_pairs` |
| Particle RI | Single `m_p_xyz` | List of `m_p_xyz`: `m_p_xyz_list` (supports anisotropic sweep) |
| GRE geometry reuse | Rebuilt for every (wl, m_m, m_p) | Rebuilt per (wl_0, m_p_xyz) combination (lattice spacing depends on dpl) |
| HDF5 dataset shape | `(N_mm, N_rv, N_bc, N_ab, N_bt, N_ori)` | `(N_pairs, N_mp, N_rv, N_bc, N_ab, N_bt, N_ori)` |

### Dependencies
| Package | Status |
|---------|--------|
| `ray` | **Removed** (replaced by NumPy broadcasting) |
| `numba` | **Removed** (replaced by pure NumPy) |
| `pywin32` | **Removed** (Windows-only) |
| `scipy` | Retained — `scipy.fft` (multi-threaded FFT), `scipy.spatial.transform` (batch rotation) |
| `scikit-learn` | Retained — `KDTree` for lattice neighbor search |
| `ipympl` | Added — interactive 3D plots in notebooks |

### Lattice spacing (v0.4.0)
| Item | Old | New |
|------|-----|-----|
| Lattice spacing formula | Empirical (shape-only) | dpl-based: `d = λ₀ / (max\|m_p\| × dpl)`, default `dpl=17` |
| Volume consistency | `ve_radius ≈ r_v_base` (approximate) | `ve_radius == r_v_base` exactly (volume-preserving rescaling) |

### Spheroid mode (v0.5.0)
- For spheroids (`ab_ratio=1`, `gre_beta=0`), azimuthal orientation average is computed analytically
- DDA solves only $N_\beta$ orientations; full grid filled via constraint relations
- Verified at machine precision (~1e-16) in `test_spheroid_symmetry.ipynb`

### Orientation sampling (v0.6.0)
| Item | Old | New |
|------|-----|-----|
| Sampling method | Random (Monte Carlo) | Deterministic uniform grid on SO(3) |
| User specifies | `num_orientations` (total count) | `N_alpha_ori`, `N_beta_ori`, `N_gamma_ori` (grid divisions) |
| Polar angle (beta) | `Uniform(0, π)` (biased) | `cos(β)` equally spaced in (−1, 1) (equal-area) |

### Spheroid parameter sweep (v0.7.1 – v0.7.4)
- v0.7.1: Spheroid-mode parameter-sweep driver with HDF5 output
- v0.7.2: Multi-wavelength support added to the sweep
- v0.7.3: Sweep grid range refinements; `.h5` added to `.gitignore`
- v0.7.4: Interactive notebook replaces the read-example script for HDF5 inspection

### Paper-production sweep and solver upgrades (v0.7.5 – v0.7.8)
- v0.7.5: Production-sweep infrastructure; DDA ↔ VIEM volume-element parity rule
- v0.7.6: Auto-`dpl` per particle slot; per-iteration residual history stored; incremental-QR block-GMRES; benchmark material `n20` (m_p = 2.0) added
- v0.7.7: Bundled paper-production sweep results under `dda_results/paper/`
- v0.7.8: Manuscript-release snapshot for the JQSRT submission

### Shifted block-GMRES refractive-index sweep (v0.8.0 – v0.8.1)
- v0.8.0: `bl_gmres_shifted_mvp_fft`: one block-Arnoldi process for all refractive indices of a homogeneous isotropic particle on a common lattice
- v0.8.0: `DiscreteDipoles.set_interaction_matrix_shifted`, `solve_matrix_equation_shifted`, `compute_observables_shifted`: sweep API; observables are evaluated without assembling the solutions
- v0.8.0: `run_dda_spheroid_sweep.py --shifted-ri`: the `RI_real` axis is solved in one process per slot; the default mode is unchanged
- v0.8.0: `tests/`: pytest suite comparing the shifted path with the existing solver and with Mie theory
- v0.8.1: The per-shift least-squares problems are handled by an incremental block-Givens QR (`lsq_method="incremental"`, default). The dense method of v0.8.0 is kept as `lsq_method="dense"`. At 64 orientations the incremental method cuts the least-squares time by a factor of 2.3 to 6.8. The results of the shifted sweep change at rounding level only

### Corrected backward-scattering amplitude (v0.8.2)
- v0.8.2: `S_bk_OCBS` (*S*<sub>bak</sub> of CAS-v2) is now (−*S*<sub>11</sub> + *S*<sub>22</sub>)/2 at the backward direction. This is the corrected Eq. (28) of Moteki and Adachi (2024), given in the erratum to that paper. Up to v0.8.1 the code followed the printed equation with 1/√2 in place of 1/2. The change applies to `compute_OCBS_observable_S_bk`, to `compute_observables_shifted` and to the Mie reference `mie_compute_q_and_s`.
- Values of `S_bk_OCBS` and `S_bk_OCBS_mie` from v0.8.1 or earlier are √2 times the present ones. The phase is the same, and so is the ratio of the DDA value to the Mie value. This includes the files under `dda_results/paper/`, which are kept as produced. Multiply their backward amplitudes by 1/√2 to compare with v0.8.2.
- The forward-scattering amplitudes and the cross sections are not affected.

### Shifted refractive-index sweep for general shapes (v0.8.3)
- v0.8.3: `run_dda.py --shifted-ri` solves all isotropic entries of `m_p_xyz_list` of a slot with one shifted block-GMRES process, for any GRE shape. The default mode is unchanged. `run_dda.py --output` selects the output file.
- v0.8.3: The usage section explains how to sweep refractive index, volume, shape and orientation with the two drivers and with the API.

### Environment
- **Python**: 3.12 (Windows) → **3.13 (Linux/WSL2)**
- **Virtual environment**: `myenv` (Windows) → `.venv` (Linux, managed by `uv`)

---

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 📖 References
- Discrete Dipole Approximation
    - Chaumet, P. C. (2022). The discrete dipole approximation: A review. Mathematics, 10(17), 3049.
    - Moteki, N. (2016). Discrete dipole approximation for black carbon-containing aerosols in arbitrary mixing state: A hybrid discretization scheme. Journal of Quantitative Spectroscopy and Radiative Transfer, 178, 306–314.
    - Yurkin, M. A., & Hoekstra, A. G. (2007). The discrete dipole approximation: An overview and recent developments. Journal of Quantitative Spectroscopy and Radiative Transfer, 106(1–3), 558–589.
    - Draine, B. T., & Flatau, P. J. (1994). Discrete-dipole approximation for scattering calculations. Journal of the Optical Society of America A, 11(4), 1491–1499.
    - Chaumet, P. C., & Rahmani, A. (2009). Coupled-dipole method for magnetic and negative-refraction materials. Journal of Quantitative Spectroscopy and Radiative Transfer, 110(1–2), 22–29. (polarizability with radiative correction used in this code)

- Gaussian Random Ellipsoid
    - Muinonen, K., & Pieniluoma, T. (2011). Light scattering by Gaussian random ellipsoid particles: First results with discrete-dipole approximation. Journal of Quantitative Spectroscopy and Radiative Transfer, 112(11), 1747–1752.

- Block-Krylov Subspace Methods
  - Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. SIAM Journal on Scientific and Statistical Computing, 7(3), 856–869.
  - Simoncini, V., & Gallopoulos, E. (1996). Convergence properties of block GMRES and matrix polynomials. Linear Algebra and its Applications, 247, 97–119.
  - Moteki, N. (2026). Block-Krylov discrete-dipole and volume-integral-equation solvers for multi-orientation light scattering across the dielectric–metal contrast range. Journal of Quantitative Spectroscopy and Radiative Transfer, 362, 110046.
  - Frommer, A., & Glässner, U. (1998). Restarted GMRES for shifted linear systems. SIAM Journal on Scientific Computing, 19(1), 15–26.
  - Argentin, C., Inzhevatkin, K. G., & Yurkin, M. A. (2026). Efficient refractive-index sweeps in the discrete dipole approximation using shifted Krylov methods. arXiv:2611.15909. (shifted solve in ADDA; Sec. 5.10 of `docs/theory_note.pdf` compares it with the method of this code)
  - El Guennouni, A., Jbilou, K., & Sadok, H. (2003). A block version of BiCGSTAB for linear systems with multiple right-hand sides. Electronic Transactions on Numerical Analysis, 16, 129–142.
  - Tadano, H., Sakurai, T., & Kuramashi, Y. (2009). Block BiCGGR: a new Block Krylov subspace method for computing high accuracy solutions. JSIAM Letters, 1, 44–47. (form of the Block BiCGSTAB algorithm implemented here)
  - Gu, X. M., Carpentieri, B., Huang, T. Z., & Meng, J. (2016). Block variants of the COCG and COCR methods for solving complex symmetric linear systems with multiple right-hand sides. In Numerical Mathematics and Advanced Applications ENUMATH 2015 (pp. 305–313). Springer International Publishing.

- FFT-based acceleration
  - Goodman, J. J., Draine, B. T., & Flatau, P. J. (1991). Application of fast-Fourier-transform techniques to the discrete-dipole approximation. Optics Letters, 16(15), 1198–1200.

- Complex Amplitude Sensing (particle measurement technique)
  - Moteki, N. (2021). Measuring the complex forward-scattering amplitude of single particles by self-reference interferometry: CAS-v1 protocol. Optics Express, 29(13), 20688–20714.
  - Moteki, N., & Adachi, K. (2024). Measuring the polarized complex forward-scattering amplitudes of single particles in unbounded fluid flow: CAS-v2 protocol. Optics Express, 32(21), 36500–36522.
  - Moteki, N., & Adachi, K. (2026). Measuring the polarized complex forward-scattering amplitudes of single particles in unbounded fluid flow: CAS-v2 protocol: erratum. Optics Express (submitted). (corrected definition of *S*<sub>bak</sub>)

## 📢 Author
Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: nobuhiro.moteki@gmail.com
