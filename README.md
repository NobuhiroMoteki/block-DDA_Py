# block-DDA_Py  [![version](https://img.shields.io/badge/version-v0.3.0-blue)](https://github.com/NobuhiroMoteki/block-DDA_Py/releases)

## 📌 Description
A Python code for the Discrete Dipole Approximation (DDA) using block-Krylov type iterative solvers, with custom features for light-scattering simulations of environmental particles (e.g., mineral dust).

### Main Features
1. Supports batch calculations for many particle orientations (or many incident beams).
2. Supports birefringent particle materials (anisotropic complex refractive index along x, y, z axes).
3. Supports a parametric irregular shape model suitable for mineral dust particles: Gaussian Random Ellipsoid ([GRE](https://doi.org/10.1016/j.jqsrt.2011.02.013)).
4. The ratio of the dipole side length to the medium wavelength is automatically set (typically between 14 to 20), depending on size-parameter and shape.
5. Supports simulation of the key quantities in the Complex Amplitude Sensing version 2 ([CAS-v2](https://doi.org/10.1364/OE.533776)): the polarized complex forward-scattering amplitudes {Re*S*(0°)<sub>*s*</sub>, Im*S*(0°)<sub>*s*</sub>, Re*S*(0°)<sub>*p*</sub>, Im*S*(0°)<sub>*p*</sub>} and the complex backward-scattering amplitude *S*<sub>bak</sub>.
6. For reference, block-DDA_Py outputs a comparison of the DDA solution to the Mie solution for a volume-equivalent sphere with axes-average refractive index.
7. Supports parameter sweeps over vacuum wavelength, medium refractive index, particle refractive index (including anisotropic cases), and shape parameters, with results stored in HDF5 format.

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

## 📄 Technical Note

For a detailed description of the theory and algorithms, see [docs/theory_note.pdf](docs/theory_note.pdf).

---

## 🚀 Installation

block-DDA_Py is developed and tested with **Python 3.13 on Linux (WSL2)**.

#### 1. Clone the repository
```sh
git clone https://github.com/NobuhiroMoteki/block-DDA_Py.git
cd block-DDA_Py
```

#### 2. Create a virtual environment and install dependencies
```sh
uv venv
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
   - Number of orientations: `num_orientations`
2. Execute `test_dda.ipynb`. DDA results are compared to the Mie reference.

### Parameter sweep
1. Open `dda_results/create_h5py.ipynb` and configure the sweep parameters:
   - List of (vacuum wavelength, medium RI) pairs: `wl_m_m_pairs` — shape `(N_pairs, 2)`
   - List of particle refractive indices (supports anisotropic x/y/z): `m_p_xyz_list` — shape `(N_mp, 3)`
   - List of volume-equivalent radii: `r_v_base_list`
   - List of axis ratios: `bc_ratio_list`, `ab_ratio_list`
   - List of GRE roughness parameters: `gre_beta_list`
   - Number of randomly chosen orientations: `num_orientations`
   - Output filename: `OUTPUT_FILE` in `run_dda.py`
2. Execute `dda_results/create_h5py.ipynb` to generate the HDF5 output file.
3. Execute `run_dda.py`. The sweep loops over all combinations of `wl_m_m_pairs`, `m_p_xyz_list`, and shape parameters. GRE geometry is computed once per shape condition and reused across all wavelength/RI combinations.
4. Use `dda_results/check_h5py.ipynb` to inspect the HDF5 file contents and verify results.
5. Use `plot_dda_results.ipynb` to visualize the parameter-swept DDA results.

---

## ⚡ Performance

### Multi-core CPU parallelism

block-DDA_Py achieves multi-core parallelism through two complementary mechanisms — no external parallelization library (ray, multiprocessing, etc.) is needed.

| Layer | Mechanism | How to control |
|-------|-----------|----------------|
| **FFT** | `scipy.fft` multi-threaded 3D FFT | `_FFT_WORKERS = max(1, cpu_count − 2)` (auto); override with `DDA_FFT_WORKERS=N python run_dda.py` |
| **Block-Krylov** | All *L* orientations processed simultaneously in each Krylov iteration via NumPy/SciPy broadcasting and BLAS multi-threading | Controlled by `num_orientations` (= *L*) |

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

$$N_\text{cuboid} \approx \left(\frac{34 \times r_v \times m_m}{\lambda_0}\right)^3$$

The factor 34 = 2 × 17 arises from two steps: (1) the cuboid must span the particle diameter $2r_v$, and (2) the lattice spacing is set to $d \approx \lambda_\text{medium} / 17 = \lambda_0 / (17 m_m)$ (the ratio of dipole side length to medium wavelength is typically 14–20; 17 is used here as a representative midpoint).

#### Practical examples (λ₀ = 0.55 μm, m_m = 1.0)

| r_v (μm) | L (orientations) | N_cuboid (approx.) | Peak memory |
|----------|------------------|--------------------|-------------|
| 0.3      | 10               | ~7,000             | ~70 MB      |
| 0.3      | 100              | ~7,000             | ~545 MB     |
| 0.5      | 10               | ~30,000            | ~295 MB     |
| 0.5      | 100              | ~30,000            | ~2.4 GB     |
| 1.0      | 10               | ~240,000           | ~2.4 GB     |
| 1.0      | 50               | ~240,000           | ~9.4 GB     |

> **Tip**: To fit within available RAM, reduce `num_orientations` (= *L*). For large particles, run separate sweeps with smaller *L* and merge the HDF5 results.

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
| GRE geometry reuse | Rebuilt for every (wl, m_m, m_p) | Built once per shape; reused across all wavelength/RI combinations |
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

- Gaussian Random Ellipsoid
    - Muinonen, K., & Pieniluoma, T. (2011). Light scattering by Gaussian random ellipsoid particles: First results with discrete-dipole approximation. Journal of Quantitative Spectroscopy and Radiative Transfer, 112(11), 1747–1752.

- Block-Krylov Subspace Methods
  - El Guennouni, A., Jbilou, K., & Sadok, H. (2003). A block version of BiCGSTAB for linear systems with multiple right-hand sides. Electronic Transactions on Numerical Analysis, 16, 129–142.
  - Gu, X. M., Carpentieri, B., Huang, T. Z., & Meng, J. (2016). Block variants of the COCG and COCR methods for solving complex symmetric linear systems with multiple right-hand sides. In Numerical Mathematics and Advanced Applications ENUMATH 2015 (pp. 305–313). Springer International Publishing.

- FFT-based acceleration
  - Goodman, J. J., Draine, B. T., & Flatau, P. J. (1991). Application of fast-Fourier-transform techniques to the discrete-dipole approximation. Optics Letters, 16(15), 1198–1200.

- Complex Amplitude Sensing (particle measurement technique)
  - Moteki, N. (2021). Measuring the complex forward-scattering amplitude of single particles by self-reference interferometry: CAS-v1 protocol. Optics Express, 29(13), 20688–20714.
  - Moteki, N., & Adachi, K. (2024). Measuring the polarized complex forward-scattering amplitudes of single particles in unbounded fluid flow: CAS-v2 protocol. Optics Express, 32(21), 36500–36522.

## 📢 Author
Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: nobuhiro.moteki@gmail.com
