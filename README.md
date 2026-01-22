# block-DDA_Py

## 📌 Description
A python code for the discrete dipole approximation using block-Krylov type iterative solvers, with custom features for light-scattering simulations for enviromental particles (e.g., mineral dust)

### Main Features
1. Supports batch calculations for many paricle orientations (or many incident beams).
2. Supports birefringent particle materials.
3. Supports a parameteric irregular shape model suitable for mineral dust particles: Gaussial Random Ellipsoid ([GRE](https://doi.org/10.1016/j.jqsrt.2011.02.013)).
4. The ratio of the dipole side length to the medium wavelength is automatically set (typically between 14 to 20),depending on size-parameter and shape.
5. Supports simulation of the key quantities in the Complex Amplitude Sensing version 2 ([CAS-v2](https://doi.org/10.1364/OE.533776)): the polarized complex forward-scattering amplitudes {Re*S*(0°)<sub>*s*</sub>, Im*S*(0°)<sub>*s*</sub>, Re*S*(0°)<sub>*p*</sub>, Im*S*(0°)<sub>*p*</sub>} and the complex backward-scattering amplitude *S*<sub>bak</sub>.
6. For reference, the block-DDA_Py outputs a comparison of the DDA solution to the Mie solution for a volume-equivalent sphere with axes-average refractive index.

### Algorithms
1. block-DDA_Py uses the [block-Krylov methods](https://people.math.ethz.ch/~mhg/pub/delhipap.pdf) for iterativaly solving the matrix equation to enable batch computations for many incident beams.
2. A batch solution for many incident beams is internally converted to the batch solution for many orientations through rotational transformation.
3. An efficient memory use and fast matrix-vector multiplications using the [Barrowes algorithm](https://onlinelibrary.wiley.com/doi/abs/10.1002/mop.1348).
4. Supports multicore concurrent computing in a personal computer (thanks to the python module "[ray](https://pypi.org/project/ray/)").

### Limitations

There are some notable limitations of current block-DDA_Py (v0.2.1):

1. Supported shape model is only GRE. 
2. Current shape model supports only single-component particles (refractive index is uniform inside each particle).
3. Supported incident beam type is only a plane wave or a set of plane waves from different directions. 

I'll remove these limitations upon user's requests and application needs.


---

## 🚀 Installation

The author developed and tested current block-DDA_Py (v0.2.1) using Python 3.12.8 in Windows 11 machines.

#### 1. Clone the repository
```sh
git clone https://github.com/NobuhiroMoteki/block-DDA_Py.git
cd block-DDA_Py
```

#### 2. Install dependencies
```sh
pip install -r requirements.txt
```
---

## 🔧 Usage
### Visualization of the GRE model particle (Optional)
1. Open the JupyterNotebook file `run_gaussian_ellipsoid.ipynb` and edit the following input parameters:
   - Volume-equivalent radius of base ellipsoid: `r_v_base`
   - Ratio of semiradius of the base ellipsoid along y axis (b) to z axis (c): `bc_ratio`
   - Ratio of semiradius of the base ellipsoid along x axis (a) to y axis (b): `ab_ratio`
   - Standard deviation of GRE surface deformation: `beta`.
2. Execute the JupyterNotebook (A 3D-plot will appear).


### Single execution

1. Open the JupyterNotebook file `test_dda.ipynb` and edit the following input parameters:
   - Parameters of the GRE shape model: `r_v_base`, `bc_ratio`, `ab_ratio`, and `beta`.
   - Vacuum wavelength: `wl_0`  (length unit must be the same as `r_v_base`)
   - Medium refracrive index (real number): `m_m`
   - Particle refractive index (complex number) for each of the x, y, z-axes of particle coordinate: `m_p_x`, `m_p_y`, `m_p_z`.
   - Number of incident beams propagating from randomly choosen directions: `number_of_orientations`
2. Execute the JupyterNotebook `test_dda.ipynb`.

### Many executions (parameter sweep)
1. First of all, we define the parameter sweep condition and prepare an output-storage file (in `.hdf5` format). Open the JupyterNotebook file `.\dda_results\create_h5py.ipynb` and edit the following input parameters:
   - Vacuum wavelength: `wl_0` 
   - Particle refractive index (complex number) for each of the x, y, z-axes of particle coordinate: `m_p_x`, `m_p_y`, `m_p_z`.
   - List of the medium refracrive index (real number): `m_m`
   - List of the volume-equivalent radius of base ellipsoid: `r_v_base`
   - List of the ratio of semiradius of the base ellipsoid along y axis (b) to z axis (c): `bc_ratio`
   - List of the ratio of semiradius of the base ellipsoid along x axis (a) to y axis (b): `ab_ratio`
   - List of the standard deviation of GRE surface deformation: `beta`.
   - Number of randomly choosen orientations: `number_of_orientations`
   - Output-storage filename: You can change the filename by editing the right hand side of `filename = pcas_ocbs_simulated_data.hdf5`
   Here, "List" means a discrete set of values in the Python-list format `[a, b, c, ... ]`.
2. Execute the JupyterNotebook `.\dda_results\create_h5py.ipynb`. You can check the contents of the generated `.hdf5` file by executing the JupyterNotebook `.\dda_results\check_h5py.ipynb` (when needed). The results will be stored in the prepared `.hdf5` file.
3. Execute the `run_dda.py`. The execution will be repeated sweeping over the lists of `m_m`, `r_v_base`, `bc_ratio`, `ab_ratio`, and `beta`.
4. Use the `plot_dda_results.ipynb` (after some modifications if needed) for loading and visualizing the parameter-sweeped DDA results from the `.hdf5` file.

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
  - Barrowes, B. E., Teixeira, F. L., & Kong, J. A. (2001). Fast algorithm for matrix–vector multiply of asymmetric multilevel block‐Toeplitz matrices in 3‐D scattering. Microwave and Optical technology letters, 31(1), 28-32.

- Complex Amplitude Sensing (particle measurement technique)
  - Moteki, N. (2021). Measuring the complex forward-scattering amplitude of single particles by self-reference interferometry: CAS-v1 protocol. Optics Express, 29(13), 20688-20714.
  - Moteki, N., & Adachi, K. (2024). Measuring the polarized complex forward-scattering amplitudes of single particles in unbounded fluid flow: CAS-v2 protocol. Optics Express, 32(21), 36500-36522.



## 📢 Author
Name: Nobuhiro Moteki
GitHub: @NobuhiroMoteki
Email: nobuhiro.moteki@gmail.com


