"""Two-sphere cluster (doublet) shape model.

Equal-radius monomers aligned along the particle z-axis (convention matches
block-VIEM.jl `_doublet_along_z` in `viem_results/run_viem.jl`). The geometry
enables the DDA spheroid-mode α-expansion (cylindrical symmetry about z).

Convention (CLAUDE.md §2.4):
    R = a_eq / 2^(1/3)                          # monomer radius
    gap = gap_ratio · R,   gap_ratio = 0.1      # surface-to-surface gap
    step = 2R + gap                             # center-to-center distance
    centers: z = ±step/2,  x = y = 0

The public API mirrors `gaussian_ellipsoid_shape_model` enough for
`_build_gre_geometry` style callers to swap in a doublet by calling
`build()` once.
"""
import numpy as np


class two_sphere_cluster_shape_model:
    """Doublet of two equal-radius spheres aligned along particle z-axis."""

    def __init__(self, r_v_total, wl_0, m_p_xyz, gap_ratio=0.1, dpl=17):
        """
        Parameters
        ----------
        r_v_total : float
            Volume-equivalent radius of the whole aggregate [um]. The monomer
            radius is derived as `R = r_v_total / 2^(1/3)` so that the summed
            volume of the two spheres equals `(4/3)π · r_v_total**3`.
        wl_0 : float
            Vacuum wavelength [um].
        m_p_xyz : array_like of complex, length 3
            Particle complex refractive index along (x, y, z).
        gap_ratio : float
            Surface-to-surface gap divided by R. Default 0.1 matches VIEM.
        dpl : int
            Dipoles per wavelength inside the particle. Default 17.
        """
        m_p_xyz = np.asarray(m_p_xyz)
        self.r_v_total = float(r_v_total)
        self.gap_ratio = float(gap_ratio)
        self.R_monomer = self.r_v_total / 2.0 ** (1.0 / 3.0)
        self.gap = self.gap_ratio * self.R_monomer
        self.step = 2.0 * self.R_monomer + self.gap
        self.centers = np.array([
            [0.0, 0.0, -0.5 * self.step],
            [0.0, 0.0, +0.5 * self.step],
        ])

        self.lattice_lf = wl_0 / (np.max(np.abs(m_p_xyz)) * dpl)

        self.name = ("doublet__a_eq={:.3f}um__R={:.4f}um__gap={:.4f}um"
                     "__axis=z").format(self.r_v_total, self.R_monomer, self.gap)

    def build(self):
        """Generate lattice + inside-target mask.

        Returns
        -------
        lattice_n : ndarray of int32, shape (3,)
        lattice_lf : float
        lattice_grid_points : ndarray, shape (Nx*Ny*Nz, 3)
        is_in_target : 1-D bool array, size Nx*Ny*Nz
        """
        R = self.R_monomer
        lf = self.lattice_lf

        # Cuboid bounding box (single lattice_lf padding, matching GRE convention)
        x_lim = R + lf
        y_lim = R + lf
        z_lim = (self.step / 2.0 + R) + lf

        x_arr = np.arange(-x_lim, x_lim, lf)
        y_arr = np.arange(-y_lim, y_lim, lf)
        z_arr = np.arange(-z_lim, z_lim, lf)
        lattice_n = np.array([len(x_arr), len(y_arr), len(z_arr)], dtype=np.int32)

        x_grid, y_grid, z_grid = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
        lattice_grid_points = np.column_stack([
            x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

        # Inside-target test: point is inside if it lies within either sphere.
        # Voxel-centre test with the un-rescaled lattice_lf — the Target class
        # then applies volume-preserving rescaling so that N_occ · d_adj³ equals
        # the nominal total volume.
        dx = lattice_grid_points[:, 0]
        dy = lattice_grid_points[:, 1]
        d2_s1 = dx ** 2 + dy ** 2 + (lattice_grid_points[:, 2] - self.centers[0, 2]) ** 2
        d2_s2 = dx ** 2 + dy ** 2 + (lattice_grid_points[:, 2] - self.centers[1, 2]) ** 2
        R2 = R * R
        is_in_target = (d2_s1 <= R2) | (d2_s2 <= R2)

        return lattice_n, lf, lattice_grid_points, is_in_target
