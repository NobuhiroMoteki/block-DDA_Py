import numpy as np
import scipy.interpolate
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)
        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


setattr(Axes3D, 'arrow3D', _arrow3D)


class gaussian_ellipsoid_shape_model:
    '''
    Generate a lattice of volume grid points for a Gaussian Random Ellipsoid (GRE)
    used in DDA calculations.

    Theoretical formulae from Muinonen & Pieniluoma 2011 JQSRT.
    '''

    def __init__(self, r_v_base, bc_ratio, ab_ratio, beta):
        '''
        Parameters
        ----------
        r_v_base : float  volume-equivalent radius of base ellipsoid [um], range [0.1, 0.5]
        bc_ratio : float  b/c semi-axis ratio, range [1.0, 7.0]
        ab_ratio : float  a/b semi-axis ratio, range [1.0, 2.0]
        beta     : float  std-dev of Gaussian surface deformation, range [0, 0.3]
        '''
        self.r_v_base = r_v_base
        self.bc_ratio = bc_ratio
        self.ab_ratio = ab_ratio
        self.beta     = beta
        self.name     = None

        c = np.cbrt(self.r_v_base ** 3 / (self.ab_ratio * self.bc_ratio ** 2))
        self.lattice_lf      = (1 / 75) * np.sqrt(c / 0.05) * (self.ab_ratio * self.bc_ratio) ** (1 / 5)  # default
        #self.lattice_lf      = (1 / 150) * np.sqrt(c / 0.05) * (self.ab_ratio * self.bc_ratio) ** (1 / 5)
        self.distance_factor = 3 ** 0.5 / 2 * (bc_ratio * ab_ratio * (c / 0.1)) ** (1 / 6)


    def compute_r_points_on_GRE(self, rng):
        '''
        Sample surface points of the Gaussian Random Ellipsoid.

        Parameters
        ----------
        rng : numpy.random.Generator

        Returns
        -------
        r_points_on_GRE_surf : ndarray, shape (N, 3)
        xyz_meshes_GRE_surf  : tuple of three 2-D meshes (x, y, z)
        '''
        self.name = ("GRE_shape__r_v_base={:.2f}um__ab_ratio={:.1f}"
                     "__bc_ratio={:.1f}__beta={:.2f}".format(
                         self.r_v_base, self.ab_ratio, self.bc_ratio, self.beta))

        c  = np.cbrt(self.r_v_base ** 3 / (self.ab_ratio * self.bc_ratio ** 2))
        b  = c * self.bc_ratio
        a  = b * self.ab_ratio
        h0 = c ** 2 / a
        lc = c * 0.3

        # Coarse polar grid for the covariance matrix
        N_theta, N_phi = 25, 100
        theta = np.linspace(0, np.pi,     N_theta)
        phi   = np.linspace(0, 2 * np.pi, N_phi)
        theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')

        x0 = a * np.sin(theta_mesh) * np.cos(phi_mesh)
        y0 = b * np.sin(theta_mesh) * np.sin(phi_mesh)
        z0 = c * np.cos(theta_mesh)

        # Vectorised covariance matrix: C[i,j] = beta^2 * exp(-||r_i - r_j||^2 / (2 lc^2))
        # Using component-wise squared differences to match original IEEE 754 evaluation order.
        N = N_theta * N_phi
        px = x0.ravel(); py = y0.ravel(); pz = z0.ravel()
        d2 = ((px[:, None] - px[None, :]) ** 2
              + (py[:, None] - py[None, :]) ** 2
              + (pz[:, None] - pz[None, :]) ** 2)        # shape (N, N)
        s_vec_cov_matrix = self.beta ** 2 * np.exp(-0.5 * d2 / lc ** 2)

        s_vec_mean = np.zeros(N)
        s_samples  = rng.multivariate_normal(s_vec_mean, s_vec_cov_matrix,
                                              size=1, method='eigh').squeeze()

        # Reshape from flat (N,) back to (N_theta, N_phi)
        s = s_samples.reshape(N_theta, N_phi)

        f_interp = scipy.interpolate.RectBivariateSpline(theta, phi, s)

        # Fine grid for surface point sampling
        surface_interpolation_factor = int(4 * np.cbrt(self.bc_ratio * self.ab_ratio))
        theta_new = np.linspace(0, np.pi,     surface_interpolation_factor * N_theta)
        phi_new   = np.linspace(0, 2 * np.pi, surface_interpolation_factor * N_phi)
        theta_new_mesh, phi_new_mesh = np.meshgrid(theta_new, phi_new, indexing='ij')

        s_intp = f_interp(theta_new, phi_new)

        x0 = a * np.sin(theta_new_mesh) * np.cos(phi_new_mesh)
        y0 = b * np.sin(theta_new_mesh) * np.sin(phi_new_mesh)
        z0 = c * np.cos(theta_new_mesh)

        n0_norm = np.sqrt((x0 / a ** 2) ** 2 + (y0 / b ** 2) ** 2 + (z0 / c ** 2) ** 2)
        n0_vec  = np.array([x0 / a ** 2, y0 / b ** 2, z0 / c ** 2]) / n0_norm

        deform = h0 * ((np.exp(s_intp) - 0.5 * self.beta ** 2) - 1)
        x_GRE = x0 + deform * n0_vec[0]
        y_GRE = y0 + deform * n0_vec[1]
        z_GRE = z0 + deform * n0_vec[2]

        xyz_meshes_GRE_surf  = (x_GRE, y_GRE, z_GRE)
        r_points_on_GRE_surf = np.column_stack([x_GRE.ravel(), y_GRE.ravel(), z_GRE.ravel()])

        return r_points_on_GRE_surf, xyz_meshes_GRE_surf


    def create_cuboid_lattice_that_encloses_GRE_shape(self, r_points_on_GRE_surf):
        '''
        Create the cuboid lattice that encloses the GRE surface.

        Parameters
        ----------
        r_points_on_GRE_surf : ndarray, shape (N, 3)

        Returns
        -------
        lattice_domain      : tuple of three [min, max] pairs
        lattice_n           : ndarray of int, shape (3,)  [Nx, Ny, Nz]
        lattice_grid_points : ndarray, shape (Nx*Ny*Nz, 3)
        '''
        x_lim = np.max(np.abs(r_points_on_GRE_surf[:, 0])) + self.lattice_lf
        y_lim = np.max(np.abs(r_points_on_GRE_surf[:, 1])) + self.lattice_lf
        z_lim = np.max(np.abs(r_points_on_GRE_surf[:, 2])) + self.lattice_lf

        lattice_domain = ([-x_lim, x_lim], [-y_lim, y_lim], [-z_lim, z_lim])

        # Compute arange once per axis; reuse for both meshgrid and lattice_n
        x_arr = np.arange(-x_lim, x_lim, self.lattice_lf)
        y_arr = np.arange(-y_lim, y_lim, self.lattice_lf)
        z_arr = np.arange(-z_lim, z_lim, self.lattice_lf)
        lattice_n = np.array([len(x_arr), len(y_arr), len(z_arr)], dtype=np.int32)

        x_grid, y_grid, z_grid = np.meshgrid(x_arr, y_arr, z_arr, indexing='ij')
        lattice_grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

        return lattice_domain, lattice_n, lattice_grid_points


    def find_nearest_distance_from_the_GRE_surf(self, lattice_grid_points, r_points_on_GRE_surf):
        '''
        Find the minimum distance of each lattice point to the GRE surface.

        Returns
        -------
        dist_from_GRE : 1-D ndarray, size = Nx*Ny*Nz
        '''
        tree = KDTree(r_points_on_GRE_surf)
        dist, _ = tree.query(lattice_grid_points, k=1)
        return dist[:, 0]


    @staticmethod
    def extract_lattice_address_in_GRE_volume(lattice_lf, distance_factor, lattice_n, dist_from_GRE):
        '''
        Determine which lattice points lie inside the GRE volume using a
        three-axis ray-casting test (forward + backward scan along each axis).

        A point is marked inside along axis α if it lies strictly between the
        first surface crossing from each end — i.e. the surface voxel itself
        is excluded (matching the original `continue` logic).

        Parameters
        ----------
        lattice_lf      : float
        distance_factor : float
        lattice_n       : array_like [Nx, Ny, Nz]
        dist_from_GRE   : 1-D array, size Nx*Ny*Nz

        Returns
        -------
        lattice_grid_points_is_in_GREvol : 1-D bool array, size Nx*Ny*Nz
        '''
        nx, ny, nz = int(lattice_n[0]), int(lattice_n[1]), int(lattice_n[2])

        # Boolean mask: is this voxel near the GRE surface?
        near = (dist_from_GRE.reshape(nx, ny, nz) < distance_factor * lattice_lf)
        ni   = near.astype(np.int32)

        # Along z (axis=2)
        cumfwd_z = np.cumsum(ni, axis=2)
        cumrev_z = np.cumsum(ni[:, :, ::-1], axis=2)[:, :, ::-1]
        inside_z = ((cumfwd_z - ni) >= 1) & ((cumrev_z - ni) >= 1)

        # Along y (axis=1)
        cumfwd_y = np.cumsum(ni, axis=1)
        cumrev_y = np.cumsum(ni[:, ::-1, :], axis=1)[:, ::-1, :]
        inside_y = ((cumfwd_y - ni) >= 1) & ((cumrev_y - ni) >= 1)

        # Along x (axis=0)
        cumfwd_x = np.cumsum(ni, axis=0)
        cumrev_x = np.cumsum(ni[::-1, :, :], axis=0)[::-1, :, :]
        inside_x = ((cumfwd_x - ni) >= 1) & ((cumrev_x - ni) >= 1)

        return (inside_x & inside_y & inside_z).ravel()


    def visualize_the_generated_GRE_shape_and_incindent_beam(
            self, xyz_meshes_GRE_surf, lattice_grid_points,
            lattice_grid_points_is_in_GREvol, euler_angles_deg):
        '''
        Visualise the GRE shape and DDA dipoles in the laboratory frame.

        Parameters
        ----------
        xyz_meshes_GRE_surf              : tuple (x, y, z) of 2-D meshes
        lattice_grid_points              : ndarray, shape (N, 3)
        lattice_grid_points_is_in_GREvol : 1-D bool array
        euler_angles_deg                 : array_like, (alpha, beta, gamma) [degrees]
        '''
        lattice_in = lattice_grid_points[lattice_grid_points_is_in_GREvol]
        num_in     = np.sum(lattice_grid_points_is_in_GREvol)

        euler_rad = np.radians(euler_angles_deg)
        rotmat    = R.from_euler('ZYZ', euler_rad).as_matrix()

        # Rotate surface meshes
        surf_x, surf_y, surf_z = xyz_meshes_GRE_surf
        surf_x_L = rotmat[0, 0]*surf_x + rotmat[0, 1]*surf_y + rotmat[0, 2]*surf_z
        surf_y_L = rotmat[1, 0]*surf_x + rotmat[1, 1]*surf_y + rotmat[1, 2]*surf_z
        surf_z_L = rotmat[2, 0]*surf_x + rotmat[2, 1]*surf_y + rotmat[2, 2]*surf_z

        lattice_in_L = (rotmat @ lattice_in.T).T

        fig = plt.figure(figsize=[8, 8])
        ax  = fig.add_subplot(111, projection='3d')

        plot_alpha = np.min([30 / (np.log(num_in) + 5*self.bc_ratio*self.ab_ratio), 0.6])
        plot_s     = np.min([10  / (np.log(num_in) +   self.bc_ratio*self.ab_ratio), 8])

        ax.plot_wireframe(surf_x_L, surf_y_L, surf_z_L,
                          color='black', linewidth=0.3, alpha=0.6)
        ax.scatter(lattice_in_L[:, 0], lattice_in_L[:, 1], lattice_in_L[:, 2],
                   marker='.', s=plot_s, color='r', alpha=plot_alpha)

        axis_range = np.max(np.abs(lattice_grid_points[:, 0]))
        ax.set_xlim([-axis_range, axis_range])
        ax.set_ylim([-axis_range, axis_range])
        ax.set_zlim([-axis_range, axis_range])

        ax.set_title("{} euler_angles=({:.0f},{:.0f},{:.0f})".format(
            self.name, *euler_angles_deg), fontsize=10)
        ax.set_xlabel("x [$\\mu$m]", fontsize=13)
        ax.set_ylabel("y [$\\mu$m]", fontsize=13)
        ax.set_zlabel("z [$\\mu$m]", fontsize=13)
        ax.tick_params(labelsize=11)
        plt.tight_layout()

        scale = axis_range
        start = np.zeros(3, dtype=np.float32)
        for vec, color, style in [
            ([0, 0, 1], 'black',  'solid'),
            ([1, 0, 0], 'red',    'dashed'),
            ([0, 1, 0], 'blue',   'dashed'),
        ]:
            v = np.array(vec, dtype=np.float32) * scale
            ax.arrow3D(*start, *v, mutation_scale=20,
                       arrowstyle="-|>" if style == 'dashed' else '->', linestyle=style,
                       ec=color, fc=color)

        plt.show()


    def compute_GRE_volume_and_ve_radius(self, lattice_grid_points_is_in_GREvol):
        num_in    = np.sum(lattice_grid_points_is_in_GREvol)
        volume    = self.lattice_lf ** 3 * num_in
        ve_radius = np.cbrt(3 * volume / (4 * np.pi))
        return volume, ve_radius
