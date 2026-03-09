import numpy as np
from mvp_fft.mvp_fft import fft_init
from bl_krylov.bl_krylov import bl_bicgstab_jacobi_mvp_fft
from scipy.spatial.transform import Rotation as R
import time


class Target:

    def __init__(self, shape_name, lattice_n, lattice_lf, lattice_grid_points, lattice_grid_points_is_in_target, m_p_xyz):
        self.shape_name : str = shape_name
        self.lattice_n : np.ndarray[int] = lattice_n
        self.lattice_lf : np.float64 = lattice_lf
        self.lattice_grid_points = lattice_grid_points
        self.lattice_grid_points_is_in_target = lattice_grid_points_is_in_target
        self.m_p_xyz = m_p_xyz

        self.lattice_address_in_target = np.where(lattice_grid_points_is_in_target)[0]
        self.lattice_pos_in_target = lattice_grid_points[self.lattice_address_in_target, :]
        self.num_element_occupy = self.lattice_address_in_target.size
        self.element_vol : np.float64 = self.lattice_lf ** 3

        # Homogeneous but optionally anisotropic refractive index
        self.m_p = np.ones_like(self.lattice_pos_in_target) * m_p_xyz
        self.eper_p = self.m_p ** 2

        self.total_vol = self.element_vol * self.num_element_occupy
        self.ve_radius = np.cbrt(3 * self.total_vol / (4 * np.pi))


class IncidentField:

    def __init__(self, wl_0, m_m, euler_angles):
        '''
        Parameters
        ----------
        wl_0         : float  vacuum wavelength [um]
        m_m          : float  medium refractive index
        euler_angles : ndarray, shape (L, 3)
            ZYZ Euler angles (alpha, beta, gamma) [radian] rotating the
            particle coordinate system from the laboratory frame.
        '''
        self.wl_0 : np.float64 = wl_0
        self.m_m  : np.float64 = m_m
        self.eper_m : np.float64 = m_m ** 2
        self.k : np.float64 = 2 * np.pi * m_m / wl_0
        self.euler_angles : np.ndarray = euler_angles
        self.L : int = euler_angles.shape[0]

        # Incident / scattering geometry fixed in the laboratory frame
        u_inc_L         = np.array([0,  0,  1], dtype=np.float64)  # +z
        theta_inc_L     = np.array([1,  0,  0], dtype=np.float64)  # +x
        phi_inc_L       = np.array([0,  1,  0], dtype=np.float64)  # +y

        u_sca_fw_L      = np.array([0,  0,  1], dtype=np.float64)
        theta_sca_fw_L  = np.array([1,  0,  0], dtype=np.float64)
        phi_sca_fw_L    = np.array([0,  1,  0], dtype=np.float64)

        u_sca_bk_L      = np.array([0,  0, -1], dtype=np.float64)
        theta_sca_bk_L  = np.array([-1, 0,  0], dtype=np.float64)
        phi_sca_bk_L    = np.array([0,  1,  0], dtype=np.float64)

        # Incident polarisation (circular, sqrt(2) normalised)
        self.e0_inc_pol_theta : np.complex128 = 1   / np.sqrt(2)
        self.e0_inc_pol_phi   : np.complex128 = 1j  / np.sqrt(2)

        # Batch rotation: lab -> particle frame is the inverse of the particle
        # orientation, i.e. ZYZ(-gamma, -beta, -alpha) for each orientation l.
        angles_ZYZ = euler_angles[:, ::-1] * -1.0          # shape (L, 3)
        rotmats = R.from_euler('ZYZ', angles_ZYZ).as_matrix()  # (L, 3, 3)

        # (L,3,3) @ (3,) -> (L,3)  for every direction vector
        self.u_inc_vec_P        = rotmats @ u_inc_L
        self.theta_inc_vec_P    = rotmats @ theta_inc_L
        self.phi_inc_vec_P      = rotmats @ phi_inc_L
        self.e0_inc_vec_P       = (self.e0_inc_pol_theta * self.theta_inc_vec_P
                                   + self.e0_inc_pol_phi  * self.phi_inc_vec_P)

        self.u_sca_fw_vec_P     = rotmats @ u_sca_fw_L
        self.theta_sca_fw_vec_P = rotmats @ theta_sca_fw_L
        self.phi_sca_fw_vec_P   = rotmats @ phi_sca_fw_L

        self.u_sca_bk_vec_P     = rotmats @ u_sca_bk_L
        self.theta_sca_bk_vec_P = rotmats @ theta_sca_bk_L
        self.phi_sca_bk_vec_P   = rotmats @ phi_sca_bk_L


class DiscreteDipoles(Target, IncidentField):

    def __init__(self, target, incidentfield):
        Target.__init__(self, target.shape_name, target.lattice_n, target.lattice_lf,
                        target.lattice_grid_points, target.lattice_grid_points_is_in_target,
                        target.m_p_xyz)
        IncidentField.__init__(self, incidentfield.wl_0, incidentfield.m_m,
                               incidentfield.euler_angles)

        N = target.num_element_occupy
        self.dpl : np.float64 = (self.wl_0 / np.abs(np.max(self.m_p_xyz))) / self.lattice_lf
        self.f : int = 3

        # Physical fields and matrix quantities (filled by set_interaction_matrix)
        self.eper_r      = np.zeros((N, 3),       dtype=np.complex128)
        self.alpha_E     = np.zeros((N, 3),       dtype=np.complex128)
        self.diag_A      = np.zeros(3 * N,        dtype=np.complex128)
        self.e_inc_phase = np.zeros((self.L, N),  dtype=np.complex128)
        self.E_inc       = np.zeros((self.L, N, 3), dtype=np.complex128)
        self.B           = np.zeros((3 * N, self.L), dtype=np.complex128)

        # Solution (filled by solve_matrix_equation)
        self.X = np.zeros((3 * N, self.L),    dtype=np.complex128)
        self.P = np.zeros((self.L, N, 3),     dtype=np.complex128)
        self.E = np.zeros((self.L, N, 3),     dtype=np.complex128)

        # Solver settings
        self.itermax = 25
        self.tol     = 1e-2
        self.converge = False

        # Observable outputs
        self.C_abs           = None
        self.C_ext           = None
        self.S_fw_PCAS_theta = None
        self.S_fw_PCAS_phi   = None
        self.S_bk_OCBS_theta = None
        self.S_bk_OCBS_phi   = None
        self.S_bk_OCBS       = None
        self.S_PCAS_sp_avg      = None
        self.S_PCAS_sp_avg_SNR  = None
        self.S_PCAS_depol       = None
        self.S_PCAS_depol_SNR   = None

        print("Number of dipoles per wavelength in the particle volume: dpl= {:}".format(self.dpl))


    def set_interaction_matrix(self):

        self.eper_r = self.eper_p / self.eper_m

        # Clausius-Mossotti static polarizability, shape (N, 3)
        alpha0_E = (3 / (4 * np.pi)) * ((self.eper_r - 1) / (self.eper_r + 2)) * self.element_vol

        # CR2009 polarizability (Chaumet & Rahmani 2009, JQSRT)
        a = (3 * self.element_vol / (4 * np.pi)) ** (1 / 3)
        M_term = (8 * np.pi / 3) * ((1 - 1j * self.k * a) * np.exp(1j * self.k * a) - 1)
        self.alpha_E = alpha0_E / (1 - M_term * alpha0_E / self.element_vol)

        # Diagonal of A: 1/alpha_E, shape (3N,), order [x0,y0,z0, x1,y1,z1, ...]
        self.diag_A = (1.0 / self.alpha_E).ravel()

        # Pre-compute FFT of the interaction tensor
        self.Au_til = fft_init(self.lattice_n, self.f, self.lattice_lf, self.k)

        # Incident-field phase on each dipole: shape (L, N)
        # (L,3) @ (3,N) = (L,N),  element [l,n] = u_inc[l] · r[n]
        self.e_inc_phase = np.exp(1j * self.k * (self.u_inc_vec_P @ self.lattice_pos_in_target.T))

        # Incident field: shape (L, N, 3)
        self.E_inc = self.e_inc_phase[:, :, np.newaxis] * self.e0_inc_vec_P[:, np.newaxis, :]

        # RHS block: shape (3N, L)
        self.B = self.E_inc.reshape(self.L, 3 * self.num_element_occupy).T


    def solve_matrix_equation(self):

        start_time = time.time()
        print("Starting block-BiCGStab iterative solver...")

        self.X, iter_fin, err_fin = bl_bicgstab_jacobi_mvp_fft(
            self.lattice_n, self.f, self.lattice_address_in_target,
            self.Au_til, self.diag_A, self.B, self.tol, self.itermax)

        elapsed_time = time.time() - start_time

        if err_fin < self.tol:
            self.converge = True
            print("block-BiCGStab converged! "
                  "(iter_fin={:}, err_fin={:.4f}, solver time={:.1f}s)".format(
                      iter_fin, err_fin, elapsed_time))

            # X has shape (3N, L); reshape to (L, N, 3)
            self.P = self.X.T.reshape(self.L, self.num_element_occupy, 3)
            self.E = (self.P * (4 * np.pi)
                      / ((self.eper_r[np.newaxis, :, :] - 1) * self.element_vol))


    def compute_C_abs(self):
        self.C_abs = np.zeros(self.L, dtype=np.float64)
        if np.mean(self.eper_r.imag) >= 1e-12:
            # sum P·conj(E) over dipoles (axis=1) and components (axis=2)
            self.C_abs = 4 * np.pi * self.k * np.sum(
                np.imag(self.P * np.conj(self.E)), axis=(1, 2))
        return self.C_abs


    def compute_C_ext(self):
        self.C_ext = (4 * np.pi * self.k
                      * np.sum(np.imag(self.P * np.conj(self.E_inc)), axis=(1, 2)))
        return self.C_ext


    def compute_PCAS_observable_S_fw(self):
        # Phase factor on each dipole for forward direction: shape (L, N)
        e_fw = np.exp(1j * self.k * (self.u_sca_fw_vec_P @ self.lattice_pos_in_target.T))

        # Project dipole polarization onto scattering basis vectors: shape (L, N)
        P_proj_theta = np.einsum('lni,li->ln', self.P, self.theta_sca_fw_vec_P)
        P_proj_phi   = np.einsum('lni,li->ln', self.P, self.phi_sca_fw_vec_P)

        sqrt2 = np.sqrt(2)
        self.S_fw_PCAS_theta = self.k ** 2 * (P_proj_theta * np.conj(sqrt2      * e_fw)).sum(axis=1)
        self.S_fw_PCAS_phi   = self.k ** 2 * (P_proj_phi   * np.conj(sqrt2 * 1j * e_fw)).sum(axis=1)
        return self.S_fw_PCAS_theta, self.S_fw_PCAS_phi


    def compute_OCBS_observable_S_bk(self):
        # Phase factor on each dipole for backward direction: shape (L, N)
        e_bk = np.exp(1j * self.k * (self.u_sca_bk_vec_P @ self.lattice_pos_in_target.T))

        # Project dipole polarization onto scattering basis vectors: shape (L, N)
        P_proj_theta = np.einsum('lni,li->ln', self.P, self.theta_sca_bk_vec_P)
        P_proj_phi   = np.einsum('lni,li->ln', self.P, self.phi_sca_bk_vec_P)

        sqrt2 = np.sqrt(2)
        self.S_bk_OCBS_theta = self.k ** 2 * (P_proj_theta * np.conj(sqrt2      * e_bk)).sum(axis=1)
        self.S_bk_OCBS_phi   = self.k ** 2 * (P_proj_phi   * np.conj(sqrt2 * 1j * e_bk)).sum(axis=1)
        self.S_bk_OCBS = (-self.S_bk_OCBS_theta + self.S_bk_OCBS_phi) / sqrt2
        return self.S_bk_OCBS
