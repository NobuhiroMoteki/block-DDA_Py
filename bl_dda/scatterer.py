import numpy as np
from mvp_fft.mvp_fft import fft_init
from bl_krylov.bl_krylov import bl_gmres_mvp_fft, bl_gmres_shifted_mvp_fft
from scipy.spatial.transform import Rotation as R
import time
import warnings


def cr2009_polarizability(m_p, eper_m, element_vol, k):
    """
    Dipole polarizability of one lattice cell: Clausius-Mossotti with the
    CR2009 radiative correction (Chaumet & Rahmani 2009, JQSRT).

    Same formula, in the same order of operations, as
    ``DiscreteDipoles.set_interaction_matrix``. It is a module function so that
    the polarizability can be evaluated for a list of refractive indices on a
    fixed lattice (shifted block-GMRES refractive-index sweep).

    Parameters
    ----------
    m_p         : complex or ndarray   particle refractive index (any shape)
    eper_m      : float   medium relative permittivity, m_m**2
    element_vol : float   lattice-cell volume [um^3]
    k           : float   wavenumber in the medium [um^-1]

    Returns
    -------
    alpha_E : complex128 ndarray, same shape as ``m_p``   polarizability [um^3]
              (Gaussian units). ``1/alpha_E`` is the diagonal of the interaction
              matrix.
    """
    eper_r = np.asarray(m_p, dtype=np.complex128) ** 2 / eper_m

    # Clausius-Mossotti static polarizability
    alpha0_E = (3 / (4 * np.pi)) * ((eper_r - 1) / (eper_r + 2)) * element_vol

    # CR2009 radiative correction
    a = (3 * element_vol / (4 * np.pi)) ** (1 / 3)
    M_term = (8 * np.pi / 3) * ((1 - 1j * k * a) * np.exp(1j * k * a) - 1)
    return alpha0_E / (1 - M_term * alpha0_E / element_vol)


class Target:

    def __init__(self, shape_name, lattice_n, lattice_lf, lattice_grid_points, lattice_grid_points_is_in_target, m_p_xyz, r_v_base):
        self.shape_name : str = shape_name
        self.lattice_n : np.ndarray[int] = lattice_n
        self.lattice_grid_points_is_in_target = lattice_grid_points_is_in_target
        self.m_p_xyz = m_p_xyz
        self._r_v_base = r_v_base

        self.lattice_address_in_target = np.where(lattice_grid_points_is_in_target)[0]
        self.num_element_occupy = self.lattice_address_in_target.size

        # Rescale lattice spacing so that N_occ * d_adj³ = V_target exactly.
        # This keeps Green's function and polarizability mutually consistent.
        V_target = (4.0 / 3.0) * np.pi * r_v_base ** 3
        d_adj = np.cbrt(V_target / self.num_element_occupy)
        scale = d_adj / lattice_lf
        self.lattice_lf = d_adj
        self.lattice_grid_points = lattice_grid_points * scale

        self.lattice_pos_in_target = self.lattice_grid_points[self.lattice_address_in_target, :]
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
                        target.m_p_xyz, target._r_v_base)
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
        print("Starting block-GMRES iterative solver...")

        self.X, iter_fin, err_fin, self.err_history = bl_gmres_mvp_fft(
            self.lattice_n, self.f, self.lattice_address_in_target,
            self.Au_til, self.diag_A, self.B, self.tol, self.itermax)

        elapsed_time = time.time() - start_time

        if err_fin < self.tol:
            self.converge = True
            print("block-GMRES converged! "
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
        """
        Backward scattering amplitude of the OCBS signal, S_bak of CAS-v2.

        With the amplitude-matrix elements S_ij at the backward direction,

            S_bk_OCBS_theta = S_11 + i S_12,
            S_bk_OCBS_phi   = S_22 - i S_21,
            S_bk_OCBS       = (-S_bk_OCBS_theta + S_bk_OCBS_phi) / 2
                            = (-S_11 + S_22) / 2,

        where the last step uses the backscattering theorem S_21 = -S_12. This
        is the corrected Eq. (28) of Moteki and Adachi, Opt. Express 32, 36500
        (2024), given in its erratum. The quarter-wave plate in the return path
        has its axes at 45 deg to the linear polarizations, so each backscattered
        component reaches the output polarization with the amplitude factor
        1/sqrt(2). The printed Eq. (28) has 1/sqrt(2) in place of 1/2. Versions
        up to 0.8.1 of this code followed the printed equation, and their
        S_bk_OCBS is sqrt(2) times the present one.

        Returns
        -------
        S_bk_OCBS : ndarray, shape (L,), complex128   backward amplitude [um]
        """
        # Phase factor on each dipole for backward direction: shape (L, N)
        e_bk = np.exp(1j * self.k * (self.u_sca_bk_vec_P @ self.lattice_pos_in_target.T))

        # Project dipole polarization onto scattering basis vectors: shape (L, N)
        P_proj_theta = np.einsum('lni,li->ln', self.P, self.theta_sca_bk_vec_P)
        P_proj_phi   = np.einsum('lni,li->ln', self.P, self.phi_sca_bk_vec_P)

        sqrt2 = np.sqrt(2)
        self.S_bk_OCBS_theta = self.k ** 2 * (P_proj_theta * np.conj(sqrt2      * e_bk)).sum(axis=1)
        self.S_bk_OCBS_phi   = self.k ** 2 * (P_proj_phi   * np.conj(sqrt2 * 1j * e_bk)).sum(axis=1)
        self.S_bk_OCBS = (-self.S_bk_OCBS_theta + self.S_bk_OCBS_phi) / 2
        return self.S_bk_OCBS


    def compute_phi_averaged_PCAS_S_fw(self):
        """
        Compute the phi-averaged forward scattering amplitudes for spheroids.

        Requires that compute_PCAS_observable_S_fw() has been called first,
        and that the orientations were computed with alpha=0, gamma=0.

        For a spheroid (a=b, beta_GRE=0), the analytical phi-average is:
            <S_s>_phi = <S_p>_phi = (S_s(alpha=0) + S_p(alpha=0)) / 2

        Returns
        -------
        S_fw_phi_avg : ndarray, shape (L,), complex128
            Phi-averaged forward scattering amplitude (same for s and p).
        """
        self.S_PCAS_sp_avg = (self.S_fw_PCAS_theta + self.S_fw_PCAS_phi) / 2
        return self.S_PCAS_sp_avg


    # ── refractive-index sweep on a fixed lattice (shifted block GMRES) ──────

    def set_interaction_matrix_shifted(self, m_p_list):
        """
        Set up the linear systems of a refractive-index sweep on this lattice.

        For a homogeneous isotropic particle the interaction matrix is
        A(m) = (1/alpha_E(m)) I + A_offdiag, where A_offdiag and the right-hand
        side B do not depend on m. All refractive indices in ``m_p_list`` are
        therefore solved on the lattice of this object, which the caller must
        build for the largest |m| of the list. ``Au_til``, ``E_inc`` and ``B``
        are built once by ``set_interaction_matrix``.

        Parameters
        ----------
        m_p_list : array_like, shape (N_m,)   complex particle refractive indices

        Sets
        ----
        m_p_shifted : ndarray (N_m,) complex128   the refractive indices
        sigmas      : ndarray (N_m,) complex128   shifts 1/alpha_E(m_s) [um^-3]
        dpl_shifted : ndarray (N_m,) float64      effective dipoles per wavelength
                      inside the particle for each m_s on this lattice

        Raises
        ------
        ValueError
            If ``m_p_list`` is not a non-empty 1-D sequence of scalars, or if the
            ``m_p_xyz`` of this object is anisotropic. The diagonal of an
            anisotropic particle is not a scalar multiple of the identity.
        """
        m_p_list = np.asarray(m_p_list)
        if (m_p_list.ndim != 1 or m_p_list.size == 0
                or not np.issubdtype(m_p_list.dtype, np.number)):
            raise ValueError("m_p_list must be a non-empty 1-D sequence of scalar "
                             "(isotropic) refractive indices")
        m_p_xyz = np.asarray(self.m_p_xyz)
        if not np.all(m_p_xyz == m_p_xyz.flat[0]):
            raise ValueError("the shifted refractive-index sweep requires an isotropic "
                             "particle (equal m_p_xyz components)")

        # Au_til, e_inc_phase, E_inc and B. They do not depend on m.
        self.set_interaction_matrix()

        self.m_p_shifted = m_p_list.astype(np.complex128)             # shape: (N_m,)
        alpha_E = cr2009_polarizability(self.m_p_shifted, self.eper_m,
                                        self.element_vol, self.k)
        self.sigmas = 1.0 / alpha_E                                   # shape: (N_m,)
        self.dpl_shifted = (self.wl_0 / np.abs(self.m_p_shifted)) / self.lattice_lf

        print("Shifted sweep: N_m= {:}, effective dpl from {:.2f} to {:.2f}".format(
            self.m_p_shifted.size, self.dpl_shifted.min(), self.dpl_shifted.max()))
        # 1e-6: self.dpl is single precision when m_p_xyz is complex64 (test_dda.ipynb).
        if self.dpl_shifted.min() < self.dpl * (1 - 1e-6):
            warnings.warn("m_p_list contains |m| larger than the refractive index this "
                          "lattice was built for; the effective dpl ({:.2f}) is below the "
                          "lattice dpl ({:.2f})".format(self.dpl_shifted.min(), self.dpl))


    def _shifted_projection_operators(self):
        """
        Functionals F with  S[l] = F[:, l]^H X[:, l]  for the scattering amplitudes.

        Order: S_fw_PCAS_theta, S_fw_PCAS_phi, S_bk_OCBS_theta, S_bk_OCBS_phi.
        They reproduce ``compute_PCAS_observable_S_fw`` and
        ``compute_OCBS_observable_S_bk``. Those sum
        ``k^2 (P . e_sca) conj(c e_phase)`` over dipoles, with c = sqrt(2) for
        theta and sqrt(2) i for phi. Hence F = k^2 c e_phase e_sca (e_sca real).

        Returns
        -------
        F : ndarray, shape (4, 3N, L), complex128, row order as in ``B``
        """
        N = self.num_element_occupy
        sqrt2 = np.sqrt(2)
        e_fw = np.exp(1j * self.k * (self.u_sca_fw_vec_P @ self.lattice_pos_in_target.T))  # (L, N)
        e_bk = np.exp(1j * self.k * (self.u_sca_bk_vec_P @ self.lattice_pos_in_target.T))  # (L, N)

        def functional(coef, phase, e_sca):
            # (L, N, 3) -> (3N, L), the layout of B
            F_lni = self.k ** 2 * coef * phase[:, :, np.newaxis] * e_sca[:, np.newaxis, :]
            return F_lni.reshape(self.L, 3 * N).T

        return np.stack([
            functional(sqrt2,      e_fw, self.theta_sca_fw_vec_P),
            functional(sqrt2 * 1j, e_fw, self.phi_sca_fw_vec_P),
            functional(sqrt2,      e_bk, self.theta_sca_bk_vec_P),
            functional(sqrt2 * 1j, e_bk, self.phi_sca_bk_vec_P),
        ])


    def solve_matrix_equation_shifted(self, return_X=False, lsq_method="incremental"):
        """
        Solve all refractive indices of ``set_interaction_matrix_shifted`` with
        one shifted block-GMRES process (``bl_gmres_shifted_mvp_fft``).

        Uses ``self.tol`` and ``self.itermax`` like ``solve_matrix_equation``.
        The stopping test is applied to each refractive index separately.

        Parameters
        ----------
        return_X : bool   also keep the full solutions (``shifted_result.X_list``,
                          each of shape (3N, L)). Not needed for the observables.
        lsq_method : str  "incremental" (default) or "dense"; how the per-shift
                          least-squares problems are handled, see
                          ``bl_gmres_shifted_mvp_fft``. Both give the same result
                          up to rounding.

        Sets
        ----
        shifted_result   : ShiftedBlockGMRESResult
        converge_shifted : ndarray (N_m,) bool   True where the residual is below tol
        """
        start_time = time.time()
        print("Starting shifted block-GMRES iterative solver...")

        F = self._shifted_projection_operators()
        self.shifted_result = bl_gmres_shifted_mvp_fft(
            self.lattice_n, self.f, self.lattice_address_in_target,
            self.Au_til, self.sigmas, self.B, self.tol, self.itermax,
            proj_ops=F, return_X=return_X, lsq_method=lsq_method)
        del F

        res = self.shifted_result
        self.converge_shifted = res.errs < self.tol                   # shape: (N_m,)

        elapsed_time = time.time() - start_time
        print("shifted block-GMRES: {:}/{:} refractive indices converged "
              "(iterations {:} to {:}, max err={:.4f}, solver time={:.1f}s)".format(
                  int(self.converge_shifted.sum()), self.converge_shifted.size,
                  int(res.iters.min()), int(res.iters.max()),
                  float(res.errs.max()), elapsed_time))


    def compute_observables_shifted(self):
        """
        Observables of every refractive index of the shifted sweep, evaluated
        from the small coefficient blocks Y_s without assembling the solutions.

        With X_s = [V_1 ... V_k] Y_s, B = V_1 Lambda and orthonormal V:

            C_ext[s, l] = 4 pi k Im( (Lambda^H Y_s[:L])[l, l] )
            C_abs[s, l] = 4 pi k Im( conj(c_E(m_s)) ) ||Y_s[:, l]||^2,
                          c_E = 4 pi / ((eper_r - 1) element_vol)
            S[s, l]     = (G Y_s)[l, l],   G = F^H [V_1 ... V_k]

        ``C_abs`` is set to zero for Im(eper_r) < 1e-12, as in ``compute_C_abs``.
        Refractive indices that did not converge are returned as NaN.

        Returns
        -------
        dict of ndarray, each of shape (N_m, L):
            C_ext, C_abs                          float64 [um^2]
            S_fw_PCAS_theta, S_fw_PCAS_phi        complex128
            S_bk_OCBS_theta, S_bk_OCBS_phi        complex128
            S_bk_OCBS                             complex128
        """
        res = self.shifted_result
        N_m, L = self.sigmas.size, self.L

        C_ext = np.full((N_m, L), np.nan, dtype=np.float64)
        C_abs = np.full((N_m, L), np.nan, dtype=np.float64)
        S = np.full((4, N_m, L), np.nan + 1j * np.nan, dtype=np.complex128)

        eper_r = self.m_p_shifted ** 2 / self.eper_m                  # shape: (N_m,)
        for s in np.flatnonzero(self.converge_shifted):
            Y = res.Y_list[s]                                         # shape: (k_s*L, L)
            kL = Y.shape[0]
            C_ext[s] = 4 * np.pi * self.k * np.imag(
                np.einsum('jl,jl->l', res.Lambda.conj(), Y[:L, :]))
            if eper_r[s].imag >= 1e-12:
                c_E = 4 * np.pi / ((eper_r[s] - 1) * self.element_vol)
                C_abs[s] = (4 * np.pi * self.k * np.imag(np.conj(c_E))
                            * np.sum(np.abs(Y) ** 2, axis=0))
            else:
                C_abs[s] = 0.0
            # diagonal of G[o][:, :kL] @ Y for the four functionals
            S[:, s, :] = np.einsum('olj,jl->ol', res.G[:, :, :kL], Y)

        return {
            "C_ext": C_ext,
            "C_abs": C_abs,
            "S_fw_PCAS_theta": S[0],
            "S_fw_PCAS_phi": S[1],
            "S_bk_OCBS_theta": S[2],
            "S_bk_OCBS_phi": S[3],
            "S_bk_OCBS": (-S[2] + S[3]) / 2,
        }
