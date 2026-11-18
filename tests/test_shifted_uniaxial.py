"""Tests of the common-offset sweep of an anisotropic particle (Phase 5).

``DiscreteDipoles.set_interaction_matrix_shifted_offset`` sweeps the principal
refractive indices of a homogeneous anisotropic particle so that 1/alpha_E of
all three axes changes by the same amount. The interaction matrix then changes
by a scalar multiple of the identity, and ``bl_gmres_shifted_mvp_fft`` solves
all sweep points with one block-Arnoldi process on the reference matrix
(argument ``diag_ref``).

The reference is the existing anisotropic path: one ``DiscreteDipoles`` object
per sweep point, with ``m_p_xyz`` set to the principal indices of that point,
solved by ``bl_gmres_mvp_fft`` on the same lattice.

T8   solutions and step counts equal those of the independent solves
T9   the recomputed 1/alpha_E of each sweep point equals diag_ref + sigma_s
T10  a zero offset reproduces the existing anisotropic path
T11  observables equal those of the existing compute methods
"""
import warnings

import numpy as np
import pytest

import shifted_cases as sc
from bl_dda.scatterer import (
    DiscreteDipoles,
    IncidentField,
    Target,
    common_offset_refractive_indices,
    cr2009_polarizability,
)
from bl_krylov.bl_krylov import bl_gmres_mvp_fft, bl_gmres_shifted_mvp_fft

RTOL_X = 1e-10

# Uniaxial reference (n_o, n_o, n_e), optic axis along z. It is the sweep point
# with the largest indices, below the index 1.70 the lattice is built for.
M_REF_XYZ = np.array([1.60, 1.60, 1.68], dtype=np.complex128)
# Ordinary index at each sweep point; the last one is absorbing.
N_O_LIST = np.array([1.40, 1.50, 1.60, 1.50 + 0.05j])

SHIFTED_TO_EXISTING = {
    "C_ext": "C_ext", "C_abs": "C_abs",
    "S_fw_PCAS_theta": "S_fw_theta", "S_fw_PCAS_phi": "S_fw_phi",
    "S_bk_OCBS_theta": "S_bk_theta", "S_bk_OCBS_phi": "S_bk_phi",
    "S_bk_OCBS": "S_bk_OCBS",
}


def _rel_diff(X, X_ref):
    return np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref)


def _dipoles(geometry, m_p_xyz):
    """``DiscreteDipoles`` of the spheroid case with principal indices ``m_p_xyz``."""
    name, lattice_n, lattice_lf, grid, is_in = geometry
    target = Target(name, lattice_n, lattice_lf, grid, is_in,
                    np.asarray(m_p_xyz, dtype=np.complex128), sc.SPHEROID_D_VE / 2.0)
    inc = IncidentField(sc.SPHEROID_WL_0, sc.SPHEROID_M_M, sc.spheroid_euler_angles())
    return sc.quiet(DiscreteDipoles, target, inc)


def _offset_sweep(geometry, m_ref_xyz, m_x_list, tol, itermax, **kwargs):
    dd = _dipoles(geometry, m_ref_xyz)
    dd.tol, dd.itermax = tol, itermax
    sc.quiet(dd.set_interaction_matrix_shifted_offset, m_x_list)
    sc.quiet(dd.solve_matrix_equation_shifted, **kwargs)
    return dd


def _independent(geometry, m_p_xyz, tol, itermax):
    """Existing anisotropic path for one sweep point: (dd, X, n_iter)."""
    dd = _dipoles(geometry, m_p_xyz)
    n_iter, _ = sc.solve_existing(dd, tol, itermax)
    return dd, dd.X, n_iter


@pytest.fixture(scope="module")
def geometry():
    return sc.spheroid_common_geometry()


@pytest.fixture(scope="module")
def sweep(geometry):
    """Common-offset sweep at tol = 1e-5 with the solutions kept."""
    return _offset_sweep(geometry, M_REF_XYZ, N_O_LIST, 1e-5, 60, return_X=True)


# ── sweep path ───────────────────────────────────────────────────────────────
def test_common_offset_path_keeps_tau_differences():
    eper_m = 1.3315 ** 2
    m_xyz = common_offset_refractive_indices(M_REF_XYZ, N_O_LIST, eper_m)   # shape: (N_m, 3)
    assert m_xyz.shape == (N_O_LIST.size, 3)
    assert np.array_equal(m_xyz[:, 0], N_O_LIST)
    # Uniaxial stays uniaxial: the two ordinary axes are equal.
    np.testing.assert_allclose(m_xyz[:, 1], m_xyz[:, 0], rtol=1e-14)
    tau = 1.0 / (m_xyz ** 2 / eper_m - 1.0)
    tau_ref = 1.0 / (M_REF_XYZ ** 2 / eper_m - 1.0)
    np.testing.assert_allclose(tau[:, 2] - tau[:, 0],
                               np.full(N_O_LIST.size, tau_ref[2] - tau_ref[0]), rtol=1e-12)
    # The reference point is reproduced, and n_e - n_o is not constant.
    np.testing.assert_allclose(m_xyz[2], M_REF_XYZ, rtol=1e-14)
    dn = (m_xyz[:3, 2] - m_xyz[:3, 0]).real
    assert dn[0] < dn[1] < dn[2]
    # An absorbing ordinary index gives an absorbing extraordinary index.
    assert m_xyz[3, 2].imag > 0


# ── T9 ───────────────────────────────────────────────────────────────────────
def test_t9_recomputed_diagonal_equals_reference_plus_offset(geometry, sweep):
    N = sweep.num_element_occupy
    assert sweep.diag_ref_shifted.shape == (3 * N,)
    assert sweep.m_p_xyz_shifted.shape == (N_O_LIST.size, 3)
    assert np.array_equal(sweep.m_p_shifted, N_O_LIST)
    for s, m_xyz in enumerate(sweep.m_p_xyz_shifted):
        diag = 1.0 / cr2009_polarizability(m_xyz, sweep.eper_m, sweep.element_vol, sweep.k)
        expected = sweep.diag_ref_shifted + sweep.sigmas[s]
        assert np.max(np.abs(np.tile(diag, N) - expected) / np.abs(expected)) < 1e-12
        # Same check against the diagonal the existing path builds.
        dd_s = _dipoles(geometry, m_xyz)
        sc.quiet(dd_s.set_interaction_matrix)
        assert np.max(np.abs(dd_s.diag_A - expected) / np.abs(expected)) < 1e-12
        # Lattice, Green's tensor and incident field do not depend on the indices.
        assert np.array_equal(dd_s.Au_til, sweep.Au_til)
        assert np.array_equal(dd_s.B, sweep.B)
    assert sweep.sigmas[2] == 0.0                       # the reference point


# ── T8 ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lsq_method", ("incremental", "dense"))
@pytest.mark.parametrize("tol, itermax", sc.SPHEROID_SOLVER_SETTINGS)
def test_t8_sweep_matches_independent_anisotropic_solves(geometry, tol, itermax, lsq_method):
    dd = _offset_sweep(geometry, M_REF_XYZ, N_O_LIST, tol, itermax,
                       return_X=True, lsq_method=lsq_method)
    res = dd.shifted_result
    assert dd.converge_shifted.all()
    n_iters = []
    for s, m_xyz in enumerate(dd.m_p_xyz_shifted):
        _, X_ref, n_iter = _independent(geometry, m_xyz, tol, itermax)
        assert res.iters[s] == n_iter
        assert _rel_diff(res.X_list[s], X_ref) < RTOL_X
        n_iters.append(n_iter)
    assert len(set(n_iters)) > 1                        # the sweep points differ


# ── T10 ──────────────────────────────────────────────────────────────────────
def test_t10_zero_offset_reproduces_existing_anisotropic_path(geometry):
    tol, itermax = 1e-5, 60
    dd = _offset_sweep(geometry, M_REF_XYZ, [M_REF_XYZ[0]], tol, itermax, return_X=True)
    assert dd.sigmas[0] == 0.0
    np.testing.assert_allclose(dd.m_p_xyz_shifted[0], M_REF_XYZ, rtol=1e-14)
    dd_ref, X_ref, n_iter = _independent(geometry, M_REF_XYZ, tol, itermax)
    assert dd.shifted_result.iters[0] == n_iter
    assert _rel_diff(dd.shifted_result.X_list[0], X_ref) < RTOL_X
    obs, obs_ref = dd.compute_observables_shifted(), sc.compute_observables(dd_ref)
    for name, name_ref in SHIFTED_TO_EXISTING.items():
        np.testing.assert_allclose(obs[name][0], obs_ref[name_ref], rtol=1e-10, atol=1e-18,
                                   err_msg=name)


# ── T11 ──────────────────────────────────────────────────────────────────────
def test_t11_observables_match_existing_compute_methods(geometry, sweep):
    obs = sweep.compute_observables_shifted()
    for s, m_xyz in enumerate(sweep.m_p_xyz_shifted):
        dd_s, _, _ = _independent(geometry, m_xyz, 1e-5, 60)
        obs_ref = sc.compute_observables(dd_s)
        for name, name_ref in SHIFTED_TO_EXISTING.items():
            np.testing.assert_allclose(obs[name][s], obs_ref[name_ref], rtol=1e-10, atol=1e-18,
                                       err_msg=f"{name}, sweep point {s}")
    # Only the absorbing sweep point has a nonzero absorption cross section.
    assert np.all(obs["C_abs"][:3] == 0.0)
    assert np.all(obs["C_abs"][3] > 0.0)
    # The anisotropy is visible: an isotropic particle with the ordinary index differs.
    dd_iso, _, _ = _independent(geometry, np.full(3, N_O_LIST[2]), 1e-5, 60)
    S_iso = sc.compute_observables(dd_iso)["S_fw_theta"]
    assert np.max(np.abs(obs["S_fw_PCAS_theta"][2] - S_iso) / np.abs(S_iso)) > 1e-3


def test_solutions_are_kept_only_when_c_abs_needs_them(geometry):
    # Absorbing anisotropic sweep: X is kept although return_X was not requested.
    dd = _offset_sweep(geometry, M_REF_XYZ, N_O_LIST, 1e-2, 25)
    assert dd.shifted_result.X_list is not None
    # Non-absorbing sweep: no solutions are assembled.
    dd = _offset_sweep(geometry, M_REF_XYZ, N_O_LIST[:3], 1e-2, 25)
    assert dd.shifted_result.X_list is None
    assert np.all(dd.compute_observables_shifted()["C_abs"] == 0.0)


# ── relation to the isotropic sweep ──────────────────────────────────────────
def test_isotropic_reference_gives_the_isotropic_shifted_sweep(geometry):
    """With equal principal indices the offset sweep is the sweep of Phase 2."""
    tol, itermax = 1e-5, 60
    m_list = np.array(sc.SPHEROID_M_LIST)
    m_max = max(sc.SPHEROID_M_LIST, key=abs)
    dd_off = _offset_sweep(geometry, np.full(3, m_max), m_list, tol, itermax, return_X=True)
    np.testing.assert_allclose(dd_off.m_p_xyz_shifted, np.repeat(m_list[:, None], 3, axis=1),
                               rtol=1e-14)

    dd_iso = sc.make_spheroid_dipoles(geometry, m_max)
    dd_iso.tol, dd_iso.itermax = tol, itermax
    sc.quiet(dd_iso.set_interaction_matrix_shifted, m_list)
    sc.quiet(dd_iso.solve_matrix_equation_shifted, return_X=True)
    assert dd_iso.diag_ref_shifted is None and dd_iso.m_p_xyz_shifted is None

    assert np.array_equal(dd_off.shifted_result.iters, dd_iso.shifted_result.iters)
    for X_off, X_iso in zip(dd_off.shifted_result.X_list, dd_iso.shifted_result.X_list):
        assert _rel_diff(X_off, X_iso) < RTOL_X
    obs_off, obs_iso = dd_off.compute_observables_shifted(), dd_iso.compute_observables_shifted()
    for name in obs_iso:
        np.testing.assert_allclose(obs_off[name], obs_iso[name], rtol=1e-10, atol=1e-18,
                                   err_msg=name)


def test_solver_reference_diagonal_is_equivalent_to_a_constant_shift(geometry):
    """A constant diag_ref c with shifts sigma - c solves the same systems as sigma alone."""
    dd = sc.make_spheroid_dipoles(geometry, max(sc.SPHEROID_M_LIST, key=abs))
    sc.quiet(dd.set_interaction_matrix_shifted, sc.SPHEROID_M_LIST)
    args = (dd.lattice_n, dd.f, dd.lattice_address_in_target, dd.Au_til)
    res = bl_gmres_shifted_mvp_fft(*args, dd.sigmas, dd.B, 1e-5, 60, return_X=True)
    c = dd.sigmas[1]
    res_ref = bl_gmres_shifted_mvp_fft(*args, dd.sigmas - c, dd.B, 1e-5, 60, return_X=True,
                                       diag_ref=np.full(dd.B.shape[0], c))
    assert np.array_equal(res.iters, res_ref.iters)
    for X, X_ref in zip(res.X_list, res_ref.X_list):
        assert _rel_diff(X_ref, X) < RTOL_X
    # The single-system solver with the same full diagonal agrees as well.
    X_single, iter_fin, _, _ = bl_gmres_mvp_fft(
        *args, np.full(dd.B.shape[0], dd.sigmas[0]), dd.B, 1e-5, 60)
    assert res_ref.iters[0] == iter_fin + 1
    assert _rel_diff(res_ref.X_list[0], X_single) < RTOL_X

    with pytest.raises(ValueError):
        bl_gmres_shifted_mvp_fft(*args, dd.sigmas, dd.B, 1e-5, 60, diag_ref=np.ones(5))


# ── input checks ─────────────────────────────────────────────────────────────
def test_invalid_or_unphysical_sweeps_raise(geometry):
    dd = _dipoles(geometry, M_REF_XYZ)
    for bad in ([], [[1.5, 1.5], [1.6, 1.6]], ["a", "b"]):
        with pytest.raises(ValueError):
            sc.quiet(dd.set_interaction_matrix_shifted_offset, bad)
    # tau of the extraordinary axis changes sign: no physical refractive index.
    dd = _dipoles(geometry, np.array([1.5, 1.5, 2.5]))
    with pytest.raises(ValueError):
        sc.quiet(dd.set_interaction_matrix_shifted_offset, [3.0])


def test_sweep_above_the_reference_index_warns(geometry):
    dd = _dipoles(geometry, np.array([1.50, 1.50, 1.55]))
    with pytest.warns(UserWarning, match="effective dpl"):
        sc.quiet(dd.set_interaction_matrix_shifted_offset, [1.50, 1.60])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sc.quiet(dd.set_interaction_matrix_shifted_offset, [1.40, 1.50])
