"""Tests of the shifted refractive-index sweep API of ``DiscreteDipoles`` (Phase 2).

T5  observables projected from Y_s equal those of the existing ``compute_*``
    methods applied to the assembled X_s, and those of independent solves
T6  the sphere of test_dda.ipynb agrees with Mie theory as the existing solver does
T7  anisotropic or non-scalar input raises ``ValueError``
plus checks of ``cr2009_polarizability`` and of the optical theorem.
"""
import warnings

import numpy as np
import pytest

import shifted_cases as sc
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import DiscreteDipoles, IncidentField, Target, cr2009_polarizability

RTOL_OBS = 1e-10

# shifted-observable name -> name used by sc.compute_observables
NAME_MAP = {
    "C_ext": "C_ext",
    "C_abs": "C_abs",
    "S_fw_PCAS_theta": "S_fw_theta",
    "S_fw_PCAS_phi": "S_fw_phi",
    "S_bk_OCBS_theta": "S_bk_theta",
    "S_bk_OCBS_phi": "S_bk_phi",
    "S_bk_OCBS": "S_bk_OCBS",
}


def _max_rel(a, b):
    """Largest element-wise relative difference of two arrays of shape (L,)."""
    return np.max(np.abs(a - b) / np.abs(b))


def _shifted_sweep(geometry, m_lattice, m_list, tol, itermax, return_X=False):
    """Run the shifted sweep on ``geometry``; returns (dd, observables dict)."""
    dd = sc.make_spheroid_dipoles(geometry, m_lattice)
    dd.tol, dd.itermax = tol, itermax
    sc.quiet(dd.set_interaction_matrix_shifted, m_list)
    sc.quiet(dd.solve_matrix_equation_shifted, return_X=return_X)
    return dd, dd.compute_observables_shifted()


@pytest.fixture(scope="module")
def geometry():
    return sc.spheroid_common_geometry()


@pytest.fixture(scope="module")
def sweep(geometry):
    """Shifted sweep of the spheroid case at tol = 1e-5, with the X_s kept."""
    m_max = max(sc.SPHEROID_M_LIST, key=abs)
    return _shifted_sweep(geometry, m_max, sc.SPHEROID_M_LIST, 1e-5, 60, return_X=True)


# ── polarizability ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("m_m", [1.0, 1.3315])
def test_polarizability_matches_set_interaction_matrix(geometry, m_m):
    """cr2009_polarizability reproduces alpha_E of the existing method."""
    name, lattice_n, lattice_lf, grid, is_in = geometry
    inc = IncidentField(sc.SPHEROID_WL_0, m_m, sc.spheroid_euler_angles())
    for m_p_xyz in (np.array([1.5, 1.5, 1.5]),
                    np.array([1.5 + 0.1j] * 3),
                    np.array([1.48, 1.48, 1.56 + 0.02j])):        # anisotropic, element-wise
        target = Target(name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, sc.SPHEROID_D_VE / 2)
        dd = sc.quiet(DiscreteDipoles, target, inc)
        sc.quiet(dd.set_interaction_matrix)
        alpha = cr2009_polarizability(m_p_xyz, dd.eper_m, dd.element_vol, dd.k)   # shape: (3,)
        assert alpha.dtype == np.complex128
        np.testing.assert_allclose(alpha, dd.alpha_E[0], rtol=1e-14, atol=0.0)
        np.testing.assert_allclose(1.0 / alpha, dd.diag_A[:3], rtol=1e-14, atol=0.0)
        # scalar input
        alpha0 = cr2009_polarizability(m_p_xyz[0], dd.eper_m, dd.element_vol, dd.k)
        assert alpha0 == pytest.approx(dd.alpha_E[0, 0], rel=1e-14)


def test_sigmas_equal_existing_diagonal(geometry, sweep):
    dd_shift, _ = sweep
    for s, m_p in enumerate(sc.SPHEROID_M_LIST):
        dd = sc.make_spheroid_dipoles(geometry, m_p)
        sc.quiet(dd.set_interaction_matrix)
        assert dd_shift.sigmas[s] == pytest.approx(dd.diag_A[0], rel=1e-14)
        assert np.array_equal(dd_shift.B, dd.B)
        assert np.array_equal(dd_shift.Au_til, dd.Au_til)
    # Effective dpl: 17 or more for every m, largest for the smallest |m|.
    assert dd_shift.dpl_shifted.shape == (len(sc.SPHEROID_M_LIST),)
    assert np.all(dd_shift.dpl_shifted >= dd_shift.dpl * (1 - 1e-12))


# ── T5 ───────────────────────────────────────────────────────────────────────
def test_t5_projected_observables_match_existing_compute_methods(geometry, sweep):
    """Projection from Y_s vs existing compute_* applied to the assembled X_s."""
    dd_shift, obs = sweep
    assert np.all(dd_shift.converge_shifted)
    for s, m_p in enumerate(sc.SPHEROID_M_LIST):
        dd = sc.make_spheroid_dipoles(geometry, m_p)
        sc.quiet(dd.set_interaction_matrix)
        # Same two lines as solve_matrix_equation, fed with the shifted solution.
        X = dd_shift.shifted_result.X_list[s]
        dd.P = X.T.reshape(dd.L, dd.num_element_occupy, 3)
        dd.E = dd.P * (4 * np.pi) / ((dd.eper_r[np.newaxis, :, :] - 1) * dd.element_vol)
        ref = sc.compute_observables(dd)
        for name, ref_name in NAME_MAP.items():
            assert obs[name].shape == (len(sc.SPHEROID_M_LIST), dd.L)
            if name == "C_abs" and m_p.imag == 0:
                assert np.all(obs[name][s] == 0.0) and np.all(ref[ref_name] == 0.0)
            else:
                assert _max_rel(obs[name][s], ref[ref_name]) < RTOL_OBS, (name, m_p)


@pytest.mark.parametrize("tol, itermax", sc.SPHEROID_SOLVER_SETTINGS)
def test_t5_observables_match_independent_solves(geometry, tol, itermax):
    """End to end: shifted sweep vs one existing solve per refractive index."""
    m_max = max(sc.SPHEROID_M_LIST, key=abs)
    dd_shift, obs = _shifted_sweep(geometry, m_max, sc.SPHEROID_M_LIST, tol, itermax)
    assert np.all(dd_shift.converge_shifted)
    assert dd_shift.shifted_result.X_list is None                # nothing assembled
    for s, m_p in enumerate(sc.SPHEROID_M_LIST):
        dd = sc.make_spheroid_dipoles(geometry, m_p)
        n_iter, _ = sc.solve_existing(dd, tol, itermax)
        assert dd_shift.shifted_result.iters[s] == n_iter
        ref = sc.compute_observables(dd)
        for name, ref_name in NAME_MAP.items():
            if name == "C_abs" and m_p.imag == 0:
                assert np.all(obs[name][s] == 0.0)
            else:
                assert _max_rel(obs[name][s], ref[ref_name]) < RTOL_OBS, (name, m_p)


def test_optical_theorem_links_forward_amplitudes_and_extinction(sweep):
    """C_ext = (2 pi / k) Im(S_fw_theta + S_fw_phi), since F_theta + F_phi = 2 k^2 B."""
    dd_shift, obs = sweep
    lhs = obs["C_ext"]
    rhs = (2 * np.pi / dd_shift.k) * np.imag(obs["S_fw_PCAS_theta"] + obs["S_fw_PCAS_phi"])
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_unconverged_refractive_indices_are_nan(geometry):
    # itermax = 5 at tol = 1e-2: only m = 1.70 needs a sixth step.
    m_max = max(sc.SPHEROID_M_LIST, key=abs)
    dd_shift, obs = _shifted_sweep(geometry, m_max, sc.SPHEROID_M_LIST, 1e-2, 5)
    expected = np.array([abs(m) < 1.7 for m in sc.SPHEROID_M_LIST])
    assert np.array_equal(dd_shift.converge_shifted, expected)
    for name in NAME_MAP:
        assert np.all(np.isnan(obs[name][~expected]))
        assert np.all(np.isfinite(obs[name][expected]))


# ── T6 ───────────────────────────────────────────────────────────────────────
def test_t6_sphere_agrees_with_mie_like_the_existing_solver():
    """test_dda.ipynb sphere through the shifted path.

    The batch holds the notebook index and a lower one, on the notebook lattice.
    For each index the shifted result must equal the existing solver on that
    lattice. For the notebook index it must also meet the 1 % Mie tolerance.
    """
    # complex64 as in the notebook: the index actually solved is the float32-rounded value.
    m_p_xyz = np.array([sc.SPHERE_M_P] * 3, dtype=np.complex64)
    m_notebook = complex(m_p_xyz[0])
    m_list = [m_notebook, 1.40 + 0.01j]
    geometry = sc.quiet(sc.build_geometry, sc.SPHERE_R_V, 1.0, sc.SPHERE_WL_0, m_p_xyz,
                        sc.SPHERE_RNG_SEED)
    name, lattice_n, lattice_lf, grid, is_in = geometry
    euler = np.column_stack([np.zeros(4), np.arccos(np.linspace(0.0, 1.0, 4)), np.zeros(4)])
    inc = IncidentField(sc.SPHERE_WL_0, sc.SPHERE_M_M, euler)

    def dipoles(m_xyz):
        target = Target(name, lattice_n, lattice_lf, grid, is_in, m_xyz, sc.SPHERE_R_V)
        return sc.quiet(DiscreteDipoles, target, inc)

    dd_shift = dipoles(m_p_xyz)                               # notebook defaults tol=1e-2, itermax=25
    sc.quiet(dd_shift.set_interaction_matrix_shifted, m_list)
    sc.quiet(dd_shift.solve_matrix_equation_shifted)
    obs = dd_shift.compute_observables_shifted()
    assert np.all(dd_shift.converge_shifted)
    area = np.pi * dd_shift.ve_radius ** 2                    # [um^2]

    for s, m_p in enumerate(m_list):
        dd = dipoles(np.array([m_p] * 3))
        sc.quiet(dd.set_interaction_matrix)
        sc.quiet(dd.solve_matrix_equation)
        assert dd.converge
        ref = sc.compute_observables(dd)
        _, Q_abs_mie, Q_ext_mie, S_fw_mie, _ = mie_compute_q_and_s(
            sc.SPHERE_WL_0, sc.SPHERE_M_M, dd.ve_radius, m_p, nang=3)

        def mie_errors(C_abs, C_ext, S_theta, S_phi):
            return np.array([
                np.abs(C_abs / area / Q_abs_mie - 1),
                np.abs(C_ext / area / Q_ext_mie - 1),
                np.abs(S_theta.real / S_fw_mie.real - 1),
                np.abs(S_theta.imag / S_fw_mie.imag - 1),
                np.abs(S_phi.real / S_fw_mie.real - 1),
                np.abs(S_phi.imag / S_fw_mie.imag - 1),
            ])                                                # shape: (6, L)

        err_shift = mie_errors(obs["C_abs"][s], obs["C_ext"][s],
                               obs["S_fw_PCAS_theta"][s], obs["S_fw_PCAS_phi"][s])
        err_exist = mie_errors(ref["C_abs"], ref["C_ext"], ref["S_fw_theta"], ref["S_fw_phi"])
        # Same Mie error as the existing solver on the same lattice.
        np.testing.assert_allclose(err_shift, err_exist, rtol=1e-6, atol=1e-10)
        for name_s, name_r in NAME_MAP.items():
            assert _max_rel(obs[name_s][s], ref[name_r]) < RTOL_OBS, (name_s, m_p)
        if m_p == m_notebook:
            assert np.all(err_shift < sc.SPHERE_MIE_TOL)


# ── T7 ───────────────────────────────────────────────────────────────────────
def test_t7_anisotropic_or_nonscalar_input_raises(geometry):
    name, lattice_n, lattice_lf, grid, is_in = geometry
    inc = IncidentField(sc.SPHEROID_WL_0, sc.SPHEROID_M_M, sc.spheroid_euler_angles())
    aniso = Target(name, lattice_n, lattice_lf, grid, is_in,
                   np.array([1.48, 1.48, 1.56]), sc.SPHEROID_D_VE / 2)
    dd = sc.quiet(DiscreteDipoles, aniso, inc)
    with pytest.raises(ValueError, match="isotropic"):
        dd.set_interaction_matrix_shifted([1.4, 1.5])

    dd = sc.make_spheroid_dipoles(geometry, 1.7)
    with pytest.raises(ValueError):
        dd.set_interaction_matrix_shifted([[1.4, 1.4, 1.5], [1.5, 1.5, 1.6]])   # per-axis indices
    with pytest.raises(ValueError):
        dd.set_interaction_matrix_shifted([])
    with pytest.raises(ValueError):
        dd.set_interaction_matrix_shifted(1.5)                                   # not a sequence


def test_index_above_lattice_index_warns(geometry):
    dd = sc.make_spheroid_dipoles(geometry, 1.5)                 # lattice built for 1.70
    with pytest.warns(UserWarning, match="effective dpl"):
        sc.quiet(dd.set_interaction_matrix_shifted, [1.4, 1.6])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sc.quiet(dd.set_interaction_matrix_shifted, [1.4, 1.5])
