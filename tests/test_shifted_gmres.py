"""Tests of the shifted block GMRES ``bl_gmres_shifted_mvp_fft`` (Phase 1).

The reference is the existing ``bl_gmres_mvp_fft``, called once per refractive
index on the common lattice of the spheroid case in ``shifted_cases.py``.

T1  all shifts in one process reproduce the independent solutions and iteration counts
T2  a single shift reproduces ``bl_gmres_mvp_fft``
T3  truncation at ``itermax`` is reported through ``errs``
T4  a zero right-hand side does not raise
plus checks of the true residual, of the projection operators and of Λ.

The per-shift least squares has two implementations (``lsq_method``):
"incremental" (default) and "dense". T1 and T3 run with both, and the two are
compared with each other and with ``numpy.linalg.lstsq``.
"""
import numpy as np
import pytest

import shifted_cases as sc
from bl_krylov.bl_krylov import (
    _block_mvp_noprec,
    _build_scatter_index,
    _shifted_lstsq_dense,
    _shifted_lstsq_hessenberg,
    bl_gmres_mvp_fft,
    bl_gmres_shifted_mvp_fft,
)

LSQ_METHODS = ("incremental", "dense")

RTOL_X = 1e-10   # relative Frobenius tolerance on solutions


def _rel_diff(X, X_ref):
    return np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref)


@pytest.fixture(scope="module")
def system():
    """Common-lattice linear systems of the spheroid case.

    Returns a dict with the solver arguments shared by all refractive indices
    (``n``, ``f``, ``address``, ``Au_til``, ``B``), the shifts ``sigmas`` of
    shape (N_m,), and the full diagonals ``diag_A`` of shape (N_m, 3*N_occ).
    """
    geometry = sc.spheroid_common_geometry()
    dds = [sc.make_spheroid_dipoles(geometry, m_p) for m_p in sc.SPHEROID_M_LIST]
    for dd in dds:
        sc.quiet(dd.set_interaction_matrix)
    dd0 = dds[0]
    for dd in dds:
        # Isotropic particle: the diagonal is a scalar multiple of the identity.
        assert np.all(dd.diag_A == dd.diag_A[0])
        # Lattice, Green's tensor and incident field do not depend on m.
        assert np.array_equal(dd.Au_til, dd0.Au_til)
        assert np.array_equal(dd.B, dd0.B)
    return {
        "n": dd0.lattice_n,
        "f": dd0.f,
        "address": dd0.lattice_address_in_target,
        "Au_til": dd0.Au_til,
        "B": dd0.B,                                              # shape: (3*N_occ, L)
        "sigmas": np.array([dd.diag_A[0] for dd in dds]),        # shape: (N_m,)
        "diag_A": np.array([dd.diag_A for dd in dds]),           # shape: (N_m, 3*N_occ)
    }


def _solve_independent(system, s, tol, itermax):
    """Existing solver for shift index ``s``: (X, n_iter, err_fin, err_history)."""
    X, iter_fin, err_fin, err_history = bl_gmres_mvp_fft(
        system["n"], system["f"], system["address"], system["Au_til"],
        system["diag_A"][s], system["B"], tol, itermax)
    return X, iter_fin + 1, err_fin, err_history


def _solve_shifted(system, sigmas, tol, itermax, **kwargs):
    return bl_gmres_shifted_mvp_fft(
        system["n"], system["f"], system["address"], system["Au_til"],
        sigmas, system["B"], tol, itermax, **kwargs)


def _true_residual(system, s, X):
    """||B - (sigma_s I + A_off) X||_F / ||B||_F via the existing MVP."""
    full_idx = _build_scatter_index(system["address"], system["f"])
    AX = _block_mvp_noprec(system["n"], system["f"], system["Au_til"], full_idx,
                           system["diag_A"][s], X)
    return np.linalg.norm(system["B"] - AX) / np.linalg.norm(system["B"])


# ── T1 ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lsq_method", LSQ_METHODS)
@pytest.mark.parametrize("tol, itermax", sc.SPHEROID_SOLVER_SETTINGS)
def test_t1_all_shifts_match_independent_solves(system, tol, itermax, lsq_method):
    res = _solve_shifted(system, system["sigmas"], tol, itermax, return_X=True,
                         lsq_method=lsq_method)
    L = system["B"].shape[1]
    for s in range(len(system["sigmas"])):
        X_ref, n_iter, err_fin, err_history = _solve_independent(system, s, tol, itermax)
        assert res.iters[s] == n_iter
        assert _rel_diff(res.X_list[s], X_ref) < RTOL_X
        assert res.errs[s] < tol
        assert res.errs[s] == pytest.approx(err_fin, rel=1e-6)
        # Residual history: identical while active, NaN afterwards.
        np.testing.assert_allclose(res.err_history[:n_iter, s], err_history, rtol=1e-6)
        assert np.all(np.isnan(res.err_history[n_iter:, s]))
        assert res.Y_list[s].shape == (n_iter * L, L)
    assert res.err_history.shape == (res.iters.max(), len(system["sigmas"]))


# ── T2 ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("s", range(len(sc.SPHEROID_M_LIST)))
def test_t2_single_shift_matches_existing_solver(system, s):
    tol, itermax = 1e-5, 60
    res = _solve_shifted(system, system["sigmas"][s:s + 1], tol, itermax, return_X=True)
    X_ref, n_iter, err_fin, _ = _solve_independent(system, s, tol, itermax)
    assert res.iters[0] == n_iter
    assert _rel_diff(res.X_list[0], X_ref) < RTOL_X
    assert res.errs[0] == pytest.approx(err_fin, rel=1e-6)
    # A scalar shift is accepted as well as a length-1 array.
    res_scalar = _solve_shifted(system, system["sigmas"][s], tol, itermax, return_X=True)
    assert np.array_equal(res_scalar.X_list[0], res.X_list[0])


# ── T3 ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lsq_method", LSQ_METHODS)
def test_t3_itermax_truncation_is_reported(system, lsq_method):
    # itermax = 5 at tol = 1e-2: m = 1.70 needs 6 steps, the others at most 5.
    tol, itermax = 1e-2, 5
    res = _solve_shifted(system, system["sigmas"], tol, itermax, return_X=True,
                         lsq_method=lsq_method)
    n_iter_full = np.array(
        [_solve_independent(system, s, tol, 25)[1] for s in range(len(system["sigmas"]))])
    unconverged = n_iter_full > itermax
    assert unconverged.any() and not unconverged.all()
    assert np.all(res.errs[unconverged] >= tol)
    assert np.all(res.errs[~unconverged] < tol)
    assert np.all(res.iters[unconverged] == itermax)
    assert np.array_equal(res.iters[~unconverged], n_iter_full[~unconverged])
    # The reported residual of a truncated shift is still the true residual.
    for s in np.flatnonzero(unconverged):
        assert _true_residual(system, s, res.X_list[s]) == pytest.approx(res.errs[s], rel=1e-6)

    # All shifts truncated.
    res = _solve_shifted(system, system["sigmas"], 1e-5, 3, lsq_method=lsq_method)
    assert np.all(res.errs >= 1e-5)
    assert np.all(res.iters == 3)
    assert res.err_history.shape == (3, len(system["sigmas"]))
    assert res.X_list is None


# ── T4 ───────────────────────────────────────────────────────────────────────
def test_t4_zero_rhs_does_not_raise(system):
    B0 = np.zeros_like(system["B"])
    L = B0.shape[1]
    N_m = len(system["sigmas"])
    F = np.ones((2,) + B0.shape, dtype=np.complex128)
    res = bl_gmres_shifted_mvp_fft(
        system["n"], system["f"], system["address"], system["Au_til"],
        system["sigmas"], B0, 1e-5, 10, proj_ops=F, return_X=True)
    assert np.all(res.iters == 0) and np.all(res.errs == 0.0)
    assert all(np.all(X == 0) and X.shape == B0.shape for X in res.X_list)
    assert all(Y.shape == (0, L) for Y in res.Y_list)
    assert res.G.shape == (2, L, 0)
    # The projection formula still evaluates (to zero) without special-casing.
    assert np.all(res.G[0] @ res.Y_list[0] == 0) and len(res.Y_list) == N_m
    # Same convention as the existing solver for a zero right-hand side.
    X, iter_fin, err_fin, _ = bl_gmres_mvp_fft(
        system["n"], system["f"], system["address"], system["Au_til"],
        system["diag_A"][0], B0, 1e-5, 10)
    assert np.all(X == 0) and iter_fin == 0 and err_fin == 0.0


# ── additional checks ────────────────────────────────────────────────────────
def test_reported_residual_is_true_residual(system):
    tol, itermax = 1e-5, 60
    res = _solve_shifted(system, system["sigmas"], tol, itermax, return_X=True)
    for s in range(len(system["sigmas"])):
        assert _true_residual(system, s, res.X_list[s]) == pytest.approx(res.errs[s], rel=1e-6)


def test_projection_operators_reproduce_functionals(system):
    """F^H X_s equals G[:, :, :k_s L] Y_s, and B^H X_s equals Λ^H Y_s[:L]."""
    tol, itermax = 1e-5, 60
    rng = np.random.default_rng(0)
    B = system["B"]
    L = B.shape[1]
    F = rng.standard_normal((3,) + B.shape) + 1j * rng.standard_normal((3,) + B.shape)
    res = _solve_shifted(system, system["sigmas"], tol, itermax, proj_ops=F, return_X=True)
    assert res.G.shape == (3, L, res.iters.max() * L)
    for s, (X, Y) in enumerate(zip(res.X_list, res.Y_list)):
        kL = res.iters[s] * L
        for o in range(F.shape[0]):
            direct = F[o].conj().T @ X                               # shape: (L, L)
            projected = res.G[o][:, :kL] @ Y
            assert _rel_diff(projected, direct) < 1e-12
        assert _rel_diff(res.Lambda.conj().T @ Y[:L], B.conj().T @ X) < 1e-12

    # Without proj_ops no G is returned, and the solutions are unchanged.
    res0 = _solve_shifted(system, system["sigmas"], tol, itermax, return_X=True)
    assert res0.G is None
    assert all(np.array_equal(a, b) for a, b in zip(res0.X_list, res.X_list))


def test_shift_order_and_duplicates_do_not_matter(system):
    """Each shift is solved independently of the others in the batch."""
    tol, itermax = 1e-5, 60
    sig = system["sigmas"]
    res = _solve_shifted(system, sig, tol, itermax, return_X=True)
    perm = np.array([2, 0, 3, 1, 2])                                  # reordered, one repeated
    res_p = _solve_shifted(system, sig[perm], tol, itermax, return_X=True)
    for i, s in enumerate(perm):
        assert res_p.iters[i] == res.iters[s]
        assert np.array_equal(res_p.X_list[i], res.X_list[s])


def test_hessenberg_growth_beyond_initial_capacity(system):
    """More than 16 steps exercises the on-demand growth of Hbar."""
    tol, itermax = 1e-9, 60
    s = 2                                                             # m = 1.70, slowest
    res = _solve_shifted(system, system["sigmas"][s:s + 1], tol, itermax, return_X=True)
    X_ref, n_iter, _, _ = _solve_independent(system, s, tol, itermax)
    assert n_iter > 16
    assert res.iters[0] == n_iter
    assert _rel_diff(res.X_list[0], X_ref) < RTOL_X


# ── least-squares implementations ────────────────────────────────────────────
@pytest.mark.parametrize("tol, itermax", sc.SPHEROID_SOLVER_SETTINGS + ((1e-9, 60),))
def test_incremental_and_dense_least_squares_agree(system, tol, itermax):
    """Same step counts, residual histories, coefficients, projections and solutions."""
    rng = np.random.default_rng(1)
    B = system["B"]
    F = rng.standard_normal((2,) + B.shape) + 1j * rng.standard_normal((2,) + B.shape)
    res_i = _solve_shifted(system, system["sigmas"], tol, itermax, proj_ops=F,
                           return_X=True, lsq_method="incremental")
    res_d = _solve_shifted(system, system["sigmas"], tol, itermax, proj_ops=F,
                           return_X=True, lsq_method="dense")
    assert np.array_equal(res_i.iters, res_d.iters)
    assert np.all(res_i.errs < tol)
    # Residual norms lose relative accuracy as they approach rounding level.
    np.testing.assert_allclose(res_i.errs, res_d.errs, rtol=1e-6)
    assert np.array_equal(np.isnan(res_i.err_history), np.isnan(res_d.err_history))
    np.testing.assert_allclose(res_i.err_history, res_d.err_history, rtol=1e-6)
    # The Arnoldi process does not depend on the least-squares method.
    assert np.array_equal(res_i.G, res_d.G)
    assert np.array_equal(res_i.Lambda, res_d.Lambda)
    for s in range(len(system["sigmas"])):
        assert _rel_diff(res_i.Y_list[s], res_d.Y_list[s]) < RTOL_X
        assert _rel_diff(res_i.X_list[s], res_d.X_list[s]) < RTOL_X


def test_structured_least_squares_matches_lstsq():
    """Both helpers solve min ‖E_1 Λ − (Hbar_k + σ Ibar_k) Y‖_F for a random block Hessenberg."""
    rng = np.random.default_rng(2)
    L, k_cap = 4, 7
    Hbar = (rng.standard_normal(((k_cap + 1) * L, k_cap * L))
            + 1j * rng.standard_normal(((k_cap + 1) * L, k_cap * L)))
    for j in range(k_cap):                              # zero below the block subdiagonal
        Hbar[(j + 2) * L:, j * L:(j + 1) * L] = 0.0
    Lambda = rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))
    sigma = 0.7 - 0.3j
    for k in (1, 2, 5, k_cap):                          # k < k_cap: Hbar is larger than needed
        kL = k * L
        A = Hbar[:kL + L, :kL].copy()
        A[np.arange(kL), np.arange(kL)] += sigma
        rhs = np.zeros((kL + L, L), dtype=np.complex128)
        rhs[:L] = Lambda
        Y_ref = np.linalg.lstsq(A, rhs, rcond=None)[0]
        res_ref = np.linalg.norm(rhs - A @ Y_ref)
        for solve in (_shifted_lstsq_dense, _shifted_lstsq_hessenberg):
            Y, res = solve(Hbar, Lambda, sigma, k)
            assert Y.shape == (kL, L)
            assert _rel_diff(Y, Y_ref) < 1e-12
            assert res == pytest.approx(res_ref, rel=1e-12)


def test_invalid_arguments(system):
    with pytest.raises(ValueError):
        _solve_shifted(system, np.ones((2, 2)), 1e-5, 10)
    with pytest.raises(ValueError):
        _solve_shifted(system, system["sigmas"], 1e-5, 0)
    with pytest.raises(ValueError):
        _solve_shifted(system, system["sigmas"], 1e-5, 10, lsq_method="qr")
