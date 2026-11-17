"""Regression tests of the EXISTING solver path (Phase 0 of the shifted-GMRES work).

These tests do not exercise any new code. They pin down the behaviour that the
shifted solver must reproduce and that later phases must not change:

- the Mie comparison of ``test_dda.ipynb`` (project validation, 1 % tolerance);
- the consistency of the reported block-GMRES residual with the true residual;
- agreement with the stored reference data, when the file is present.
"""
import numpy as np
import pytest

import shifted_cases as sc
from bl_krylov.bl_krylov import _block_mvp_noprec, _build_scatter_index

RTOL_ARRAY = 1e-10   # tolerance for solutions and derived observables


@pytest.fixture(scope="module")
def current():
    """Reference quantities recomputed with the code under test."""
    return sc.collect_baseline()


def test_sphere_agrees_with_mie(current):
    """test_dda.ipynb: DDA vs Mie within 1 % for every orientation."""
    tol = sc.SPHERE_MIE_TOL
    S_mie = current["sphere/S_fw_mie"]
    assert np.all(np.abs(current["sphere/Q_abs"] / current["sphere/Q_abs_mie"] - 1) < tol)
    assert np.all(np.abs(current["sphere/Q_ext"] / current["sphere/Q_ext_mie"] - 1) < tol)
    for name in ("sphere/S_fw_theta", "sphere/S_fw_phi"):
        S = current[name]                                     # shape: (L,)
        assert np.all(np.abs(S.real / S_mie.real - 1) < tol)
        assert np.all(np.abs(S.imag / S_mie.imag - 1) < tol)


@pytest.mark.parametrize("j", range(len(sc.SPHEROID_SOLVER_SETTINGS)))
def test_reported_residual_is_true_residual(current, j):
    """The residual reported by block-GMRES equals ||B - A X||_F / ||B||_F."""
    tol, _ = sc.SPHEROID_SOLVER_SETTINGS[j]
    geometry = sc.spheroid_common_geometry()
    for s, m_p in enumerate(sc.SPHEROID_M_LIST):
        dd = sc.make_spheroid_dipoles(geometry, m_p)
        sc.quiet(dd.set_interaction_matrix)
        full_idx = _build_scatter_index(dd.lattice_address_in_target, dd.f)
        X = current[f"spheroid/tol{j}/X"][s]                  # shape: (3*N_occ, L)
        AX = _block_mvp_noprec(dd.lattice_n, dd.f, dd.Au_til, full_idx, dd.diag_A, X)
        res = np.linalg.norm(dd.B - AX) / np.linalg.norm(dd.B)
        err_fin = current[f"spheroid/tol{j}/err_fin"][s]
        assert res < tol
        assert res == pytest.approx(err_fin, rel=1e-6)


def test_absorption_vanishes_for_real_index(current):
    """C_abs is exactly zero for real m and positive for absorbing m."""
    m_list = current["spheroid/m_list"]
    C_abs = current["spheroid/tol1/C_abs"]                    # shape: (N_m, L)
    assert np.all(C_abs[m_list.imag == 0] == 0.0)
    assert np.all(C_abs[m_list.imag > 0] > 0.0)


def test_matches_stored_baseline(current):
    """Outputs agree with .claude/validation/shifted_krylov/baseline_phase0.npz."""
    if not sc.BASELINE_FILE.exists():
        pytest.skip(f"no stored baseline ({sc.BASELINE_FILE}); "
                    "run tests/generate_shifted_baseline.py")
    stored = np.load(sc.BASELINE_FILE)
    for key, now in current.items():
        ref = stored[key]
        now = np.asarray(now)
        if np.issubdtype(now.dtype, np.integer):
            assert np.array_equal(now, ref), key
        elif key.endswith("/X"):
            # shape: (N_m, 3*N_occ, L); relative Frobenius difference per refractive index
            diff = np.linalg.norm(now - ref, axis=(1, 2)) / np.linalg.norm(ref, axis=(1, 2))
            assert np.all(diff < RTOL_ARRAY), (key, diff)
        elif key.endswith("/err_fin"):
            # A residual of order tol carries a relative rounding error of order eps / tol.
            np.testing.assert_allclose(now, ref, rtol=1e-6, err_msg=key)
        else:
            scale = np.max(np.abs(ref))
            np.testing.assert_allclose(now, ref, rtol=RTOL_ARRAY,
                                       atol=RTOL_ARRAY * scale, err_msg=key)
