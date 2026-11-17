"""Shared test cases for the shifted block-GMRES work (refractive-index sweeps).

Two fixed problems are defined here and reused by the baseline generator
(``generate_shifted_baseline.py``) and by the pytest modules.

1. ``spheroid``: a small oblate spheroid solved for several refractive indices
   on ONE common lattice. The lattice is built for the largest ``|m|`` of the
   batch, which is the lattice policy of the shifted solver. Each refractive
   index is solved independently with the existing block-GMRES path
   (``DiscreteDipoles.solve_matrix_equation``). These are the reference
   solutions the shifted solver must reproduce.

2. ``sphere``: the sphere of ``test_dda.ipynb``, compared with Mie theory. This
   is the project's validation of the existing code path.

All lengths are in micrometres [um]. Cross sections are in [um^2]. Scattering
amplitudes are dimensionless.
"""
import contextlib
import io
import pathlib

import numpy as np

from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import DiscreteDipoles, IncidentField, Target
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Reference data live outside git (global convention: .claude/validation/).
VALIDATION_DIR = REPO_ROOT / ".claude" / "validation" / "shifted_krylov"
BASELINE_FILE = VALIDATION_DIR / "baseline_phase0.npz"

# ── case 1: oblate spheroid, several refractive indices on a common lattice ──
SPHEROID_WL_0 = 0.638          # vacuum wavelength [um]
SPHEROID_M_M = 1.0             # medium refractive index
SPHEROID_D_VE = 0.35           # volume-equivalent diameter [um]
SPHEROID_BC_RATIO = 2.0        # b/c (oblate)
SPHEROID_L = 5                 # number of orientations (right-hand sides)
SPHEROID_RNG_SEED = 12345
# Three dielectric indices spanning the LUT range and one absorbing index.
SPHEROID_M_LIST = (1.35 + 0.0j, 1.50 + 0.0j, 1.70 + 0.0j, 1.50 + 0.1j)
# (tol, itermax): the production setting and a tight setting.
SPHEROID_SOLVER_SETTINGS = ((1e-2, 25), (1e-5, 60))

# ── case 2: sphere of test_dda.ipynb ─────────────────────────────────────────
SPHERE_R_V = 0.2               # volume-equivalent radius [um]
SPHERE_WL_0 = 0.834            # vacuum wavelength [um]
SPHERE_M_M = 1.329             # medium refractive index
SPHERE_M_P = 1.5 + 0.01j       # particle refractive index
SPHERE_RNG_SEED = 1234
SPHERE_MIE_TOL = 0.01          # tolerance stated in the test_dda.ipynb cell comments

OBSERVABLE_NAMES = ("C_ext", "C_abs", "S_fw_theta", "S_fw_phi",
                    "S_bk_theta", "S_bk_phi", "S_bk_OCBS")


def quiet(func, *args, **kwargs):
    """Call ``func`` with stdout suppressed (the DDA classes print progress)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def build_geometry(r_v_base, bc_ratio, wl_0, m_p_xyz, rng_seed, ab_ratio=1.0, beta=0.0):
    """Build the lattice of a Gaussian random ellipsoid.

    Parameters
    ----------
    r_v_base : float   volume-equivalent radius [um]
    bc_ratio : float   b/c semi-axis ratio
    wl_0     : float   vacuum wavelength [um]
    m_p_xyz  : ndarray, shape (3,)   refractive index that sets the lattice spacing
    rng_seed : int

    Returns
    -------
    (name, lattice_n, lattice_lf, grid, is_in) : the arguments of ``Target``
    """
    rng = np.random.default_rng(rng_seed)
    gre = gaussian_ellipsoid_shape_model(r_v_base, bc_ratio, ab_ratio, beta, wl_0, m_p_xyz)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return gre.name, lattice_n, gre.lattice_lf, grid, is_in


def spheroid_euler_angles():
    """Orientation grid of the spheroid sweep: alpha = gamma = 0, cos(beta) equidistant.

    Returns
    -------
    euler : ndarray, shape (L, 3)   ZYZ Euler angles [rad]
    """
    beta = np.arccos(np.linspace(0.0, 1.0, SPHEROID_L))  # shape: (L,)
    return np.column_stack([np.zeros(SPHEROID_L), beta, np.zeros(SPHEROID_L)])


def spheroid_common_geometry():
    """Common lattice of the spheroid case, built for the largest |m| of the batch."""
    m_max = max(SPHEROID_M_LIST, key=abs)
    m_p_xyz = np.array([m_max, m_max, m_max])
    return quiet(build_geometry, SPHEROID_D_VE / 2.0, SPHEROID_BC_RATIO,
                 SPHEROID_WL_0, m_p_xyz, SPHEROID_RNG_SEED)


def make_spheroid_dipoles(geometry, m_p):
    """``DiscreteDipoles`` of the spheroid case for refractive index ``m_p``."""
    name, lattice_n, lattice_lf, grid, is_in = geometry
    m_p_xyz = np.array([m_p, m_p, m_p])
    target = Target(name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, SPHEROID_D_VE / 2.0)
    inc = IncidentField(SPHEROID_WL_0, SPHEROID_M_M, spheroid_euler_angles())
    return quiet(DiscreteDipoles, target, inc)


def solve_existing(dd, tol, itermax):
    """Solve with the existing block-GMRES path and return iteration data.

    Returns
    -------
    n_iter      : int     number of block-Arnoldi steps (1-based count)
    err_history : ndarray, shape (n_iter,)   relative residual per step
    """
    dd.tol, dd.itermax = tol, itermax
    quiet(dd.set_interaction_matrix)
    quiet(dd.solve_matrix_equation)
    if not dd.converge:
        raise RuntimeError(f"existing block-GMRES did not converge (tol={tol}, itermax={itermax})")
    err_history = np.asarray(dd.err_history, dtype=np.float64)
    return err_history.size, err_history


def compute_observables(dd):
    """Observables of a solved ``DiscreteDipoles`` object.

    Returns
    -------
    dict of ndarray, each of shape (L,). Keys are ``OBSERVABLE_NAMES``.
    """
    S_fw_theta, S_fw_phi = dd.compute_PCAS_observable_S_fw()
    S_bk = dd.compute_OCBS_observable_S_bk()
    return {
        "C_ext": dd.compute_C_ext(),          # [um^2]
        "C_abs": dd.compute_C_abs(),          # [um^2]
        "S_fw_theta": S_fw_theta,
        "S_fw_phi": S_fw_phi,
        "S_bk_theta": dd.S_bk_OCBS_theta,
        "S_bk_phi": dd.S_bk_OCBS_phi,
        "S_bk_OCBS": S_bk,
    }


def spheroid_reference():
    """Reference solutions of the spheroid case from the existing solver.

    Returns
    -------
    dict of ndarray. Keys:
      ``spheroid/lattice_n``            (3,) int
      ``spheroid/n_occ``                () int
      ``spheroid/lattice_lf``           () float [um], after the volume rescaling in Target
      ``spheroid/m_list``               (N_m,) complex
      ``spheroid/sigma``                (N_m,) complex, 1/alpha_E for each m [um^-3]
      ``spheroid/tol{j}/n_iter``        (N_m,) int
      ``spheroid/tol{j}/err_fin``       (N_m,) float
      ``spheroid/tol{j}/X``             (N_m, 3*N_occ, L) complex
      ``spheroid/tol{j}/<observable>``  (N_m, L)
    where j indexes ``SPHEROID_SOLVER_SETTINGS``.
    """
    geometry = spheroid_common_geometry()
    out = {"spheroid/m_list": np.asarray(SPHEROID_M_LIST, dtype=np.complex128)}
    for j, (tol, itermax) in enumerate(SPHEROID_SOLVER_SETTINGS):
        n_iter, err_fin, X, sigma = [], [], [], []
        obs = {name: [] for name in OBSERVABLE_NAMES}
        for m_p in SPHEROID_M_LIST:
            dd = make_spheroid_dipoles(geometry, m_p)
            k, err_history = solve_existing(dd, tol, itermax)
            n_iter.append(k)
            err_fin.append(err_history[-1])
            X.append(dd.X.copy())
            sigma.append(dd.diag_A[0])
            for name, val in compute_observables(dd).items():
                obs[name].append(np.array(val))
        prefix = f"spheroid/tol{j}/"
        out[prefix + "n_iter"] = np.asarray(n_iter, dtype=np.int64)
        out[prefix + "err_fin"] = np.asarray(err_fin, dtype=np.float64)
        out[prefix + "X"] = np.asarray(X)                    # shape: (N_m, 3*N_occ, L)
        for name in OBSERVABLE_NAMES:
            out[prefix + name] = np.asarray(obs[name])       # shape: (N_m, L)
    out["spheroid/sigma"] = np.asarray(sigma, dtype=np.complex128)
    out["spheroid/lattice_n"] = np.asarray(dd.lattice_n, dtype=np.int64)
    out["spheroid/n_occ"] = np.int64(dd.num_element_occupy)
    out["spheroid/lattice_lf"] = np.float64(dd.lattice_lf)
    return out


def sphere_reference():
    """DDA solution of the ``test_dda.ipynb`` sphere and its Mie reference.

    The setup mirrors the notebook cell by cell, including the complex64
    ``m_p_xyz`` array and the 3 x 4 x 3 Euler-angle grid (L = 36).

    Returns
    -------
    dict of ndarray. Keys:
      ``sphere/n_occ``, ``sphere/n_iter``              () int
      ``sphere/Q_abs``, ``sphere/Q_ext``               (L,) float, DDA efficiencies
      ``sphere/S_fw_theta``, ``sphere/S_fw_phi``       (L,) complex, DDA amplitudes
      ``sphere/Q_abs_mie``, ``sphere/Q_ext_mie``       () float
      ``sphere/S_fw_mie``                              () complex
    """
    m_p_xyz = np.array([SPHERE_M_P, SPHERE_M_P, SPHERE_M_P], dtype=np.complex64)
    m_p_avg = SPHERE_M_P
    geometry = quiet(build_geometry, SPHERE_R_V, 1.0, SPHERE_WL_0, m_p_xyz, SPHERE_RNG_SEED)
    name, lattice_n, lattice_lf, grid, is_in = geometry
    target = Target(name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, SPHERE_R_V)

    n_alpha, n_beta, n_gamma = 3, 4, 3
    alpha = np.linspace(0, 2 * np.pi, n_alpha, endpoint=False)
    beta = np.arccos(np.linspace(1 - 1 / n_beta, -1 + 1 / n_beta, n_beta))
    gamma = np.linspace(0, 2 * np.pi, n_gamma, endpoint=False)
    aa, bb, gg = np.meshgrid(alpha, beta, gamma, indexing="ij")
    euler = np.column_stack([aa.ravel(), bb.ravel(), gg.ravel()])  # shape: (36, 3)

    dd = quiet(DiscreteDipoles, target, IncidentField(SPHERE_WL_0, SPHERE_M_M, euler))
    quiet(dd.set_interaction_matrix)
    quiet(dd.solve_matrix_equation)          # notebook defaults: tol=1e-2, itermax=25
    if not dd.converge:
        raise RuntimeError("existing block-GMRES did not converge for the test_dda.ipynb sphere")

    _, Q_abs_mie, Q_ext_mie, S_fw_mie, _ = mie_compute_q_and_s(
        SPHERE_WL_0, SPHERE_M_M, dd.ve_radius, m_p_avg, nang=3)
    area = np.pi * dd.ve_radius ** 2         # geometric cross section [um^2]
    S_fw_theta, S_fw_phi = dd.compute_PCAS_observable_S_fw()
    return {
        "sphere/n_occ": np.int64(dd.num_element_occupy),
        "sphere/n_iter": np.int64(len(dd.err_history)),
        "sphere/Q_abs": dd.compute_C_abs() / area,
        "sphere/Q_ext": dd.compute_C_ext() / area,
        "sphere/S_fw_theta": np.array(S_fw_theta),
        "sphere/S_fw_phi": np.array(S_fw_phi),
        "sphere/Q_abs_mie": np.float64(Q_abs_mie),
        "sphere/Q_ext_mie": np.float64(Q_ext_mie),
        "sphere/S_fw_mie": np.complex128(np.ravel(S_fw_mie)[0]),
    }


def collect_baseline():
    """All Phase-0 reference quantities as one flat dict of ndarray."""
    out = {}
    out.update(spheroid_reference())
    out.update(sphere_reference())
    return out
