"""Spheroid-only DDA parameter sweep (multi-wavelength).

Sweeps over (wl_0, m_m) pairs x D_ve x RI_real x log10(AR) on a regular grid.
Orientations: (cos_theta_o_half [0,1], phi_o [0,pi]) — reduced domain.
Analytical expansion from beta-only DDA solves is applied before writing.

Output HDF5 schema follows .claude/spheroid_h5_schema_spec.md for compatibility
with the downstream consumer (build_spheroid_lut.py).  Each wavelength gets its
own HDF5 group (e.g. wl_0p453/); shared grid axes live at the root level.

Output: dda_results/dda_results_spheroid_sweep.h5
"""

import itertools
import datetime
import pathlib
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

# ══════════════════════════════════════════════════════════════════════════════
# Settings — edit this section
# ══════════════════════════════════════════════════════════════════════════════
RNG_SEED    = 12345
MAX_TRY     = 1
OUTPUT_FILE = "dda_results/dda_results_spheroid_sweep.h5"

# (wavelength [um], medium refractive index) pairs
MEDIUM_CONDITIONS = [
    (0.453, 1.0),
    (0.638, 1.0),
    (0.834, 1.0),
]

M_IMAG = 0.0     # imaginary part of particle refractive index (fixed)

# Swept parameters: (min, max, N_grid)  — all grids are equidistant
D_VE_RANGE     = (0.26, 0.56, 16)      # volume-equivalent diameter [um]
RI_REAL_RANGE  = (1.2, 1.7, 21)      # Re(m_p)
LOG10_AR_RANGE = (-0.8, 0.8, 17)     # log10(AR), AR = bc_ratio

# Orientation grid on reduced domain
N_COS_THETA_O_HALF = 13   # cos(theta_o) in [0, 1], equidistant
N_PHI_O            = 21   # phi_o in [0, pi], equidistant
# ══════════════════════════════════════════════════════════════════════════════


def _log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}")


def _wl_group_name(wl_0: float) -> str:
    """Convert wavelength to HDF5 group name: 0.453 -> 'wl_0p453'."""
    return f"wl_{str(wl_0).replace('.', 'p')}"


# ── grid generation ──────────────────────────────────────────────────────────
D_ve_grid     = np.linspace(*D_VE_RANGE)                     # shape (N_Dve,)
RI_real_grid  = np.linspace(*RI_REAL_RANGE)                   # shape (N_RI,)
log_AR_grid   = np.linspace(*LOG10_AR_RANGE)                  # shape (N_AR,)
AR_grid       = 10.0 ** log_AR_grid                           # bc_ratio values

cos_theta_o_half_grid = np.linspace(0, 1, N_COS_THETA_O_HALF)  # shape (N_u_half,)
phi_o_grid            = np.linspace(0, np.pi, N_PHI_O)         # shape (N_ph,)

N_Dve    = len(D_ve_grid)
N_RI     = len(RI_real_grid)
N_AR     = len(log_AR_grid)
N_u_half = len(cos_theta_o_half_grid)
N_ph     = len(phi_o_grid)


# ── helper functions ─────────────────────────────────────────────────────────

def _build_spheroid_geometry(r_v_base, bc_ratio, wl_0, m_p_xyz):
    """Build spheroid lattice geometry (ab_ratio=1, gre_beta=0).

    Returns (name, lattice_n, lattice_lf, grid, is_in).
    """
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(
        r_v_base, bc_ratio, 1.0, 0.0, wl_0, m_p_xyz)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist  = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return gre.name, lattice_n, gre.lattice_lf, grid, is_in


def _run_dda_spheroid(target, wl_0, m_m, cos_theta_o_half, phi_o):
    """DDA solve for spheroids on reduced-domain orientation grid.

    Solves N_u_half orientations at phi_o=0 (alpha=0), then expands
    analytically to the full (N_u_half, N_ph) grid.

    Orientation mapping:
        theta_o = arccos(cos_theta_o)  <->  beta  (polar Euler angle)
        phi_o                          <->  alpha (azimuthal Euler angle)

    Analytical expansion (spheroid symmetry):
        S_s(phi_o) = A + B * exp(2j * phi_o)
        S_p(phi_o) = A - B * exp(2j * phi_o)
    where A = (S_s(0) + S_p(0))/2,  B = (S_s(0) - S_p(0))/2.

    Parameters
    ----------
    cos_theta_o_half : ndarray, shape (N_u_half,)
        cos(theta_o) grid on [0, 1].
    phi_o : ndarray, shape (N_ph,)
        phi_o grid on [0, pi] [rad].

    Returns
    -------
    S_fw_theta : ndarray, shape (N_u_half, N_ph), complex128
    S_fw_phi   : ndarray, shape (N_u_half, N_ph), complex128
    converged  : bool
    """
    N_u = len(cos_theta_o_half)
    N_p = len(phi_o)

    # Convert cos_theta_o to beta (Euler polar angle)
    beta = np.arccos(cos_theta_o_half)  # shape (N_u,)
    euler_beta_only = np.column_stack([
        np.zeros(N_u), beta, np.zeros(N_u),
    ])  # shape (N_u, 3): alpha=0, gamma=0

    for i_try in range(1, MAX_TRY + 1):
        inc = IncidentField(wl_0, m_m, euler_beta_only)
        dd  = DiscreteDipoles(target, inc)
        dd.set_interaction_matrix()
        dd.solve_matrix_equation()

        _log(f"    try {i_try}/{MAX_TRY} [spheroid, L={N_u}]: "
             f"{'converged' if dd.converge else 'not converged'}")
        if dd.converge:
            break
    else:
        nan_2d = np.full((N_u, N_p), np.nan)
        return nan_2d, nan_2d, False

    # Observables at phi_o=0 (alpha=0)
    S_s_0, S_p_0 = dd.compute_PCAS_observable_S_fw()  # shape (N_u,)

    # Analytical expansion to (N_u_half, N_ph)
    exp_2p = np.exp(2j * phi_o)  # shape (N_ph,)

    A_fw = (S_s_0 + S_p_0) / 2  # shape (N_u,)
    B_fw = (S_s_0 - S_p_0) / 2  # shape (N_u,)

    # S_fw_theta[j, i] = A[j] + B[j] * exp(2j * phi_o[i])
    # shape (N_u_half, N_ph)
    S_fw_theta = A_fw[:, np.newaxis] + B_fw[:, np.newaxis] * exp_2p[np.newaxis, :]
    S_fw_phi   = A_fw[:, np.newaxis] - B_fw[:, np.newaxis] * exp_2p[np.newaxis, :]

    return S_fw_theta, S_fw_phi, True


# ── HDF5 file creation ───────────────────────────────────────────────────────

def _create_h5(filepath):
    """Create output HDF5 file with shared grids and per-wavelength groups."""
    shape_5d   = (N_Dve, N_RI, N_AR, N_u_half, N_ph)
    shape_conv = (N_Dve, N_RI, N_AR)

    with h5py.File(filepath, "w") as f:
        # Root-level provenance attributes
        f.attrs['m_m']               = MEDIUM_CONDITIONS[0][1]
        f.attrs['m_imag']            = M_IMAG
        f.attrs['rng_seed']          = RNG_SEED
        f.attrs['solver_tol']        = 1e-2
        f.attrs['block_dda_version'] = '0.7.2'

        # Shared grid axis arrays (at root)
        f.create_dataset("D_ve_grid",              data=D_ve_grid,              dtype=np.float64)
        f.create_dataset("RI_real_grid",            data=RI_real_grid,           dtype=np.float64)
        f.create_dataset("log_AR_grid",             data=log_AR_grid,            dtype=np.float64)
        f.create_dataset("cos_theta_o_half_grid",   data=cos_theta_o_half_grid,  dtype=np.float64)
        f.create_dataset("phi_o_grid",              data=phi_o_grid,             dtype=np.float64)

        # Per-wavelength groups
        for wl_0, m_m in MEDIUM_CONDITIONS:
            grp_name = _wl_group_name(wl_0)
            grp = f.create_group(grp_name)
            grp.attrs['wl_0'] = wl_0
            grp.attrs['m_m']  = m_m

            for name in ("S_fw_theta_re", "S_fw_theta_im",
                          "S_fw_phi_re",   "S_fw_phi_im"):
                ds = grp.create_dataset(name, shape=shape_5d, dtype=np.float64)
                ds[:] = np.nan

            grp.create_dataset("converged", shape=shape_conv, dtype=bool,
                               data=np.zeros(shape_conv, dtype=bool))

    _log(f"Created {filepath}")
    _log(f"  Wavelengths: {[wl for wl, _ in MEDIUM_CONDITIONS]}")
    _log(f"  Grid: N_Dve={N_Dve}, N_RI={N_RI}, N_AR={N_AR}")
    _log(f"  Orientation: N_cos_theta_o_half={N_u_half}, N_phi_o={N_ph}")
    n_per_wl = N_Dve * N_RI * N_AR
    _log(f"  DDA conditions per wavelength: {n_per_wl}  "
         f"(total: {n_per_wl * len(MEDIUM_CONDITIONS)})")


# ── main sweep ───────────────────────────────────────────────────────────────

if not pathlib.Path(OUTPUT_FILE).exists():
    _create_h5(OUTPUT_FILE)

with h5py.File(OUTPUT_FILE, "r+") as h5:
    for wl_0, m_m in MEDIUM_CONDITIONS:
        grp_name = _wl_group_name(wl_0)

        # Create group if resuming a file that was created before this
        # wavelength was added
        if grp_name not in h5:
            shape_5d   = (N_Dve, N_RI, N_AR, N_u_half, N_ph)
            shape_conv = (N_Dve, N_RI, N_AR)
            grp = h5.create_group(grp_name)
            grp.attrs['wl_0'] = wl_0
            grp.attrs['m_m']  = m_m
            for name in ("S_fw_theta_re", "S_fw_theta_im",
                          "S_fw_phi_re",   "S_fw_phi_im"):
                ds = grp.create_dataset(name, shape=shape_5d, dtype=np.float64)
                ds[:] = np.nan
            grp.create_dataset("converged", shape=shape_conv, dtype=bool,
                               data=np.zeros(shape_conv, dtype=bool))

        grp = h5[grp_name]
        _log(f"── Wavelength {wl_0} um (group: {grp_name}) ──")

        for (i_dve, D_ve), (i_ri, RI_real), (i_ar, bc_ratio) in \
                itertools.product(
                    enumerate(D_ve_grid),
                    enumerate(RI_real_grid),
                    enumerate(AR_grid)):

            idx3 = (i_dve, i_ri, i_ar)

            # Skip if already computed (converged flag is True)
            if grp['converged'][idx3]:
                continue

            r_v_base = D_ve / 2.0
            m_p = RI_real + 1j * M_IMAG
            m_p_xyz = np.array([m_p, m_p, m_p])  # isotropic

            # Build geometry
            print("=" * 64)
            try:
                gre_name, lattice_n, lattice_lf, grid, is_in = \
                    _build_spheroid_geometry(r_v_base, bc_ratio, wl_0, m_p_xyz)
            except (ValueError, IndexError):
                _log(f"[{grp_name}] D_ve={D_ve:.4f} um  RI={RI_real:.3f}  "
                     f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                     f"-- geometry failed, skipping")
                continue

            n_occ = int(np.sum(is_in))
            if n_occ == 0:
                _log(f"[{grp_name}] D_ve={D_ve:.4f} um  RI={RI_real:.3f}  "
                     f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                     f"-- no dipoles, skipping")
                continue

            _log(f"[{grp_name}] D_ve={D_ve:.4f} um  RI={RI_real:.3f}+{M_IMAG:.4f}j  "
                 f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                 f"d={lattice_lf:.5f} um  N_dip={n_occ}  L_solve={N_u_half}")

            target = Target(gre_name, lattice_n, lattice_lf,
                            grid, is_in, m_p_xyz, r_v_base)

            try:
                S_fw_theta, S_fw_phi, converged = \
                    _run_dda_spheroid(target, wl_0, m_m,
                                      cos_theta_o_half_grid, phi_o_grid)
            except KeyboardInterrupt:
                _log("Interrupted -- file closed cleanly.")
                raise SystemExit(0)

            # Write results
            N = slice(None)
            grp['S_fw_theta_re'][idx3 + (N, N)] = S_fw_theta.real
            grp['S_fw_theta_im'][idx3 + (N, N)] = S_fw_theta.imag
            grp['S_fw_phi_re'  ][idx3 + (N, N)] = S_fw_phi.real
            grp['S_fw_phi_im'  ][idx3 + (N, N)] = S_fw_phi.imag
            grp['converged'    ][idx3]           = converged

            _log(f"  converged={converged}  "
                 f"S_fw_t(mean)={np.nanmean(S_fw_theta):.4g}  "
                 f"S_fw_p(mean)={np.nanmean(S_fw_phi):.4g}")

_log("All conditions completed.")
