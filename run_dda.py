"""General DDA parameter sweep over the lists stored in the output HDF5 file.

The file is created by dda_results/create_h5py.ipynb. It holds the lists of
(wl_0, m_m) pairs, particle refractive indices m_p_xyz, volume-equivalent radii
and GRE shape parameters, and the orientation grid. This script loops over all
combinations and fills in the results. A finished condition is skipped, so the
run can be interrupted and resumed.

Options:
  --output PATH   output HDF5 file (default: OUTPUT_FILE below)
  --shifted-ri    solve all ISOTROPIC entries of m_p_xyz_list of a
                  (shape, wavelength) slot with ONE shifted block-GMRES process.
                  They then share the lattice built for the largest |m| among the
                  isotropic entries of the list, so their values differ from the
                  default mode at the level of the discretisation error. The
                  anisotropic entries are solved one at a time, as in the default
                  mode. The two modes cannot be mixed in one output file.
"""
import argparse
import itertools
import datetime
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

# ── settings ──────────────────────────────────────────────────────────────────
RNG_SEED    = 12345   # GRE shape uses RNG_SEED
MAX_TRY     = 4       # max DDA retries when solver does not converge
OUTPUT_FILE = "dda_results/pcas_ocbs_simulated_data.hdf5"

_parser = argparse.ArgumentParser(description="block-DDA_Py general parameter sweep")
_parser.add_argument("--output", default=None, help="override the output HDF5 path")
_parser.add_argument("--shifted-ri", action="store_true",
                     help="solve the isotropic entries of m_p_xyz_list of a slot with one "
                          "shifted block-GMRES process on a common lattice (built for the "
                          "largest |m| among them). Use for NEW output files only.")
_cli, _ = _parser.parse_known_args()
if _cli.output is not None:
    OUTPUT_FILE = _cli.output
SHIFTED_RI = bool(_cli.shifted_ri)
# ──────────────────────────────────────────────────────────────────────────────


def _log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}")


def _build_gre_geometry(r_v_base, bc_ratio, ab_ratio, gre_beta, wl_0, m_p_xyz):
    """Build GRE lattice geometry (deterministic: uses RNG_SEED).

    Returns (name, lattice_n, lattice_lf, grid, is_in).
    Lattice spacing is set by dpl; depends on wl_0 and m_p_xyz.
    """
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(r_v_base, bc_ratio, ab_ratio, gre_beta,
                                         wl_0, m_p_xyz)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist  = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return gre.name, lattice_n, gre.lattice_lf, grid, is_in


def _is_spheroid(ab_ratio, gre_beta):
    """Detect spheroid condition: a=b (ab_ratio==1) and smooth surface (beta==0)."""
    return ab_ratio == 1.0 and gre_beta == 0.0


def _is_isotropic(m_p_xyz):
    """True if the three principal refractive indices are equal."""
    return m_p_xyz[0] == m_p_xyz[1] == m_p_xyz[2]


def _generate_euler_grid(N_alpha, N_beta, N_gamma):
    """Generate deterministic Euler angle grid, uniform on SO(3).

    alpha: equally spaced in [0, 2pi)
    beta:  cos(beta) equally spaced in (-1, 1) (equal-area midpoints)
    gamma: equally spaced in [0, 2pi)

    Returns
    -------
    euler_angles : ndarray, shape (N_alpha * N_beta * N_gamma, 3)
        Columns: [alpha, beta, gamma] in radians.
        Ordering: alpha slowest, gamma fastest.
    """
    alpha = np.linspace(0, 2 * np.pi, N_alpha, endpoint=False)
    cos_beta = np.linspace(1 - 1 / N_beta, -1 + 1 / N_beta, N_beta)
    beta = np.arccos(cos_beta)
    gamma = np.linspace(0, 2 * np.pi, N_gamma, endpoint=False)

    aa, bb, gg = np.meshgrid(alpha, beta, gamma, indexing='ij')
    return np.column_stack([aa.ravel(), bb.ravel(), gg.ravel()])


def _run_dda_general(target, wl_0, m_m, euler_angles):
    """DDA solve for all orientations (general particles).

    Returns (C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, converged).
    """
    num_ori = euler_angles.shape[0]
    for i_try in range(1, MAX_TRY + 1):
        inc = IncidentField(wl_0, m_m, euler_angles)
        dd  = DiscreteDipoles(target, inc)
        dd.set_interaction_matrix()
        dd.solve_matrix_equation()

        _log(f"    try {i_try}/{MAX_TRY}: {'converged ✓' if dd.converge else 'not converged'}")
        if dd.converge:
            return (dd.compute_C_abs(), dd.compute_C_ext(),
                    *dd.compute_PCAS_observable_S_fw(),
                    dd.compute_OCBS_observable_S_bk(),
                    True)

    nan_r = np.full(num_ori, np.nan)
    nan_c = np.full(num_ori, np.nan + 0j)
    return nan_r, nan_r, nan_c, nan_c, nan_c, False


def _run_dda_spheroid(target, wl_0, m_m, N_alpha, N_beta, N_gamma):
    """DDA solve for spheroids: solve only N_beta orientations at alpha=0,
    then fill the full (N_alpha, N_beta, N_gamma) grid analytically.

    For forward scattering:
        S_s(alpha) = A + B * exp(i2*alpha)
        S_p(alpha) = A - B * exp(i2*alpha)
    where A = (S_s(0) + S_p(0))/2,  B = (S_s(0) - S_p(0))/2.

    For backward scattering (OCBS):
        S_bk(alpha) = S_bk(0) * exp(i2*alpha)

    C_ext and C_abs are alpha-invariant (verified in test_spheroid_symmetry.ipynb).

    Returns (C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, converged).
    Each array has shape (N_alpha * N_beta * N_gamma,).
    """
    num_full = N_alpha * N_beta * N_gamma

    # Beta-only grid for DDA solve
    euler_beta_only = _beta_only_euler_grid(N_beta)   # shape (N_beta, 3)

    for i_try in range(1, MAX_TRY + 1):
        inc = IncidentField(wl_0, m_m, euler_beta_only)
        dd  = DiscreteDipoles(target, inc)
        dd.set_interaction_matrix()
        dd.solve_matrix_equation()

        _log(f"    try {i_try}/{MAX_TRY} [spheroid, L={N_beta}]: "
             f"{'converged ✓' if dd.converge else 'not converged'}")
        if dd.converge:
            break
    else:
        nan_r = np.full(num_full, np.nan)
        nan_c = np.full(num_full, np.nan + 0j)
        return nan_r, nan_r, nan_c, nan_c, nan_c, False

    # Compute observables at alpha=0 for each beta
    S_s_0, S_p_0 = dd.compute_PCAS_observable_S_fw()   # shape (N_beta,)
    C_abs_0 = dd.compute_C_abs()                        # shape (N_beta,)
    C_ext_0 = dd.compute_C_ext()                        # shape (N_beta,)
    S_bk_0  = dd.compute_OCBS_observable_S_bk()         # shape (N_beta,)

    return (*_expand_spheroid_grid(C_abs_0, C_ext_0, S_s_0, S_p_0, S_bk_0,
                                   N_alpha, N_beta, N_gamma),
            True)


def _beta_only_euler_grid(N_beta):
    """Euler angles (alpha=0, beta, gamma=0) of the spheroid mode, shape (N_beta, 3) [rad]."""
    cos_beta = np.linspace(1 - 1 / N_beta, -1 + 1 / N_beta, N_beta)
    beta = np.arccos(cos_beta)
    return np.column_stack([
        np.zeros(N_beta), beta, np.zeros(N_beta),
    ])  # shape (N_beta, 3)


def _expand_spheroid_grid(C_abs_0, C_ext_0, S_s_0, S_p_0, S_bk_0, N_alpha, N_beta, N_gamma):
    """Fill the full (N_alpha, N_beta, N_gamma) grid from the alpha=0 values of a spheroid.

    Inputs have shape (N_beta,). See ``_run_dda_spheroid`` for the relations.
    Returns (C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk), each of shape
    (N_alpha * N_beta * N_gamma,).
    """
    # Analytical expansion to full grid
    alpha = np.linspace(0, 2 * np.pi, N_alpha, endpoint=False)
    exp_2a = np.exp(2j * alpha)   # shape (N_alpha,)

    A_fw = (S_s_0 + S_p_0) / 2   # shape (N_beta,)
    B_fw = (S_s_0 - S_p_0) / 2   # shape (N_beta,)

    # S_s[i_a, i_b, i_g] = A[i_b] + B[i_b] * exp(2j*alpha[i_a])
    S_s_3d = (A_fw[np.newaxis, :, np.newaxis]
              + B_fw[np.newaxis, :, np.newaxis] * exp_2a[:, np.newaxis, np.newaxis])
    S_p_3d = (A_fw[np.newaxis, :, np.newaxis]
              - B_fw[np.newaxis, :, np.newaxis] * exp_2a[:, np.newaxis, np.newaxis])
    S_bk_3d = S_bk_0[np.newaxis, :, np.newaxis] * exp_2a[:, np.newaxis, np.newaxis]

    # C_ext and C_abs are alpha-invariant
    C_ext_3d = C_ext_0[np.newaxis, :, np.newaxis]
    C_abs_3d = C_abs_0[np.newaxis, :, np.newaxis]

    # Broadcast to (N_alpha, N_beta, N_gamma) and flatten
    shape_full = (N_alpha, N_beta, N_gamma)
    return (np.broadcast_to(C_abs_3d, shape_full).ravel(),
            np.broadcast_to(C_ext_3d, shape_full).ravel(),
            np.broadcast_to(S_s_3d,   shape_full).ravel(),
            np.broadcast_to(S_p_3d,   shape_full).ravel(),
            np.broadcast_to(S_bk_3d,  shape_full).ravel())


def _run_dda_shifted(target, wl_0, m_m, m_list, euler_angles, spheroid_mode,
                     N_alpha, N_beta, N_gamma):
    """One shifted block-GMRES solve for all refractive indices of ``m_list``.

    ``target`` holds the common lattice, built for the largest |m| of the
    isotropic entries of the sweep. In spheroid mode only the beta-only grid is
    solved and the full orientation grid is filled analytically, as in
    ``_run_dda_spheroid``.

    Parameters
    ----------
    m_list : ndarray, shape (N_m,), complex   isotropic refractive indices

    Returns
    -------
    list of N_m tuples (C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, converged), each
    array of shape (num_orientations,). A refractive index that did not converge
    gets NaN arrays, as in the default mode.
    """
    euler_solve = _beta_only_euler_grid(N_beta) if spheroid_mode else euler_angles
    inc = IncidentField(wl_0, m_m, euler_solve)
    dd  = DiscreteDipoles(target, inc)
    dd.set_interaction_matrix_shifted(m_list)
    dd.solve_matrix_equation_shifted()
    obs = dd.compute_observables_shifted()            # dict of arrays (N_m, L_solve)
    _log(f"    shifted solve [L={euler_solve.shape[0]}]: "
         f"{int(dd.converge_shifted.sum())}/{len(m_list)} refractive indices converged")

    num_full = N_alpha * N_beta * N_gamma
    results = []
    for s in range(len(m_list)):
        if not dd.converge_shifted[s]:
            nan_r = np.full(num_full, np.nan)
            nan_c = np.full(num_full, np.nan + 0j)
            results.append((nan_r, nan_r, nan_c, nan_c, nan_c, False))
            continue
        values = (obs["C_abs"][s], obs["C_ext"][s], obs["S_fw_PCAS_theta"][s],
                  obs["S_fw_PCAS_phi"][s], obs["S_bk_OCBS"][s])
        if spheroid_mode:
            values = _expand_spheroid_grid(*values, N_alpha, N_beta, N_gamma)
        results.append((*values, True))
    return results


def _write_condition(sd, idx6, euler_angles, C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk,
                     wl_0, m_m, r_v_base, m_p_xyz):
    """Write the results of one condition and its Mie reference to the HDF5 group ``sd``."""
    # Mie reference for volume-equivalent sphere
    m_p_avg = complex(np.mean(m_p_xyz))
    _, Q_abs_mie, Q_ext_mie, S_fw_mie, S_bk_mie = \
        mie_compute_q_and_s(wl_0, m_m, r_v_base, m_p_avg, nang=3)
    C_abs_mie = Q_abs_mie * np.pi * r_v_base**2
    C_ext_mie = Q_ext_mie * np.pi * r_v_base**2

    # Write results to HDF5
    N = slice(None)
    sd['Euler_angles'   ][idx6 + (N, N)] = euler_angles
    sd['C_abs'          ][idx6 + (N,)]   = C_abs
    sd['C_ext'          ][idx6 + (N,)]   = C_ext
    sd['S_fw_PCAS_theta'][idx6 + (N,)]   = S_fw_theta
    sd['S_fw_PCAS_phi'  ][idx6 + (N,)]   = S_fw_phi
    sd['S_bk_OCBS'      ][idx6 + (N,)]   = S_bk
    sd['C_abs_mie'      ][idx6]           = C_abs_mie
    sd['C_ext_mie'      ][idx6]           = C_ext_mie
    sd['S_fw_PCAS_mie'  ][idx6]           = S_fw_mie
    sd['S_bk_OCBS_mie'  ][idx6]           = S_bk_mie

    _log(f"  C_ext(mean)={np.nanmean(C_ext):.4e}  "
         f"S_fw_θ(mean)={np.nanmean(S_fw_theta):.4g}  "
         f"S_fw_φ(mean)={np.nanmean(S_fw_phi):.4g}  "
         f"S_bk(mean)={np.nanmean(S_bk):.4g}")


# ── main ──────────────────────────────────────────────────────────────────────
with h5py.File(OUTPUT_FILE, "r+") as h5:
    t  = h5['target']
    sd = t['simulated_data']

    N_alpha = int(t.attrs['N_alpha_ori'])
    N_beta  = int(t.attrs['N_beta_ori'])
    N_gamma = int(t.attrs['N_gamma_ori'])
    num_orientations = N_alpha * N_beta * N_gamma

    wl_m_m_pairs     = t['wl_m_m_pairs'][:]
    m_p_xyz_list     = t['m_p_xyz_list'][:]
    r_v_base_list    = t['r_v_base_list'][:]
    bc_ratio_list    = t['bc_ratio_list'][:]
    ab_ratio_list    = t['ab_ratio_list'][:]
    gre_beta_list    = t['gre_beta_list'][:]

    # Full Euler angle grid (same for all conditions)
    euler_angles = _generate_euler_grid(N_alpha, N_beta, N_gamma)

    # A file keeps the mode it was started with: the two modes use different lattices
    # for the isotropic refractive indices.
    file_shifted = bool(t.attrs.get('shifted_ri', False))
    if SHIFTED_RI != file_shifted and np.any(sd['S_fw_PCAS_mie'][...].imag != 0.0):
        raise SystemExit(
            f"{OUTPUT_FILE} holds results of the "
            f"{'shifted' if file_shifted else 'default'} mode and cannot be resumed "
            f"{'with' if SHIFTED_RI else 'without'} --shifted-ri. Use a new output file.")
    if SHIFTED_RI:
        t.attrs['shifted_ri']     = True
        t.attrs['lattice_policy'] = 'max_abs_m_of_isotropic_entries'
        iso_idx = [i for i, m in enumerate(m_p_xyz_list) if _is_isotropic(m)]
        # Lattice-defining index: fixed by the whole list, so a resumed run uses the
        # same lattice as an uninterrupted one.
        m_lattice = (max((m_p_xyz_list[i][0] for i in iso_idx), key=abs)
                     if iso_idx else None)

    for (i_rv, r_v_base), (i_bc, bc_ratio), (i_ab, ab_ratio), (i_bt, gre_beta) in \
            itertools.product(enumerate(r_v_base_list), enumerate(bc_ratio_list),
                              enumerate(ab_ratio_list), enumerate(gre_beta_list)):

        shape_idx4 = (i_rv, i_bc, i_ab, i_bt)
        sd['r_ve'][shape_idx4] = r_v_base
        spheroid_mode = _is_spheroid(ab_ratio, gre_beta)

        for i_pair, (wl_0, m_m) in enumerate(wl_m_m_pairs):

            if SHIFTED_RI:
                # All isotropic refractive indices of this slot that are not computed yet.
                pending = [i for i in iso_idx
                           if sd['S_fw_PCAS_mie'][(i_pair, i) + shape_idx4].imag == 0.0]
                if pending:
                    m_list = np.array([m_p_xyz_list[i][0] for i in pending])   # shape (N_m,)
                    m_lat_xyz = np.array([m_lattice, m_lattice, m_lattice])
                    gre_name, lattice_n, lattice_lf, grid, is_in = \
                        _build_gre_geometry(r_v_base, bc_ratio, ab_ratio, gre_beta, wl_0, m_lat_xyz)

                    print("─" * 64)
                    mode_tag = f", spheroid, L_solve={N_beta}" if spheroid_mode else ""
                    _log(f"wl_0={wl_0:.4f} μm  m_m={m_m:.4f}  m_p={m_list}  |  "
                         f"r_v_base={r_v_base:.3f}  bc={bc_ratio:.1f}  "
                         f"ab={ab_ratio:.1f}  β={gre_beta:.2f}  "
                         f"d={lattice_lf:.5f} μm  N_ori={num_orientations} "
                         f"[shifted, lattice for m={m_lattice}{mode_tag}]")

                    target = Target(gre_name, lattice_n, lattice_lf, grid, is_in,
                                    m_lat_xyz, r_v_base)
                    try:
                        results = _run_dda_shifted(target, wl_0, m_m, m_list, euler_angles,
                                                   spheroid_mode, N_alpha, N_beta, N_gamma)
                    except KeyboardInterrupt:
                        _log("Interrupted – file closed cleanly.")
                        raise SystemExit(0)

                    for i_mp, (C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, _) in zip(pending, results):
                        _write_condition(sd, (i_pair, i_mp) + shape_idx4, euler_angles,
                                         C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk,
                                         wl_0, m_m, r_v_base, m_p_xyz_list[i_mp])

            for i_mp, m_p_xyz in enumerate(m_p_xyz_list):
                idx6 = (i_pair, i_mp) + shape_idx4

                # Skip if already computed
                if sd['S_fw_PCAS_mie'][idx6].imag != 0.0:
                    _log(f"Skip: pair={i_pair} m_p={i_mp} shape={shape_idx4} (already computed)")
                    continue

                # Build GRE geometry
                gre_name, lattice_n, lattice_lf, grid, is_in = \
                    _build_gre_geometry(r_v_base, bc_ratio, ab_ratio, gre_beta, wl_0, m_p_xyz)

                print("─" * 64)
                mode_tag = f" [spheroid, L_solve={N_beta}]" if spheroid_mode else ""
                _log(f"wl_0={wl_0:.4f} μm  m_m={m_m:.4f}  m_p_xyz={m_p_xyz}  |  "
                     f"r_v_base={r_v_base:.3f}  bc={bc_ratio:.1f}  "
                     f"ab={ab_ratio:.1f}  β={gre_beta:.2f}  "
                     f"d={lattice_lf:.5f} μm  N_ori={num_orientations}{mode_tag}")

                target = Target(gre_name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, r_v_base)

                try:
                    if spheroid_mode:
                        C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, _ = \
                            _run_dda_spheroid(target, wl_0, m_m, N_alpha, N_beta, N_gamma)
                    else:
                        C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, _ = \
                            _run_dda_general(target, wl_0, m_m, euler_angles)
                except KeyboardInterrupt:
                    _log("Interrupted – file closed cleanly.")
                    raise SystemExit(0)

                _write_condition(sd, idx6, euler_angles, C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk,
                                 wl_0, m_m, r_v_base, m_p_xyz)
