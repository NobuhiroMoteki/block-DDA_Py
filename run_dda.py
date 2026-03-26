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
    cos_beta = np.linspace(1 - 1 / N_beta, -1 + 1 / N_beta, N_beta)
    beta = np.arccos(cos_beta)
    euler_beta_only = np.column_stack([
        np.zeros(N_beta), beta, np.zeros(N_beta),
    ])  # shape (N_beta, 3)

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
            np.broadcast_to(S_bk_3d,  shape_full).ravel(),
            True)


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

    for (i_rv, r_v_base), (i_bc, bc_ratio), (i_ab, ab_ratio), (i_bt, gre_beta) in \
            itertools.product(enumerate(r_v_base_list), enumerate(bc_ratio_list),
                              enumerate(ab_ratio_list), enumerate(gre_beta_list)):

        shape_idx4 = (i_rv, i_bc, i_ab, i_bt)
        sd['r_ve'][shape_idx4] = r_v_base
        spheroid_mode = _is_spheroid(ab_ratio, gre_beta)

        for i_pair, (wl_0, m_m) in enumerate(wl_m_m_pairs):

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
