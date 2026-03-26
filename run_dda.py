import itertools
import datetime
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

# ── settings ──────────────────────────────────────────────────────────────────
RNG_SEED    = 12345   # GRE shape uses RNG_SEED; Euler angles use RNG_SEED + 1
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


def _generate_euler_angles(num_orientations, spheroid_mode):
    """Generate Euler angles for DDA orientations.

    Parameters
    ----------
    num_orientations : int
    spheroid_mode    : bool
        If True, alpha=0 and gamma=0; only beta is sampled.

    Returns
    -------
    euler_angles : ndarray, shape (num_orientations, 3)
        Columns: [alpha, beta, gamma] in radians.
    """
    rng = np.random.default_rng(RNG_SEED + 1)

    if spheroid_mode:
        # For spheroids, phi-average is analytical; sample only beta (polar angle).
        # Uniform distribution on sphere: cos(beta) ~ Uniform(-1, 1).
        cos_beta = rng.uniform(-1, 1, num_orientations)
        beta = np.arccos(cos_beta)
        euler_angles = np.column_stack([
            np.zeros(num_orientations),   # alpha = 0 (phi-average done analytically)
            beta,
            np.zeros(num_orientations),   # gamma irrelevant for spheroid (a=b)
        ])
    else:
        euler_angles = np.column_stack([
            rng.uniform(0, 2 * np.pi, num_orientations),   # alpha
            rng.uniform(0,     np.pi, num_orientations),   # beta
            rng.uniform(0, 2 * np.pi, num_orientations),   # gamma
        ])
    return euler_angles


def _run_dda(target, wl_0, m_m, num_orientations, spheroid_mode=False):
    """Attempt DDA solve up to MAX_TRY times with fresh orientations.

    Euler-angle rng uses RNG_SEED + 1 so orientations are reproducible and
    identical across all (wl_0, m_m, m_p_xyz) combinations for the same shape.

    When spheroid_mode is True, only beta (polar angle) is sampled.
    The phi-averaged forward scattering amplitudes are computed analytically:
        <S_s>_phi = <S_p>_phi = (S_fw_theta(alpha=0) + S_fw_phi(alpha=0)) / 2

    Returns (euler_angles, C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, converged).
    """
    for i_try in range(1, MAX_TRY + 1):
        euler_angles = _generate_euler_angles(num_orientations, spheroid_mode)
        inc = IncidentField(wl_0, m_m, euler_angles)
        dd  = DiscreteDipoles(target, inc)
        dd.set_interaction_matrix()
        dd.solve_matrix_equation()

        _log(f"    try {i_try}/{MAX_TRY}: {'converged ✓' if dd.converge else 'not converged'}")
        if dd.converge:
            return (euler_angles,
                    dd.compute_C_abs(), dd.compute_C_ext(),
                    *dd.compute_PCAS_observable_S_fw(),
                    dd.compute_OCBS_observable_S_bk(),
                    True)

    nan_r = np.full(num_orientations, np.nan)
    nan_c = np.full(num_orientations, np.nan + 0j)
    return euler_angles, nan_r, nan_r, nan_c, nan_c, nan_c, False


# ── main ──────────────────────────────────────────────────────────────────────
with h5py.File(OUTPUT_FILE, "r+") as h5:
    t  = h5['target']
    sd = t['simulated_data']

    num_orientations = int(t.attrs['num_orientations'])
    wl_m_m_pairs     = t['wl_m_m_pairs'][:]      # (N_pairs, 2): columns = [wl_0, m_m]
    m_p_xyz_list     = t['m_p_xyz_list'][:]       # (N_m_p,  3): particle refractive index
    r_v_base_list    = t['r_v_base_list'][:]
    bc_ratio_list    = t['bc_ratio_list'][:]
    ab_ratio_list    = t['ab_ratio_list'][:]
    gre_beta_list    = t['gre_beta_list'][:]

    for (i_rv, r_v_base), (i_bc, bc_ratio), (i_ab, ab_ratio), (i_bt, gre_beta) in \
            itertools.product(enumerate(r_v_base_list), enumerate(bc_ratio_list),
                              enumerate(ab_ratio_list), enumerate(gre_beta_list)):

        shape_idx4 = (i_rv, i_bc, i_ab, i_bt)
        sd['r_ve'][shape_idx4] = r_v_base
        spheroid_mode = _is_spheroid(ab_ratio, gre_beta)

        for i_pair, (wl_0, m_m) in enumerate(wl_m_m_pairs):

            for i_mp, m_p_xyz in enumerate(m_p_xyz_list):
                idx6 = (i_pair, i_mp) + shape_idx4

                # Skip if already computed (S_fw_PCAS_mie is non-zero imag when done)
                if sd['S_fw_PCAS_mie'][idx6].imag != 0.0:
                    _log(f"Skip: pair={i_pair} m_p={i_mp} shape={shape_idx4} (already computed)")
                    continue

                # Build GRE geometry (lattice spacing depends on wl_0 and m_p_xyz via dpl)
                gre_name, lattice_n, lattice_lf, grid, is_in = \
                    _build_gre_geometry(r_v_base, bc_ratio, ab_ratio, gre_beta, wl_0, m_p_xyz)

                print("─" * 64)
                mode_tag = " [spheroid]" if spheroid_mode else ""
                _log(f"wl_0={wl_0:.4f} μm  m_m={m_m:.4f}  m_p_xyz={m_p_xyz}  |  "
                     f"r_v_base={r_v_base:.3f}  bc={bc_ratio:.1f}  "
                     f"ab={ab_ratio:.1f}  β={gre_beta:.2f}  "
                     f"d={lattice_lf:.5f} μm  N_ori={num_orientations}{mode_tag}")

                target = Target(gre_name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, r_v_base)

                try:
                    euler_angles, C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, _ = \
                        _run_dda(target, wl_0, m_m, num_orientations, spheroid_mode)
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
