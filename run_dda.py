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


def _build_gre_geometry(r_v_base, bc_ratio, ab_ratio, gre_beta):
    """Build GRE lattice geometry (deterministic: uses RNG_SEED).

    Returns (name, lattice_n, lattice_lf, grid, is_in, r_ve).
    Independent of m_p_xyz, wl_0, m_m — called once per shape combination.
    """
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(r_v_base, bc_ratio, ab_ratio, gre_beta)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist  = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    r_ve = np.cbrt(3.0 * gre.lattice_lf**3 * int(is_in.sum()) / (4 * np.pi))
    return gre.name, lattice_n, gre.lattice_lf, grid, is_in, r_ve


def _run_dda(target, wl_0, m_m, num_orientations):
    """Attempt DDA solve up to MAX_TRY times with fresh random orientations.

    Euler-angle rng uses RNG_SEED + 1 so orientations are reproducible and
    identical across all (wl_0, m_m, m_p_xyz) combinations for the same shape.

    Returns (euler_angles, C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, converged).
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    euler_angles = None

    for i_try in range(1, MAX_TRY + 1):
        euler_angles = np.column_stack([
            rng.uniform(0, 2 * np.pi, num_orientations),   # alpha
            rng.uniform(0,     np.pi, num_orientations),   # beta
            rng.uniform(0, 2 * np.pi, num_orientations),   # gamma
        ])
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

        # Build GRE geometry once per shape (independent of wl_0, m_m, m_p_xyz)
        gre_name, lattice_n, lattice_lf, grid, is_in, r_ve = \
            _build_gre_geometry(r_v_base, bc_ratio, ab_ratio, gre_beta)
        sd['r_ve'][shape_idx4] = r_ve

        for i_pair, (wl_0, m_m) in enumerate(wl_m_m_pairs):

            for i_mp, m_p_xyz in enumerate(m_p_xyz_list):
                idx6 = (i_pair, i_mp) + shape_idx4

                # Skip if already computed (S_fw_PCAS_mie is non-zero imag when done)
                if sd['S_fw_PCAS_mie'][idx6].imag != 0.0:
                    _log(f"Skip: pair={i_pair} m_p={i_mp} shape={shape_idx4} (already computed)")
                    continue

                print("─" * 64)
                _log(f"wl_0={wl_0:.4f} μm  m_m={m_m:.4f}  m_p_xyz={m_p_xyz}  |  "
                     f"r_v_base={r_v_base:.3f}  bc={bc_ratio:.1f}  "
                     f"ab={ab_ratio:.1f}  β={gre_beta:.2f}  "
                     f"r_ve={r_ve:.4f} μm  N_ori={num_orientations}")

                # Target depends on GRE geometry + m_p_xyz (not wl_0, m_m)
                target = Target(gre_name, lattice_n, lattice_lf, grid, is_in, m_p_xyz)

                try:
                    euler_angles, C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk, _ = \
                        _run_dda(target, wl_0, m_m, num_orientations)
                except KeyboardInterrupt:
                    _log("Interrupted – file closed cleanly.")
                    raise SystemExit(0)

                # Mie reference for volume-equivalent sphere
                m_p_avg = complex(np.mean(m_p_xyz))
                _, Q_abs_mie, Q_ext_mie, S_fw_mie, S_bk_mie = \
                    mie_compute_q_and_s(wl_0, m_m, r_ve, m_p_avg, nang=3)
                C_abs_mie = Q_abs_mie * np.pi * r_ve**2
                C_ext_mie = Q_ext_mie * np.pi * r_ve**2

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
