"""Paper-production sweep runner for block-DDA_Py (CLAUDE.md §10-8).

Reads a paper sweep HDF5 file (created by `dda_results/paper/{shape}_{material}.py`),
iterates over every (wl × m_p × r_v × bc × ab × β) slot, and fills in the
`/target/simulated_data/` datasets in place. Skips slots where
`S_fw_PCAS_mie.imag != 0` so interrupted runs can be resumed.

This is the paper-grade analogue of `run_dda.py`:
    - calls `bl_gmres_mvp_fft` (VIEM-parity, unpreconditioned) directly
    - unified convergence criterion tol=1e-5, maxiter=100 (CLAUDE.md §7.5)
    - supports both `shape_kind="gre"` (GRE family via gaussian_ellipsoid)
      and `shape_kind="doublet"` (two_sphere_cluster) via the HDF5 attr
    - supports spheroid-mode α-expansion when the shape is axially
      symmetric about z (attr `spheroid_mode=1`), solving only L=N_β
      orientations and filling the full (N_α·N_β·N_γ) grid analytically
    - non-convergence → NaN observables + `S_fw_PCAS_mie` intentionally
      left unwritten so the resume logic retries (or a future run with
      different tol/maxiter picks it up)

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_paper_sweep.py <sweep.hdf5>
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import os
import sys
import time
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from shape_model.two_sphere_cluster import two_sphere_cluster_shape_model
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles
from bl_krylov.bl_krylov import bl_gmres_mvp_fft
from utils.rss_monitor import RSSMonitor
from utils.dpl_calibration import (
    dpl_for_slot, material_label, shape_label, get_n_tet_target,
)


RNG_SEED    = 12345
SOLVER_TOL  = 1e-5
MAXITER     = 200   # v0.7.6+ (VIEM-synced): raised from 100 to allow paper
                    # sweeps with heavier index contrast to reach tol.


def _log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _decode_attr(val, default):
    if val is None:
        return default
    if isinstance(val, bytes):
        return val.decode()
    return str(val)


def _build_gre_target(r_v, bc, ab, beta, wl_0, m_p_xyz, dpl):
    gre = gaussian_ellipsoid_shape_model(r_v, bc, ab, beta, wl_0, m_p_xyz, dpl=dpl)
    rng = np.random.default_rng(RNG_SEED)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return Target(gre.name, lattice_n, gre.lattice_lf, grid, is_in, m_p_xyz, r_v)


def _build_doublet_target(r_v, wl_0, m_p_xyz, dpl):
    mdl = two_sphere_cluster_shape_model(r_v, wl_0, m_p_xyz, dpl=dpl)
    lattice_n, lf, grid, is_in = mdl.build()
    return Target(mdl.name, lattice_n, lf, grid, is_in, m_p_xyz, r_v)


def _build_target(shape_kind, r_v, bc, ab, beta, wl_0, m_p_xyz, dpl):
    if shape_kind == "doublet":
        return _build_doublet_target(r_v, wl_0, m_p_xyz, dpl)
    return _build_gre_target(r_v, bc, ab, beta, wl_0, m_p_xyz, dpl)


def _generate_euler_grid(N_alpha, N_beta, N_gamma):
    alpha = np.linspace(0, 2 * np.pi, N_alpha, endpoint=False)
    cos_beta = np.linspace(1 - 1 / N_beta, -1 + 1 / N_beta, N_beta)
    beta = np.arccos(cos_beta)
    gamma = np.linspace(0, 2 * np.pi, N_gamma, endpoint=False)
    aa, bb, gg = np.meshgrid(alpha, beta, gamma, indexing='ij')
    return np.column_stack([aa.ravel(), bb.ravel(), gg.ravel()])


def _solve_observables(target, inc):
    """Set up A and B, call GMRES, compute observables.

    Returns a dict with arrays (C_abs, C_ext, S_fw_theta, S_fw_phi, S_bk)
    of length inc.L, plus diagnostics (converged, iters, err) and per-stage
    timings (t_setup_s, t_solve_s).
    """
    t0 = time.time()
    dd = DiscreteDipoles(target, inc)
    dd.set_interaction_matrix()
    t_setup = time.time() - t0

    t0 = time.time()
    X, iter_fin, err_fin, err_history = bl_gmres_mvp_fft(
        dd.lattice_n, dd.f, dd.lattice_address_in_target,
        dd.Au_til, dd.diag_A, dd.B,
        SOLVER_TOL, MAXITER,
    )
    t_solve = time.time() - t0

    converged = bool(err_fin < SOLVER_TOL)

    if not converged:
        nan_r = np.full(dd.L, np.nan)
        nan_c = np.full(dd.L, np.nan + 0j)
        return dict(C_abs=nan_r, C_ext=nan_r,
                    S_fw_theta=nan_c, S_fw_phi=nan_c, S_bk=nan_c,
                    converged=False, iters=int(iter_fin + 1),
                    err=float(err_fin), err_history=err_history,
                    t_setup_s=t_setup, t_solve_s=t_solve)

    dd.X = X
    dd.P = dd.X.T.reshape(dd.L, dd.num_element_occupy, 3)
    dd.E = (dd.P * (4 * np.pi)
            / ((dd.eper_r[np.newaxis, :, :] - 1) * dd.element_vol))
    C_abs = dd.compute_C_abs()
    C_ext = dd.compute_C_ext()
    S_fw_theta, S_fw_phi = dd.compute_PCAS_observable_S_fw()
    S_bk = dd.compute_OCBS_observable_S_bk()
    return dict(C_abs=C_abs, C_ext=C_ext,
                S_fw_theta=S_fw_theta, S_fw_phi=S_fw_phi, S_bk=S_bk,
                converged=True, iters=int(iter_fin + 1),
                err=float(err_fin), err_history=err_history,
                t_setup_s=t_setup, t_solve_s=t_solve)


def _solve_spheroid(target, wl_0, m_m, N_alpha, N_beta, N_gamma):
    """Spheroid α-expansion: solve L=N_β orientations at α=0, then expand
    analytically to the full (N_α, N_β, N_γ) grid.

    For forward scattering:
        S_s(α) = A + B · exp(2iα),  S_p(α) = A − B · exp(2iα)
        with A = (S_s(0)+S_p(0))/2, B = (S_s(0)−S_p(0))/2.
    For backward (OCBS): S_bk(α) = S_bk(0) · exp(2iα).
    C_ext and C_abs are α-invariant.

    Returns a dict with the same keys as `_solve_observables` but the
    observable arrays are expanded to length N_α·N_β·N_γ.
    """
    num_full = N_alpha * N_beta * N_gamma

    cos_beta = np.linspace(1 - 1 / N_beta, -1 + 1 / N_beta, N_beta)
    beta = np.arccos(cos_beta)
    euler_beta_only = np.column_stack([
        np.zeros(N_beta), beta, np.zeros(N_beta),
    ])
    inc = IncidentField(wl_0, m_m, euler_beta_only)

    r = _solve_observables(target, inc)

    if not r["converged"]:
        nan_r = np.full(num_full, np.nan)
        nan_c = np.full(num_full, np.nan + 0j)
        r.update(C_abs=nan_r, C_ext=nan_r,
                 S_fw_theta=nan_c, S_fw_phi=nan_c, S_bk=nan_c)
        return r

    alpha = np.linspace(0, 2 * np.pi, N_alpha, endpoint=False)
    exp_2a = np.exp(2j * alpha)

    S_s_0 = r["S_fw_theta"]
    S_p_0 = r["S_fw_phi"]
    A_fw = (S_s_0 + S_p_0) / 2
    B_fw = (S_s_0 - S_p_0) / 2

    S_s_3d = (A_fw[np.newaxis, :, np.newaxis]
              + B_fw[np.newaxis, :, np.newaxis] * exp_2a[:, np.newaxis, np.newaxis])
    S_p_3d = (A_fw[np.newaxis, :, np.newaxis]
              - B_fw[np.newaxis, :, np.newaxis] * exp_2a[:, np.newaxis, np.newaxis])
    S_bk_3d = r["S_bk"][np.newaxis, :, np.newaxis] * exp_2a[:, np.newaxis, np.newaxis]

    C_ext_3d = r["C_ext"][np.newaxis, :, np.newaxis]
    C_abs_3d = r["C_abs"][np.newaxis, :, np.newaxis]

    shape_full = (N_alpha, N_beta, N_gamma)
    r.update(
        C_abs=np.broadcast_to(C_abs_3d, shape_full).ravel().copy(),
        C_ext=np.broadcast_to(C_ext_3d, shape_full).ravel().copy(),
        S_fw_theta=np.broadcast_to(S_s_3d, shape_full).ravel().copy(),
        S_fw_phi=np.broadcast_to(S_p_3d, shape_full).ravel().copy(),
        S_bk=np.broadcast_to(S_bk_3d, shape_full).ravel().copy(),
    )
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5path", help="paper sweep HDF5 (overwrites /simulated_data)")
    args = ap.parse_args()

    if not os.path.isfile(args.h5path):
        print(f"error: HDF5 file not found: {args.h5path}", file=sys.stderr)
        sys.exit(2)

    _log(f"Opening {args.h5path}  (paper sweep runner, solver=bl_gmres_mvp_fft)")
    _log(f"  tol={SOLVER_TOL}  maxiter={MAXITER}")

    with h5py.File(args.h5path, "r+") as h5:
        t  = h5["target"]
        sd = t["simulated_data"]
        if "cost" not in t:
            _log("ERROR: /target/cost/ group missing. "
                 "This HDF5 was created with an older _common.py — "
                 "regenerate via the per-shape creator script and re-run.")
            raise SystemExit(2)
        cst = t["cost"]

        N_alpha = int(t.attrs["N_alpha_ori"])
        N_beta  = int(t.attrs["N_beta_ori"])
        N_gamma = int(t.attrs["N_gamma_ori"])
        num_orientations = N_alpha * N_beta * N_gamma

        shape_kind = _decode_attr(t.attrs.get("shape_kind"), "gre")
        spheroid_mode_file = int(t.attrs.get("spheroid_mode", 0))

        wl_m_m_pairs  = t["wl_m_m_pairs"][:]
        m_p_xyz_list  = t["m_p_xyz_list"][:]
        r_v_base_list = t["r_v_base_list"][:]
        bc_ratio_list = t["bc_ratio_list"][:]
        ab_ratio_list = t["ab_ratio_list"][:]
        gre_beta_list = t["gre_beta_list"][:]

        euler_angles = _generate_euler_grid(N_alpha, N_beta, N_gamma)

        monitor = RSSMonitor(interval=0.2).start()
        try:
            t_sweep_start = time.time()

            for (i_rv, r_v), (i_bc, bc), (i_ab, ab), (i_bt, bt) in \
                    itertools.product(enumerate(r_v_base_list),
                                      enumerate(bc_ratio_list),
                                      enumerate(ab_ratio_list),
                                      enumerate(gre_beta_list)):

                shape_idx4 = (i_rv, i_bc, i_ab, i_bt)
                sd["r_ve"][shape_idx4] = r_v

                # Spheroid mode: doublet (axis-z) or GRE with ab_ratio=1 & β=0
                spheroid_mode = (shape_kind == "doublet") or (ab == 1.0 and bt == 0.0)
                if spheroid_mode_file and not spheroid_mode:
                    spheroid_mode = True

                for i_pair, (wl_0, m_m) in enumerate(wl_m_m_pairs):
                    for i_mp, m_p_xyz in enumerate(m_p_xyz_list):
                        idx6 = (i_pair, i_mp) + shape_idx4

                        # Resume flag: S_fw_mie.imag != 0 ⇒ already computed.
                        if sd["S_fw_PCAS_mie"][idx6].imag != 0.0:
                            _log(f"Skip: pair={i_pair} m_p={i_mp} "
                                 f"shape={shape_idx4} (already computed)")
                            continue

                        # Auto-dpl: pick dpl so that DDA's N_occ matches
                        # VIEM's n_tet for this (shape, material, a_eq) slot
                        # within 1.5× (CLAUDE.md §3, v0.7.6+).
                        shape_lbl = shape_label(shape_kind, bc, ab, bt)
                        m_scalar = complex(np.mean(m_p_xyz))
                        mat_lbl  = material_label(m_scalar)
                        n_tet_target = get_n_tet_target(shape_lbl, mat_lbl, r_v)
                        dpl_slot = dpl_for_slot(shape_lbl, mat_lbl, r_v,
                                                m_p_xyz, wl_0=wl_0)

                        monitor.reset()
                        t_slot_start = time.time()

                        t0 = time.time()
                        tgt = _build_target(shape_kind, r_v, bc, ab, bt,
                                            wl_0, m_p_xyz, dpl=dpl_slot)
                        t_build = time.time() - t0

                        print("─" * 72)
                        mode_tag = (f" [spheroid, L_solve={N_beta}]"
                                    if spheroid_mode
                                    else f" [general, L_solve={num_orientations}]")
                        ratio = tgt.num_element_occupy / max(n_tet_target, 1)
                        _log(f"wl_0={wl_0:.4f}  m_m={m_m:.3f}  "
                             f"m_p_xyz={m_p_xyz}  "
                             f"r_v={r_v:.3f}  bc={bc:.1f}  ab={ab:.1f}  "
                             f"β={bt:.2f}  "
                             f"[{shape_lbl}/{mat_lbl}] dpl={dpl_slot:.2f}  "
                             f"d={tgt.lattice_lf:.5f}μm  "
                             f"N_cub={int(np.prod(tgt.lattice_n))}  "
                             f"N_occ={tgt.num_element_occupy}  "
                             f"(target n_tet={n_tet_target}, ratio={ratio:.2f}×)"
                             f"{mode_tag}")

                        try:
                            if spheroid_mode:
                                res = _solve_spheroid(tgt, wl_0, m_m,
                                                     N_alpha, N_beta, N_gamma)
                            else:
                                inc = IncidentField(wl_0, m_m, euler_angles)
                                res = _solve_observables(tgt, inc)
                        except KeyboardInterrupt:
                            _log("Interrupted — HDF5 flushed cleanly.")
                            raise SystemExit(130)

                        t_total = time.time() - t_slot_start
                        peak_rss = monitor.peak_bytes
                        flag = "✓" if res["converged"] else "✗"
                        _log(f"  solver {flag} iters={res['iters']} "
                             f"err={res['err']:.3e}  "
                             f"t_build={t_build:.2f}s "
                             f"t_setup={res['t_setup_s']:.2f}s "
                             f"t_solve={res['t_solve_s']:.2f}s "
                             f"t_total={t_total:.1f}s  "
                             f"peak_RSS={peak_rss/1024**3:.2f}GB")

                        # Mie reference (volume-equivalent sphere)
                        m_p_avg = complex(np.mean(m_p_xyz))
                        _, Q_abs_m, Q_ext_m, S_fw_m, S_bk_m = \
                            mie_compute_q_and_s(wl_0, m_m, r_v, m_p_avg, nang=3)
                        geom = np.pi * r_v ** 2
                        C_abs_m = Q_abs_m * geom
                        C_ext_m = Q_ext_m * geom

                        N = slice(None)
                        sd["Euler_angles"   ][idx6 + (N, N)] = euler_angles
                        sd["C_abs"          ][idx6 + (N,)]   = res["C_abs"]
                        sd["C_ext"          ][idx6 + (N,)]   = res["C_ext"]
                        sd["S_fw_PCAS_theta"][idx6 + (N,)]   = res["S_fw_theta"]
                        sd["S_fw_PCAS_phi"  ][idx6 + (N,)]   = res["S_fw_phi"]
                        sd["S_bk_OCBS"      ][idx6 + (N,)]   = res["S_bk"]
                        sd["C_abs_mie"      ][idx6]          = C_abs_m
                        sd["C_ext_mie"      ][idx6]          = C_ext_m
                        sd["S_fw_PCAS_mie"  ][idx6]          = S_fw_m
                        sd["S_bk_OCBS_mie"  ][idx6]          = S_bk_m

                        # Cost diagnostics
                        cst["t_build_s"     ][idx6] = t_build
                        cst["t_setup_s"     ][idx6] = res["t_setup_s"]
                        cst["t_solve_s"     ][idx6] = res["t_solve_s"]
                        cst["t_total_s"     ][idx6] = t_total
                        cst["peak_rss_bytes"][idx6] = int(peak_rss)
                        cst["n_cuboid"      ][idx6] = int(np.prod(tgt.lattice_n))
                        cst["n_occ"         ][idx6] = int(tgt.num_element_occupy)
                        cst["lattice_lf"    ][idx6] = float(tgt.lattice_lf)
                        cst["iters"         ][idx6] = int(res["iters"])
                        cst["converged"     ][idx6] = 1 if res["converged"] else 0
                        cst["solver_err"    ][idx6] = float(res["err"])
                        # residual_history: NaN-pad to MAXITER-wide fixed length
                        hist = np.full(MAXITER, np.nan, dtype=np.float64)
                        eh = res["err_history"]
                        hist[:eh.size] = eh
                        cst["residual_history"][idx6 + (slice(None),)] = hist
                        h5.flush()

                        if res["converged"]:
                            _log(f"  C_ext(mean)={np.nanmean(res['C_ext']):.4e}  "
                                 f"S_fw_θ(mean)={np.nanmean(res['S_fw_theta']):.4g}  "
                                 f"S_bk(mean)={np.nanmean(res['S_bk']):.4g}")

            t_sweep = time.time() - t_sweep_start
            _log("═" * 72)
            _log(f"Sweep complete.  total wall time = {t_sweep:.1f}s "
                 f"({t_sweep/60:.1f} min)")
        finally:
            monitor.stop()


if __name__ == "__main__":
    main()
