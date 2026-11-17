"""Scaling benchmark of the shifted block GMRES for refractive-index sweeps.

For one spheroid slot (D_ve, b/c) the real refractive index is swept over
N_m values. Three ways of solving the sweep are timed.

  A  per-m lattice, one existing block-GMRES solve per m. This is the default
     mode of run_dda_spheroid_sweep.py.
  B  common lattice (built for the largest |m|), one existing solve per m.
  C  common lattice, ONE shifted block-GMRES process for all m. This is
     run_dda_spheroid_sweep.py --shifted-ri.

B and C solve the same linear systems, so B/C isolates the gain of the shifted
solver. A/C is the gain over the current production path and includes the
change of lattice.

The run loops over D_ve x L x tol x N_m. For N_m > 1 the refractive indices are
linspace(ri_lo, ri_hi, N_m). For N_m = 1 the single index is ri_hi. Solves of
A and B are done once per distinct m and reused for every N_m that contains it.
Orientations follow the spheroid sweep: alpha = gamma = 0 and cos(beta)
equidistant on [0, 1].

Speed-up model (plan section 5.1). With t_mvp the time of one block MVP and
t_gs the time of one block inner product and update, a block-GMRES run of K
steps costs  T(K) = K t_mvp + K (K + 1) / 2 t_gs.  The model solver speed-up is
sum_s T(K_s) / T(K_max), with t_mvp and t_gs measured in run C.

Output HDF5 layout (times in seconds, sizes in bytes):
    /                       attrs: settings, host, versions
    /D_ve_<v>/              attrs: D_ve [um], bc_ratio, t_geom_common, n_occ_common, lattice_n
      m_values              (N_dist,) complex   all distinct refractive indices
      A_t_geom, A_n_occ     (N_dist,)           per-m lattice build
      L_<L>/tol_<tol>/
        B_t_init, B_t_solve, B_t_obs, B_n_iter, B_converged      (N_dist,)
        A_t_init, A_t_solve, A_t_obs, A_n_iter, A_converged      (N_dist,)  if A was run
        B_peak_rss, A_peak_rss                                   scalars (max over m)
        N_m_<n>/            attrs: C timings, peak RSS, speed-ups, agreement with B
          m_index           (n,) index into m_values
          iters, errs       (n,)

Usage (from the repository root):
    PYTHONPATH=. .venv/bin/python scripts/run_shift_scaling.py out.h5
    PYTHONPATH=. .venv/bin/python scripts/run_shift_scaling.py out.h5 \\
        --d-ve 0.6 --L 13 --tol 1e-2 --n-m 1 15          # a short run
"""
import argparse
import contextlib
import datetime
import io
import os
import platform
import subprocess
import time

import h5py
import numpy as np
import scipy

from bl_dda.scatterer import DiscreteDipoles, IncidentField, Target
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from utils.rss_monitor import RSSMonitor

RNG_SEED = 12345


def _log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""


def _tag(value):
    """0.6 -> '0p6', 1e-05 -> '1e-05'."""
    return f"{value:g}".replace(".", "p")


def build_geometry(r_v_base, bc_ratio, wl_0, m_lattice):
    """Spheroid lattice with the spacing set by ``m_lattice``. Returns Target arguments."""
    rng = np.random.default_rng(RNG_SEED)
    m_p_xyz = np.array([m_lattice, m_lattice, m_lattice])
    gre = gaussian_ellipsoid_shape_model(r_v_base, bc_ratio, 1.0, 0.0, wl_0, m_p_xyz)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return gre.name, lattice_n, gre.lattice_lf, grid, is_in


def make_dipoles(geometry, m_p, r_v_base, wl_0, m_m, euler, tol, itermax):
    name, lattice_n, lattice_lf, grid, is_in = geometry
    target = Target(name, lattice_n, lattice_lf, grid, is_in, np.array([m_p, m_p, m_p]), r_v_base)
    dd = _quiet(DiscreteDipoles, target, IncidentField(wl_0, m_m, euler))
    dd.tol, dd.itermax = tol, itermax
    return dd


def solve_independent(dd, mon):
    """One existing solve. Returns dict of times, iteration count and S_fw."""
    mon.reset()
    t0 = time.perf_counter()
    _quiet(dd.set_interaction_matrix)
    t1 = time.perf_counter()
    _quiet(dd.solve_matrix_equation)
    t2 = time.perf_counter()
    S_theta, S_phi = dd.compute_PCAS_observable_S_fw()
    t3 = time.perf_counter()
    return {
        "t_init": t1 - t0, "t_solve": t2 - t1, "t_obs": t3 - t2,
        "n_iter": len(dd.err_history), "converged": bool(dd.converge),
        "S": np.concatenate([S_theta, S_phi]),                  # shape: (2L,)
        "peak_rss": mon.peak_bytes,
    }


def solve_shifted(dd, m_list, mon):
    """One shifted solve of all ``m_list``. Returns dict of times, iterations and S_fw."""
    mon.reset()
    t0 = time.perf_counter()
    _quiet(dd.set_interaction_matrix_shifted, m_list)
    t1 = time.perf_counter()
    _quiet(dd.solve_matrix_equation_shifted)
    t2 = time.perf_counter()
    obs = dd.compute_observables_shifted()
    t3 = time.perf_counter()
    res = dd.shifted_result
    return {
        "t_init": t1 - t0, "t_solve": t2 - t1, "t_obs": t3 - t2,
        "iters": res.iters, "errs": res.errs, "converged": dd.converge_shifted,
        "timings": res.timings,
        "S": np.concatenate([obs["S_fw_PCAS_theta"], obs["S_fw_PCAS_phi"]], axis=1),  # (N_m, 2L)
        "peak_rss": mon.peak_bytes,
    }


def model_speedup(iters, timings):
    """sum_s T(K_s) / T(K_max) with T(K) = K t_mvp + K (K + 1) / 2 t_gs."""
    k_fin = int(np.max(iters))
    t_mvp = timings["mvp"] / k_fin
    t_gs = timings["gs"] / (k_fin * (k_fin + 1) / 2)
    cost = lambda K: K * t_mvp + K * (K + 1) / 2 * t_gs
    return float(np.sum(cost(np.asarray(iters, dtype=np.float64))) / cost(float(k_fin)))


def main():
    parser = argparse.ArgumentParser(
        description="Scaling benchmark of the shifted block GMRES (refractive-index sweep)")
    parser.add_argument("h5path", help="output HDF5 file (created or overwritten)")
    parser.add_argument("--d-ve", type=float, nargs="+", default=[0.6, 1.0],
                        help="volume-equivalent diameters [um]")
    parser.add_argument("--bc-ratio", type=float, default=2.0, help="b/c of the spheroid")
    parser.add_argument("--wl", type=float, default=0.638, help="vacuum wavelength [um]")
    parser.add_argument("--m-m", type=float, default=1.0, help="medium refractive index")
    parser.add_argument("--ri-range", type=float, nargs=2, default=[1.35, 1.70],
                        help="range of Re(m_p)")
    parser.add_argument("--m-imag", type=float, default=0.0, help="Im(m_p), fixed")
    parser.add_argument("--n-m", type=int, nargs="+", default=[1, 2, 4, 8, 15, 30],
                        help="numbers of refractive indices")
    parser.add_argument("--L", type=int, nargs="+", default=[5, 13, 64],
                        help="numbers of orientations (right-hand sides)")
    parser.add_argument("--tol", type=float, nargs="+", default=[1e-2, 1e-5])
    parser.add_argument("--itermax", type=int, default=100)
    parser.add_argument("--per-m-lattice-L", type=int, nargs="*", default=[13],
                        help="values of L for which method A (per-m lattice) is also run")
    args = parser.parse_args()

    ri_lo, ri_hi = args.ri_range
    m_sets = {n: (np.linspace(ri_lo, ri_hi, n) if n > 1 else np.array([ri_hi])) + 1j * args.m_imag
              for n in args.n_m}
    m_values = np.unique(np.round(np.concatenate(list(m_sets.values())), 12))   # sorted by Re
    m_index = {n: np.array([int(np.argmin(np.abs(m_values - m))) for m in ms])
               for n, ms in m_sets.items()}
    m_max = m_values[np.argmax(np.abs(m_values))]
    run_A = sorted(set(args.per_m_lattice_L) & set(args.L))

    mon = RSSMonitor(interval=0.05).start()
    _log(f"shift-scaling benchmark -> {args.h5path}")
    _log(f"D_ve={args.d_ve}  b/c={args.bc_ratio}  wl={args.wl}  m_m={args.m_m}  "
         f"RI={ri_lo}..{ri_hi}+{args.m_imag}j  N_m={args.n_m}  L={args.L}  tol={args.tol}  "
         f"itermax={args.itermax}  distinct m={m_values.size}  method A at L={run_A}")

    with h5py.File(args.h5path, "w") as h5:
        h5.attrs.update({
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_sha": _git_sha(), "host": platform.node(), "cpu_count": os.cpu_count() or 0,
            "DDA_FFT_WORKERS": os.environ.get("DDA_FFT_WORKERS", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "wl_0": args.wl, "m_m": args.m_m, "bc_ratio": args.bc_ratio,
            "m_imag": args.m_imag, "itermax": args.itermax, "rng_seed": RNG_SEED,
        })

        for D_ve in args.d_ve:
            r_v = D_ve / 2.0
            g_slot = h5.create_group(f"D_ve_{_tag(D_ve)}")
            g_slot.create_dataset("m_values", data=m_values)
            _log("=" * 78)

            t0 = time.perf_counter()
            geom_common = _quiet(build_geometry, r_v, args.bc_ratio, args.wl, m_max)
            t_geom_common = time.perf_counter() - t0
            n_occ_common = int(np.sum(geom_common[4]))
            g_slot.attrs.update({"D_ve": D_ve, "bc_ratio": args.bc_ratio,
                                 "t_geom_common": t_geom_common, "n_occ_common": n_occ_common,
                                 "lattice_n": np.asarray(geom_common[1], dtype=np.int64)})
            _log(f"D_ve={D_ve} um: common lattice {[int(v) for v in geom_common[1]]}  "
                 f"N_occ={n_occ_common}  ({t_geom_common:.1f} s)")

            geom_A, A_t_geom, A_n_occ = {}, np.full(m_values.size, np.nan), np.zeros(m_values.size, int)
            if run_A:
                for i, m in enumerate(m_values):
                    t0 = time.perf_counter()
                    geom_A[i] = _quiet(build_geometry, r_v, args.bc_ratio, args.wl, m)
                    A_t_geom[i] = time.perf_counter() - t0
                    A_n_occ[i] = int(np.sum(geom_A[i][4]))
                _log(f"  per-m lattices: N_occ {A_n_occ.min()}..{A_n_occ.max()}  "
                     f"(total {np.sum(A_t_geom):.1f} s)")
            g_slot.create_dataset("A_t_geom", data=A_t_geom)
            g_slot.create_dataset("A_n_occ", data=A_n_occ)

            for L in args.L:
                euler = np.column_stack([np.zeros(L), np.arccos(np.linspace(0.0, 1.0, L)),
                                         np.zeros(L)])                       # shape: (L, 3)
                for tol in args.tol:
                    g = g_slot.create_group(f"L_{L}/tol_{_tag(tol)}")
                    _log("-" * 78)
                    _log(f"D_ve={D_ve}  L={L}  tol={tol:g}")

                    # B: common lattice, one existing solve per distinct m
                    rB = [solve_independent(make_dipoles(geom_common, m, r_v, args.wl, args.m_m,
                                                         euler, tol, args.itermax), mon)
                          for m in m_values]
                    for key in ("t_init", "t_solve", "t_obs", "n_iter", "converged"):
                        g.create_dataset("B_" + key, data=np.array([r[key] for r in rB]))
                    g.attrs["B_peak_rss"] = max(r["peak_rss"] for r in rB)
                    B_iter = np.array([r["n_iter"] for r in rB])
                    B_solve = np.array([r["t_solve"] for r in rB])
                    B_total = np.array([r["t_init"] + r["t_solve"] + r["t_obs"] for r in rB])
                    _log(f"  B: iterations {B_iter.min()}..{B_iter.max()}  "
                         f"solve {B_solve.min():.2f}..{B_solve.max():.2f} s per m  "
                         f"all converged={all(r['converged'] for r in rB)}")

                    # A: per-m lattice, one existing solve per distinct m
                    rA = None
                    if L in run_A:
                        rA = [solve_independent(make_dipoles(geom_A[i], m, r_v, args.wl, args.m_m,
                                                             euler, tol, args.itermax), mon)
                              for i, m in enumerate(m_values)]
                        for key in ("t_init", "t_solve", "t_obs", "n_iter", "converged"):
                            g.create_dataset("A_" + key, data=np.array([r[key] for r in rA]))
                        g.attrs["A_peak_rss"] = max(r["peak_rss"] for r in rA)
                        A_solve = np.array([r["t_solve"] for r in rA])
                        A_total = A_t_geom + np.array(
                            [r["t_init"] + r["t_solve"] + r["t_obs"] for r in rA])

                    # C: common lattice, one shifted solve per N_m
                    for n in args.n_m:
                        idx = m_index[n]
                        dd = make_dipoles(geom_common, m_max, r_v, args.wl, args.m_m,
                                          euler, tol, args.itermax)
                        rC = solve_shifted(dd, m_values[idx], mon)
                        del dd
                        tm = rC["timings"]
                        S_B = np.array([rB[i]["S"] for i in idx])              # shape: (n, 2L)
                        ok = rC["converged"] & np.array([rB[i]["converged"] for i in idx])
                        max_rel = (float(np.max(np.abs(rC["S"][ok] - S_B[ok]) / np.abs(S_B[ok])))
                                   if ok.any() else np.nan)
                        C_total = t_geom_common + rC["t_init"] + rC["t_solve"] + rC["t_obs"]
                        out = {
                            "t_init": rC["t_init"], "t_solve": rC["t_solve"], "t_obs": rC["t_obs"],
                            "t_mvp": tm["mvp"], "t_gs": tm["gs"], "t_lsq": tm["lsq"],
                            "t_proj": tm["proj"], "lsq_fraction": tm["lsq"] / rC["t_solve"],
                            "peak_rss": rC["peak_rss"], "total": C_total,
                            "iters_equal_B": bool(np.array_equal(rC["iters"], B_iter[idx])),
                            "max_rel_S_vs_B": max_rel,
                            "ideal_speedup": float(np.sum(B_iter[idx]) / np.max(B_iter[idx])),
                            "model_speedup": model_speedup(rC["iters"], tm),
                            "solve_speedup_B": float(np.sum(B_solve[idx]) / rC["t_solve"]),
                            "total_speedup_B": float(
                                (t_geom_common + np.sum(B_total[idx])) / C_total),
                        }
                        if rA is not None:
                            out["solve_speedup_A"] = float(np.sum(A_solve[idx]) / rC["t_solve"])
                            out["total_speedup_A"] = float(np.sum(A_total[idx]) / C_total)
                        gn = g.create_group(f"N_m_{n}")
                        gn.attrs.update(out)
                        gn.create_dataset("m_index", data=idx)
                        gn.create_dataset("iters", data=rC["iters"])
                        gn.create_dataset("errs", data=rC["errs"])
                        extra = (f"  A/C solve {out['solve_speedup_A']:.2f} total "
                                 f"{out['total_speedup_A']:.2f}" if rA is not None else "")
                        _log(f"  C N_m={n:<3d} iters {rC['iters'].min()}..{rC['iters'].max()}  "
                             f"solve {rC['t_solve']:.2f} s (lsq {100 * out['lsq_fraction']:.1f}%)  "
                             f"B/C solve {out['solve_speedup_B']:.2f} "
                             f"[ideal {out['ideal_speedup']:.2f}, model {out['model_speedup']:.2f}]"
                             f"{extra}  RSS {rC['peak_rss'] / 2**30:.2f} GiB  "
                             f"dS {max_rel:.1e}  iters=B: {out['iters_equal_B']}")
                    h5.flush()

    mon.stop()
    _log(f"wrote {args.h5path}")


if __name__ == "__main__":
    main()
