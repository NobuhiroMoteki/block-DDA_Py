"""Compare the two least-squares methods of the shifted block GMRES.

``bl_gmres_shifted_mvp_fft`` handles the per-shift least-squares problems
either by an incremental block-Givens QR (``lsq_method="incremental"``, the
default) or by a dense QR from scratch at every step (``lsq_method="dense"``).
This script solves the same refractive-index sweep with both and reports the
solver time, the time of the least-squares part, the peak RSS and the
agreement of the results.

The particle and the refractive indices are those of
``scripts/run_shift_scaling.py``: an oblate spheroid on the lattice of the
largest refractive index, with ``N_m`` equispaced real indices.

Output HDF5 layout:
    attrs: provenance (host, git SHA, thread settings, wl_0 [um], m_m, bc_ratio)
    D_ve_<d>/L_<L>/tol_<tol>/N_m_<n>/
        attrs per method (prefix ``dense_`` / ``incremental_``):
            t_solve, t_mvp, t_gs, t_lsq, t_proj [s], peak_rss [bytes]
        attrs: iters_equal (bool), max_rel_S (max relative difference of S_fw)
        iters   (N_m,)  block-Arnoldi steps per refractive index

Usage (from the repository root):
    PYTHONPATH=. .venv/bin/python scripts/run_shift_lsq_compare.py out.h5
    PYTHONPATH=. .venv/bin/python scripts/run_shift_lsq_compare.py out.h5 \\
        --d-ve 0.6 --L 13 --tol 1e-2 --n-m 15            # a short run
"""
import argparse
import datetime
import os
import platform
import time

import h5py
import numpy as np
import scipy

from run_shift_scaling import _git_sha, _log, _quiet, _tag, build_geometry, make_dipoles
from utils.rss_monitor import RSSMonitor

METHODS = ("dense", "incremental")


def solve(dd, m_list, lsq_method, mon):
    """One shifted solve. Returns solver times [s], peak RSS [bytes], iterations and S_fw."""
    mon.reset()
    t0 = time.perf_counter()
    _quiet(dd.solve_matrix_equation_shifted, lsq_method=lsq_method)
    t_solve = time.perf_counter() - t0
    obs = dd.compute_observables_shifted()
    res = dd.shifted_result
    return {
        "t_solve": t_solve, "timings": res.timings, "peak_rss": mon.peak_bytes,
        "iters": res.iters, "converged": bool(np.all(dd.converge_shifted)),
        "S": np.concatenate([obs["S_fw_PCAS_theta"], obs["S_fw_PCAS_phi"]], axis=1),  # (N_m, 2L)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Dense vs incremental least squares in the shifted block GMRES")
    parser.add_argument("h5path", help="output HDF5 file (created or overwritten)")
    parser.add_argument("--d-ve", type=float, nargs="+", default=[0.6, 1.0],
                        help="volume-equivalent diameters [um]")
    parser.add_argument("--bc-ratio", type=float, default=2.0, help="b/c of the spheroid")
    parser.add_argument("--wl", type=float, default=0.638, help="vacuum wavelength [um]")
    parser.add_argument("--m-m", type=float, default=1.0, help="medium refractive index")
    parser.add_argument("--ri-range", type=float, nargs=2, default=[1.35, 1.70],
                        help="range of Re(m_p)")
    parser.add_argument("--n-m", type=int, nargs="+", default=[15, 30],
                        help="numbers of refractive indices")
    parser.add_argument("--L", type=int, nargs="+", default=[13, 64],
                        help="numbers of orientations (right-hand sides)")
    parser.add_argument("--tol", type=float, nargs="+", default=[1e-2, 1e-5])
    parser.add_argument("--itermax", type=int, default=100)
    args = parser.parse_args()

    ri_lo, ri_hi = args.ri_range
    mon = RSSMonitor(interval=0.05).start()
    _log(f"least-squares comparison -> {args.h5path}")
    _log(f"D_ve={args.d_ve}  b/c={args.bc_ratio}  wl={args.wl}  m_m={args.m_m}  "
         f"RI={ri_lo}..{ri_hi}  N_m={args.n_m}  L={args.L}  tol={args.tol}")

    with h5py.File(args.h5path, "w") as h5:
        h5.attrs.update({
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git_sha": _git_sha(), "host": platform.node(), "cpu_count": os.cpu_count() or 0,
            "DDA_FFT_WORKERS": os.environ.get("DDA_FFT_WORKERS", ""),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "numpy_version": np.__version__, "scipy_version": scipy.__version__,
            "wl_0": args.wl, "m_m": args.m_m, "bc_ratio": args.bc_ratio,
            "itermax": args.itermax,
        })
        for D_ve in args.d_ve:
            r_v = D_ve / 2.0
            geometry = _quiet(build_geometry, r_v, args.bc_ratio, args.wl, ri_hi + 0.0j)
            _log("=" * 78)
            _log(f"D_ve={D_ve} um: lattice {[int(v) for v in geometry[1]]}  "
                 f"N_occ={int(np.sum(geometry[4]))}")
            for L in args.L:
                euler = np.column_stack([np.zeros(L), np.arccos(np.linspace(0.0, 1.0, L)),
                                         np.zeros(L)])                       # shape: (L, 3)
                for tol in args.tol:
                    dd = make_dipoles(geometry, ri_hi + 0.0j, r_v, args.wl, args.m_m,
                                      euler, tol, args.itermax)
                    for n_m in args.n_m:
                        m_list = np.linspace(ri_lo, ri_hi, n_m) + 0.0j       # shape: (N_m,)
                        _quiet(dd.set_interaction_matrix_shifted, m_list)
                        out = {method: solve(dd, m_list, method, mon) for method in METHODS}
                        d, i = out["dense"], out["incremental"]
                        iters_equal = bool(np.array_equal(d["iters"], i["iters"]))
                        max_rel_S = float(np.max(np.abs(i["S"] - d["S"]) / np.abs(d["S"])))

                        g = h5.create_group(
                            f"D_ve_{_tag(D_ve)}/L_{L}/tol_{_tag(tol)}/N_m_{n_m}")
                        g.create_dataset("iters", data=i["iters"])
                        g.attrs.update({"iters_equal": iters_equal, "max_rel_S": max_rel_S})
                        for method, r in out.items():
                            g.attrs.update({
                                f"{method}_t_solve": r["t_solve"],
                                f"{method}_peak_rss": r["peak_rss"],
                                f"{method}_converged": r["converged"],
                                **{f"{method}_t_{key}": val for key, val in r["timings"].items()},
                            })
                        h5.flush()
                        _log(f"D_ve={D_ve} L={L:3d} tol={tol:g} N_m={n_m:2d} "
                             f"iters {i['iters'].min()}..{i['iters'].max()}  "
                             f"dense: solve {d['t_solve']:.2f} s, lsq {d['timings']['lsq']:.2f} s "
                             f"({100 * d['timings']['lsq'] / d['t_solve']:.1f}%)  "
                             f"incremental: solve {i['t_solve']:.2f} s, "
                             f"lsq {i['timings']['lsq']:.2f} s "
                             f"({100 * i['timings']['lsq'] / i['t_solve']:.1f}%)  "
                             f"RSS {d['peak_rss'] / 2**30:.2f} / {i['peak_rss'] / 2**30:.2f} GiB  "
                             f"dS {max_rel_S:.1e}  iters equal: {iters_equal}  "
                             f"converged: {d['converged'] and i['converged']}")
    mon.stop()
    _log(f"wrote {args.h5path}")


if __name__ == "__main__":
    main()
