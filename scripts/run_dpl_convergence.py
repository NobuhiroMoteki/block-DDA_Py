"""dpl-convergence study for the paper (CLAUDE.md §4).

Python analogue of `block-VIEM.jl/viem_results/paper/run_lc_convergence.jl`:
for one (shape, material) at a_eq = 0.1 μm and a single representative
orientation (ZYZ identity), build the DDA lattice at five dpl values
and record the converged observables (Q_ext, Q_sca, Q_abs, S_fw_θ, S_fw_φ,
S_bk) plus per-dpl setup/solve wall time. Results land in
`dda_results/paper/convergence_{shape}_{material}.hdf5`.

dpl plays the role of VIEM's lc factor:
    VIEM lc_factors = [1.5, 1.0, 0.7, 0.5, 0.35]     (coarse→fine)
    DDA dpl_list    = [10,  14,  17,  24,  34]        (coarse→fine)
CLAUDE.md §4 fixes the DDA dpl grid; 17 is the production central value.

Solver: `bl_gmres_mvp_fft` (unpreconditioned, VIEM parity) with
    tol = 1e-5, maxiter = 100.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_dpl_convergence.py \\
                                  <shape> <material>

    shape    ∈ {sphere, oblate, gre}
    material ∈ {n15, n317, Au}
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import datetime
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles
from bl_krylov.bl_krylov import bl_gmres_mvp_fft
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from dda_results.paper._common import (
    N_LOW, N_HIGH, N_AU, WL_PAPER, M_M_PAPER,
)
from utils.rss_monitor import RSSMonitor


# ── Settings (mirror VIEM run_lc_convergence.jl) ─────────────────────────────
RNG_SEED    = 12345
SOLVER_TOL  = 1e-5
MAXITER     = 200                            # matches run_paper_sweep.py (v0.7.6)
A_EQ_CONV   = 0.1                            # CLAUDE.md §4 representative size
DPL_LIST    = [10, 14, 17, 24, 34]           # CLAUDE.md §4
SINGLE_ORIENT = np.array([[0.0, 0.0, 0.0]])  # ZYZ identity

SHAPE_PARAMS = {
    "sphere": dict(bc_ratio=1.0, ab_ratio=1.0, gre_beta=0.0),
    "oblate": dict(bc_ratio=3.0, ab_ratio=1.0, gre_beta=0.0),
    "gre":    dict(bc_ratio=1.0, ab_ratio=1.0, gre_beta=0.2),
}
MATERIAL_M_P = {
    "n15":  N_LOW,
    "n317": N_HIGH,
    "Au":   N_AU,
}


def _log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _build_gre_target(r_v_base, bc, ab, beta, wl_0, m_p_xyz, dpl):
    """Build a Target at a specific dpl for the GRE family (sphere/oblate/gre).

    sphere: (bc, ab, beta) = (1, 1, 0)
    oblate: (bc, ab, beta) = (3, 1, 0)
    gre:    (bc, ab, beta) = (1, 1, 0.2)
    """
    gre = gaussian_ellipsoid_shape_model(
        r_v_base, bc, ab, beta, wl_0, m_p_xyz, dpl=int(dpl))
    rng = np.random.default_rng(RNG_SEED)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return Target(gre.name, lattice_n, gre.lattice_lf, grid, is_in, m_p_xyz, r_v_base)


def solve_one(shape, m_p_xyz, dpl, monitor, wl_0=WL_PAPER, m_m=M_M_PAPER):
    """Solve DDA at one dpl and return observables + diagnostics.

    `monitor` is an RSSMonitor instance; we reset it at slot entry and
    record its `peak_bytes` on exit.
    """
    sp = SHAPE_PARAMS[shape]

    monitor.reset()
    t_slot_start = time.time()

    t0 = time.time()
    tgt = _build_gre_target(A_EQ_CONV, sp["bc_ratio"], sp["ab_ratio"],
                            sp["gre_beta"], wl_0, m_p_xyz, dpl)
    inc = IncidentField(wl_0, m_m, SINGLE_ORIENT)
    dd  = DiscreteDipoles(tgt, inc)
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
    if converged:
        dd.X = X
        dd.P = dd.X.T.reshape(dd.L, dd.num_element_occupy, 3)
        dd.E = (dd.P * (4 * np.pi)
                / ((dd.eper_r[np.newaxis, :, :] - 1) * dd.element_vol))
        dd.compute_C_abs()
        dd.compute_C_ext()
        dd.compute_PCAS_observable_S_fw()
        dd.compute_OCBS_observable_S_bk()
        C_abs      = float(dd.C_abs[0])
        C_ext      = float(dd.C_ext[0])
        S_fw_theta = complex(dd.S_fw_PCAS_theta[0])
        S_fw_phi   = complex(dd.S_fw_PCAS_phi[0])
        S_bk       = complex(dd.S_bk_OCBS[0])
    else:
        C_abs = C_ext = float("nan")
        S_fw_theta = S_fw_phi = S_bk = complex("nan")

    t_total = time.time() - t_slot_start
    peak_rss = monitor.peak_bytes

    return {
        "n_cuboid": int(np.prod(tgt.lattice_n)),
        "n_occ":   int(tgt.num_element_occupy),
        "r_ve":    float(tgt.ve_radius),
        "lattice_lf": float(tgt.lattice_lf),
        "iters":   int(iter_fin + 1),     # 1-based count
        "converged": converged,
        "err":     float(err_fin),
        "err_history": err_history,
        "t_setup": float(t_setup),
        "t_solve": float(t_solve),
        "t_total": float(t_total),
        "peak_rss_bytes": int(peak_rss),
        "C_abs":   C_abs,
        "C_ext":   C_ext,
        "C_sca":   C_ext - C_abs,
        "S_fw_theta": S_fw_theta,
        "S_fw_phi":   S_fw_phi,
        "S_bk":       S_bk,
    }


def mie_reference(m_p_xyz):
    """Mie of volume-equivalent sphere at A_EQ_CONV."""
    m_p_avg = complex(np.mean(m_p_xyz))
    Q_sca, Q_abs, Q_ext, S_fw, S_bk = mie_compute_q_and_s(
        WL_PAPER, M_M_PAPER, A_EQ_CONV, m_p_avg, nang=3)
    geom = np.pi * A_EQ_CONV ** 2
    return {
        "Q_abs": float(Q_abs), "Q_ext": float(Q_ext), "Q_sca": float(Q_sca),
        "C_abs": float(Q_abs * geom),
        "C_ext": float(Q_ext * geom),
        "C_sca": float(Q_sca * geom),
        "S_fw":  complex(S_fw),
        "S_bk":  complex(S_bk),
    }


def write_convergence_h5(path, shape, material, m_p_xyz, results, mie):
    """Write HDF5 whose schema mirrors VIEM's convergence_* files."""
    sp = SHAPE_PARAMS[shape]
    geom_area = np.pi * A_EQ_CONV ** 2

    dpls        = np.asarray([r["dpl"]        for r in results], dtype=np.float64)
    n_cuboid    = np.asarray([r["n_cuboid"]   for r in results], dtype=np.int64)
    n_occ       = np.asarray([r["n_occ"]      for r in results], dtype=np.int64)
    r_ve        = np.asarray([r["r_ve"]       for r in results], dtype=np.float64)
    lattice_lf  = np.asarray([r["lattice_lf"] for r in results], dtype=np.float64)
    iters       = np.asarray([r["iters"]      for r in results], dtype=np.int64)
    conv        = np.asarray([1 if r["converged"] else 0 for r in results], dtype=np.int64)
    t_setup     = np.asarray([r["t_setup"]    for r in results], dtype=np.float64)
    t_solve     = np.asarray([r["t_solve"]    for r in results], dtype=np.float64)
    t_total     = np.asarray([r["t_total"]    for r in results], dtype=np.float64)
    peak_rss    = np.asarray([r["peak_rss_bytes"] for r in results], dtype=np.int64)
    C_abs       = np.asarray([r["C_abs"]      for r in results], dtype=np.float64)
    C_ext       = np.asarray([r["C_ext"]      for r in results], dtype=np.float64)
    C_sca       = np.asarray([r["C_sca"]      for r in results], dtype=np.float64)
    S_fw_theta  = np.asarray([r["S_fw_theta"] for r in results], dtype=np.complex128)
    S_fw_phi    = np.asarray([r["S_fw_phi"]   for r in results], dtype=np.complex128)
    S_bk        = np.asarray([r["S_bk"]       for r in results], dtype=np.complex128)

    with h5py.File(path, "w") as f:
        g = f.create_group("target")
        g.attrs["shape"]        = shape
        g.attrs["material"]     = material
        g.attrs["wl_0_um"]      = WL_PAPER
        g.attrs["m_m"]          = M_M_PAPER
        g.attrs["a_eq_um"]      = A_EQ_CONV
        g.attrs["bc_ratio"]     = sp["bc_ratio"]
        g.attrs["ab_ratio"]     = sp["ab_ratio"]
        g.attrs["gre_beta"]     = sp["gre_beta"]
        g.attrs["n_dpl_points"] = len(DPL_LIST)
        g.attrs["solver"]       = "bl_gmres_mvp_fft (unpreconditioned, VIEM parity)"
        g.attrs["solver_tol"]     = SOLVER_TOL
        g.attrs["solver_maxiter"] = MAXITER
        g.attrs["units"]        = "C:[um^2], S:[um], lattice_lf:[um], t:[s]"
        g.attrs["orientation"]  = "ZYZ Euler (0,0,0): incidence +z, LHC polarization"

        g.create_dataset("m_p_xyz", data=np.asarray(m_p_xyz, dtype=np.complex128))

        c = g.create_group("dpl_convergence")
        c.attrs["description"] = (
            "DDA convergence study at a_eq=0.1 μm, one representative "
            "ZYZ-identity orientation, over dpl ∈ {10,14,17,24,34}. "
            "Analogue of VIEM's lc_convergence group (coarse→fine order).")
        c.create_dataset("dpl",        data=dpls)
        c.create_dataset("lattice_lf", data=lattice_lf)
        c.create_dataset("n_cuboid",   data=n_cuboid)
        c.create_dataset("n_occ",      data=n_occ)
        c.create_dataset("r_ve",       data=r_ve)
        c.create_dataset("iters",      data=iters)
        c.create_dataset("converged",  data=conv)
        c.create_dataset("t_setup",    data=t_setup)
        c.create_dataset("t_solve",    data=t_solve)
        c.create_dataset("t_total",    data=t_total)
        c.create_dataset("peak_rss_bytes", data=peak_rss)
        # residual_history: (n_dpl, MAXITER) float64, NaN-padded
        hist = np.full((len(results), MAXITER), np.nan, dtype=np.float64)
        for i, r in enumerate(results):
            eh = r["err_history"]
            hist[i, :eh.size] = eh
        c.create_dataset("residual_history", data=hist)
        c.create_dataset("C_abs",      data=C_abs)
        c.create_dataset("C_ext",      data=C_ext)
        c.create_dataset("C_sca",      data=C_sca)
        c.create_dataset("Q_abs",      data=C_abs / geom_area)
        c.create_dataset("Q_ext",      data=C_ext / geom_area)
        c.create_dataset("Q_sca",      data=C_sca / geom_area)
        c.create_dataset("S_fw_theta", data=S_fw_theta)
        c.create_dataset("S_fw_phi",   data=S_fw_phi)
        c.create_dataset("S_bk",       data=S_bk)

        r = g.create_group("reference")
        r.attrs["definition"] = ("Mie of volume-equivalent sphere (radius a_eq); "
                                 "exact for shape=sphere, approximation otherwise")
        r.create_dataset("C_abs_mie", data=mie["C_abs"])
        r.create_dataset("C_ext_mie", data=mie["C_ext"])
        r.create_dataset("C_sca_mie", data=mie["C_sca"])
        r.create_dataset("Q_abs_mie", data=mie["Q_abs"])
        r.create_dataset("Q_ext_mie", data=mie["Q_ext"])
        r.create_dataset("Q_sca_mie", data=mie["Q_sca"])
        r.create_dataset("S_fw_mie",  data=mie["S_fw"])
        r.create_dataset("S_bk_mie",  data=mie["S_bk"])


def main():
    ap = argparse.ArgumentParser(description="dpl-convergence study (CLAUDE.md §4)")
    ap.add_argument("shape",    choices=sorted(SHAPE_PARAMS.keys()))
    ap.add_argument("material", choices=sorted(MATERIAL_M_P.keys()))
    args = ap.parse_args()

    shape, material = args.shape, args.material
    m_p = MATERIAL_M_P[material]
    m_p_xyz = np.array([m_p, m_p, m_p], dtype=np.complex128)
    sp = SHAPE_PARAMS[shape]

    _log(f"dpl convergence: shape={shape}  material={material}")
    print(f"  m_p     = {m_p_xyz.tolist()}")
    print(f"  a_eq    = {A_EQ_CONV} μm")
    print(f"  (bc, ab, β_gre) = ({sp['bc_ratio']}, {sp['ab_ratio']}, {sp['gre_beta']})")
    print(f"  dpl grid = {DPL_LIST}")
    print(f"  solver   = bl_gmres_mvp_fft  (tol={SOLVER_TOL}, maxiter={MAXITER})")
    print("─" * 124)
    print(f"{'dpl':>4} {'d[μm]':>10} {'N_cuboid':>10} {'N_occ':>10} "
          f"{'iters':>6} {'t_setup[s]':>11} {'t_solve[s]':>11} "
          f"{'t_total[s]':>11} {'peak_RSS[GB]':>13} "
          f"{'C_ext[μm²]':>14} {'C_abs[μm²]':>14}")
    print("─" * 124)

    monitor = RSSMonitor(interval=0.2).start()
    try:
        results = []
        for dpl in DPL_LIST:
            r = solve_one(shape, m_p_xyz, dpl, monitor)
            r["dpl"] = dpl
            results.append(r)
            flag = "✓" if r["converged"] else "✗"
            print(f"{dpl:>4d} {r['lattice_lf']:>10.5f} {r['n_cuboid']:>10d} "
                  f"{r['n_occ']:>10d} {r['iters']:>5d}{flag} "
                  f"{r['t_setup']:>11.2f} {r['t_solve']:>11.2f} "
                  f"{r['t_total']:>11.2f} "
                  f"{r['peak_rss_bytes']/1024**3:>13.3f} "
                  f"{r['C_ext']:>14.4e} {r['C_abs']:>14.4e}", flush=True)
    finally:
        monitor.stop()

    print("─" * 124)
    mie = mie_reference(m_p_xyz)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "dda_results", "paper",
                       f"convergence_{shape}_{material}.hdf5")
    out = os.path.normpath(out)
    write_convergence_h5(out, shape, material, m_p_xyz, results, mie)
    _log(f"Wrote {out}")
    print(f"  Mie reference: C_ext={mie['C_ext']:.4e}  C_abs={mie['C_abs']:.4e}  μm²")


if __name__ == "__main__":
    main()
