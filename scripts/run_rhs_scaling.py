"""RHS-scaling diagnostic for block-Krylov (CLAUDE.md §7.5).

For every shape slot in a paper sweep HDF5, runs BOTH block-BiCGSTAB and
block-GMRES (unpreconditioned, VIEM-parity variants from
bl_krylov.bl_krylov) against L ∈ L_LIST distinct orientations and records
iteration count + wall time. Output is written under /target/rhs_scaling/
with the same schema as block-VIEM.jl's viem_results/paper/run_rhs_scaling.jl
so the two diagnostics can be compared side-by-side.

Schema under /target/rhs_scaling/:
    L_values:       (nL,)            int       # [1, 2, 4, 8, 16, 32]
    n_cuboid:       (N_rv,N_bc,N_ab,N_bt)        int
    n_occ:          (N_rv,N_bc,N_ab,N_bt)        int
    bicgstab/
      iters:                   (nL, N_rv, N_bc, N_ab, N_bt)  int
      converged:               (nL, N_rv, N_bc, N_ab, N_bt)  int (0/1)
      t_total_s:               (nL, N_rv, N_bc, N_ab, N_bt)  float64
      t_end2end_per_orient_s:  (nL, N_rv, N_bc, N_ab, N_bt)  float64
    gmres/
      (same datasets)

Diagnostic only — does NOT touch any production-data datasets
(C_ext, S_fw, etc.).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/run_rhs_scaling.py <sweep.hdf5>
"""
import os
import sys
import time
import argparse
import datetime
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from shape_model.two_sphere_cluster import two_sphere_cluster_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles
from bl_krylov.bl_krylov import bl_gmres_mvp_fft


RNG_SEED    = 12345
# Convergence criterion is unified between block-DDA_Py and block-VIEM.jl
# (v0.7.6, 2026-04-24) at tol=1e-5 / maxiter=200.  tol=1e-5 is still 2–3
# orders below the DDA/VIEM discretization error.  maxiter=200 was chosen
# after observing that some high-index / large-r_v cases need >100 iters
# at small L (n317 × r_v=0.4 at L=5, dpl-convergence Au stagnation);
# 200 keeps the GMRES Krylov basis within machine memory for sphere-only
# RHS-scaling.  Production path is GMRES-only (BiCGSTAB deprecated for
# paper comparison, VIEM-side decision synced).
SOLVER_TOL  = 1e-5
MAXITER     = 200
L_LIST      = [1, 2, 4, 8, 16, 32, 64, 128]   # VIEM v0.7.1 parity
METHODS     = [
    ("gmres",    bl_gmres_mvp_fft),
]


def _log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _build_target(shape_kind, r_v_base, bc, ab, beta, wl_0, m_p_xyz):
    """Build DDA Target for one shape slot."""
    if shape_kind == "doublet":
        mdl = two_sphere_cluster_shape_model(r_v_base, wl_0, m_p_xyz)
        lattice_n, lf, grid, is_in = mdl.build()
        name = mdl.name
    else:
        mdl = gaussian_ellipsoid_shape_model(r_v_base, bc, ab, beta, wl_0, m_p_xyz)
        rng = np.random.default_rng(RNG_SEED)
        r_pts, _ = mdl.compute_r_points_on_GRE(rng)
        _, lattice_n, grid = mdl.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
        dist = mdl.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
        is_in = mdl.extract_lattice_address_in_GRE_volume(
            mdl.lattice_lf, mdl.distance_factor, lattice_n, dist)
        lf = mdl.lattice_lf
        name = mdl.name
    return Target(name, lattice_n, lf, grid, is_in, m_p_xyz, r_v_base)


def pick_orientations(L):
    """Deterministic uniform-sphere Euler sequence; larger L nests smaller L as prefix.

    VIEM parity: fixed RNG seed, uniform on SO(3) via (α, acos(2u-1), γ).
    """
    rng = np.random.default_rng(RNG_SEED)
    out = np.empty((L, 3), dtype=np.float64)
    for i in range(L):
        a = 2 * np.pi * rng.random()
        b = np.arccos(2 * rng.random() - 1)
        g = 2 * np.pi * rng.random()
        out[i] = (a, b, g)
    return out


def measure_one(target, wl_0, m_m, method_fn, L):
    """Solve for L orientations with method_fn and return timing + iter count."""
    eulers = pick_orientations(L)
    inc = IncidentField(wl_0, m_m, eulers)
    dd = DiscreteDipoles(target, inc)
    dd.set_interaction_matrix()

    t0 = time.time()
    X, iter_fin, err_fin, err_history = method_fn(
        dd.lattice_n, dd.f, dd.lattice_address_in_target,
        dd.Au_til, dd.diag_A, dd.B,
        SOLVER_TOL, MAXITER,
    )
    t_total = time.time() - t0
    return {
        "iters":       int(iter_fin + 1),   # report count (1-based)
        "converged":   bool(err_fin < SOLVER_TOL),
        "t_total":     float(t_total),
        "t_per":       float(t_total / max(L, 1)),
        "err_history": err_history,
    }


def _write_results(t_grp, results, n_cuboid_arr, n_occ_arr):
    """Overwrite /target/rhs_scaling/ with results. `results` is a dict
    keyed by method name."""
    if "rhs_scaling" in t_grp:
        del t_grp["rhs_scaling"]
    g = t_grp.create_group("rhs_scaling")
    g.attrs["description"] = (
        "Block-Krylov scaling diagnostic over L ∈ L_values, per method "
        "(bicgstab, gmres, unpreconditioned variants). Per-orientation "
        "sets are nested (L=1 ⊂ L=2 ⊂ L=4 ⊂ L=8 ⊂ L=16 ⊂ L=32), drawn "
        "from a fixed-seed uniform-sphere Euler sequence.")
    g.attrs["units"]           = "t:[s]"
    g.attrs["solver_tol"]      = SOLVER_TOL
    g.attrs["solver_maxiter"]  = MAXITER
    g.attrs["solver_variant"]  = "unpreconditioned (VIEM-parity)"

    g.create_dataset("L_values", data=np.asarray(L_LIST, dtype=np.int64))
    g.create_dataset("n_cuboid", data=n_cuboid_arr)
    g.create_dataset("n_occ",    data=n_occ_arr)

    for name, arrays in results.items():
        sg = g.create_group(name)
        sg.attrs["solver_function"] = f"bl_krylov.{arrays['fn_name']}"
        sg.create_dataset("iters",                   data=arrays["iters"])
        sg.create_dataset("converged",               data=arrays["converged"])
        sg.create_dataset("t_total_s",               data=arrays["t_total"])
        sg.create_dataset("t_end2end_per_orient_s",  data=arrays["t_per"])
        sg.create_dataset("residual_history",        data=arrays["err_history"])


def main():
    parser = argparse.ArgumentParser(description="RHS-scaling diagnostic for block-DDA_Py")
    parser.add_argument("h5path", help="sweep HDF5 file (run_dda.py schema, updated in place)")
    args = parser.parse_args()

    if not os.path.isfile(args.h5path):
        print(f"error: HDF5 not found: {args.h5path}", file=sys.stderr)
        sys.exit(2)

    _log(f"Opening {args.h5path}  (rhs-scaling diagnostic)")
    with h5py.File(args.h5path, "r+") as f:
        t = f["target"]

        n_alpha = int(t.attrs["N_alpha_ori"])
        n_beta  = int(t.attrs["N_beta_ori"])
        n_gamma = int(t.attrs["N_gamma_ori"])
        shape_kind_attr = t.attrs.get("shape_kind", "gre")
        shape_kind = (shape_kind_attr.decode()
                      if isinstance(shape_kind_attr, bytes)
                      else str(shape_kind_attr))

        wl_pairs = t["wl_m_m_pairs"][:]
        m_p_list = t["m_p_xyz_list"][:]
        r_v_list = t["r_v_base_list"][:]
        bc_list  = t["bc_ratio_list"][:]
        ab_list  = t["ab_ratio_list"][:]
        bt_list  = t["gre_beta_list"][:]

        n_rv, n_bc, n_ab, n_bt = len(r_v_list), len(bc_list), len(ab_list), len(bt_list)

        # Worst-case build geometry (VIEM parity)
        wl_min  = float(np.min(wl_pairs[:, 0]))
        m_p_max = float(np.max(np.abs(m_p_list)))
        m_p_worst = np.array([m_p_max, m_p_max, m_p_max], dtype=np.complex64)

        # Operating point — first HDF5 entry
        wl_0_op = float(wl_pairs[0, 0])
        m_m_op  = float(wl_pairs[0, 1])
        m_p_op  = np.asarray(m_p_list[0], dtype=np.complex64)

        _log(f"shape_kind={shape_kind}  "
             f"worst-case wl_0={wl_min:.4f} |m_p|_max={m_p_max:.4f}")
        _log(f"operating point: wl_0={wl_0_op:.4f}  m_m={m_m_op:.4f}  m_p={m_p_op.tolist()}")
        _log(f"L_LIST={L_LIST}  methods={[n for n,_ in METHODS]}  "
             f"tol={SOLVER_TOL}  maxiter={MAXITER}")

        nL = len(L_LIST)
        n_cuboid_arr = np.zeros((n_rv, n_bc, n_ab, n_bt), dtype=np.int64)
        n_occ_arr    = np.zeros((n_rv, n_bc, n_ab, n_bt), dtype=np.int64)

        results = {
            name: {
                "fn_name":     fn.__name__,
                "iters":       np.zeros((nL, n_rv, n_bc, n_ab, n_bt), dtype=np.int64),
                "converged":   np.zeros((nL, n_rv, n_bc, n_ab, n_bt), dtype=np.int64),
                "t_total":     np.zeros((nL, n_rv, n_bc, n_ab, n_bt), dtype=np.float64),
                "t_per":       np.zeros((nL, n_rv, n_bc, n_ab, n_bt), dtype=np.float64),
                "err_history": np.full((nL, n_rv, n_bc, n_ab, n_bt, MAXITER),
                                       np.nan, dtype=np.float64),
            } for name, fn in METHODS
        }

        for i_rv in range(n_rv):
            for i_bc in range(n_bc):
                for i_ab in range(n_ab):
                    for i_bt in range(n_bt):
                        r_v  = float(r_v_list[i_rv])
                        bc   = float(bc_list[i_bc])
                        ab   = float(ab_list[i_ab])
                        beta = float(bt_list[i_bt])

                        _log("─" * 70)
                        _log(f"shape=({i_rv},{i_bc},{i_ab},{i_bt})  "
                             f"r_v={r_v}  bc={bc}  ab={ab}  β={beta}")

                        # Build worst-case Target once; reused for all L / methods
                        t_mesh0 = time.time()
                        tgt = _build_target(
                            shape_kind, r_v, bc, ab, beta, wl_min, m_p_worst)
                        t_mesh = time.time() - t_mesh0
                        n_cuboid_arr[i_rv,i_bc,i_ab,i_bt] = int(np.prod(tgt.lattice_n))
                        n_occ_arr   [i_rv,i_bc,i_ab,i_bt] = tgt.num_element_occupy
                        _log(f"  target  N_cuboid={int(np.prod(tgt.lattice_n))}  "
                             f"N_occ={tgt.num_element_occupy}  ({t_mesh:.2f}s)")

                        for (name, fn) in METHODS:
                            _log(f"  method = {name}")
                            for iL, L in enumerate(L_LIST):
                                r = results[name]
                                # Skip degenerate case: RHS count exceeds DOF
                                # (block-QR becomes rank-deficient when L > 3·N_occ).
                                if L > 3 * tgt.num_element_occupy:
                                    _log(f"    L={L:<3d}  skipped (L > 3·N_occ"
                                         f"={3*tgt.num_element_occupy}, degenerate)")
                                    r["iters"]    [iL,i_rv,i_bc,i_ab,i_bt] = 0
                                    r["converged"][iL,i_rv,i_bc,i_ab,i_bt] = 0
                                    r["t_total"]  [iL,i_rv,i_bc,i_ab,i_bt] = np.nan
                                    r["t_per"]    [iL,i_rv,i_bc,i_ab,i_bt] = np.nan
                                    continue
                                m = measure_one(tgt, wl_0_op, m_m_op, fn, L)
                                r["iters"]    [iL,i_rv,i_bc,i_ab,i_bt] = m["iters"]
                                r["converged"][iL,i_rv,i_bc,i_ab,i_bt] = 1 if m["converged"] else 0
                                r["t_total"]  [iL,i_rv,i_bc,i_ab,i_bt] = m["t_total"]
                                r["t_per"]    [iL,i_rv,i_bc,i_ab,i_bt] = m["t_per"]
                                eh = m["err_history"]
                                r["err_history"][iL,i_rv,i_bc,i_ab,i_bt, :eh.size] = eh
                                conv = "✓" if m["converged"] else "✗"
                                _log(f"    L={L:<3d}  iters={m['iters']:<4d} "
                                     f"{conv}  t_total={m['t_total']:>7.2f}s  "
                                     f"t/ori={m['t_per']:>7.3f}s")

        _write_results(t, results, n_cuboid_arr, n_occ_arr)
        _log("─" * 70)
        _log(f"Wrote /target/rhs_scaling/ to {args.h5path}")


if __name__ == "__main__":
    main()
