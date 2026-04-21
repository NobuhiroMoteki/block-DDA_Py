"""Pre-run resource estimator for block-DDA_Py paper sweeps.

Reads the same HDF5 file consumed by `run_dda.py` (created by
`dda_results/create_h5py.ipynb`) and prints N_cuboid, N_occ, peak RSS,
estimated setup / solve / total wall time per shape slot, plus sweep
totals. Warns if any single condition exceeds 24 h or if peak RSS
exceeds machine memory.

Parallel to `block-VIEM.jl/viem_results/estimate_cost.jl` — same
threshold / same layout so DDA and VIEM estimates can be compared.

Cost model (README §Performance):
    Peak RSS   ≈ (1152 + 768·L) · N_cuboid   bytes      (exact, from Goodman FFT layout)
    T_solve    ≈ N_iter · c1 · L · N_cuboid · log(N_cuboid)  seconds
    T_setup    ≈ c2 · N_cuboid                            seconds (Green's function + 1 FFT)

Calibration defaults from README benchmark
  (r_v=0.5 μm, N_cuboid ≈ 100k, L=800, general mode ≈ 6 min)
are conservative.  Override via env vars when calibrating against a
heavier slot.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/estimate_cost.py <sweep.hdf5>

Env vars (optional):
    DDA_N_ITER_EST                 assumed BiCGSTAB iterations   (default 50)
    DDA_T_ITER_SEC_PER_UNIT        seconds per (L · N · log N)   (default 1e-8)
    DDA_T_SETUP_SEC_PER_NCUBOID    seconds per N_cuboid          (default 1e-5)
    DDA_MEM_LIMIT_GB               override available memory     (default /proc/meminfo)
"""
import os
import sys
import argparse
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from shape_model.two_sphere_cluster import two_sphere_cluster_shape_model


RNG_SEED = 12345
DPL_DEFAULT = 17

N_ITER_EST              = int(os.getenv("DDA_N_ITER_EST", "50"))
T_ITER_SEC_PER_UNIT     = float(os.getenv("DDA_T_ITER_SEC_PER_UNIT", "1e-8"))
T_SETUP_SEC_PER_NCUBOID = float(os.getenv("DDA_T_SETUP_SEC_PER_NCUBOID", "1e-5"))

ESCALATION_HOURS = 24.0
RSS_FRACTION_LIMIT = 0.9


def machine_available_gb():
    if "DDA_MEM_LIMIT_GB" in os.environ:
        return float(os.environ["DDA_MEM_LIMIT_GB"])
    try:
        with open("/proc/meminfo") as fp:
            for line in fp:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("nan")


def _is_spheroid(shape_kind, ab_ratio, gre_beta):
    return shape_kind == "doublet" or (ab_ratio == 1.0 and gre_beta == 0.0)


def _build_shape_lattice(shape_kind, r_v, bc, ab, gre_beta, wl_min, m_p_max):
    """Build the cuboid lattice for one shape slot with worst-case (λ_min, |m_p|_max)
    and return (lattice_n, N_occ)."""
    m_p_xyz = np.array([m_p_max, m_p_max, m_p_max], dtype=np.complex64)
    if shape_kind == "doublet":
        mdl = two_sphere_cluster_shape_model(r_v, wl_min, m_p_xyz)
        lattice_n, _, _, is_in = mdl.build()
    else:
        mdl = gaussian_ellipsoid_shape_model(r_v, bc, ab, gre_beta, wl_min, m_p_xyz)
        rng = np.random.default_rng(RNG_SEED)
        r_pts, _ = mdl.compute_r_points_on_GRE(rng)
        _, lattice_n, grid = mdl.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
        dist = mdl.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
        is_in = mdl.extract_lattice_address_in_GRE_volume(
            mdl.lattice_lf, mdl.distance_factor, lattice_n, dist)
    return lattice_n, int(is_in.sum())


def _estimate_slot(n_cuboid, L_solve, L_orient):
    rss_bytes  = (1152 + 768 * L_solve) * n_cuboid
    rss_gb     = rss_bytes / 1024 ** 3
    t_setup_s  = T_SETUP_SEC_PER_NCUBOID * n_cuboid
    n_log_n    = n_cuboid * np.log(max(n_cuboid, 2))
    t_iter_s   = T_ITER_SEC_PER_UNIT * L_solve * n_log_n
    t_solve_s  = N_ITER_EST * t_iter_s
    t_per_ori_s = t_solve_s / max(L_orient, 1)
    t_total_s  = t_setup_s + t_solve_s
    return rss_gb, t_setup_s, t_solve_s, t_per_ori_s, t_total_s


def _fmt_time(s):
    if s < 60:    return f"{s:.1f}s"
    if s < 3600:  return f"{s/60:.1f}m"
    if s < 86400: return f"{s/3600:.2f}h"
    return f"{s/86400:.2f}d"


def _fmt_gb(g):
    if g < 1.0:   return f"{g*1024:.0f} MB"
    return f"{g:.2f} GB"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("h5path", help="sweep HDF5 file (same schema as run_dda.py)")
    args = parser.parse_args()

    if not os.path.isfile(args.h5path):
        print(f"error: HDF5 file not found: {args.h5path}", file=sys.stderr)
        sys.exit(2)

    mem_gb = machine_available_gb()

    bar = "=" * 120
    rule = "─" * 120

    print(bar)
    print("block-DDA_Py  pre-run estimator")
    print(f"file       : {args.h5path}")
    print(f"calib (env): N_ITER_EST={N_ITER_EST}  "
          f"T_ITER_SEC_PER_UNIT={T_ITER_SEC_PER_UNIT:g}  "
          f"T_SETUP_SEC_PER_NCUBOID={T_SETUP_SEC_PER_NCUBOID:g}")
    mem_s = "?" if np.isnan(mem_gb) else f"{mem_gb:.1f} GB"
    print(f"machine    : MemAvailable ≈ {mem_s}")
    print(f"escalation : any slot t_total > {ESCALATION_HOURS} h "
          f"or RSS > {RSS_FRACTION_LIMIT*100:.0f}% of MemAvailable → flagged ⚠️")
    print(bar)

    with h5py.File(args.h5path, "r") as f:
        t = f["target"]

        n_alpha = int(t.attrs["N_alpha_ori"])
        n_beta  = int(t.attrs["N_beta_ori"])
        n_gamma = int(t.attrs["N_gamma_ori"])
        shape_kind = (t.attrs["shape_kind"].decode()
                      if isinstance(t.attrs.get("shape_kind", "gre"), bytes)
                      else str(t.attrs.get("shape_kind", "gre")))

        wl_pairs = t["wl_m_m_pairs"][:]         # (N_pairs, 2)
        m_p_list = t["m_p_xyz_list"][:]         # (N_mp, 3)
        r_v_list = t["r_v_base_list"][:]
        bc_list  = t["bc_ratio_list"][:]
        ab_list  = t["ab_ratio_list"][:]
        bt_list  = t["gre_beta_list"][:]

        n_pairs = wl_pairs.shape[0]
        n_mp    = m_p_list.shape[0]
        n_rv, n_bc, n_ab, n_bt = len(r_v_list), len(bc_list), len(ab_list), len(bt_list)

        wl_min  = float(np.min(wl_pairs[:, 0]))
        m_p_max = float(np.max(np.abs(m_p_list)))
        n_inner = n_pairs * n_mp
        n_orient_total = n_alpha * n_beta * n_gamma

        print(f"sweep       : shape_kind={shape_kind}  {n_pairs} wl-pairs × "
              f"{n_mp} m_p × {n_rv} r_v × {n_bc} bc × {n_ab} ab × {n_bt} β")
        print(f"orientations: N_α={n_alpha} N_β={n_beta} N_γ={n_gamma} "
              f"(L_total={n_orient_total})")
        print(f"worst-case  : wl_0_min={wl_min:.4f} μm  |m_p|_max={m_p_max:.4f}")
        print()

        hdr = ("{:<13} {:>8} {:>8} {:>8} {:>8} {:>9} {:>8} {:>4} {:>9} {:>9} {:>10} {:>10}"
               .format("shape_idx", "r_v(μm)", "bc", "ab", "β_gre",
                       "N_cub", "N_occ", "L", "RSS",
                       "t_setup", "t_per_ori", "t_total"))
        print(rule)
        print(hdr)
        print(rule)

        slots = []
        any_warn = False
        for i_rv in range(n_rv):
            for i_bc in range(n_bc):
                for i_ab in range(n_ab):
                    for i_bt in range(n_bt):
                        r_v  = float(r_v_list[i_rv])
                        bc   = float(bc_list[i_bc])
                        ab   = float(ab_list[i_ab])
                        beta = float(bt_list[i_bt])

                        lattice_n, n_occ = _build_shape_lattice(
                            shape_kind, r_v, bc, ab, beta, wl_min, m_p_max)
                        n_cub = int(np.prod(lattice_n))

                        spheroid = _is_spheroid(shape_kind, ab, beta)
                        L_solve = n_beta if spheroid else n_orient_total

                        rss_gb, t_setup, t_solve, t_per_ori, t_total = _estimate_slot(
                            n_cub, L_solve, n_orient_total)

                        warn_t   = t_total * n_inner > ESCALATION_HOURS * 3600
                        warn_rss = (not np.isnan(mem_gb)) and rss_gb > RSS_FRACTION_LIMIT * mem_gb
                        any_warn = any_warn or warn_t or warn_rss

                        flag = " ⚠️" if (warn_t or warn_rss) else ""
                        mode = "S" if spheroid else "G"
                        print("({:1d},{:1d},{:1d},{:1d})   {:>8.4f} {:>8.2f} {:>8.2f} "
                              "{:>8.2f} {:>9d} {:>8d} {:>3d}{} {:>9} {:>9} {:>10} {:>10}{}"
                              .format(i_rv, i_bc, i_ab, i_bt,
                                      r_v, bc, ab, beta,
                                      n_cub, n_occ, L_solve, mode,
                                      _fmt_gb(rss_gb),
                                      _fmt_time(t_setup),
                                      _fmt_time(t_per_ori),
                                      _fmt_time(t_total * n_inner),
                                      flag))
                        slots.append((r_v, bc, ab, beta, rss_gb, t_total, spheroid))

        print(rule)
        t_sweep = sum(ts * n_inner for *_, ts, _ in [(r, b, a, bt, ts, sp) for r, b, a, bt, rss, ts, sp in slots])
        rss_peak = max(s[4] for s in slots)
        print(f"SWEEP TOTAL : {len(slots)} shape slots × {n_inner} (wl × m_p) inner pts each")
        print(f"  estimated wall time      = {_fmt_time(t_sweep)}")
        mem_s2 = "?" if np.isnan(mem_gb) else f"{mem_gb:.1f} GB"
        print(f"  estimated peak RSS       = {_fmt_gb(rss_peak)}  (machine: {mem_s2})")
        print(f"  L columns: S = spheroid mode (L=N_β={n_beta}), "
              f"G = general mode (L={n_orient_total})")

        print()
        if any_warn:
            print(f"⚠️  Some slots exceed the escalation threshold "
                  f"({ESCALATION_HOURS} h or {RSS_FRACTION_LIMIT*100:.0f}% of MemAvailable).")
            print("   Confirm with the user before launching run_dda.py.")
            sys.exit(1)
        print("✓ All slots within escalation thresholds.")
        print(bar)


if __name__ == "__main__":
    main()
