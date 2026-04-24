"""Shared HDF5 schema writer for the paper-production DDA sweeps.

This is the Python analogue of block-VIEM.jl's
`viem_results/paper/_common.jl`. It parameterises the sweep HDF5 creation
by `(shape, material, size list)` so the per-(shape, material) scripts
under `dda_results/paper/` stay short and declarative.

The HDF5 schema is **bit-for-bit compatible** with block-VIEM.jl's
`create_paper_h5` output: same `/target/` datasets and attributes,
same dtypes, same shapes — so the downstream analysis scripts can
load either solver's results without modification (CLAUDE.md §6).

Material/size constants mirror block-VIEM.jl v0.7.1+ `_common.jl`:
  - N_LOW / N_HIGH / N_AU              (ComplexF64 refractive indices @ λ=0.638 μm)
  - A_EQ_FULL / A_EQ_AU                (volume-equivalent radii)
  - WL_PAPER = 0.638 μm, M_M_PAPER = 1.0
  - N_ALPHA_DEFAULT / N_BETA_DEFAULT / N_GAMMA_DEFAULT
"""
from __future__ import annotations

import os
import numpy as np
import h5py

# ──────────────────────────────────────────────────────────────────────
#  Material constants  (all at λ₀ = 0.638 μm; CLAUDE.md §2)
# ──────────────────────────────────────────────────────────────────────
N_LOW  = 1.5  + 0.01j                # n15 — low-index dielectric
N_20   = 2.0  + 0.0j                 # n20 — mid-index dielectric (non-absorbing,
                                     #       paper "high"; replaced n317 for
                                     #       DDA convergence in v0.7.6)
N_AU   = 0.17525 + 3.4830j           # Au, Johnson & Christy 1972 @ 0.638 μm
# Retained for reference / backward-compat of historical n317 HDF5 files:
N_HIGH = 3.17 + 0.16j                # n317 — legacy high-index (superseded by n20)

WL_PAPER  = 0.638                    # vacuum wavelength [μm]
M_M_PAPER = 1.0                      # vacuum background

# Volume-equivalent radii [μm]
# a_eq=0.5 μm removed (block-VIEM.jl v0.7.1) — for GRE × n317 and similar
# heavy cases the wavelength-driven lattice spacing blows up N_cuboid and
# pushes a single slot past 24 h wall time.
A_EQ_FULL = [0.05, 0.10, 0.20, 0.40]   # low / high-index
A_EQ_AU   = [0.05, 0.10, 0.20]         # Au only (≤ 0.2 μm); a_eq=0.5 dropped
                                       # because |m_p|≈3.49 combined with
                                       # the wavelength constraint makes N_cuboid
                                       # prohibitive (>24 h per slot).

# Multi-orientation grid (CLAUDE.md §2.5)
# Spheroid mode (sphere, oblate, doublet): block solver gets L = N_β = 5.
# GRE mode (β_gre ≠ 0): block solver gets L = N_α × N_β × N_γ = 100.
N_ALPHA_DEFAULT = 4
N_BETA_DEFAULT  = 5
N_GAMMA_DEFAULT = 5

# Hard-wired block-Krylov maxiter matching run_paper_sweep.py and VIEM
# (v0.7.6, 2026-04-24: bumped 100 → 200 for heavier cases; see CLAUDE.md §7.5).
# residual_history is NaN-padded to this width.
MAXITER_DEFAULT = 200


# ──────────────────────────────────────────────────────────────────────
#  Schema writer
# ──────────────────────────────────────────────────────────────────────
def create_paper_h5(filename,
                    *,
                    m_p,
                    a_eq_list,
                    bc_ratio,
                    ab_ratio,
                    gre_beta,
                    shape_kind="gre",
                    N_alpha=N_ALPHA_DEFAULT,
                    N_beta=N_BETA_DEFAULT,
                    N_gamma=N_GAMMA_DEFAULT,
                    light_source="(paper run)",
                    polarization="left-handed circular: "
                                 "E0_theta=1/sqrt(2), E0_phi=1j/sqrt(2)",
                    overwrite=False):
    """Write a DDA-side paper-sweep HDF5 file.

    Parameters
    ----------
    filename : str
        Output HDF5 path.
    m_p : complex or array-like of length 3
        Particle refractive index. Scalar → isotropic; length-3 → anisotropic
        along (x, y, z).
    a_eq_list : array-like
        Volume-equivalent radii [μm].
    bc_ratio, ab_ratio, gre_beta : float
        GRE shape parameters. For `shape_kind="doublet"` these are ignored by
        the runner but still written for schema parity with VIEM.
    shape_kind : {"gre", "doublet"}
        "gre" — GRE family driven by (bc_ratio, ab_ratio, gre_beta).
        "doublet" — touching equal-sphere doublet, monomer radius
        R = a_eq / 2^(1/3), gap = 0.1 R along z (CLAUDE.md §2.4).
    N_alpha, N_beta, N_gamma : int
        ZYZ Euler orientation grid divisions. The block solver receives
        L = N_beta when spheroid mode applies, else N_alpha·N_beta·N_gamma.
    light_source, polarization : str
        Provenance strings attached to /target/.
    overwrite : bool
        If False and `filename` exists, raises FileExistsError.
    """
    if shape_kind not in ("gre", "doublet"):
        raise ValueError(f'shape_kind must be "gre" or "doublet", got {shape_kind!r}')

    if os.path.isfile(filename) and not overwrite:
        raise FileExistsError(
            f"{filename} already exists. Pass overwrite=True to replace it.")

    # wl_m_m_pairs shape (1, 2); single λ for the paper
    wl_m_m_pairs = np.array([[WL_PAPER, M_M_PAPER]], dtype=np.float64)

    # m_p_xyz_list shape (1, 3): one entry, anisotropic along (x, y, z)
    if np.ndim(m_p) == 0:
        m_p_vec = np.array([m_p, m_p, m_p], dtype=np.complex128)
    else:
        m_p_vec = np.asarray(m_p, dtype=np.complex128)
        if m_p_vec.shape != (3,):
            raise ValueError(f"m_p must be scalar or length-3; got shape {m_p_vec.shape}")
    m_p_xyz_list = m_p_vec.reshape(1, 3)

    r_v_base_list = np.asarray(a_eq_list, dtype=np.float64)
    bc_ratio_list = np.array([bc_ratio], dtype=np.float64)
    ab_ratio_list = np.array([ab_ratio], dtype=np.float64)
    gre_beta_list = np.array([gre_beta], dtype=np.float64)

    num_orientations = N_alpha * N_beta * N_gamma

    # Spheroid α-expansion applies to any axially symmetric particle:
    # GRE with ab_ratio = 1 ∧ gre_beta = 0, OR a doublet along particle z.
    spheroid_mode = (shape_kind == "doublet") \
                    or (ab_ratio == 1.0 and gre_beta == 0.0)

    N_pairs = wl_m_m_pairs.shape[0]
    N_m_p   = m_p_xyz_list.shape[0]
    N_rv    = r_v_base_list.size
    N_bc    = bc_ratio_list.size
    N_ab    = ab_ratio_list.size
    N_bt    = gre_beta_list.size

    shape_geo  = (N_rv, N_bc, N_ab, N_bt)
    shape_cond = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt)
    shape_ori  = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, num_orientations)
    shape_ang  = (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, num_orientations, 3)

    with h5py.File(filename, "w") as f:
        grp = f.create_group("target")
        grp.attrs["light_source"]       = light_source
        grp.attrs["polarization_state"] = polarization
        grp.attrs["shape_kind"]         = shape_kind
        grp.attrs["N_alpha_ori"]        = int(N_alpha)
        grp.attrs["N_beta_ori"]         = int(N_beta)
        grp.attrs["N_gamma_ori"]        = int(N_gamma)
        grp.attrs["num_orientations"]   = int(num_orientations)
        grp.attrs["spheroid_mode"]      = 1 if spheroid_mode else 0

        grp.create_dataset("wl_m_m_pairs",  data=wl_m_m_pairs)
        grp.create_dataset("m_p_xyz_list",  data=m_p_xyz_list)
        grp.create_dataset("r_v_base_list", data=r_v_base_list)
        grp.create_dataset("bc_ratio_list", data=bc_ratio_list)
        grp.create_dataset("ab_ratio_list", data=ab_ratio_list)
        grp.create_dataset("gre_beta_list", data=gre_beta_list)

        sd = grp.create_group("simulated_data")
        sd.attrs["scattering_code"] = "block-DDA_Py (discrete dipole approximation)"
        sd.attrs["orientation"] = (
            "Deterministic ZYZ Euler angle grid (alpha, beta, gamma). "
            "alpha: equally spaced in [0,2pi), N_alpha divisions. "
            "beta: cos(beta) equally spaced in (-1,1), N_beta equal-area divisions. "
            "gamma: equally spaced in [0,2pi), N_gamma divisions. "
            "Ordering: alpha slowest, gamma fastest. "
            "Spheroid mode (ab_ratio=1 & gre_beta=0, or doublet): DDA solves "
            "only N_beta orientations; full grid filled analytically."
        )
        sd.attrs["units"]        = "r_ve:[um], euler_angles:[rad], C:[um^2], S:[um]"
        sd.attrs["S_definition"] = (
            "S(0)_theta = S11(0)+1j*S12(0), "
            "S(0)_phi = S22(0)-1j*S21(0), "
            "S(180) = (S11+S22+1j*S12-1j*S21)(180)/sqrt(2), "
            "per Mishchenko 2000"
        )

        def _ds(parent, name, shape, dtype, definition):
            d = parent.create_dataset(name, shape, dtype=dtype)
            d.attrs["definition"] = definition
            return d

        _ds(sd, "r_ve",            shape_geo,  np.float64,
            "volume-equivalent radius [um] from discretised particle volume")
        _ds(sd, "Euler_angles",    shape_ang,  np.float64,
            "ZYZ Euler angles (alpha,beta,gamma) rotating particle frame "
            "from lab frame [rad]")
        _ds(sd, "C_abs",           shape_ori,  np.float64,
            "absorption cross section per orientation [um^2]")
        _ds(sd, "C_ext",           shape_ori,  np.float64,
            "extinction cross section per orientation [um^2]")
        _ds(sd, "S_fw_PCAS_theta", shape_ori,  np.complex128,
            "forward-scattering amplitude S(0)_theta per orientation [um]")
        _ds(sd, "S_fw_PCAS_phi",   shape_ori,  np.complex128,
            "forward-scattering amplitude S(0)_phi per orientation [um]")
        _ds(sd, "S_bk_OCBS",       shape_ori,  np.complex128,
            "backward-scattering amplitude S(180) per orientation [um]")
        _ds(sd, "C_abs_mie",       shape_cond, np.float64,
            "Mie C_abs of volume-equivalent sphere [um^2]")
        _ds(sd, "C_ext_mie",       shape_cond, np.float64,
            "Mie C_ext of volume-equivalent sphere [um^2]")
        _ds(sd, "S_fw_PCAS_mie",   shape_cond, np.complex128,
            "Mie S(0) of volume-equivalent sphere [um]")
        _ds(sd, "S_bk_OCBS_mie",   shape_cond, np.complex128,
            "Mie S(180) of volume-equivalent sphere [um]")

        # Cost / diagnostic group for direct DDA-vs-VIEM comparison.
        # Populated by scripts/run_paper_sweep.py. Dimensioned on
        # (wl-pair, m_p, r_v, bc, ab, β) — one entry per slot solved.
        cst = grp.create_group("cost")
        cst.attrs["description"] = (
            "Per-slot cost and solver diagnostics recorded by the "
            "paper sweep runner. Enables direct end-to-end comparison "
            "between block-DDA_Py and block-VIEM.jl on identical slots.")
        cst.attrs["units"] = ("t_*:[s], peak_rss_bytes:[B], "
                              "lattice_lf:[um], solver_err:[1]")

        _ds(cst, "t_build_s",      shape_cond, np.float64,
            "wall time for shape geometry build (lattice + inside-target mask)")
        _ds(cst, "t_setup_s",      shape_cond, np.float64,
            "wall time for polarizability + FFT-init interaction matrix setup")
        _ds(cst, "t_solve_s",      shape_cond, np.float64,
            "wall time for the block-Krylov solve call")
        _ds(cst, "t_total_s",      shape_cond, np.float64,
            "end-to-end wall time (build + setup + solve + observables)")
        _ds(cst, "peak_rss_bytes", shape_cond, np.int64,
            "peak resident-set size [bytes] during this slot, "
            "sampled from /proc/self/status:VmRSS")
        _ds(cst, "n_cuboid",       shape_cond, np.int64,
            "lattice Nx*Ny*Nz (total cuboid cell count)")
        _ds(cst, "n_occ",          shape_cond, np.int64,
            "number of occupied dipoles (= 3·N_occ DoF)")
        _ds(cst, "lattice_lf",     shape_cond, np.float64,
            "lattice spacing d (after volume-preserving rescale in Target) [um]")
        _ds(cst, "iters",          shape_cond, np.int64,
            "block-Krylov iteration count at termination (spheroid-mode "
            "runs report the single-solve count)")
        _ds(cst, "converged",      shape_cond, np.int8,
            "1 if solver reached tol, 0 otherwise")
        _ds(cst, "solver_err",     shape_cond, np.float64,
            "final relative residual ‖B − A X‖_F / ‖B‖_F at termination")
        # Residual history: fixed-width (MAXITER) NaN-padded per-iter residuals.
        # MAXITER is hard-wired to 100 to match run_paper_sweep.py / VIEM.
        shape_cond_hist = shape_cond + (MAXITER_DEFAULT,)
        _ds(cst, "residual_history", shape_cond_hist, np.float64,
            "per-iteration relative residual, NaN-padded to length MAXITER "
            "(entry [...,k] is the residual after iteration k+1; NaN if "
            "k+1 > iter_fin). Enables convergence-profile figures.")

    print(f"Created {filename}")
    print(f"  shape_kind: {shape_kind}  "
          f"(bc={bc_ratio}  ab={ab_ratio}  β_gre={gre_beta}, "
          f"spheroid_mode={bool(spheroid_mode)})")
    print(f"  m_p  : {m_p_vec.tolist()}")
    print(f"  a_eq : {r_v_base_list.tolist()} μm")
    print(f"  N_α={N_alpha}  N_β={N_beta}  N_γ={N_gamma}  "
          f"→ {num_orientations} nominal orientations")
    print(f"  block-Krylov L (per shape slot) = "
          + (f"{N_beta} (spheroid mode)" if spheroid_mode
             else f"{num_orientations} (GRE mode)"))
    return filename
