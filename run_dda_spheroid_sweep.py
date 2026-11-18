"""Spheroid-only DDA parameter sweep (multi-wavelength).

Sweeps over (wl_0, m_m) pairs x log10(D_ve) x RI_real x log10(AR) on a regular grid.
Orientations: (cos_theta_o_half [0,1], phi_o [0,pi]) — reduced domain.
Analytical expansion from beta-only DDA solves is applied before writing.

Output HDF5 schema follows .claude/spheroid_h5_schema_spec.md for compatibility
with the downstream consumer (build_spheroid_lut.py).  Each wavelength gets its
own HDF5 group (e.g. wl_0p453/); shared grid axes live at the root level.

Output: dda_results/dda_results_spheroid_sweep.h5

With --shifted-ri the RI_real axis is solved in ONE shifted block-GMRES process per
(wl, D_ve, AR) slot instead of one solve per RI. All RI of a slot then share the
lattice built for the largest |m| of the RI axis, so the values differ from the
default mode at the level of the discretisation error. The HDF5 schema is the same.
The two modes cannot be mixed in one output file.
"""

import argparse
import itertools
import datetime
import hashlib
import os
import pathlib
import struct
import subprocess
import numpy as np
import h5py

from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

# ══════════════════════════════════════════════════════════════════════════════
# Settings — edit this section
# ══════════════════════════════════════════════════════════════════════════════
RNG_SEED    = 12345
MAX_TRY     = 1

# Host-medium + grid preset. Thin CLI (for the pcas_lut_schema adapter): --preset
# and --output; falls back to the DDA_SWEEP_PRESET env var, then "air". No args (or
# preset=air) reproduces the original run byte-for-byte; "liquid" targets the
# liquid CAS-v2 setup (water host at the two operating wavelengths);
# "liquid_prolate" is the same host but sweeps PROLATE spheroids over a narrow
# low-contrast RI band, for bacteria-like aquatic particles (see below).
_parser = argparse.ArgumentParser(
    description="block-DDA_Py spheroid parameter sweep (pcas_lut_schema producer)")
_parser.add_argument("--preset", choices=["air", "liquid", "liquid512", "liquid_prolate"],
                     default=os.environ.get("DDA_SWEEP_PRESET", "air"))
_parser.add_argument("--output", default=None, help="override the output HDF5 path")
_parser.add_argument("--delta-n-eff", type=float, default=0.0,
                     help="effective UNIAXIAL birefringence n_e - n_o (optic axis = c = spheroid "
                          "symmetry axis). 0 = isotropic (default). Nonzero builds an anisotropic "
                          "m_p_xyz=(n_o,n_o,n_e), mean-preserving about the RI axis (n_o=RI-dn/3). "
                          "Uniaxial-along-c stays axisymmetric so the analytic phi-expansion holds.")
_parser.add_argument("--shifted-ri", action="store_true",
                     help="solve all RI_real of a (wl, D_ve, AR) slot with one shifted "
                          "block-GMRES process on a lattice common to the RI axis (built for "
                          "the largest |m|). Isotropic particles only: not allowed with a "
                          "nonzero --delta-n-eff. Use for NEW output files only.")
_cli, _ = _parser.parse_known_args()
DELTA_N_EFF = float(_cli.delta_n_eff)
SHIFTED_RI  = bool(_cli.shifted_ri)
if SHIFTED_RI and DELTA_N_EFF != 0.0:
    raise SystemExit("--shifted-ri requires an isotropic particle: with a nonzero "
                     "--delta-n-eff the diagonal of the interaction matrix is not a scalar "
                     "shift. Drop --shifted-ri to use the per-RI solver.")

_PRESET = _cli.preset.lower()
if _PRESET == "air":
    OUTPUT_FILE = "dda_results/dda_results_spheroid_sweep.h5"
    # (wavelength [um], medium refractive index) pairs
    MEDIUM_CONDITIONS = [
        (0.453, 1.0),
        (0.638, 1.0),
        (0.834, 1.0),
    ]
elif _PRESET == "liquid":
    OUTPUT_FILE = "dda_results/dda_results_spheroid_sweep_liquid.h5"
    # Water host m_m at the two CAS-v2 wavelengths (637 nm, 773 nm).
    MEDIUM_CONDITIONS = [
        (0.637, 1.3315),
        (0.773, 1.3300),
    ]
elif _PRESET == "liquid512":
    # Third CAS-v2 wavelength (PCAS_Bayes_for_liquid_2wls, 2026-10-13): a SEPARATE table at 512 nm
    # with the same grid as "liquid". Water m_m from Hale & Querry (1973), linearly interpolated
    # (the same source gives 1.3315 / 1.3300 at 637 / 773 nm).
    OUTPUT_FILE = "dda_results/dda_results_spheroid_sweep_liquid512.h5"
    MEDIUM_CONDITIONS = [
        (0.512, 1.3345),
    ]
elif _PRESET == "liquid_prolate":
    OUTPUT_FILE = "dda_results/dda_results_spheroid_sweep_liquid_prolate.h5"
    # Same water host as "liquid"; the grid overrides below make it PROLATE.
    MEDIUM_CONDITIONS = [
        (0.637, 1.3315),
        (0.773, 1.3300),
    ]
else:
    raise SystemExit(
        f"unknown preset {_PRESET!r} "
        "(use 'air', 'liquid', 'liquid512' or 'liquid_prolate')")

if _cli.output is not None:
    OUTPUT_FILE = _cli.output

M_IMAG = 0.0     # imaginary part of particle refractive index (fixed)

# Swept parameters: (min, max, N_grid).  D_ve is LOG10-equidistant (stored as the
# log_D_ve_grid axis); RI_real and log_AR are linear-equidistant.
D_VE_RANGE     = (0.20, 1.20, 50)      # volume-equivalent diameter [um] endpoints; log10-spaced
D_VE_SPACING   = "log10"               # grid spacing for D_ve: "log10" -> log_D_ve_grid axis
RI_REAL_RANGE  = (1.35, 1.70, 15)    # Re(m_p), 0.025 step
LOG10_AR_RANGE = (0.0, 1.55, 16)     # log10(AR), AR = b/c = 1/beta; oblate-only (beta 1.0 -> 0.028; extended for strongly-oblate Kaolinite)

# The "liquid_prolate" preset re-points the shape axis to PROLATE (AR = b/c < 1, so the
# symmetry axis c is the LONG one) and narrows RI to the low-contrast band of aquatic
# bacteria. Target: marine bacterioplankton, whose relative refractive index is ~1.055
# (Stramski, Bricaud & Morel 2001, Appl. Opt. 40, 2929, Table 1 "HBAC") and whose natural
# cell volumes are 0.042-0.074 um^3, i.e. D_ve ~ 0.43-0.52 um, in small rod-shaped cells
# (Schut et al. 1993, Appl. Environ. Microbiol. 59, 2150).
#   - RI 1.34-1.47 covers relative 1.055 +- 3 sigma (sigma = 0.015, so relative 1.010-1.100)
#     at BOTH host indices: 1.010 x 1.3300 = 1.3433 and 1.100 x 1.3315 = 1.4646. The mean maps
#     to 1.4047 at 637 nm and 1.4032 at 773 nm. 8 nodes (step 0.019, finer than the oblate
#     LUT's 0.025) leaves the consumer's cubic B-spline margin over its 4-knot minimum.
#     CAUTION: sigma = 0.015 is an ESTIMATE of cell-to-cell spread, not a value from the two
#     cited papers -- Stramski et al. tabulate only the mean. Widen the axis if it is revised.
#   - log10(AR) -log10(3)..0 is elongation (length/width = 1/AR) from 1 to 3, the range reported
#     for natural marine cocci and short rods. Filamentous freshwater morphotypes reach
#     elongation >10 and are NOT covered -- they need a wider axis and a regenerated LUT.
#   - D_ve and the orientation grid are left identical to the oblate LUT so both can be used
#     in one mixture with matching node clamping. At the median D_ve of 0.48 um, elongation
#     3 gives 0.33 x 1.00 um -- the real dimensions of a marine rod.
# NOTE the geometry builder's documented bc_ratio range is [1.0, 7.0] (oblate). Prolate is
# mathematically sound there (volume is preserved, b = c*bc < c), and the realized lattice
# was checked at elongation 1-3: volume within 4% and aspect ratio within 9% of intent,
# the latter dominated by +-1 lattice cell at dpl=17. Its surface_interpolation_factor
# = int(4*cbrt(bc*ab)) does drop from 4 to 2 as the particle elongates, which is the wrong
# direction, but with gre_beta=0 the surface is an exact analytic ellipsoid so the coarser
# sampling did not degrade the realized shape. Re-check before extending to filaments.
if _PRESET == "liquid_prolate":
    RI_REAL_RANGE  = (1.34, 1.47, 8)     # Re(m_p), 0.0186 step; low-contrast bacteria band
    LOG10_AR_RANGE = (-0.4771, 0.0, 9)   # log10(AR = b/c); PROLATE, -log10(3): elongation 1 -> 3

# Orientation grid on reduced domain
N_COS_THETA_O_HALF = 13   # cos(theta_o) in [0, 1], equidistant
N_PHI_O            = 21   # phi_o in [0, pi], equidistant
# ══════════════════════════════════════════════════════════════════════════════


def _log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}")


def _wl_group_name(wl_0: float) -> str:
    """Convert wavelength to HDF5 group name: 0.453 -> 'wl_0p453'."""
    return f"wl_{str(wl_0).replace('.', 'p')}"


# ── grid generation ──────────────────────────────────────────────────────────
# D_ve is log10-equidistant: the stored axis is log_D_ve_grid (equidistant, per
# the pcas_lut_schema gridded-spheroid contract); D_ve_grid = 10**log_D_ve_grid.
_d_lo, _d_hi, _n_dve = D_VE_RANGE
log_D_ve_grid = np.linspace(np.log10(_d_lo), np.log10(_d_hi), _n_dve)  # shape (N_Dve,), equidistant
D_ve_grid     = 10.0 ** log_D_ve_grid                        # shape (N_Dve,) [um]
RI_real_grid  = np.linspace(*RI_REAL_RANGE)                   # shape (N_RI,)
log_AR_grid   = np.linspace(*LOG10_AR_RANGE)                  # shape (N_AR,)
AR_grid       = 10.0 ** log_AR_grid                           # bc_ratio values

cos_theta_o_half_grid = np.linspace(0, 1, N_COS_THETA_O_HALF)  # shape (N_u_half,)
phi_o_grid            = np.linspace(0, np.pi, N_PHI_O)         # shape (N_ph,)

N_Dve    = len(D_ve_grid)
N_RI     = len(RI_real_grid)
N_AR     = len(log_AR_grid)
N_u_half = len(cos_theta_o_half_grid)
N_ph     = len(phi_o_grid)


# ── helper functions ─────────────────────────────────────────────────────────

def _build_spheroid_geometry(r_v_base, bc_ratio, wl_0, m_p_xyz):
    """Build spheroid lattice geometry (ab_ratio=1, gre_beta=0).

    Returns (name, lattice_n, lattice_lf, grid, is_in).
    """
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(
        r_v_base, bc_ratio, 1.0, 0.0, wl_0, m_p_xyz)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist  = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(
        gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return gre.name, lattice_n, gre.lattice_lf, grid, is_in


def _run_dda_spheroid(target, wl_0, m_m, cos_theta_o_half, phi_o):
    """DDA solve for spheroids on reduced-domain orientation grid.

    Solves N_u_half orientations at phi_o=0 (alpha=0), then expands
    analytically to the full (N_u_half, N_ph) grid.

    Orientation mapping:
        theta_o = arccos(cos_theta_o)  <->  beta  (polar Euler angle)
        phi_o                          <->  alpha (azimuthal Euler angle)

    Analytical expansion (spheroid symmetry):
        S_s(phi_o) = A + B * exp(2j * phi_o)
        S_p(phi_o) = A - B * exp(2j * phi_o)
    where A = (S_s(0) + S_p(0))/2,  B = (S_s(0) - S_p(0))/2.

    Parameters
    ----------
    cos_theta_o_half : ndarray, shape (N_u_half,)
        cos(theta_o) grid on [0, 1].
    phi_o : ndarray, shape (N_ph,)
        phi_o grid on [0, pi] [rad].

    Returns
    -------
    S_fw_theta : ndarray, shape (N_u_half, N_ph), complex128
    S_fw_phi   : ndarray, shape (N_u_half, N_ph), complex128
    converged  : bool
    """
    N_u = len(cos_theta_o_half)
    N_p = len(phi_o)

    # Convert cos_theta_o to beta (Euler polar angle)
    beta = np.arccos(cos_theta_o_half)  # shape (N_u,)
    euler_beta_only = np.column_stack([
        np.zeros(N_u), beta, np.zeros(N_u),
    ])  # shape (N_u, 3): alpha=0, gamma=0

    for i_try in range(1, MAX_TRY + 1):
        inc = IncidentField(wl_0, m_m, euler_beta_only)
        dd  = DiscreteDipoles(target, inc)
        dd.set_interaction_matrix()
        dd.solve_matrix_equation()

        _log(f"    try {i_try}/{MAX_TRY} [spheroid, L={N_u}]: "
             f"{'converged' if dd.converge else 'not converged'}")
        if dd.converge:
            break
    else:
        nan_2d = np.full((N_u, N_p), np.nan)
        return nan_2d, nan_2d, False

    # Observables at phi_o=0 (alpha=0)
    S_s_0, S_p_0 = dd.compute_PCAS_observable_S_fw()  # shape (N_u,)

    # Analytical expansion to (N_u_half, N_ph)
    exp_2p = np.exp(2j * phi_o)  # shape (N_ph,)

    A_fw = (S_s_0 + S_p_0) / 2  # shape (N_u,)
    B_fw = (S_s_0 - S_p_0) / 2  # shape (N_u,)

    # S_fw_theta[j, i] = A[j] + B[j] * exp(2j * phi_o[i])
    # shape (N_u_half, N_ph)
    S_fw_theta = A_fw[:, np.newaxis] + B_fw[:, np.newaxis] * exp_2p[np.newaxis, :]
    S_fw_phi   = A_fw[:, np.newaxis] - B_fw[:, np.newaxis] * exp_2p[np.newaxis, :]

    return S_fw_theta, S_fw_phi, True


def _run_dda_spheroid_shifted(target, wl_0, m_m, m_p_list, cos_theta_o_half, phi_o):
    """Shifted block-GMRES solve of several refractive indices on one lattice.

    Counterpart of `_run_dda_spheroid` for --shifted-ri. The lattice of `target`
    is shared by all refractive indices in `m_p_list`. The reduced-domain
    orientation grid and the analytical phi_o expansion are the same.

    Parameters
    ----------
    m_p_list : ndarray, shape (N_m,), complex128
        Isotropic particle refractive indices to solve.
    cos_theta_o_half : ndarray, shape (N_u_half,)
    phi_o : ndarray, shape (N_ph,)   [rad]

    Returns
    -------
    S_fw_theta : ndarray, shape (N_m, N_u_half, N_ph), complex128
    S_fw_phi   : ndarray, shape (N_m, N_u_half, N_ph), complex128
        NaN for refractive indices that did not converge.
    converged  : ndarray, shape (N_m,), bool
    """
    N_u = len(cos_theta_o_half)

    beta = np.arccos(cos_theta_o_half)  # shape (N_u,)
    euler_beta_only = np.column_stack([
        np.zeros(N_u), beta, np.zeros(N_u),
    ])  # shape (N_u, 3): alpha=0, gamma=0

    inc = IncidentField(wl_0, m_m, euler_beta_only)
    dd  = DiscreteDipoles(target, inc)
    dd.set_interaction_matrix_shifted(m_p_list)
    dd.solve_matrix_equation_shifted()
    obs = dd.compute_observables_shifted()

    # Observables at phi_o=0 (alpha=0), shape (N_m, N_u)
    S_s_0 = obs["S_fw_PCAS_theta"]
    S_p_0 = obs["S_fw_PCAS_phi"]

    # Analytical expansion to (N_m, N_u_half, N_ph)
    exp_2p = np.exp(2j * phi_o)  # shape (N_ph,)
    A_fw = (S_s_0 + S_p_0) / 2   # shape (N_m, N_u)
    B_fw = (S_s_0 - S_p_0) / 2   # shape (N_m, N_u)
    S_fw_theta = A_fw[:, :, np.newaxis] + B_fw[:, :, np.newaxis] * exp_2p[np.newaxis, np.newaxis, :]
    S_fw_phi   = A_fw[:, :, np.newaxis] - B_fw[:, :, np.newaxis] * exp_2p[np.newaxis, np.newaxis, :]

    return S_fw_theta, S_fw_phi, dd.converge_shifted


def _sweep_wavelength_shifted(grp, grp_name, wl_0, m_m):
    """--shifted-ri sweep of one wavelength group: one solve per (D_ve, AR) slot.

    The lattice of a slot is always built for the largest |m| of the WHOLE RI axis,
    also when only part of the axis is left to solve after a resume. The stored
    values therefore do not depend on how a run was interrupted.
    """
    m_p_grid  = RI_real_grid + 1j * M_IMAG                    # shape (N_RI,), complex128
    m_lattice = m_p_grid[np.argmax(np.abs(m_p_grid))]
    m_p_xyz   = np.array([m_lattice, m_lattice, m_lattice])   # lattice-defining index

    for (i_dve, D_ve), (i_ar, bc_ratio) in \
            itertools.product(enumerate(D_ve_grid), enumerate(AR_grid)):

        # Skip the slot only if every RI is already converged
        done = grp['converged'][i_dve, :, i_ar]               # shape (N_RI,)
        if np.all(done):
            continue
        todo = np.flatnonzero(~done)                          # RI indices left to solve

        r_v_base = D_ve / 2.0

        print("=" * 64)
        try:
            gre_name, lattice_n, lattice_lf, grid, is_in = \
                _build_spheroid_geometry(r_v_base, bc_ratio, wl_0, m_p_xyz)
        except (ValueError, IndexError):
            _log(f"[{grp_name}] D_ve={D_ve:.4f} um  "
                 f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                 f"-- geometry failed, skipping")
            continue

        n_occ = int(np.sum(is_in))
        if n_occ == 0:
            _log(f"[{grp_name}] D_ve={D_ve:.4f} um  "
                 f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                 f"-- no dipoles, skipping")
            continue

        _log(f"[{grp_name}] D_ve={D_ve:.4f} um  "
             f"RI={RI_real_grid[todo[0]]:.3f}..{RI_real_grid[todo[-1]]:.3f}+{M_IMAG:.4f}j "
             f"({len(todo)}/{N_RI} shifts, lattice for |m|={abs(m_lattice):.3f})  "
             f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
             f"d={lattice_lf:.5f} um  N_dip={n_occ}  L_solve={N_u_half}")

        target = Target(gre_name, lattice_n, lattice_lf,
                        grid, is_in, m_p_xyz, r_v_base)

        try:
            S_fw_theta, S_fw_phi, converged = \
                _run_dda_spheroid_shifted(target, wl_0, m_m, m_p_grid[todo],
                                          cos_theta_o_half_grid, phi_o_grid)
        except KeyboardInterrupt:
            _log("Interrupted -- file closed cleanly.")
            raise SystemExit(0)

        # Write results (NaN and converged=False for RI that did not converge)
        N = slice(None)
        for j, i_ri in enumerate(todo):
            idx3 = (i_dve, int(i_ri), i_ar)
            grp['S_fw_theta_re'][idx3 + (N, N)] = S_fw_theta[j].real
            grp['S_fw_theta_im'][idx3 + (N, N)] = S_fw_theta[j].imag
            grp['S_fw_phi_re'  ][idx3 + (N, N)] = S_fw_phi[j].real
            grp['S_fw_phi_im'  ][idx3 + (N, N)] = S_fw_phi[j].imag
            grp['converged'    ][idx3]           = bool(converged[j])

        _log(f"  converged={int(converged.sum())}/{len(todo)}  "
             f"S_fw_t(mean)={np.nanmean(S_fw_theta):.4g}  "
             f"S_fw_p(mean)={np.nanmean(S_fw_phi):.4g}")


# ── provenance (pcas_lut_schema contract) ────────────────────────────────────

# vendored from pcas_lut_schema/reference/canonical_hash.py (contract v0.1.0)
# language-independent, bit-exact byte encoding; do not edit — re-vendor from source.
def _canonical_encode(v):
    if v is None:
        return b"n;"
    if isinstance(v, bool):
        return b"b1;" if v else b"b0;"
    if isinstance(v, int):
        return b"i" + str(v).encode("ascii") + b";"
    if isinstance(v, float):
        return b"f" + struct.pack(">d", v)
    if isinstance(v, str):
        b = v.encode("utf-8")
        return b"s" + str(len(b)).encode("ascii") + b":" + b
    if isinstance(v, (list, tuple)):
        return b"l" + str(len(v)).encode("ascii") + b":" + b"".join(_canonical_encode(x) for x in v)
    if isinstance(v, dict):
        items = sorted(v.items(), key=lambda kv: str(kv[0]))
        body = b"".join(_canonical_encode(str(k)) + _canonical_encode(val) for k, val in items)
        return b"d" + str(len(items)).encode("ascii") + b":" + body
    raise TypeError(f"canonical_hash: unsupported type {type(v).__name__}")


def _provenance_attrs():
    """Provenance block required by the pcas_lut_schema contract (v0.1.0).

    Records the producer git SHA, a dirty-tree flag (tracked files only), the
    canonical config hash (contract byte encoding), and a UTC timestamp, so a
    generated LUT is traceable to the exact producer version and settings.
    """
    repo_dir = pathlib.Path(__file__).resolve().parent

    def _git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_dir), *args],
                stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""

    config = {
        "preset": _PRESET,
        "medium_conditions": MEDIUM_CONDITIONS,
        "m_imag": M_IMAG,
        "d_ve_range": D_VE_RANGE,
        "d_ve_spacing": D_VE_SPACING,
        "ri_real_range": RI_REAL_RANGE,
        "log10_ar_range": LOG10_AR_RANGE,
        "n_cos_theta_o_half": N_COS_THETA_O_HALF,
        "n_phi_o": N_PHI_O,
        "rng_seed": RNG_SEED,
    }
    # Only record the uniaxial effective birefringence when nonzero, so an isotropic
    # (dn=0) run keeps the SAME config_hash as the existing isotropic LUTs.
    if DELTA_N_EFF != 0.0:
        config["delta_n_eff"] = DELTA_N_EFF
    # Same rule for the shifted RI sweep: recorded only when on, so the default
    # mode keeps its config_hash. The lattice policy changes the stored values.
    if SHIFTED_RI:
        config["shifted_ri"] = True
        config["lattice_policy"] = "max_abs_m"
    config_hash = hashlib.sha256(_canonical_encode(config)).hexdigest()

    return {
        "producer_repo":    "block-DDA_Py",
        "producer_version": "0.8.1",
        "git_sha":          _git("rev-parse", "HEAD"),
        # tracked-file changes only; untracked artifacts do not count as dirty
        "git_dirty":        bool(_git("status", "--porcelain", "--untracked-files=no")),
        "contract_version": "0.1.0",
        "config_hash":      config_hash,
        "created_utc":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


# ── HDF5 file creation ───────────────────────────────────────────────────────

def _create_h5(filepath):
    """Create output HDF5 file with shared grids and per-wavelength groups."""
    shape_5d   = (N_Dve, N_RI, N_AR, N_u_half, N_ph)
    shape_conv = (N_Dve, N_RI, N_AR)

    with h5py.File(filepath, "w") as f:
        # Root-level provenance attributes
        f.attrs['m_m']               = MEDIUM_CONDITIONS[0][1]
        f.attrs['m_imag']            = M_IMAG
        f.attrs['delta_n_eff']       = DELTA_N_EFF
        f.attrs['rng_seed']          = RNG_SEED
        f.attrs['solver_tol']        = 1e-2
        f.attrs['block_dda_version'] = '0.8.1'
        # Written only in --shifted-ri mode; their absence marks a per-RI-lattice file.
        if SHIFTED_RI:
            f.attrs['shifted_ri']     = True
            f.attrs['lattice_policy'] = 'max_abs_m'

        # Provenance block (pcas_lut_schema contract) — additive '/provenance' group
        prov = f.create_group("provenance")
        for k, v in _provenance_attrs().items():
            prov.attrs[k] = v

        # Shared grid axis arrays (at root)
        f.create_dataset("log_D_ve_grid",          data=log_D_ve_grid,          dtype=np.float64)
        f.create_dataset("RI_real_grid",            data=RI_real_grid,           dtype=np.float64)
        f.create_dataset("log_AR_grid",             data=log_AR_grid,            dtype=np.float64)
        f.create_dataset("cos_theta_o_half_grid",   data=cos_theta_o_half_grid,  dtype=np.float64)
        f.create_dataset("phi_o_grid",              data=phi_o_grid,             dtype=np.float64)

        # Per-wavelength groups
        for wl_0, m_m in MEDIUM_CONDITIONS:
            grp_name = _wl_group_name(wl_0)
            grp = f.create_group(grp_name)
            grp.attrs['wl_0'] = wl_0
            grp.attrs['m_m']  = m_m

            for name in ("S_fw_theta_re", "S_fw_theta_im",
                          "S_fw_phi_re",   "S_fw_phi_im"):
                ds = grp.create_dataset(name, shape=shape_5d, dtype=np.float64)
                ds[:] = np.nan

            grp.create_dataset("converged", shape=shape_conv, dtype=bool,
                               data=np.zeros(shape_conv, dtype=bool))

    _log(f"Created {filepath}")
    _log(f"  Wavelengths: {[wl for wl, _ in MEDIUM_CONDITIONS]}")
    _log(f"  Grid: N_Dve={N_Dve}, N_RI={N_RI}, N_AR={N_AR}")
    _log(f"  Orientation: N_cos_theta_o_half={N_u_half}, N_phi_o={N_ph}")
    n_per_wl = N_Dve * N_RI * N_AR
    _log(f"  DDA conditions per wavelength: {n_per_wl}  "
         f"(total: {n_per_wl * len(MEDIUM_CONDITIONS)})")


# ── main sweep ───────────────────────────────────────────────────────────────

if not pathlib.Path(OUTPUT_FILE).exists():
    _create_h5(OUTPUT_FILE)

with h5py.File(OUTPUT_FILE, "r+") as h5:
    # The two lattice policies must not be mixed in one file.
    _file_shifted = bool(h5.attrs.get('shifted_ri', False))
    if _file_shifted != SHIFTED_RI:
        raise SystemExit(
            f"{OUTPUT_FILE} was created {'with' if _file_shifted else 'without'} "
            f"--shifted-ri and cannot be resumed {'with' if SHIFTED_RI else 'without'} it "
            "(the RI lattice policy differs). Use a new --output file.")

    for wl_0, m_m in MEDIUM_CONDITIONS:
        grp_name = _wl_group_name(wl_0)

        # Create group if resuming a file that was created before this
        # wavelength was added
        if grp_name not in h5:
            shape_5d   = (N_Dve, N_RI, N_AR, N_u_half, N_ph)
            shape_conv = (N_Dve, N_RI, N_AR)
            grp = h5.create_group(grp_name)
            grp.attrs['wl_0'] = wl_0
            grp.attrs['m_m']  = m_m
            for name in ("S_fw_theta_re", "S_fw_theta_im",
                          "S_fw_phi_re",   "S_fw_phi_im"):
                ds = grp.create_dataset(name, shape=shape_5d, dtype=np.float64)
                ds[:] = np.nan
            grp.create_dataset("converged", shape=shape_conv, dtype=bool,
                               data=np.zeros(shape_conv, dtype=bool))

        grp = h5[grp_name]
        _log(f"── Wavelength {wl_0} um (group: {grp_name}) ──")

        if SHIFTED_RI:
            _sweep_wavelength_shifted(grp, grp_name, wl_0, m_m)
            continue

        for (i_dve, D_ve), (i_ri, RI_real), (i_ar, bc_ratio) in \
                itertools.product(
                    enumerate(D_ve_grid),
                    enumerate(RI_real_grid),
                    enumerate(AR_grid)):

            idx3 = (i_dve, i_ri, i_ar)

            # Skip if already computed (converged flag is True)
            if grp['converged'][idx3]:
                continue

            r_v_base = D_ve / 2.0
            # Uniaxial effective birefringence: optic axis = z = c (spheroid symmetry axis),
            # mean-preserving about the RI axis so (2 n_o + n_e)/3 = RI_real. dn=0 -> isotropic.
            n_o = RI_real - DELTA_N_EFF / 3.0
            n_e = RI_real + 2.0 * DELTA_N_EFF / 3.0
            m_p_o = n_o + 1j * M_IMAG
            m_p_e = n_e + 1j * M_IMAG
            m_p_xyz = np.array([m_p_o, m_p_o, m_p_e])  # (x,y in-plane = n_o ; z = c = n_e)

            # Build geometry
            print("=" * 64)
            try:
                gre_name, lattice_n, lattice_lf, grid, is_in = \
                    _build_spheroid_geometry(r_v_base, bc_ratio, wl_0, m_p_xyz)
            except (ValueError, IndexError):
                _log(f"[{grp_name}] D_ve={D_ve:.4f} um  RI={RI_real:.3f}  "
                     f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                     f"-- geometry failed, skipping")
                continue

            n_occ = int(np.sum(is_in))
            if n_occ == 0:
                _log(f"[{grp_name}] D_ve={D_ve:.4f} um  RI={RI_real:.3f}  "
                     f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                     f"-- no dipoles, skipping")
                continue

            _log(f"[{grp_name}] D_ve={D_ve:.4f} um  RI={RI_real:.3f}+{M_IMAG:.4f}j  "
                 f"AR={bc_ratio:.4f} (log={log_AR_grid[i_ar]:.3f})  "
                 f"d={lattice_lf:.5f} um  N_dip={n_occ}  L_solve={N_u_half}")

            target = Target(gre_name, lattice_n, lattice_lf,
                            grid, is_in, m_p_xyz, r_v_base)

            try:
                S_fw_theta, S_fw_phi, converged = \
                    _run_dda_spheroid(target, wl_0, m_m,
                                      cos_theta_o_half_grid, phi_o_grid)
            except KeyboardInterrupt:
                _log("Interrupted -- file closed cleanly.")
                raise SystemExit(0)

            # Write results
            N = slice(None)
            grp['S_fw_theta_re'][idx3 + (N, N)] = S_fw_theta.real
            grp['S_fw_theta_im'][idx3 + (N, N)] = S_fw_theta.imag
            grp['S_fw_phi_re'  ][idx3 + (N, N)] = S_fw_phi.real
            grp['S_fw_phi_im'  ][idx3 + (N, N)] = S_fw_phi.imag
            grp['converged'    ][idx3]           = converged

            _log(f"  converged={converged}  "
                 f"S_fw_t(mean)={np.nanmean(S_fw_theta):.4g}  "
                 f"S_fw_p(mean)={np.nanmean(S_fw_phi):.4g}")

_log("All conditions completed.")
