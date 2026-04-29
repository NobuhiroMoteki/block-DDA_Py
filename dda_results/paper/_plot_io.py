"""HDF5 loaders for paper-production results (block-DDA_Py / block-VIEM.jl / MSTM).

Normalizes the axis order across the two solvers so that downstream plotting
code sees a single layout regardless of the source.

Normalized return shapes
------------------------
sweep observables (per-orientation):       (N_rv, N_orient)         [μm² or μm]
sweep Mie reference (scalar per a_eq):     (N_rv,)                  [complex for amplitudes]
Euler angles per orientation:              (N_rv, N_orient, 3)      [rad]
cost diagnostics:                          (N_rv,)
convergence sweep observables:             (N_steps,)               [scalar reference (single orient)]
convergence Mie reference:                 scalar (or None)
rhs_scaling per (L, a_eq):                 (N_L, N_rv)
MSTM observables:                          (N_beta, N_rv)
MSTM beta grid:                            (N_beta,)

Side conventions
----------------
- DDA HDF5 sweep dataset shape: (N_pairs, N_m_p, N_rv, N_bc, N_ab, N_bt, N_orient)
  → squeeze singletons → (N_rv, N_orient).
- VIEM HDF5 sweep dataset shape: (N_orient, N_pairs, N_m_p, N_bc, N_rv, N_ab, N_bt)
  (Julia column-major; axes mirrored from DDA) → squeeze → (N_orient, N_rv) → .T.
- DDA rhs_scaling: (L, N_rv, 1, 1, 1) → squeeze → (L, N_rv).
- VIEM rhs_scaling: (1, 1, 1, N_rv, L) → squeeze → (N_rv, L) → .T.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal

import numpy as np
import h5py


Side = Literal["dda", "viem"]

# ----- sweep observables (production HDF5) -----------------------------------

_OBS_REAL = ("C_ext", "C_abs")
_OBS_CPLX = ("S_fw_PCAS_theta", "S_fw_PCAS_phi", "S_bk_OCBS")
_OBS_RENAME = {
    "C_ext":           "C_ext",
    "C_abs":           "C_abs",
    "S_fw_PCAS_theta": "S_fw_theta",
    "S_fw_PCAS_phi":   "S_fw_phi",
    "S_bk_OCBS":       "S_bk",
}
_MIE_REAL = ("C_ext_mie", "C_abs_mie")
_MIE_CPLX = ("S_fw_PCAS_mie", "S_bk_OCBS_mie")
_MIE_RENAME = {
    "C_ext_mie":      "C_ext_mie",
    "C_abs_mie":      "C_abs_mie",
    "S_fw_PCAS_mie":  "S_fw_mie",
    "S_bk_OCBS_mie":  "S_bk_mie",
}


def _norm_per_orient(arr: np.ndarray, side: Side) -> np.ndarray:
    """Normalize a per-orientation observable to (N_rv, N_orient)."""
    sq = np.squeeze(arr)
    if sq.ndim != 2:
        raise ValueError(f"expected 2 non-singleton axes for per-orientation array, got {sq.shape}")
    if side == "dda":
        return sq            # (N_rv, N_orient)
    else:
        return sq.T          # was (N_orient, N_rv)


def _norm_mie(arr: np.ndarray) -> np.ndarray:
    """Normalize a Mie reference scalar-per-rv to (N_rv,)."""
    sq = np.squeeze(arr)
    if sq.ndim == 0:
        return sq[None]
    if sq.ndim == 1:
        return sq
    raise ValueError(f"expected ≤1 non-singleton axis for Mie array, got {sq.shape}")


def _norm_euler(arr: np.ndarray, side: Side) -> np.ndarray:
    """Normalize Euler grid to (N_rv, N_orient, 3)."""
    sq = np.squeeze(arr)
    if side == "dda":
        # DDA: (N_rv, N_orient, 3)
        if sq.ndim != 3:
            raise ValueError(f"DDA Euler: expected 3 non-singleton axes, got {sq.shape}")
        return sq
    else:
        # VIEM: (3, N_orient, N_rv) — Julia transpose of (N_rv, N_orient, 3)
        if sq.ndim != 3:
            raise ValueError(f"VIEM Euler: expected 3 non-singleton axes, got {sq.shape}")
        return np.transpose(sq, (2, 1, 0))


def _norm_cost(arr: np.ndarray) -> np.ndarray:
    """Normalize a cost diagnostic to (N_rv,)."""
    sq = np.squeeze(arr)
    if sq.ndim == 0:
        return sq[None]
    if sq.ndim != 1:
        raise ValueError(f"expected 1 non-singleton axis for cost array, got {sq.shape}")
    return sq


def load_paper(path: str | Path, side: Side) -> dict:
    """Load a production paper-sweep HDF5 file (one shape × one material).

    Returns
    -------
    dict with keys:
      side, path, shape_kind, spheroid_mode, m_p, wavelength_um
      a_eq:        (N_rv,)
      r_ve:        (N_rv,)
      Euler:       (N_rv, N_orient, 3)
      C_ext, C_abs: (N_rv, N_orient)
      S_fw_theta, S_fw_phi, S_bk: (N_rv, N_orient) complex
      C_ext_mie, C_abs_mie: (N_rv,)
      S_fw_mie, S_bk_mie:   (N_rv,) complex
      cost: dict of (N_rv,) arrays
    """
    p = Path(path)
    out: dict = {"side": side, "path": str(p)}
    with h5py.File(p, "r") as f:
        t = f["target"]
        sd = t["simulated_data"]
        out["shape_kind"]    = t.attrs.get("shape_kind", "?")
        out["spheroid_mode"] = bool(int(t.attrs.get("spheroid_mode", 0)))
        out["wavelength_um"] = float(np.squeeze(t["wl_m_m_pairs"][:])[0])
        m_p_arr = np.squeeze(t["m_p_xyz_list"][:])
        if m_p_arr.ndim == 0:
            out["m_p"] = complex(m_p_arr) * np.ones(3, dtype=np.complex128)
        else:
            out["m_p"] = m_p_arr.astype(np.complex128).ravel()[:3]

        out["a_eq"] = np.squeeze(t["r_v_base_list"][:]).astype(np.float64).ravel()
        # r_ve is the post-rescale effective radius; same shape squeeze rules as Mie scalar
        out["r_ve"] = _norm_mie(sd["r_ve"][:])
        out["Euler"] = _norm_euler(sd["Euler_angles"][:], side)

        for name in _OBS_REAL + _OBS_CPLX:
            out[_OBS_RENAME[name]] = _norm_per_orient(sd[name][:], side)
        for name in _MIE_REAL + _MIE_CPLX:
            out[_MIE_RENAME[name]] = _norm_mie(sd[name][:])

        # cost
        cost: dict = {}
        if "cost" in t:
            for k in t["cost"].keys():
                if k == "residual_history":
                    continue                # 2-D, used by Phase 4/5; skip here
                cost[k] = _norm_cost(t["cost"][k][:])
        out["cost"] = cost

    return out


# ----- convergence (a_eq=0.1, single orientation) ---------------------------

_CONV_REAL = ("C_ext", "C_abs", "C_sca", "Q_ext", "Q_abs", "Q_sca")
_CONV_CPLX = ("S_fw_theta", "S_fw_phi", "S_bk")


def load_convergence(path: str | Path, side: Side) -> dict:
    """Load dpl-convergence (DDA) or lc-convergence (VIEM) HDF5.

    Returns
    -------
    dict with keys:
      side, path
      step_axis_name: 'dpl' (DDA) or 'lc_factor' (VIEM)
      step_values:   (N_steps,)
      lattice_lf:    (N_steps,)         [μm]
      n_dof:         (N_steps,)         (= n_occ for DDA, n_dof for VIEM)
      n_cuboid:      (N_steps,)         (= n_cuboid for DDA, n_tet for VIEM)
      r_ve:          (N_steps,)
      iters, converged: (N_steps,)
      C_ext, C_abs, C_sca, Q_ext, Q_abs, Q_sca: (N_steps,)
      S_fw_theta, S_fw_phi, S_bk:               (N_steps,) complex
      reference: dict of scalar Mie values (Q_ext_mie, ..., S_fw_mie, S_bk_mie) or None
    """
    p = Path(path)
    out: dict = {"side": side, "path": str(p)}
    with h5py.File(p, "r") as f:
        t = f["target"]
        if side == "dda":
            grp = t["dpl_convergence"]
            step_name, step_key = "dpl", "dpl"
        else:
            grp = t["lc_convergence"]
            step_name, step_key = "lc_factor", "lc_factor"
        out["step_axis_name"] = step_name
        out["step_values"] = np.asarray(grp[step_key][:]).ravel()

        for k in ("iters", "converged", "r_ve", "solver_err",
                  *_CONV_REAL, *_CONV_CPLX):
            if k in grp:
                out[k] = np.asarray(grp[k][:]).ravel()

        # Side-specific extras (kept under unified names)
        if side == "dda":
            out["lattice_lf"] = np.asarray(grp["lattice_lf"][:]).ravel()
            out["n_dof"]      = np.asarray(grp["n_occ"][:]).ravel()
            out["n_cuboid"]   = np.asarray(grp["n_cuboid"][:]).ravel()
        else:
            out["lattice_lf"] = np.asarray(grp["lc"][:]).ravel()
            out["n_dof"]      = np.asarray(grp["n_dof"][:]).ravel()
            out["n_cuboid"]   = np.asarray(grp["n_tet"][:]).ravel()

        # Reference (Mie) — common schema across both sides; scalar datasets.
        if "reference" in t:
            ref = t["reference"]
            ref_out = {}
            for k in ref.keys():
                v = np.asarray(ref[k][()])
                ref_out[k] = (complex(v) if np.iscomplexobj(v) else float(v))
            out["reference"] = ref_out
        else:
            out["reference"] = None
    return out


# ----- Exact references (MSTM doublet, T-matrix oblate) ---------------------
#
# Both producers are required to emit a common HDF5 schema (see
# `exact_reference_schema.md` next to this file). The only differences are
# (i) the file-name prefix (`mstm_doublet_*.hdf5` vs `tmm_oblate_*.hdf5`) and
# (ii) optional shape-specific attributes/datasets (gap_um, R_monomer_um for
# doublet; aspect_ratio, semi-axis lengths for oblate). Observables and the
# (β, a_eq) grid layout are identical, so a single internal loader handles
# both; `load_mstm` and `load_tmm` are thin wrappers preserving call-site
# intent.

_EXACT_OBS_KEYS = (
    "C_ext", "C_abs", "C_sca",
    "Q_ext", "Q_abs", "Q_sca",
    "S_fw_theta", "S_fw_phi", "S_fw_mean", "S_bk",
)


def _load_exact_reference(path: str | Path) -> dict | None:
    """Load an exact-reference HDF5 (MSTM-compatible schema).

    See `exact_reference_schema.md` for the canonical schema. Required keys:
      /target/a_eq_um           (N_rv,)
      /target/beta_rad          (N_beta,)
      /target/m_p               (3,) complex
      /target/observables/*     (N_beta, N_rv) — see _EXACT_OBS_KEYS

    Optional keys:
      /target/diagnostics/{n_iterations, converged}
      shape-specific extras forwarded into the returned dict if present.

    Returns
    -------
    dict with keys:
      path:        str
      a_eq:        (N_rv,)              [μm]
      beta_rad:    (N_beta,)            [rad]
      m_p:         (3,) complex
      C_ext, C_abs, C_sca, Q_ext, Q_abs, Q_sca: (N_beta, N_rv)
      S_fw_theta, S_fw_phi, S_fw_mean, S_bk:     (N_beta, N_rv) complex
      n_iter, converged: (N_beta, N_rv) (if diagnostics present)
      shape_kind:  str (from /target attrs, if present)
      extras:      dict of any extra /target datasets (gap_um, aspect_ratio, …)
    or None if `path` does not exist.
    """
    p = Path(path)
    if not p.is_file():
        return None
    out: dict = {"path": str(p)}
    with h5py.File(p, "r") as f:
        t = f["target"]
        out["a_eq"]     = np.asarray(t["a_eq_um"][:]).ravel()
        out["beta_rad"] = np.asarray(t["beta_rad"][:]).ravel()
        out["m_p"]      = np.asarray(t["m_p"][:]).ravel()
        sk = t.attrs.get("shape_kind")
        if sk is not None:
            out["shape_kind"] = sk.decode() if isinstance(sk, (bytes, np.bytes_)) else str(sk)

        obs = t["observables"]
        for k in _EXACT_OBS_KEYS:
            if k in obs:
                out[k] = np.asarray(obs[k][:])   # already (N_beta, N_rv)

        if "diagnostics" in t:
            d = t["diagnostics"]
            if "n_iterations" in d:
                out["n_iter"]    = np.asarray(d["n_iterations"][:])
            if "converged" in d:
                out["converged"] = np.asarray(d["converged"][:])

        # Forward any shape-specific scalar/1-D datasets at /target/* without
        # interpreting them — callers that care (e.g. doublet's gap_um) read
        # them out of `extras`.
        extras: dict = {}
        for k in t.keys():
            if k in ("a_eq_um", "beta_rad", "m_p", "observables", "diagnostics"):
                continue
            ds = t[k]
            if isinstance(ds, h5py.Dataset):
                extras[k] = np.asarray(ds[()])
        if extras:
            out["extras"] = extras
    return out


def load_mstm(path: str | Path) -> dict | None:
    """Load MSTM doublet exact-reference HDF5 (multi-sphere T-matrix).

    Producer: ``~/Julia/MSTMforCAS.jl`` (block-VIEM side product).
    File-name pattern: ``mstm_doublet_<material>.hdf5``.
    Schema: shared MSTM-compatible layout — see `_load_exact_reference`.
    """
    return _load_exact_reference(path)


def load_tmm(path: str | Path) -> dict | None:
    """Load oblate-spheroid exact-reference HDF5 (T-matrix / EBCM).

    Producer: ``~/Julia/TransitionMatrices.jl`` (block-VIEM side product).
    File-name pattern: ``tmm_oblate_<material>.hdf5``.
    Schema: shared MSTM-compatible layout — see `_load_exact_reference`.
    """
    return _load_exact_reference(path)


# ----- RHS scaling ---------------------------------------------------------

def load_rhs_scaling(path: str | Path, side: Side) -> dict | None:
    """Load /target/rhs_scaling/gmres group, normalized to (N_L, N_rv).

    Returns None if the group is absent.
    """
    p = Path(path)
    if not p.is_file():
        return None
    with h5py.File(p, "r") as f:
        t = f["target"]
        if "rhs_scaling" not in t:
            return None
        rs = t["rhs_scaling"]
        out: dict = {"side": side,
                     "L_values": np.asarray(rs["L_values"][:]).ravel()}

        # n_dof / n_cuboid (per a_eq, no L axis): squeeze to (N_rv,)
        if side == "dda":
            n_occ_key, n_cub_key = "n_occ", "n_cuboid"
        else:
            n_occ_key, n_cub_key = "n_dof", "n_tet"
        if n_occ_key in rs:
            out["n_dof"] = np.squeeze(rs[n_occ_key][:])
        if n_cub_key in rs:
            out["n_cuboid"] = np.squeeze(rs[n_cub_key][:])

        if "gmres" not in rs:
            return out
        g = rs["gmres"]

        def _norm_L_rv(arr: np.ndarray) -> np.ndarray:
            sq = np.squeeze(arr)
            if sq.ndim == 1:        # only one a_eq survived
                return sq[:, None] if side == "dda" else sq[None, :]
            if sq.ndim != 2:
                raise ValueError(f"rhs_scaling array has unexpected shape {sq.shape}")
            return sq if side == "dda" else sq.T

        for k in ("iters", "converged", "t_total_s",
                  "t_end2end_per_orient_s", "peak_rss_bytes"):
            if k in g:
                out[k] = _norm_L_rv(g[k][:])
    return out
