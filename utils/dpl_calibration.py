"""Per-slot dpl calibration for VIEM-matched DDA paper sweeps.

CLAUDE.md §3 (v0.7.6+) drops the fixed-dpl convention: the production
paper sweep picks `dpl` per slot so that DDA's `n_occ` matches VIEM's
`n_tet` within 1.5× (ideally as close as possible).

The target `n_tet` values are taken from `viem_results/estimate_cost.jl`
run on the current (v0.7.6) VIEM paper schemas (2026-04-24). They are
cached here as `VIEM_N_TET_TABLE` so the DDA runner can be invoked
without Julia on the path.

The DDA lattice spacing d = λ/(|m_p|·dpl). After the volume-preserving
rescale in `bl_dda.scatterer.Target`, `N_occ · d_adj³ = V_target` exactly.
Consequently

    n_occ ≈ V · (|m_p|·dpl/λ)³   ⇒   dpl = (λ/|m_p|) · (n_occ/V)^(1/3)

is the formula used to invert `n_tet → dpl`. For non-spherical shapes
(oblate, GRE, doublet) the discrete interior-check introduces a 5–35 %
surface-area perturbation on top of the ideal n_occ; empirically this
stays within the 1.5× tolerance for every paper slot, so a single
closed-form dpl suffices (no iterative refinement).
"""
from __future__ import annotations

import numpy as np


# ──────────────────────────────────────────────────────────────────────
#  VIEM n_tet lookup (v0.7.6, 2026-04-24)
#
#  Source: `julia --project=. viem_results/estimate_cost.jl
#             viem_results/paper/{shape}_{material}.hdf5`
#  on ~/Julia/block-VIEM.jl at v0.7.6.
#  n_tet values quoted are the worst-case (smallest λ / largest |m_p|)
#  adaptive-lc mesh build — i.e. exactly what VIEM will produce in the
#  production sweep.
# ──────────────────────────────────────────────────────────────────────

VIEM_N_TET_TABLE = {
    "sphere": {
        # a_eq = 0.05, 0.10, 0.20, 0.40 μm
        "n15": [652, 652, 2229, 16671],
        "n20": [652, 710, 5151, 39167],
        # Au: no 0.40 (wavelength-constrained lattice would exceed 24 h budget)
        "Au":  [652, 3673, 26459],
    },
    "oblate": {
        "n15": [5377, 5377, 5377, 17046],
        "n20": [5377, 5377, 5365, 39107],
        "Au":  [5377, 5377, 26160],
    },
    "gre": {
        "n15": [19853, 19853, 19853, 19853],
        "n20": [19853, 19853, 19853, 39167],
        "Au":  [19853, 19853, 26459],
    },
    "doublet": {
        "n15": [1437, 1437, 2417, 17612],
        "n20": [1437, 1437, 5478, 41441],
        "Au":  [1437, 3735, 27573],
    },
}

_A_EQ_INDEX_FULL = {0.05: 0, 0.10: 1, 0.20: 2, 0.40: 3}
_A_EQ_INDEX_AU   = {0.05: 0, 0.10: 1, 0.20: 2}


# ──────────────────────────────────────────────────────────────────────
#  Material / shape inference from HDF5 attributes
# ──────────────────────────────────────────────────────────────────────

_MATERIAL_BY_M_P = {
    # key is (round(n_r, 5), round(n_i, 5)) — 5 digits needed because
    # 0.17525 stored as float64 is 0.17524999999999996, which rounds to
    # 0.1752 at 4 digits but correctly to 0.17525 at 5.
    (1.5,    0.01):   "n15",
    (2.0,    0.0):    "n20",
    (3.17,   0.16):   "n317",          # legacy, not in VIEM_N_TET_TABLE
    (0.17525, 3.483): "Au",
}


def material_label(m_p_value):
    """Map a scalar complex refractive index to the paper label (n15 / n20 / Au)."""
    z = complex(m_p_value)
    key = (round(z.real, 5), round(z.imag, 5))
    if key in _MATERIAL_BY_M_P:
        return _MATERIAL_BY_M_P[key]
    raise ValueError(f"unknown material m_p={m_p_value} "
                     f"(expected n15=1.5+0.01i, n20=2.0+0.0i, Au=0.17525+3.483i)")


def shape_label(shape_kind, bc_ratio, ab_ratio, gre_beta):
    """Map (shape_kind, GRE params) to the paper shape label."""
    if shape_kind == "doublet":
        return "doublet"
    if bc_ratio == 1.0 and ab_ratio == 1.0 and gre_beta == 0.0:
        return "sphere"
    if bc_ratio == 3.0 and ab_ratio == 1.0 and gre_beta == 0.0:
        return "oblate"
    if bc_ratio == 1.0 and ab_ratio == 1.0 and abs(gre_beta - 0.2) < 1e-9:
        return "gre"
    raise ValueError(f"cannot map shape (kind={shape_kind}, bc={bc_ratio}, "
                     f"ab={ab_ratio}, β={gre_beta}) to paper label")


def a_eq_index(material, a_eq):
    """Index into VIEM_N_TET_TABLE[shape][material] for the given a_eq."""
    table = _A_EQ_INDEX_AU if material == "Au" else _A_EQ_INDEX_FULL
    key = round(float(a_eq), 3)
    if key not in table:
        raise ValueError(f"a_eq={a_eq} μm not in VIEM paper grid for {material}; "
                         f"valid: {sorted(table.keys())}")
    return table[key]


def get_n_tet_target(shape, material, a_eq):
    """VIEM's n_tet for the given (shape, material, a_eq) slot."""
    if shape not in VIEM_N_TET_TABLE:
        raise ValueError(f"no VIEM data for shape={shape!r}")
    if material not in VIEM_N_TET_TABLE[shape]:
        raise ValueError(
            f"no VIEM data for material={material!r} (shape={shape}). "
            f"Supported: {list(VIEM_N_TET_TABLE[shape].keys())}")
    return VIEM_N_TET_TABLE[shape][material][a_eq_index(material, a_eq)]


# ──────────────────────────────────────────────────────────────────────
#  dpl formula
# ──────────────────────────────────────────────────────────────────────

def dpl_for_target_n_occ(n_occ_target, a_eq, m_p_xyz, wl_0=0.638):
    """Return the DDA `dpl` whose volume-preserving lattice yields
    `n_occ ≈ n_occ_target`.

    Derivation:  V = (4π/3)·a_eq³, d = λ/(|m_p|·dpl),
        n_occ ≈ V / d³   ⇒   dpl = (λ / |m_p|) · (n_occ / V)^(1/3).
    """
    m_p_max = float(np.max(np.abs(np.asarray(m_p_xyz))))
    V = (4.0 * np.pi / 3.0) * a_eq ** 3
    return (wl_0 / m_p_max) * (n_occ_target / V) ** (1.0 / 3.0)


def dpl_for_slot(shape, material, a_eq, m_p_xyz, wl_0=0.638):
    """Convenience: look up the VIEM n_tet for the slot and return the
    matching DDA dpl."""
    n_tet = get_n_tet_target(shape, material, a_eq)
    return dpl_for_target_n_occ(n_tet, a_eq, m_p_xyz, wl_0=wl_0)
