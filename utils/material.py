"""Material complex refractive indices used in the paper sweep.

Single-wavelength hardcoded constants for λ₀ = 0.638 μm (CLAUDE.md §2.2).
VIEM-side values are matched exactly so the two solvers see identical inputs.

For future multi-wavelength extension, replace the Au hardcode with an
interpolation over Johnson & Christy 1972 tabulated (n, k).
"""
import numpy as np

WAVELENGTH_UM = 0.638

MATERIALS = {
    "low":  1.5   + 0.01j,      # low refractive index reference (n15)
    "n20":  2.0   + 0.0j,       # mid index, non-absorbing (paper "high", v0.7.6+)
    "high": 3.17  + 0.16j,      # legacy high refractive index (n317, superseded)
    "Au":   0.17525 + 3.4830j,  # Johnson & Christy 1972 at 638 nm
}


def get_m_p(material_id, wl_0=WAVELENGTH_UM):
    """Return the complex refractive index for a paper-sweep material.

    Parameters
    ----------
    material_id : str
        One of "low", "high", "Au".
    wl_0 : float
        Wavelength [um]; for the paper sweep the only supported value is 0.638.

    Returns
    -------
    m_p : complex
    """
    if wl_0 != WAVELENGTH_UM:
        raise ValueError(
            f"Only λ₀ = {WAVELENGTH_UM} μm is supported by the hardcoded table; "
            f"got {wl_0} μm. Extend MATERIALS or add J&C interpolation.")
    try:
        return MATERIALS[material_id]
    except KeyError:
        raise ValueError(
            f"Unknown material_id {material_id!r}; "
            f"expected one of {list(MATERIALS.keys())}")


def get_m_p_xyz(material_id, wl_0=WAVELENGTH_UM, dtype=np.complex64):
    """Same as get_m_p but returns an isotropic (mx, my, mz) complex64 array
    ready to hand to Target / shape models."""
    m = get_m_p(material_id, wl_0)
    return np.array([m, m, m], dtype=dtype)
