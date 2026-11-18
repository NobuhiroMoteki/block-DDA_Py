"""Tests of the backward scattering amplitude S_bk_OCBS (S_bak of CAS-v2).

The definition is the corrected Eq. (28) of Moteki and Adachi, Opt. Express
32, 36500 (2024), given in the erratum to that paper:

    S_bak = (-S_11 + S_22) / 2   at the backward direction.

The printed Eq. (28) has 1/sqrt(2) in place of 1/2. Versions of block-DDA_Py
up to 0.8.1 followed the printed equation. The tests below fail for a factor
sqrt(2) in the DDA value, in the Mie value, or in one of the two.
"""
import numpy as np

import shifted_cases as sc
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import DiscreteDipoles, IncidentField, Target

WL_0 = 0.638                    # vacuum wavelength [um]
M_P = 1.5 + 0.0j                # particle refractive index
RNG_SEED = 12345


def _small_sphere(size_parameter, euler):
    """Solved ``DiscreteDipoles`` of a sphere in air with x = 2 pi r / wl_0."""
    r_v = size_parameter * WL_0 / (2 * np.pi)                  # [um]
    m_p_xyz = np.array([M_P, M_P, M_P])
    name, lattice_n, lattice_lf, grid, is_in = sc.quiet(
        sc.build_geometry, r_v, 1.0, WL_0, m_p_xyz, RNG_SEED)
    target = Target(name, lattice_n, lattice_lf, grid, is_in, m_p_xyz, r_v)
    dd = sc.quiet(DiscreteDipoles, target, IncidentField(WL_0, 1.0, euler))
    dd.tol, dd.itermax = 1e-5, 60
    sc.quiet(dd.set_interaction_matrix)
    sc.quiet(dd.solve_matrix_equation)
    assert dd.converge
    return dd


def test_mie_backward_amplitude_equals_forward_amplitude_in_rayleigh_limit():
    """A dipole radiates equally forward and backward: S_bak -> S(0 deg) for x -> 0."""
    r_p = 0.001                                                 # [um], x = 0.0098
    _, _, _, S_fw, S_bk = mie_compute_q_and_s(WL_0, 1.0, r_p, M_P, nang=3)
    assert abs(complex(S_bk) / complex(S_fw) - 1) < 1e-3


def test_dda_backward_amplitude_of_small_sphere_matches_mie():
    euler = np.array([[0.0, 0.0, 0.0], [0.3, 0.9, 0.2], [1.1, 2.0, 0.5]])   # shape: (L, 3)
    dd = _small_sphere(0.6, euler)
    S_bk = dd.compute_OCBS_observable_S_bk()                    # shape: (L,)
    S_fw_theta, S_fw_phi = dd.compute_PCAS_observable_S_fw()
    _, _, _, S_fw_mie, S_bk_mie = mie_compute_q_and_s(WL_0, 1.0, dd.ve_radius, M_P, nang=3)
    S_fw_mie, S_bk_mie = complex(S_fw_mie), complex(S_bk_mie)
    # Measured deviations: 2.9 % (backward) and 1.6 % (forward) with 57 dipoles.
    assert np.max(np.abs(S_bk / S_bk_mie - 1)) < 0.05
    assert np.max(np.abs(S_fw_theta / S_fw_mie - 1)) < 0.05
    assert np.max(np.abs(S_fw_phi / S_fw_mie - 1)) < 0.05
    # Ratio of backward to forward amplitude, independent of the Mie code.
    ratio = np.abs(S_bk) / np.abs((S_fw_theta + S_fw_phi) / 2)
    assert np.all(np.abs(ratio / (abs(S_bk_mie) / abs(S_fw_mie)) - 1) < 0.05)


def test_backward_amplitude_is_half_the_difference_of_its_components():
    dd = _small_sphere(0.6, np.array([[0.2, 0.7, 0.4]]))
    S_bk = dd.compute_OCBS_observable_S_bk()
    expected = (-dd.S_bk_OCBS_theta + dd.S_bk_OCBS_phi) / 2
    assert np.array_equal(S_bk, expected)
    assert np.array_equal(dd.S_bk_OCBS, expected)


def test_shifted_sweep_uses_the_same_definition():
    geometry = sc.spheroid_common_geometry()
    m_max = max(sc.SPHEROID_M_LIST, key=abs)
    dd = sc.make_spheroid_dipoles(geometry, m_max)
    dd.tol, dd.itermax = 1e-5, 60
    sc.quiet(dd.set_interaction_matrix_shifted, sc.SPHEROID_M_LIST)
    sc.quiet(dd.solve_matrix_equation_shifted)
    obs = dd.compute_observables_shifted()
    assert np.array_equal(obs["S_bk_OCBS"],
                          (-obs["S_bk_OCBS_theta"] + obs["S_bk_OCBS_phi"]) / 2)
    # Same value as the existing path for the lattice-defining refractive index.
    s = sc.SPHEROID_M_LIST.index(m_max)
    dd_ref = sc.make_spheroid_dipoles(geometry, m_max)
    sc.solve_existing(dd_ref, 1e-5, 60)
    np.testing.assert_allclose(obs["S_bk_OCBS"][s], dd_ref.compute_OCBS_observable_S_bk(),
                               rtol=1e-10)
