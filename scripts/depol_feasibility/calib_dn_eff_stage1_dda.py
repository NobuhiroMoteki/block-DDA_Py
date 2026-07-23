"""Stage 1: uniaxial oblate spheroid (optic axis = c = symmetry axis, so axisymmetric ->
S_fw_theta=A+B e^{2i phi_o}, S_fw_phi=A-B e^{2i phi_o} is exact). Kaolinite beta~0.1,
D_ve=0.48um. Sweep delta_n = n_e - n_o (<0, optically negative). For each delta_n, solve at
the cos_theta_o_half grid (phi_o=0) and save A(theta), B(theta) = (S_s(0)+/-S_p(0))/2.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

WL, M_M = 0.637, 1.3315
R_V, BC = 0.24, 10.0
DPL = 17
N_MEAN = 1.5425                        # material RI (strategy-B Kaolinite)
U = np.linspace(0.0, 1.0, 13)          # cos_theta_o_half grid
DELTA_NS = [0.0, -0.10, -0.20, -0.30, -0.40, -0.50]
SP = str(__import__("pathlib").Path(__file__).resolve().parent)


def build(m_p_xyz):
    rng = np.random.default_rng(12345)
    gre = gaussian_ellipsoid_shape_model(R_V, BC, 1.0, 0.0, WL, m_p_xyz, dpl=DPL)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, ln, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(gre.lattice_lf, gre.distance_factor, ln, dist)
    return Target(gre.name, ln, gre.lattice_lf, grid, is_in, m_p_xyz, R_V)


out = {"U": U, "delta_ns": np.array(DELTA_NS), "n_mean": N_MEAN}
for dn in DELTA_NS:
    n_o = N_MEAN - dn / 3.0               # keep mean ~ N_MEAN (n_o twice, n_e once)
    n_e = n_o + dn
    m = np.array([n_o, n_o, n_e], dtype=np.complex128)   # optic axis = z = c
    target = build(m)
    beta = np.arccos(U)
    euler = np.column_stack([np.zeros_like(beta), beta, np.zeros_like(beta)])  # phi_o=0
    dd = DiscreteDipoles(target, IncidentField(WL, M_M, euler))
    dd.set_interaction_matrix(); dd.solve_matrix_equation()
    if not dd.converge:
        print(f"dn={dn} NOT CONVERGED"); continue
    Ss0, Sp0 = dd.compute_PCAS_observable_S_fw()
    A = (np.asarray(Ss0) + np.asarray(Sp0)) / 2
    B = (np.asarray(Ss0) - np.asarray(Sp0)) / 2
    out[f"A_{dn}"] = A; out[f"B_{dn}"] = B
    print(f"dn={dn:+.2f} (n_o={n_o:.4f},n_e={n_e:.4f}): |B/A| at phi=0 median={np.median(np.abs(B)/(np.abs(A)+1e-12)):.3f}")
np.savez(f"{SP}/calib_uniaxial_AB.npz", **out)
print(f"[saved] {SP}/calib_uniaxial_AB.npz")
