"""Decisive test: does Kaolinite's REAL biaxial dielectric tensor (n_alpha/n_beta/n_gamma,
optic axes at the crystal orientation of the platelet) lift the forward depol |B/A| toward
the observed ~0.58, which no dielectric SHAPE can reach (spheroid max 0.18)? block-DDA takes
a diagonal anisotropic m_p_xyz in the lattice frame; the platelet short axis (c ~ z) gets the
out-of-plane index, the two in-plane axes get the other two (biaxial => in-plane n_x != n_y,
which directly makes S_s != S_p). We scale the real birefringence by a factor s to map |B/A|
vs delta_n and find where it reaches 0.58. Oblate spheroid beta~0.1, D_ve=0.48um, WL=0.637.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

RNG_SEED = 12345
WL, M_M = 0.637, 1.3315
R_V, BC = 0.24, 10.0          # D_ve=0.48um, oblate beta~0.1
DPL = 16
N_ORI = 30

# Kaolinite biaxial indices (representative): n_alpha<n_beta<n_gamma
N_A, N_B, N_G = 1.553, 1.559, 1.570
N_MEAN = (N_A + N_B + N_G) / 3.0
# lattice-frame assignment: x,y in-plane (n_gamma, n_beta), z = platelet normal (n_alpha, low)
DEV = np.array([N_G - N_MEAN, N_B - N_MEAN, N_A - N_MEAN])   # (x,y,z) deviations


def build(ab_mpxyz, dpl):
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(R_V, BC, 1.0, 0.0, WL, ab_mpxyz, dpl=dpl)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, ln, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(gre.lattice_lf, gre.distance_factor, ln, dist)
    return Target(gre.name, ln, gre.lattice_lf, grid, is_in, ab_mpxyz, R_V)


def depol(scale):
    m_p_xyz = (N_MEAN + scale * DEV).astype(np.complex128)   # scaled biaxial tensor
    target = build(m_p_xyz, DPL)
    rng = np.random.default_rng(7)
    euler = np.column_stack([rng.uniform(0, 2*np.pi, N_ORI),
                             np.arccos(rng.uniform(-1, 1, N_ORI)),
                             rng.uniform(0, 2*np.pi, N_ORI)])
    dd = DiscreteDipoles(target, IncidentField(WL, M_M, euler))
    dd.set_interaction_matrix(); dd.solve_matrix_equation()
    if not dd.converge:
        return None
    Ss, Sp = dd.compute_PCAS_observable_S_fw()
    BA = np.abs(np.asarray(Ss)-np.asarray(Sp))/(np.abs(np.asarray(Ss)+np.asarray(Sp))+1e-12)
    return np.median(BA), np.percentile(BA, 90), BA.max()


print(f"Kaolinite biaxial: n_a/n_b/n_g = {N_A}/{N_B}/{N_G}, mean={N_MEAN:.3f}")
print(f"real in-plane delta_n = n_x-n_y = {DEV[0]-DEV[1]:.4f}, out-of-plane spread = {DEV[0]-DEV[2]:.4f}")
print("observed Kaolinite |B/A| ~0.58 ; isotropic spheroid ~0.12-0.15")
print(f"{'scale':>6} {'m_p_xyz (x,y,z)':>26} {'|B/A| med':>10} {'|B/A| p90':>10} {'|B/A| max':>10}")
for s in (0.0, 1.0, 5.0, 10.0, 20.0, 40.0):
    m = N_MEAN + s * DEV
    r = depol(s)
    if r is None:
        print(f"{s:6.0f}   NOT CONVERGED"); continue
    print(f"{s:6.0f} {f'({m[0]:.3f},{m[1]:.3f},{m[2]:.3f})':>26} {r[0]:10.3f} {r[1]:10.3f} {r[2]:10.3f}")
print("\nscale=1 is the REAL kaolinite tensor. The scale where |B/A|->0.58 tells how many x")
print("the real birefringence would need to be -- if that is huge, birefringence is ruled out too.")
