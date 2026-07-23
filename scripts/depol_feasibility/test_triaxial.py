"""Feasibility: can a TRIAXIAL ellipsoid (a != b) reach the observed Kaolinite depol
magnitude that the axisymmetric spheroid (|B/A|~0.1) cannot? At Kaolinite's oblate
flattening (bc_ratio=10, beta~0.1) and D_ve~0.48um, sweep the in-plane elongation
ab_ratio and look at the |B/A| = |S_s-S_p|/|S_s+S_p| distribution over full orientations.
Isotropic n=1.55, WL=0.637. (Triaxial is NOT axisymmetric, so we solve the DDA at full
random Euler orientations -- the A+/-B*exp(2i phi) shortcut does not apply.)
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

RNG_SEED = 12345
WL, M_M, N_O = 0.637, 1.3315, 1.55
R_V, BC = 0.24, 10.0          # D_ve=0.48um, oblate flattening beta~0.1
DPL = 15
N_ORI = 40                    # random full orientations


def build(ab, dpl):
    m = np.array([N_O, N_O, N_O], dtype=np.complex128)
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(R_V, BC, ab, 0.0, WL, m, dpl=dpl)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, ln, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(gre.lattice_lf, gre.distance_factor, ln, dist)
    return Target(gre.name, ln, gre.lattice_lf, grid, is_in, m, R_V)


def depol(ab):
    target = build(ab, DPL)
    rng = np.random.default_rng(7)
    # full ZYZ Euler orientations (alpha, beta, gamma), uniform on SO(3)
    alpha = rng.uniform(0, 2*np.pi, N_ORI)
    cbeta = rng.uniform(-1, 1, N_ORI); beta = np.arccos(cbeta)
    gamma = rng.uniform(0, 2*np.pi, N_ORI)
    euler = np.column_stack([alpha, beta, gamma])
    dd = DiscreteDipoles(target, IncidentField(WL, M_M, euler))
    dd.set_interaction_matrix(); dd.solve_matrix_equation()
    if not dd.converge:
        return None, target.num_element_occupy
    Ss, Sp = dd.compute_PCAS_observable_S_fw()
    Ss, Sp = np.asarray(Ss), np.asarray(Sp)
    BA = np.abs(Ss - Sp) / (np.abs(Ss + Sp) + 1e-12)
    return dict(med=np.median(BA), p90=np.percentile(BA, 90), mx=BA.max()), target.num_element_occupy


print(f"D_ve={2*R_V}um, bc_ratio={BC} (beta={1/BC:.2f}), isotropic n={N_O}, {N_ORI} full orientations")
print("target: observed Kaolinite needs |B/A| ~4x the spheroid (~0.1 -> ~0.4)")
print(f"{'ab_ratio':>8} {'a:b:c':>12} {'Ndip':>6} {'|B/A| med':>10} {'|B/A| p90':>10} {'|B/A| max':>10}")
for ab in (1.0, 1.3, 1.6, 2.0):
    r, nd = depol(ab)
    if r is None:
        print(f"{ab:8.1f}  NOT CONVERGED"); continue
    a=ab*BC; print(f"{ab:8.1f} {f'{a:.0f}:{BC:.0f}:1':>12} {nd:6d} {r['med']:10.3f} {r['p90']:10.3f} {r['mx']:10.3f}")
print("\nab_ratio=1.0 is the spheroid baseline. If |B/A| rises toward ~0.4 with ab_ratio,")
print("triaxial elongation supplies the missing depol magnitude.")
