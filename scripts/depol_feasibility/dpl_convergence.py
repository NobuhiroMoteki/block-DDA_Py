"""DDA dipole-resolution convergence of the forward depolarization B_fw for oblate
spheroids. Question: is the |B_fw| peak/decline at high AR real, or a thin-axis
under-resolution artifact? Sweep dpl at AR=6 (near the apparent peak) and AR=11
(Kaolinite, past the peak); if |B_fw| at AR=11 keeps rising with dpl while AR=6 is
stable, the LUT (dpl=17) under-resolves the thin plate and the decline is an artifact.
Isotropic n=1.55, D_ve=0.5um, WL=0.637.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

RNG_SEED = 12345
WL, M_M, N_O, D_VE = 0.637, 1.3315, 1.55, 0.5
COS_THETA = np.linspace(0.0, 1.0, 5)     # a few polar tilts (fast)


def build(r_v_base, AR, wl_0, m_p_xyz, dpl):
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(r_v_base, AR, 1.0, 0.0, wl_0, m_p_xyz, dpl=dpl)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, ln, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(gre.lattice_lf, gre.distance_factor, ln, dist)
    return gre.name, ln, gre.lattice_lf, grid, is_in


def AB(AR, dpl):
    m = np.array([N_O, N_O, N_O], dtype=np.complex128)
    name, ln, lf, grid, is_in = build(D_VE / 2, AR, WL, m, dpl)
    target = Target(name, ln, lf, grid, is_in, m, D_VE / 2)
    ndip = target.num_element_occupy
    beta = np.arccos(COS_THETA)
    euler = np.column_stack([np.zeros_like(beta), beta, np.zeros_like(beta)])
    dd = DiscreteDipoles(target, IncidentField(WL, M_M, euler))
    dd.set_interaction_matrix(); dd.solve_matrix_equation()
    if not dd.converge:
        return None, ndip
    Ss0, Sp0 = dd.compute_PCAS_observable_S_fw()
    A = (np.asarray(Ss0) + np.asarray(Sp0)) / 2
    B = (np.asarray(Ss0) - np.asarray(Sp0)) / 2
    # thin-axis dipole count: half-thickness c / lattice spacing (post-rescale)
    c = (D_VE / 2) * (1.0 / (AR ** 2)) ** (1.0 / 3.0)
    n_across = 2 * c / target.lattice_lf
    return dict(magBA=np.median(np.abs(B) / (np.abs(A) + 1e-12)),
                absB=np.median(np.abs(B)), absA=np.median(np.abs(A)),
                n_across=n_across), ndip


for AR in (6.0, 11.0):
    print(f"\n===== AR={AR} (beta={1/AR:.3f}), D_ve={D_VE}um, isotropic n={N_O} =====")
    print(f"{'dpl':>4} {'Ndip':>7} {'dip across thin axis':>20} {'|B/A| med':>10} {'|B| med':>9} {'|A| med':>9}")
    for dpl in (12, 17, 25, 34):
        r, nd = AB(AR, dpl)
        if r is None:
            print(f"{dpl:4d} {nd:7d}   NOT CONVERGED"); continue
        print(f"{dpl:4d} {nd:7d} {r['n_across']:20.1f} {r['magBA']:10.3f} {r['absB']:9.4f} {r['absA']:9.4f}")
print("\nIf |B| at AR=11 keeps rising with dpl (esp. while AR=6 is flat) -> the LUT (dpl=17)")
print("under-resolves the thin plate and the |B_fw| decline at high AR is a resolution artifact.")
