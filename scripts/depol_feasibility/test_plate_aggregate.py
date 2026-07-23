"""Feasibility: does a rigid AGGREGATE of oblate plates (edge-face 'house of cards' /
random cluster) reach the observed Kaolinite depol |B/A| ~0.6 that single smooth bodies
(spheroid/disk/triaxial/booklet, all ~0.12) cannot? Plate aggregates need DDA (MSTM is
spheres-only). Each plate = oblate ellipsoid (in-plane radius A_p, half-thickness C_p);
plates are placed in a compact touching cluster at various orientations; the union solid
volume is normalized to r_v (= D_ve/2) so the total mass matches the spheroid comparison.
Isotropic n=1.55, WL=0.637.
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from scipy.spatial.transform import Rotation as Rot
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

WL, M_M, N_O = 0.637, 1.3315, 1.55
R_V = 0.24          # solid D_ve = 0.48um; Target normalizes UNION solid volume to this
DPL = 15
N_ORI = 24
AR_P = 7.0          # individual plate aspect ratio (A_p/C_p)


def in_plate(pts, center, rotmat, A_p, C_p):
    q = (pts - center) @ rotmat            # into plate frame (rotmat cols = plate axes)
    return (q[:, 0]**2 + q[:, 1]**2) / A_p**2 + q[:, 2]**2 / C_p**2 <= 1.0


def build_aggregate(plates, dpl):
    """plates: list of (center[3], euler_zyz[3]). All plates share A_p,C_p."""
    m = np.array([N_O, N_O, N_O], dtype=np.complex128)
    lf = WL / (N_O * dpl)
    C_p = 0.06                              # provisional plate half-thickness (Target rescales)
    A_p = AR_P * C_p
    R = A_p + 0.5 * A_p                     # bounding half-extent for the cluster
    lim = R + lf
    ax = np.arange(-lim, lim, lf)
    xg, yg, zg = np.meshgrid(ax, ax, ax, indexing="ij")
    grid = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    solid = np.zeros(grid.shape[0], dtype=bool)
    for center, euler in plates:
        Rm = Rot.from_euler("zyz", euler).as_matrix()
        solid |= in_plate(grid, np.asarray(center) * A_p, Rm, A_p, C_p)
    return Target("agg", np.array([len(ax)]*3, dtype=np.int32), lf, grid, solid, m, R_V), int(solid.sum())


def depol(plates, dpl=DPL):
    target, nsolid = build_aggregate(plates, dpl)
    rng = np.random.default_rng(7)
    euler = np.column_stack([rng.uniform(0, 2*np.pi, N_ORI),
                             np.arccos(rng.uniform(-1, 1, N_ORI)),
                             rng.uniform(0, 2*np.pi, N_ORI)])
    dd = DiscreteDipoles(target, IncidentField(WL, M_M, euler))
    dd.set_interaction_matrix(); dd.solve_matrix_equation()
    if not dd.converge:
        return None, target.num_element_occupy
    Ss, Sp = dd.compute_PCAS_observable_S_fw()
    BA = np.abs(np.asarray(Ss)-np.asarray(Sp))/(np.abs(np.asarray(Ss)+np.asarray(Sp))+1e-12)
    return dict(med=np.median(BA), p90=np.percentile(BA, 90), mx=BA.max()), target.num_element_occupy


# configurations (centers in units of A_p; eulers in rad)
d2 = np.pi/2
configs = {
    "single plate":              [([0,0,0], [0,0,0])],
    "2 plates edge-face (T)":    [([0,0,0], [0,0,0]), ([0.5,0,0.3], [0,d2,0])],
    "3 plates house-of-cards":   [([0,0,0],[0,0,0]), ([0.4,0,0.2],[0,d2,0]), ([0,0.4,0.2],[d2,d2,0])],
    "4 plates random cluster":   [([0,0,0],[0.6,1.1,0.3]), ([0.4,0.1,0],[2.1,0.7,1.4]),
                                  ([-0.2,0.3,0.1],[1.2,2.0,0.5]), ([0.1,-0.3,0.2],[0.3,1.5,2.2])],
}
print(f"solid D_ve={2*R_V}um, plate AR={AR_P}, n={N_O}, WL={WL}, {N_ORI} orientations")
print("target: observed Kaolinite |B/A| ~0.6 (large |S|); single smooth body caps ~0.12")
print(f"{'config':>28} {'Ndip':>6} {'|B/A| med':>10} {'|B/A| p90':>10} {'|B/A| max':>10}")
for name, plates in configs.items():
    r, nd = depol(plates)
    if r is None:
        print(f"{name:>28} {nd:6d}   NOT CONVERGED"); continue
    print(f"{name:>28} {nd:6d} {r['med']:10.3f} {r['p90']:10.3f} {r['mx']:10.3f}")
print("\nIf plate aggregates lift |B/A| toward ~0.6, multiple scattering among differently-")
print("oriented plates supplies the observed depol -> Kaolinite is measured as aggregates.")
