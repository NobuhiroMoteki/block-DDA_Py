"""Feasibility: does a STACKED-BOOKLET structure (oblate envelope filled with N solid
plates separated by water gaps, layered along c) boost the forward depol |B/A| beyond the
smooth-ellipsoid cap ~0.12? Kaolinite forms layered stacks; a periodic high/low-index
layering is a form-birefringence structure (effective negative uniaxial, optic axis = c,
the sign we need). Gaps = absent dipoles = the water medium. Target normalizes the SOLID
volume to r_v (=D_ve/2), so all configs have the same solid mass as the spheroid.
Isotropic plate n=1.55, WL=0.637, 30 full orientations. Compare to N=1 (solid spheroid).
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

RNG_SEED = 12345
WL, M_M, N_O = 0.637, 1.3315, 1.55
R_V = 0.24            # solid D_ve = 0.48um (Target normalizes solid volume to this)
DPL = 20
N_ORI = 30


def build_booklet(AR_env, N_layers, fill, dpl):
    """Oblate envelope (in-plane semi-axis A = AR_env*C) with N_layers solid plates along z,
    solid fraction `fill` per period. Returns a Target (solid volume rescaled to R_V)."""
    m = np.array([N_O, N_O, N_O], dtype=np.complex128)
    lf = WL / (N_O * dpl)
    # provisional envelope size (Target rescales anyway); pick C so lattice resolves layers
    C = 0.10                      # half-thickness of the envelope (um, provisional)
    A = AR_env * C
    xlim, zlim = A + lf, C + lf
    xa = np.arange(-xlim, xlim, lf); ya = np.arange(-xlim, xlim, lf); za = np.arange(-zlim, zlim, lf)
    ln = np.array([len(xa), len(ya), len(za)], dtype=np.int32)
    xg, yg, zg = np.meshgrid(xa, ya, za, indexing="ij")
    grid = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    inside = (grid[:, 0]**2 + grid[:, 1]**2) / A**2 + grid[:, 2]**2 / C**2 <= 1.0
    if N_layers == 1:
        solid = inside                                   # solid spheroid
    else:
        period = 2 * C / N_layers
        zrel = (grid[:, 2] + C) % period                 # position within period [0, period)
        in_plate = zrel < fill * period                  # solid band; rest is water gap
        solid = inside & in_plate
    return Target("booklet", ln, lf, grid, solid, m, R_V), int(solid.sum())


def depol(AR_env, N_layers, fill, dpl=DPL):
    target, nsolid = build_booklet(AR_env, N_layers, fill, dpl)
    rng = np.random.default_rng(7)
    alpha = rng.uniform(0, 2*np.pi, N_ORI)
    beta = np.arccos(rng.uniform(-1, 1, N_ORI)); gamma = rng.uniform(0, 2*np.pi, N_ORI)
    euler = np.column_stack([alpha, beta, gamma])
    dd = DiscreteDipoles(target, IncidentField(WL, M_M, euler))
    dd.set_interaction_matrix(); dd.solve_matrix_equation()
    if not dd.converge:
        return None, target.num_element_occupy
    Ss, Sp = dd.compute_PCAS_observable_S_fw()
    BA = np.abs(np.asarray(Ss) - np.asarray(Sp)) / (np.abs(np.asarray(Ss) + np.asarray(Sp)) + 1e-12)
    return dict(med=np.median(BA), p90=np.percentile(BA, 90), mx=BA.max()), target.num_element_occupy


print(f"solid D_ve={2*R_V}um, plate n={N_O}, water gaps, WL={WL}, {N_ORI} orientations")
print("smooth spheroid cap ~0.12 ; observed Kaolinite needs |B/A| ~0.4")
print(f"{'config':>28} {'Ndip':>6} {'|B/A| med':>10} {'|B/A| p90':>10} {'|B/A| max':>10}")
configs = [
    ("solid spheroid AR_env=5",       5.0, 1, 1.0),
    ("booklet AR_env=5 N=3 f=0.5",    5.0, 3, 0.5),
    ("booklet AR_env=5 N=5 f=0.5",    5.0, 5, 0.5),
    ("booklet AR_env=5 N=5 f=0.3",    5.0, 5, 0.3),
    ("booklet AR_env=8 N=5 f=0.5",    8.0, 5, 0.5),
    ("booklet AR_env=8 N=7 f=0.4",    8.0, 7, 0.4),
]
for name, ar, n, f in configs:
    r, nd = depol(ar, n, f)
    if r is None:
        print(f"{name:>28} {nd:6d}   NOT CONVERGED"); continue
    print(f"{name:>28} {nd:6d} {r['med']:10.3f} {r['p90']:10.3f} {r['mx']:10.3f}")
print("\nIf the booklet |B/A| rises well above the solid spheroid (~0.12) toward ~0.4,")
print("the layered form-birefringence structure supplies the missing depol.")
