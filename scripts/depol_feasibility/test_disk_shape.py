"""Shape test: does a flat DISK (circular cylinder, sharp edges) produce the observed
Kaolinite forward-amplitude anisotropy (~1.3-1.6, INCREASING with size) that a smooth
oblate spheroid cannot? Isotropic RI n=1.56 (no birefringence).

Disk is axially symmetric about z, so the analytic expansion S_s/S_p = A +/- B*exp(2i phi_o)
still holds -- directly comparable to the spheroid at the same AR = (in-plane radius)/(half-thickness).
"""
import sys, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

RNG_SEED = 12345
WL = 0.637
M_M = 1.3315
N_O = 1.56
COS_THETA = np.linspace(0.0, 1.0, 7)


def build_disk(r_v_base, AR, wl_0, m_p_xyz, dpl=17):
    """Flat circular cylinder: radius R, half-thickness c, AR = R/c, isotropic volume r_v_base."""
    lf = wl_0 / (np.max(np.abs(m_p_xyz)) * dpl)
    c = r_v_base * (2.0 / (3.0 * AR**2))**(1.0/3.0)     # V=pi R^2 (2c)=(4/3)pi r_v^3, R=AR c
    R = AR * c
    xlim, zlim = R + lf, c + lf
    xa = np.arange(-xlim, xlim, lf); ya = np.arange(-xlim, xlim, lf); za = np.arange(-zlim, zlim, lf)
    lattice_n = np.array([len(xa), len(ya), len(za)], dtype=np.int32)
    xg, yg, zg = np.meshgrid(xa, ya, za, indexing='ij')
    grid = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])
    is_in = (grid[:, 0]**2 + grid[:, 1]**2 <= R**2) & (np.abs(grid[:, 2]) <= c)
    return f"disk_AR{AR}", lattice_n, lf, grid, is_in


def build_spheroid(r_v_base, bc_ratio, wl_0, m_p_xyz):
    rng = np.random.default_rng(RNG_SEED)
    gre = gaussian_ellipsoid_shape_model(r_v_base, bc_ratio, 1.0, 0.0, wl_0, m_p_xyz)
    r_pts, _ = gre.compute_r_points_on_GRE(rng)
    _, lattice_n, grid = gre.create_cuboid_lattice_that_encloses_GRE_shape(r_pts)
    dist = gre.find_nearest_distance_from_the_GRE_surf(grid, r_pts)
    is_in = gre.extract_lattice_address_in_GRE_volume(gre.lattice_lf, gre.distance_factor, lattice_n, dist)
    return gre.name, lattice_n, gre.lattice_lf, grid, is_in


def anisotropy(name, ln, lf, grid, is_in, m_p_xyz, r_v_base):
    target = Target(name, ln, lf, grid, is_in, m_p_xyz, r_v_base)
    beta = np.arccos(COS_THETA)
    euler = np.column_stack([np.zeros_like(beta), beta, np.zeros_like(beta)])
    dd = DiscreteDipoles(target, IncidentField(WL, M_M, euler))
    dd.set_interaction_matrix(); dd.solve_matrix_equation()
    if not dd.converge:
        return None, dd.num_element_occupy
    S_s0, S_p0 = dd.compute_PCAS_observable_S_fw()
    A = (np.asarray(S_s0) + np.asarray(S_p0)) / 2
    B = (np.asarray(S_s0) - np.asarray(S_p0)) / 2
    return dict(A=A, B=B, cos_theta=COS_THETA,
                magBA=np.median(np.abs(B)/(np.abs(A)+1e-12)),
                imBA=np.median(np.abs(B.imag)/(np.abs(A.imag)+1e-12))), dd.num_element_occupy


m_iso = np.array([N_O, N_O, N_O], dtype=np.complex128)
out = {}

print("== SHAPE comparison at D_ve=0.40um, isotropic n=1.56 (|ImB/ImA| ; observed pop ~1.3) ==")
print(f"{'AR':>5} {'SPHEROID Ndip':>14} {'sph |ImB/ImA|':>13} {'DISK Ndip':>10} {'disk |ImB/ImA|':>14} {'disk/sph':>9}")
for AR in (6.0, 10.0, 12.0):
    rs, ns = anisotropy(*build_spheroid(0.20, AR, WL, m_iso), m_iso, 0.20)
    rd, nd = anisotropy(*build_disk(0.20, AR, WL, m_iso), m_iso, 0.20)
    out[("sph", AR, 0.20)] = rs; out[("disk", AR, 0.20)] = rd
    print(f"{AR:5.0f} {ns:14d} {rs['imBA']:13.3f} {nd:10d} {rd['imBA']:14.3f} {rd['imBA']/rs['imBA']:9.2f}")

print("\n== DISK size-dependence at AR=10, isotropic n=1.56 (does anisotropy GROW with size?) ==")
print(f"{'D_ve[um]':>9} {'Ndip':>7} {'disk |ImB/ImA|':>14} {'disk |B/A|':>11}")
for Dve in (0.25, 0.40, 0.60, 0.90):
    rd, nd = anisotropy(*build_disk(Dve/2, 10.0, WL, m_iso), m_iso, Dve/2)
    out[("disk", 10.0, Dve/2)] = rd
    print(f"{Dve:9.2f} {nd:7d} {rd['imBA']:14.3f} {rd['magBA']:11.3f}")

np.savez("/tmp/claude-1000/-home-moteki-Python-PCAS-Bayes-for-liquid-2wls/"
         "9cf163d2-85ab-4f1f-bfd2-3f26f503cb5f/scratchpad/disk_AB.npz",
         **{f"{k[0]}_AR{k[1]}_rv{k[2]}_{f}": v
            for k, r in out.items() if r for f, v in
            (("A", r["A"]), ("B", r["B"]), ("cos_theta", r["cos_theta"]))})
print("\n[saved] disk_AB.npz")
print("Spheroid baseline (earlier): iso |ImB/ImA| ~0.31-0.35, DECREASES with size. Observed ~1.3-1.6, INCREASES.")
