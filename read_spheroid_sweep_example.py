"""Example: read a single data point from dda_results_spheroid_sweep.h5
by specifying wavelength, physical parameters, and orientation angles.

The HDF5 uses per-wavelength groups (e.g. wl_0p453/, wl_0p638/).
Grid coordinates: D_ve, RI_real, log10(AR), cos_theta_o (half-domain), phi_o.
"""

import h5py
import numpy as np

# ── 取り出したい条件を指定 ──────────────────────────────────────────────────
wl_0_target          = 0.834    # wavelength [um] (selects HDF5 group)
D_ve_target          = 0.6      # volume-equivalent diameter [um]
RI_real_target       = 1.5      # Re(m_p)
log_AR_target        = 0.0      # log10(AR), 0 = sphere
cos_theta_o_target   = 0.5      # cos(theta_o), half-domain [0, 1]
phi_o_target         = 0.5      # phi_o [rad], domain [0, pi]
# ────────────────────────────────────────────────────────────────────────────

with h5py.File("dda_results/dda_results_spheroid_sweep.h5", "r") as f:

    # Find wavelength group by toleranced matching
    grp = None
    for name in f:
        obj = f[name]
        if isinstance(obj, h5py.Group) and 'wl_0' in obj.attrs:
            if abs(obj.attrs['wl_0'] - wl_0_target) < 1e-4:
                grp = obj
                break
    if grp is None:
        wl_list = [f[n].attrs['wl_0'] for n in f
                   if isinstance(f[n], h5py.Group) and 'wl_0' in f[n].attrs]
        raise ValueError(f"Wavelength {wl_0_target} not found. "
                         f"Available: {wl_list}")

    # 各格子から最近傍インデックスを取得
    i_dve = int(np.argmin(np.abs(f["D_ve_grid"][:]              - D_ve_target)))
    i_ri  = int(np.argmin(np.abs(f["RI_real_grid"][:]           - RI_real_target)))
    i_ar  = int(np.argmin(np.abs(f["log_AR_grid"][:]            - log_AR_target)))
    i_u   = int(np.argmin(np.abs(f["cos_theta_o_half_grid"][:]  - cos_theta_o_target)))
    i_ph  = int(np.argmin(np.abs(f["phi_o_grid"][:]             - phi_o_target)))

    # 実際にヒットした格子点の値を表示
    print("=== Selected grid point ===")
    print(f"  wl_0         = {grp.attrs['wl_0']:.4f} um  "
          f"(group: {grp.name}, target {wl_0_target})")
    print(f"  D_ve         = {f['D_ve_grid'][i_dve]:.4f} um  (target {D_ve_target})")
    print(f"  RI_real      = {f['RI_real_grid'][i_ri]:.4f}      (target {RI_real_target})")
    print(f"  log10(AR)    = {f['log_AR_grid'][i_ar]:.4f}      (target {log_AR_target})")
    print(f"  AR           = {10**f['log_AR_grid'][i_ar]:.4f}")
    print(f"  cos(theta_o) = {f['cos_theta_o_half_grid'][i_u]:.4f}      "
          f"(target {cos_theta_o_target})")
    print(f"  phi_o        = {f['phi_o_grid'][i_ph]:.4f} rad  (target {phi_o_target})")
    print(f"  converged    = {bool(grp['converged'][i_dve, i_ri, i_ar])}")

    # 結果取り出し
    idx = (i_dve, i_ri, i_ar, i_u, i_ph)

    S_fw_theta = grp["S_fw_theta_re"][idx] + 1j * grp["S_fw_theta_im"][idx]
    S_fw_phi   = grp["S_fw_phi_re"][idx]   + 1j * grp["S_fw_phi_im"][idx]

    print("\n=== DDA results ===")
    print(f"  S_fw_theta = {S_fw_theta:.6e}")
    print(f"  S_fw_phi   = {S_fw_phi:.6e}")

    # 利用可能な波長一覧
    print("\n=== Available wavelengths ===")
    for name in sorted(f):
        obj = f[name]
        if isinstance(obj, h5py.Group) and 'wl_0' in obj.attrs:
            print(f"  {name}: wl_0 = {obj.attrs['wl_0']:.4f} um")
