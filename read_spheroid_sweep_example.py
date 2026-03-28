"""Example: read a single data point from dda_results_spheroid_sweep.h5
by specifying all physical parameters and orientation angles.

Grid coordinates: D_ve, RI_real, log10(AR), cos_theta_o (half-domain), phi_o.
"""

import h5py
import numpy as np

# ── 取り出したい条件を指定 ──────────────────────────────────────────────────
D_ve_target          = 0.6     # volume-equivalent diameter [um]
RI_real_target       = 1.5     # Re(m_p)
log_AR_target        = 0.0     # log10(AR), 0 = sphere
cos_theta_o_target   = 0.5     # cos(theta_o), half-domain [0, 1]
phi_o_target         = 0.5     # phi_o [rad], domain [0, pi]
# ────────────────────────────────────────────────────────────────────────────

with h5py.File("dda_results/dda_results_spheroid_sweep.h5", "r") as f:

    # 各格子から最近傍インデックスを取得
    i_dve = int(np.argmin(np.abs(f["D_ve_grid"][:]              - D_ve_target)))
    i_ri  = int(np.argmin(np.abs(f["RI_real_grid"][:]           - RI_real_target)))
    i_ar  = int(np.argmin(np.abs(f["log_AR_grid"][:]            - log_AR_target)))
    i_u   = int(np.argmin(np.abs(f["cos_theta_o_half_grid"][:]  - cos_theta_o_target)))
    i_ph  = int(np.argmin(np.abs(f["phi_o_grid"][:]             - phi_o_target)))

    # 実際にヒットした格子点の値を表示
    print("=== Selected grid point ===")
    print(f"  D_ve         = {f['D_ve_grid'][i_dve]:.4f} um  (target {D_ve_target})")
    print(f"  RI_real      = {f['RI_real_grid'][i_ri]:.4f}      (target {RI_real_target})")
    print(f"  log10(AR)    = {f['log_AR_grid'][i_ar]:.4f}      (target {log_AR_target})")
    print(f"  AR           = {10**f['log_AR_grid'][i_ar]:.4f}")
    print(f"  cos(theta_o) = {f['cos_theta_o_half_grid'][i_u]:.4f}      (target {cos_theta_o_target})")
    print(f"  phi_o        = {f['phi_o_grid'][i_ph]:.4f} rad  (target {phi_o_target})")
    print(f"  converged    = {bool(f['converged'][i_dve, i_ri, i_ar])}")

    # 結果取り出し
    idx = (i_dve, i_ri, i_ar, i_u, i_ph)

    S_fw_theta = f["S_fw_theta_re"][idx] + 1j * f["S_fw_theta_im"][idx]
    S_fw_phi   = f["S_fw_phi_re"][idx]   + 1j * f["S_fw_phi_im"][idx]

    print("\n=== DDA results ===")
    print(f"  S_fw_theta = {S_fw_theta:.6e}")
    print(f"  S_fw_phi   = {S_fw_phi:.6e}")

    # Provenance attributes
    print("\n=== Provenance ===")
    print(f"  wl_0    = {f.attrs['wl_0']} um")
    print(f"  m_m     = {f.attrs['m_m']}")
    print(f"  version = {f.attrs['block_dda_version']}")
