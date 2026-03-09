"""
Step-1 intermediate verification:
Compare build_interaction_tensor (Goodman) with legacy application_function (Barrowes).

For every non-Nyquist, non-origin position (Ix, Iy, Iz) in the doubled grid,
the 3x3 block stored in A_tensor must equal MBT_elem = -Gkm from
application_function for the corresponding physical displacement.

Run from the repo root:
    python verify_interaction_tensor.py
"""

import numpy as np
from numba import njit

# ------------------------------------------------------------------ #
#  Legacy application_function (copied verbatim from old mvp_fft.py) #
# ------------------------------------------------------------------ #
@njit
def application_function_legacy(f, n_ind, lf, k):
    Xkm = (n_ind[0, :] - n_ind[1, :]) * lf
    Xkm_abs = np.linalg.norm(Xkm)
    XkmXkm = np.outer(Xkm, Xkm)
    MBT_elem = np.zeros((3, 3), dtype=np.complex128)
    Imat = np.identity(3, dtype=np.complex128)
    if Xkm_abs == 0:
        MBT_elem = np.zeros((3, 3), dtype=np.complex128)
    else:
        Gkm = (k * k * (Imat - XkmXkm / (Xkm_abs * Xkm_abs))
               - ((1 - 1j * k * Xkm_abs) / (Xkm_abs * Xkm_abs))
               * (Imat - 3 * XkmXkm / (Xkm_abs * Xkm_abs)))
        Gkm *= np.exp(1j * k * Xkm_abs) / Xkm_abs
        MBT_elem = -Gkm.astype(np.complex128)
    return MBT_elem


# ------------------------------------------------------------------ #
#  Goodman build_interaction_tensor (from new mvp_fft.py)            #
# ------------------------------------------------------------------ #
from mvp_fft.mvp_fft import build_interaction_tensor


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
def delta_from_extended_index(I, N):
    """Map extended FFT index I to integer displacement delta."""
    if I == N:
        return None          # Nyquist: skip
    return int(I) if I < N else int(I) - 2 * N


def legacy_3x3(delta, lf, k):
    """Call application_function for integer displacement (dx, dy, dz)."""
    dx, dy, dz = delta
    # n_ind: choose any valid 1-based pair whose difference equals (dx, dy, dz).
    # We use: n_ind[0] = (max(0,d)+1),  n_ind[1] = (max(0,-d)+1)  for each component.
    n_ind = np.array([[max(0, dx) + 1, max(0, dy) + 1, max(0, dz) + 1],
                       [max(0, -dx) + 1, max(0, -dy) + 1, max(0, -dz) + 1]],
                      dtype=np.int64)
    return application_function_legacy(3, n_ind, lf, k)


# ------------------------------------------------------------------ #
#  Verification                                                       #
# ------------------------------------------------------------------ #
def verify(n, lf, k, tol=1e-10):
    Nx, Ny, Nz = n
    print(f"\n=== Verification  n={n}  lf={lf}  k={k} ===")

    # Pre-compile numba JIT (dummy call)
    _ = application_function_legacy(3,
                                     np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int64),
                                     lf, k)

    A_tensor = build_interaction_tensor(n, lf, k)

    errors = []
    skipped_nyquist = 0
    skipped_origin  = 0

    for Ix in range(2 * Nx):
        dx = delta_from_extended_index(Ix, Nx)
        if dx is None:
            skipped_nyquist += 2 * Ny * 2 * Nz
            continue
        for Iy in range(2 * Ny):
            dy = delta_from_extended_index(Iy, Ny)
            if dy is None:
                skipped_nyquist += 2 * Nz
                continue
            for Iz in range(2 * Nz):
                dz = delta_from_extended_index(Iz, Nz)
                if dz is None:
                    skipped_nyquist += 1
                    continue

                if dx == 0 and dy == 0 and dz == 0:
                    # Origin: both sides should be zero
                    expected = np.zeros((3, 3), dtype=np.complex128)
                    skipped_origin += 1
                else:
                    expected = legacy_3x3((dx, dy, dz), lf, k)

                got = A_tensor[Ix, Iy, Iz]
                # Relative tolerance: normalise by the larger of the two norms
                # to handle elements near zero gracefully.
                scale = max(np.max(np.abs(expected)), np.max(np.abs(got)), 1e-30)
                err = np.max(np.abs(got - expected)) / scale
                errors.append(err)

                if err > tol:
                    print(f"  MISMATCH at (Ix,Iy,Iz)=({Ix},{Iy},{Iz}), "
                          f"delta=({dx},{dy},{dz}):")
                    print(f"    max |got - expected| = {err:.3e}")
                    print(f"    expected[0,:] = {expected[0,:]}")
                    print(f"    got[0,:]      = {got[0,:]}")

    # Check Nyquist positions are zero
    nyquist_errs = []
    for Iz in range(2 * Nz):
        nyquist_errs.append(np.max(np.abs(A_tensor[Nx, :, Iz])))
        nyquist_errs.append(np.max(np.abs(A_tensor[:, Ny, Iz])))
    for Ix in range(2 * Nx):
        for Iy in range(2 * Ny):
            nyquist_errs.append(np.max(np.abs(A_tensor[Ix, Iy, Nz])))
    nyq_max = max(nyquist_errs)

    n_pass = sum(e <= tol for e in errors)
    n_fail = sum(e > tol for e in errors)
    max_err = max(errors) if errors else 0.0
    mean_err = np.mean(errors) if errors else 0.0

    print(f"  Positions checked  : {len(errors)} "
          f"(origin: {skipped_origin}, Nyquist zeroed: {skipped_nyquist})")
    print(f"  PASS  : {n_pass}")
    print(f"  FAIL  : {n_fail}")
    print(f"  max rel-error  = {max_err:.3e}  (tolerance {tol:.0e})")
    print(f"  mean rel-error = {mean_err:.3e}")
    print(f"  Nyquist max  = {nyq_max:.3e}  (should be 0)")

    if n_fail == 0 and nyq_max < tol:
        print("  --> ALL PASSED")
    else:
        print("  --> VERIFICATION FAILED")
    return n_fail == 0 and nyq_max < tol


if __name__ == "__main__":
    # Small grids at several parameter combinations
    cases = [
        (np.array([2, 2, 2]), 0.1,  1.0),
        (np.array([3, 2, 4]), 0.15, 2.5),
        (np.array([4, 4, 4]), 0.05, 0.8),
    ]
    all_ok = True
    for (n, lf, k) in cases:
        ok = verify(n, lf, k)
        all_ok = all_ok and ok

    print("\n" + ("=" * 50))
    if all_ok:
        print("ALL VERIFICATION CASES PASSED")
    else:
        print("SOME CASES FAILED -- check output above")
