import os
import numpy as np
from scipy.fft import fftn, ifftn

# Number of threads used by scipy.fft.
# Default: leave 2 cores free so other processes stay responsive.
# Override at runtime via the environment variable DDA_FFT_WORKERS, e.g.:
#   DDA_FFT_WORKERS=4 python run_dda.py
_FFT_WORKERS = int(os.environ.get("DDA_FFT_WORKERS",
                                   max(1, (os.cpu_count() or 1) - 2)))


def build_interaction_tensor(n, lf, k):
    """
    Construct the 3D interaction tensor A'(r) on the doubled grid (Goodman 1991).

    For each extended-grid index (Ix, Iy, Iz) the physical integer displacement
    delta_alpha is mapped as:
        Ix in [0,   N-1]      ->  delta_alpha = Ix          (positive)
        Ix = N                ->  delta_alpha = 0 (Nyquist, zeroed out)
        Ix in [N+1, 2N-1]    ->  delta_alpha = Ix - 2N      (negative, wrap-around)

    The 3x3 tensor element A'(r) equals -G(r) where G is the free-space
    Green's function dyadic:
        A'(r) = exp(ikr)/r * [k^2*(rhat x rhat - I) + (1-ikr)/r^2*(I - 3*rhat x rhat)]

    This is identical to `MBT_elem = -Gkm` returned by the legacy
    `application_function`.

    Parameters
    ----------
    n  : array_like of int, length 3  [Nx, Ny, Nz]
    lf : float   physical lattice spacing [um]
    k  : float   wavenumber in medium [um^-1]

    Returns
    -------
    A_tensor : ndarray, shape (2*Nx, 2*Ny, 2*Nz, 3, 3), dtype complex128
    """
    Nx, Ny, Nz = int(n[0]), int(n[1]), int(n[2])

    # Extended-grid indices
    Ix = np.arange(2 * Nx)
    Iy = np.arange(2 * Ny)
    Iz = np.arange(2 * Nz)

    # Map extended index -> integer displacement (in lattice units)
    # Nyquist position (Ix == Nx) maps to 0; it will be zeroed out afterward.
    dx = np.where(Ix <= Nx, Ix, Ix - 2 * Nx).astype(np.float64)
    dy = np.where(Iy <= Ny, Iy, Iy - 2 * Ny).astype(np.float64)
    dz = np.where(Iz <= Nz, Iz, Iz - 2 * Nz).astype(np.float64)

    # Physical displacement vectors; sparse meshgrid to avoid O(N) intermediate arrays
    # DX: shape (2Nx, 1, 1), DY: (1, 2Ny, 1), DZ: (1, 1, 2Nz)
    DX, DY, DZ = np.meshgrid(dx * lf, dy * lf, dz * lf, indexing='ij', sparse=True)

    # Physical distance  shape (2Nx, 2Ny, 2Nz)
    r = np.sqrt(DX**2 + DY**2 + DZ**2)

    # Temporarily replace r=0 with 1.0 to avoid division by zero;
    # the origin element is explicitly zeroed at the end.
    r_safe = np.where(r == 0.0, 1.0, r)

    # Unit direction vector components, each shape (2Nx, 2Ny, 2Nz)
    rx = DX / r_safe
    ry = DY / r_safe
    rz = DZ / r_safe

    # Stack -> r_hat shape (2Nx, 2Ny, 2Nz, 3)
    r_hat = np.stack([rx, ry, rz], axis=-1)

    # Outer product  r_hat ⊗ r_hat, shape (2Nx, 2Ny, 2Nz, 3, 3)
    rr = r_hat[..., :, np.newaxis] * r_hat[..., np.newaxis, :]

    I3 = np.eye(3, dtype=np.complex128).reshape(1, 1, 1, 3, 3)

    # A'(r) = exp(ikr)/r * [k^2*(rhat x rhat - I) + (1-ikr)/r^2*(I - 3*rhat x rhat)]
    exp_term = (np.exp(1j * k * r_safe) / r_safe)[..., np.newaxis, np.newaxis]

    term1 = k**2 * (rr - I3)
    coef2 = ((1.0 - 1j * k * r_safe) / r_safe**2)[..., np.newaxis, np.newaxis]
    term2 = coef2 * (I3 - 3.0 * rr)

    A_tensor = exp_term * (term1 + term2)

    # Self-interaction (origin): handled separately as diagonal of the full matrix
    A_tensor[0, 0, 0, :, :] = 0.0

    # Nyquist positions: physically unreachable, zero out
    A_tensor[Nx, :, :, :, :] = 0.0
    A_tensor[:, Ny, :, :, :] = 0.0
    A_tensor[:, :, Nz, :, :] = 0.0

    return A_tensor


def fft_init(n, f, lf, k):
    """
    Pre-compute the FFT of the interaction tensor for fast MVP.
    Replaces legacy MBT_fft_init.

    Parameters
    ----------
    n  : array_like of int, length 3  [Nx, Ny, Nz]
    f  : int  vector dimension; must be 3 (electric dipoles only)
    lf : float  physical lattice spacing [um]
    k  : float  wavenumber in medium [um^-1]

    Returns
    -------
    A_tensor_tilde : ndarray, shape (2*Nx, 2*Ny, 2*Nz, 3, 3), dtype complex128
        FFT of the interaction tensor along spatial axes (0, 1, 2).
        Stored in memory once; unchanged across Krylov iterations and
        incident directions.
    """
    if f != 3:
        raise ValueError("fft_init: only f=3 (electric dipoles) is supported.")
    A_tensor = build_interaction_tensor(n, lf, k)
    return fftn(A_tensor, axes=(0, 1, 2), workers=_FFT_WORKERS)


def fft_mvp(n, f, A_tensor_tilde, P_block):
    """
    Fast matrix-vector product via 3D FFT (Goodman 1991).
    Replaces legacy MBT_fft_mvp.

    Supports both single-column (1-D) and block (2-D) inputs to allow
    drop-in replacement AND vectorised multi-RHS processing.

    Parameters
    ----------
    n             : array_like of int, length 3  [Nx, Ny, Nz]
    f             : int  vector dimension; must be 3
    A_tensor_tilde: ndarray, shape (2Nx, 2Ny, 2Nz, 3, 3)  from fft_init
    P_block       : ndarray, shape (f*Nx*Ny*Nz,) or (f*Nx*Ny*Nz, L)
                    Polarisation vector(s) on the full cuboid grid.
                    Elements outside the target volume should be zero.

    Returns
    -------
    result : ndarray  same shape as P_block
        Off-diagonal contribution A_offdiag @ P_block.
    """
    Nx, Ny, Nz = int(n[0]), int(n[1]), int(n[2])

    squeeze = (P_block.ndim == 1)
    if squeeze:
        P_block = P_block[:, np.newaxis]
    L = P_block.shape[1]

    # Reshape to 5-D physical representation
    P_5d = P_block.reshape(Nx, Ny, Nz, f, L)

    # Zero-pad to doubled grid
    P_padded = np.zeros((2 * Nx, 2 * Ny, 2 * Nz, f, L), dtype=np.complex128)
    P_padded[:Nx, :Ny, :Nz, :, :] = P_5d

    # Forward 3-D FFT on spatial axes
    P_tilde = fftn(P_padded, axes=(0, 1, 2), workers=_FFT_WORKERS)

    # Element-wise tensor product in Fourier space:
    #   A_tilde(..., i, j) * P_tilde(..., j, l) -> Y_tilde(..., i, l)
    Y_tilde = A_tensor_tilde @ P_tilde

    # Inverse 3-D FFT back to physical space
    Y_padded = ifftn(Y_tilde, axes=(0, 1, 2), workers=_FFT_WORKERS)

    # Extract the valid physical region and flatten
    result = Y_padded[:Nx, :Ny, :Nz, :, :].reshape(f * Nx * Ny * Nz, L)

    if squeeze:
        return result[:, 0]
    return result
