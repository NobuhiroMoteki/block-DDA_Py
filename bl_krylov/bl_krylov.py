import numpy as np
from mvp_fft.mvp_fft import fft_mvp


def _build_scatter_index(address, f):
    """
    Build an index array that maps sparse-vector positions to full-grid positions.

    For the m-th occupied dipole:
        sparse position  : f*m + c   (c = 0..f-1)
        full-grid position: f*address[m] + c

    Parameters
    ----------
    address : 1-D int array of length N_occ
    f       : int (3 for electric dipoles)

    Returns
    -------
    full_idx : 1-D int array of length f*N_occ
        full_idx[f*m + c] = f*address[m] + c
    """
    return (np.repeat(address, f) * f
            + np.tile(np.arange(f), len(address)))


def _block_mvp(n, f, Au_til, full_idx, DIAG_A, S, jpre):
    """
    Compute  AS = jpre * (DIAG_A * S + A_offdiag * S)  for all L columns at once.

    Parameters
    ----------
    n       : array_like [Nx, Ny, Nz]
    f       : int
    Au_til  : ndarray (2Nx,2Ny,2Nz,3,3) – pre-FFT interaction tensor
    full_idx: 1-D int array from _build_scatter_index
    DIAG_A  : 1-D complex array of length f*N_occ  – diagonal of A
    S       : 2-D complex array (f*N_occ, L)
    jpre    : 1-D complex array of length f*N_occ  – Jacobi preconditioner (1/DIAG_A)

    Returns
    -------
    AS : ndarray (f*N_occ, L)
    """
    num_element_cuboid = np.prod(n)

    # Scatter: sparse (f*N_occ, L) -> full cuboid (f*prod(n), L)
    P_hat = np.zeros((f * num_element_cuboid, S.shape[1]), dtype=np.complex128)
    P_hat[full_idx, :] = S

    # Block MVP for all L columns in one call
    AP_full = fft_mvp(n, f, Au_til, P_hat)   # shape (f*prod(n), L)

    # Diagonal + gather off-diagonal; apply Jacobi preconditioner
    return (DIAG_A[:, np.newaxis] * S + AP_full[full_idx, :]) * jpre[:, np.newaxis]


def bl_cocg_rq_jacobi_mvp_fft(n, f, address, Au_til, DIAG_A, B, tol, itermax):
    """
    Block-COCG-RQ with Jacobi preconditioning and FFT-accelerated MVP.
    Solves A X = B for complex-symmetric block-Toeplitz A.

    Gu et al. 2016, arXiv: Block variants of COCG and COCR methods for solving
    complex symmetric linear systems with L right-hand sides.

    Parameters
    ----------
    n       : array_like [Nx, Ny, Nz]
    f       : int  (3 for electric dipoles)
    address : 1-D int array of occupied cuboid addresses
    Au_til  : ndarray (2Nx,2Ny,2Nz,3,3) – pre-FFT interaction tensor from fft_init
    DIAG_A  : 1-D complex array (f*N_occ,) – diagonal of A
    B       : 2-D complex array (f*N_occ, L) – RHS block; columns must be linearly independent
    tol     : float  convergence tolerance
    itermax : int    maximum iterations

    Returns
    -------
    X        : ndarray (f*N_occ, L) – solution block
    iter_fin : int   – final iteration count
    err_fin  : float – final relative residual
    """
    L = B.shape[1]
    jpre = 1.0 / DIAG_A
    full_idx = _build_scatter_index(address, f)

    B_jpre = B * jpre[:, np.newaxis]
    B_jpre_norm = np.linalg.norm(B_jpre)

    X = np.zeros_like(B)

    Q, xi = np.linalg.qr(B_jpre, mode='reduced')
    S = Q

    iter_fin = 0
    err_fin = float('inf')

    for k in range(itermax):
        AS = _block_mvp(n, f, Au_til, full_idx, DIAG_A, S, jpre)

        alpha = np.linalg.solve(S.T @ AS, Q.T @ Q)
        X = X + S @ (alpha @ xi)

        Qnew, tau = np.linalg.qr(Q - AS @ alpha, mode='reduced')
        xi = tau @ xi
        err = np.linalg.norm(xi) / B_jpre_norm
        print("iter= {:}, err= {:.4f}".format(k, err))
        iter_fin = k
        err_fin = err
        if err < tol:
            break

        beta = np.linalg.solve(Q.T @ Q,
                               tau.T @ (Qnew.T @ Qnew))
        Q = Qnew
        S = Q + S @ beta

    return X, iter_fin, err_fin


def bl_bicgstab_jacobi_mvp_fft(n, f, address, Au_til, DIAG_A, B, tol, itermax):
    """
    Block-BiCGSTAB with Jacobi preconditioning and FFT-accelerated MVP.
    Solves A X = B for general complex block-Toeplitz A.

    Tadano et al. 2009, JSIAM Letters: Block BiCGSTAB.

    Parameters
    ----------
    n       : array_like [Nx, Ny, Nz]
    f       : int  (3 for electric dipoles)
    address : 1-D int array of occupied cuboid addresses
    Au_til  : ndarray (2Nx,2Ny,2Nz,3,3) – pre-FFT interaction tensor from fft_init
    DIAG_A  : 1-D complex array (f*N_occ,) – diagonal of A
    B       : 2-D complex array (f*N_occ, L) – RHS block; columns must be linearly independent
    tol     : float  convergence tolerance
    itermax : int    maximum iterations

    Returns
    -------
    X        : ndarray (f*N_occ, L) – solution block
    iter_fin : int   – final iteration count
    err_fin  : float – final relative residual
    """
    L = B.shape[1]
    jpre = 1.0 / DIAG_A
    full_idx = _build_scatter_index(address, f)

    B_jpre = B * jpre[:, np.newaxis]
    B_jpre_norm = np.linalg.norm(B_jpre)

    X = np.zeros_like(B)
    R = B_jpre.copy()
    P = R.copy()
    R0til = R.copy()
    R0til_H = R0til.conj().T

    iter_fin = 0
    err_fin = float('inf')

    for k in range(itermax):

        # V = A * P
        V = _block_mvp(n, f, Au_til, full_idx, DIAG_A, P, jpre)

        RV = R0til_H @ V
        alpha = np.linalg.solve(RV, R0til_H @ R)
        T = R - V @ alpha

        # Z = A * T
        Z = _block_mvp(n, f, Au_til, full_idx, DIAG_A, T, jpre)

        qsi = np.trace(Z.conj().T @ T) / np.trace(Z.conj().T @ Z)
        X = X + P @ alpha + qsi * T
        R = T - qsi * Z

        err = np.linalg.norm(R) / B_jpre_norm
        print("iter= {:}, err= {:.4f}".format(k, err))
        iter_fin = k
        err_fin = err
        if err < tol:
            break

        beta = np.linalg.solve(RV, (-R0til_H @ Z))
        P = R + (P - qsi * V) @ beta

    return X, iter_fin, err_fin
