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


def _block_mvp_noprec(n, f, Au_til, full_idx, DIAG_A, S):
    """
    Compute  AS = DIAG_A * S + A_offdiag * S  for all L columns at once,
    with no Jacobi preconditioning. VIEM parity variant.
    """
    num_element_cuboid = np.prod(n)
    P_hat = np.zeros((f * num_element_cuboid, S.shape[1]), dtype=np.complex128)
    P_hat[full_idx, :] = S
    AP_full = fft_mvp(n, f, Au_til, P_hat)
    return DIAG_A[:, np.newaxis] * S + AP_full[full_idx, :]


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


def bl_bicgstab_mvp_fft(n, f, address, Au_til, DIAG_A, B, tol, itermax, verbose=False):
    """
    Block-BiCGSTAB without Jacobi preconditioning. Solves A X = B.

    Direct Python port of block-VIEM.jl's `block_bicgstab`
    (src/block_krylov.jl), which is itself a port of
    bl_bicgstab_jacobi_mvp_fft minus the preconditioner.

    Same API as bl_bicgstab_jacobi_mvp_fft; DIAG_A is used in the MVP
    (A = DIAG_A + A_offdiag) but no preconditioning is applied.

    Parameters
    ----------
    n       : array_like [Nx, Ny, Nz]
    f       : int  (3 for electric dipoles)
    address : 1-D int array of occupied cuboid addresses
    Au_til  : ndarray (2Nx,2Ny,2Nz,3,3) – pre-FFT interaction tensor
    DIAG_A  : 1-D complex array (f*N_occ,) – diagonal of A
    B       : 2-D complex array (f*N_occ, L) – RHS block
    tol     : float  convergence tolerance
    itermax : int    maximum iterations
    verbose : bool   print per-iteration residual (default False for new solvers)

    Returns
    -------
    X        : ndarray (f*N_occ, L) – solution block
    iter_fin : int   – final iteration count
    err_fin  : float – final relative residual ‖B − A X‖_F / ‖B‖_F
    """
    full_idx = _build_scatter_index(address, f)
    B_norm = np.linalg.norm(B)
    if B_norm == 0.0:
        return np.zeros_like(B), 0, 0.0

    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()
    R0til = R.copy()
    R0til_H = R0til.conj().T

    iter_fin = 0
    err_fin = float('inf')

    for k in range(itermax):
        V = _block_mvp_noprec(n, f, Au_til, full_idx, DIAG_A, P)

        RV = R0til_H @ V
        alpha = np.linalg.solve(RV, R0til_H @ R)
        T = R - V @ alpha

        Z = _block_mvp_noprec(n, f, Au_til, full_idx, DIAG_A, T)

        qsi = np.trace(Z.conj().T @ T) / np.trace(Z.conj().T @ Z)
        X = X + P @ alpha + qsi * T
        R = T - qsi * Z

        err = np.linalg.norm(R) / B_norm
        if verbose:
            print("iter= {:}, err= {:.4f}".format(k, err))
        iter_fin = k
        err_fin = err
        if err < tol:
            break

        beta = np.linalg.solve(RV, (-R0til_H @ Z))
        P = R + (P - qsi * V) @ beta

    return X, iter_fin, err_fin


def bl_gmres_mvp_fft(n, f, address, Au_til, DIAG_A, B, tol, itermax, verbose=False):
    """
    Unrestarted Block GMRES without Jacobi preconditioning.
    Solves A X = B via block-Arnoldi with thin QR of each new block.

    Simoncini & Szyld, NLAA 1996.
    Direct Python port of block-VIEM.jl's `block_gmres` (src/block_krylov.jl).

    Memory cost is O(itermax · f·N_occ · L) — the full block Krylov basis
    is retained. For long-running problems, use bl_bicgstab_mvp_fft.

    Parameters
    ----------
    n, f, address, Au_til, DIAG_A, B, tol, itermax : same as bl_bicgstab_mvp_fft
    verbose : bool   print per-iteration residual (default False)

    Returns
    -------
    X        : ndarray (f*N_occ, L) – solution block
    iter_fin : int   – final iteration count
    err_fin  : float – final relative residual ‖B − A X‖_F / ‖B‖_F
    """
    L = B.shape[1]
    N = B.shape[0]
    full_idx = _build_scatter_index(address, f)

    B_norm = np.linalg.norm(B)
    if B_norm == 0.0:
        return np.zeros_like(B), 0, 0.0

    # Initial residual (X0 = 0 ⇒ R0 = B) and its thin QR: R0 = V1 · Λ
    V1, Lambda = np.linalg.qr(B, mode='reduced')   # V1: (N,L),  Lambda: (L,L)

    Vblocks = [V1]
    # Hfull stored as a dense block-Hessenberg matrix grown block-column by block-column.
    Hfull = np.zeros(((itermax + 1) * L, itermax * L), dtype=np.complex128)

    iter_fin = 0
    err_fin = float('inf')
    last_X = np.zeros_like(B)

    for k in range(1, itermax + 1):
        iter_fin = k - 1
        W = _block_mvp_noprec(n, f, Au_til, full_idx, DIAG_A, Vblocks[k - 1])

        # Block Gram–Schmidt against prior blocks
        for i in range(1, k + 1):
            Hik = Vblocks[i - 1].conj().T @ W
            Hfull[(i - 1) * L:i * L, (k - 1) * L:k * L] = Hik
            W = W - Vblocks[i - 1] @ Hik

        # Thin QR of the remaining block
        Vk1, Hk1k = np.linalg.qr(W, mode='reduced')
        Hfull[k * L:(k + 1) * L, (k - 1) * L:k * L] = Hk1k
        Vblocks.append(Vk1)

        # Least-squares:  min ‖ H_k · Y − E1·Λ ‖_F  over the current (k+1)L × kL
        m = (k + 1) * L
        nn = k * L
        Hk_view = Hfull[:m, :nn]
        rhs = np.zeros((m, L), dtype=np.complex128)
        rhs[:L, :] = Lambda
        Y, _, _, _ = np.linalg.lstsq(Hk_view, rhs, rcond=None)
        resblk = Hk_view @ Y - rhs
        err = np.linalg.norm(resblk) / B_norm
        if verbose:
            print("iter= {:}, err= {:.4f}".format(k - 1, err))
        err_fin = err

        if err < tol or k == itermax:
            X = np.zeros((N, L), dtype=np.complex128)
            for j in range(1, k + 1):
                X += Vblocks[j - 1] @ Y[(j - 1) * L:j * L, :]
            return X, iter_fin, err_fin

    return last_X, iter_fin, err_fin
