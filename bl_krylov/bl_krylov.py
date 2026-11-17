import time
from typing import NamedTuple, Optional

import numpy as np
import scipy.linalg
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
    X           : ndarray (f*N_occ, L) – solution block
    iter_fin    : int   – final iteration count
    err_fin     : float – final relative residual ‖B − A X‖_F / ‖B‖_F
    err_history : ndarray (iter_fin+1,) float64 – per-iteration residuals
                  (same scale as err_fin, indexed 0..iter_fin)
    """
    full_idx = _build_scatter_index(address, f)
    B_norm = np.linalg.norm(B)
    if B_norm == 0.0:
        return np.zeros_like(B), 0, 0.0, np.zeros(1, dtype=np.float64)

    X = np.zeros_like(B)
    R = B.copy()
    P = R.copy()
    R0til = R.copy()
    R0til_H = R0til.conj().T

    iter_fin = 0
    err_fin = float('inf')
    err_history = []

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
        err_history.append(float(err))
        if verbose:
            print("iter= {:}, err= {:.4f}".format(k, err))
        iter_fin = k
        err_fin = err
        if err < tol:
            break

        beta = np.linalg.solve(RV, (-R0til_H @ Z))
        P = R + (P - qsi * V) @ beta

    return X, iter_fin, err_fin, np.asarray(err_history, dtype=np.float64)


def bl_gmres_mvp_fft(n, f, address, Au_til, DIAG_A, B, tol, itermax, verbose=False):
    """
    Unrestarted Block GMRES without Jacobi preconditioning.
    Solves A X = B via block-Arnoldi with thin QR of each new block, and
    **incremental block-Givens QR triangularisation** of the Hessenberg —
    the residual Frobenius norm is available at each step without solving
    a least-squares problem (Saad & Schultz 1986; Simoncini & Szyld,
    NLAA 1996).

    Python port of block-VIEM.jl's `block_gmres` (src/block_krylov.jl)
    including its incremental-QR optimisation (VIEM 2026-04-24).

    Previous lstsq-based implementation cost O((kL)³) per iteration;
    this version costs O(kL²) per iteration for rotation propagation
    plus one 2L×L block QR (O(L³)).  Speedup is material for L=100
    GRE sweeps.

    Maintained state across iterations:
      R_tri    — (itermax·L, itermax·L) upper-triangular factor of H
                 after block-Givens triangularisation (stored for final
                 back-substitution)
      b_hat    — ((itermax+1)·L, L) transformed RHS; init [Λ; 0; …].
                 Residual Frobenius norm at iteration k equals
                 ‖b_hat[kL:(k+1)L, :]‖_F.
      Qstore   — list of 2L×2L orthogonal Q factors, one per iter k,
                 obtained by QR of the 2L×L super-block
                 [H_{kk}; H_{k+1,k}].  Applied from the left to
                 future H-columns and the corresponding row range of
                 b_hat.

    Memory cost is O(itermax · f·N_occ · L) — the full block Krylov
    basis is retained. For long-running problems, use bl_bicgstab_mvp_fft.

    Parameters
    ----------
    n, f, address, Au_til, DIAG_A, B, tol, itermax : same as bl_bicgstab_mvp_fft
    verbose : bool   print per-iteration residual (default False)

    Returns
    -------
    X           : ndarray (f*N_occ, L) – solution block
    iter_fin    : int   – final iteration count (0-based, as before)
    err_fin     : float – final relative residual ‖B − A X‖_F / ‖B‖_F
    err_history : ndarray (iter_fin+1,) float64 – per-iteration residuals
    """
    L = B.shape[1]
    N = B.shape[0]
    full_idx = _build_scatter_index(address, f)

    B_norm = np.linalg.norm(B)
    if B_norm == 0.0:
        return np.zeros_like(B), 0, 0.0, np.zeros(1, dtype=np.float64)

    # Initial residual (X0 = 0 ⇒ R0 = B) and its thin QR: R0 = V1 · Λ
    V1, Lambda = np.linalg.qr(B, mode='reduced')   # V1: (N,L),  Lambda: (L,L)

    Vblocks = [V1]

    # Incremental block-Givens QR state
    R_tri = np.zeros((itermax * L, itermax * L), dtype=np.complex128)
    b_hat = np.zeros(((itermax + 1) * L, L), dtype=np.complex128)
    b_hat[:L, :] = Lambda
    Qstore = [None] * itermax                  # 2L×2L full Q per iter

    iter_fin = 0
    err_fin = float('inf')
    err_history = []

    for k in range(1, itermax + 1):
        iter_fin = k - 1
        W = _block_mvp_noprec(n, f, Au_til, full_idx, DIAG_A, Vblocks[k - 1])

        # Block Gram–Schmidt: build H_col of shape ((k+1)L, L).
        H_col = np.zeros(((k + 1) * L, L), dtype=np.complex128)
        for i in range(1, k + 1):
            Hik = Vblocks[i - 1].conj().T @ W
            H_col[(i - 1) * L:i * L, :] = Hik
            W = W - Vblocks[i - 1] @ Hik

        # Thin QR of the remaining block → V_{k+1}, H_{k+1,k}
        Vk1, Hk1k = np.linalg.qr(W, mode='reduced')
        H_col[k * L:(k + 1) * L, :] = Hk1k
        Vblocks.append(Vk1)

        # Apply previously stored block-Givens rotations (j = 1..k-1).
        # Rotation j acts on rows (j-1)L : (j+1)L of the current H-column.
        for j in range(1, k):
            rlo = (j - 1) * L
            rhi = (j + 1) * L
            H_col[rlo:rhi, :] = Qstore[j - 1].conj().T @ H_col[rlo:rhi, :]

        # New block-Givens: QR the 2L×L super-block [H_kk; H_{k+1,k}].
        # Use mode='complete' to obtain full 2L×2L Q (mode='reduced' returns
        # only 2L×L, which would discard the rotation on the off-diagonal
        # block-row that must act on b_hat).
        super_lo = (k - 1) * L
        super_hi = (k + 1) * L
        Qmat, Rmat = np.linalg.qr(H_col[super_lo:super_hi, :], mode='complete')
        # Qmat: (2L, 2L), Rmat: (2L, L); take the top L×L of Rmat as the
        # new diagonal block and zero the sub-diagonal.
        Qstore[k - 1] = Qmat
        H_col[super_lo:super_lo + L, :] = Rmat[:L, :]
        H_col[super_lo + L:super_hi, :] = 0.0

        # Store k-th block column of R_tri (only first k block-rows non-zero).
        R_tri[:k * L, (k - 1) * L:k * L] = H_col[:k * L, :]

        # Apply new rotation to b_hat on the same row range.
        b_hat[super_lo:super_hi, :] = Qmat.conj().T @ b_hat[super_lo:super_hi, :]

        # Residual Frobenius norm: rows kL : (k+1)L of b_hat.
        err = np.linalg.norm(b_hat[k * L:(k + 1) * L, :]) / B_norm
        err_history.append(float(err))
        if verbose:
            print("iter= {:}, err= {:.4e}".format(k - 1, err))
        err_fin = err

        if err < tol or k == itermax:
            # Upper-triangular solve  R_k · Y = b_hat[:kL, :]
            R_top = R_tri[:k * L, :k * L]
            Y = scipy.linalg.solve_triangular(R_top, b_hat[:k * L, :],
                                              lower=False)
            # Reconstruct X = Σ_j V_j · Y_j
            X = np.zeros((N, L), dtype=np.complex128)
            for j in range(1, k + 1):
                X += Vblocks[j - 1] @ Y[(j - 1) * L:j * L, :]
            return X, iter_fin, err_fin, np.asarray(err_history, dtype=np.float64)

    # Unreachable; the loop always returns on its final iteration.
    return (np.zeros_like(B), iter_fin, err_fin,
            np.asarray(err_history, dtype=np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# Shifted block GMRES: one block-Arnoldi process for many scalar shifts
# ══════════════════════════════════════════════════════════════════════════════

class ShiftedBlockGMRESResult(NamedTuple):
    """Output of :func:`bl_gmres_shifted_mvp_fft`.

    N_m is the number of shifts, L the number of right-hand sides, k_s the
    number of block-Arnoldi steps taken for shift s and k_fin = max_s k_s.
    """
    Y_list: list              # length N_m; Y_s has shape (k_s*L, L), X_s = [V_1 … V_{k_s}] Y_s
    Lambda: np.ndarray        # shape (L, L); R factor of the thin QR  B = V_1 Λ
    G: Optional[np.ndarray]   # shape (n_obs, L, k_fin*L); F^H [V_1 … V_{k_fin}], or None
    iters: np.ndarray         # shape (N_m,) int; k_s (1-based count of block-Arnoldi steps)
    errs: np.ndarray          # shape (N_m,) float; final ‖B − A_s X_s‖_F / ‖B‖_F
    err_history: np.ndarray   # shape (k_fin, N_m) float; NaN after shift s has stopped
    X_list: Optional[list]    # length N_m; X_s has shape (f*N_occ, L), or None
    timings: dict             # wall time [s] per part: 'mvp', 'gs', 'lsq', 'proj'


def _block_mvp_offdiag(n, f, Au_til, full_idx, S):
    """
    Compute  A_offdiag * S  for all L columns at once (no diagonal term).

    This is `_block_mvp_noprec` without the `DIAG_A * S` term. A_offdiag depends
    on the lattice, the wavelength and the medium only, not on the particle
    refractive index.

    Parameters
    ----------
    n, f, Au_til, full_idx : same as `_block_mvp_noprec`
    S : 2-D complex array (f*N_occ, L)

    Returns
    -------
    AS : ndarray (f*N_occ, L)
    """
    num_element_cuboid = np.prod(n)
    P_hat = np.zeros((f * num_element_cuboid, S.shape[1]), dtype=np.complex128)
    P_hat[full_idx, :] = S
    AP_full = fft_mvp(n, f, Au_til, P_hat)
    return AP_full[full_idx, :]


def bl_gmres_shifted_mvp_fft(n, f, address, Au_til, sigmas, B, tol, itermax,
                             proj_ops=None, return_X=False, verbose=False):
    """
    Unrestarted shifted block GMRES without preconditioning.

    Solves the family of systems

        (sigma_s I + A_offdiag) X_s = B,      s = 1 … N_m,

    with ONE block-Arnoldi process. The block Krylov space of A_offdiag and B
    equals that of sigma_s I + A_offdiag and B for every shift, so the basis
    V_1, V_2, … and the block Hessenberg matrix are shared:

        A_offdiag [V_1 … V_k] = [V_1 … V_{k+1}] Hbar_k,
        (sigma_s I + A_offdiag) [V_1 … V_k] = [V_1 … V_{k+1}] (Hbar_k + sigma_s Ibar_k).

    Here Hbar_k is (k+1)L × kL and Ibar_k is the kL identity with L zero rows
    appended. With X_0 = 0 and B = V_1 Λ, the GMRES iterate of shift s is

        X_s = [V_1 … V_k] Y_s,   Y_s = argmin_Y ‖E_1 Λ − (Hbar_k + sigma_s Ibar_k) Y‖_F,

    and the minimum equals ‖B − (sigma_s I + A_offdiag) X_s‖_F. Each X_s is
    therefore the iterate that `bl_gmres_mvp_fft` would produce for shift s on
    its own, and the stopping test below is the same one. The matrix–vector
    products and the block Gram–Schmidt are done once for all shifts. Per shift
    only a dense least-squares problem of size (k+1)L × kL is solved.

    In the DDA, sigma_s = 1/alpha_E(m_s) for a homogeneous isotropic particle
    and B is the incident field, both on a lattice common to all m_s.

    A shift stops at the first step k where its relative residual is below
    `tol`, or at k = itermax. Its Y_s is frozen at that step. The process ends
    when every shift has stopped.

    The least-squares problems are solved from scratch at every step by a dense
    QR of the augmented matrix [Hbar_k + sigma_s Ibar_k | E_1 Λ]. This costs
    O((kL)^3) per shift and step, which is small against the MVP as long as kL
    is a few hundred.

    Parameters
    ----------
    n, f, address, Au_til : same as `bl_gmres_mvp_fft`
    sigmas   : 1-D complex array (N_m,)   the shifts
    B        : 2-D complex array (f*N_occ, L)   right-hand sides, shared by all shifts
    tol      : float   relative residual tolerance, applied per shift
    itermax  : int     maximum number of block-Arnoldi steps
    proj_ops : None or 3-D complex array (n_obs, f*N_occ, L)
               Functionals F_o. If given, G[o] = F_o^H [V_1 … V_{k_fin}] is
               accumulated so that F_o^H X_s = G[o][:, :k_s*L] @ Y_s without
               assembling X_s.
    return_X : bool    also assemble and return the full solutions X_s
    verbose  : bool    print the residual range of the active shifts per step

    Returns
    -------
    ShiftedBlockGMRESResult
        Named tuple (Y_list, Lambda, G, iters, errs, err_history, X_list,
        timings). See the class for shapes. `iters` counts block-Arnoldi steps
        from 1, so it equals `iter_fin + 1` of `bl_gmres_mvp_fft`. A shift with
        `errs[s] >= tol` did not converge within `itermax`. `timings` holds the
        wall time [s] spent in the matrix–vector products ('mvp'), the block
        Gram–Schmidt and thin QR ('gs'), the per-shift least squares ('lsq')
        and the projections F^H V_k ('proj').
    """
    sigmas = np.atleast_1d(np.asarray(sigmas, dtype=np.complex128))
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a 1-D array of shifts")
    if itermax < 1:
        raise ValueError("itermax must be at least 1")
    N_m = sigmas.size
    L = B.shape[1]
    N = B.shape[0]
    n_obs = 0 if proj_ops is None else proj_ops.shape[0]
    full_idx = _build_scatter_index(address, f)

    B_norm = np.linalg.norm(B)
    if B_norm == 0.0:
        # Zero right-hand side: the solution is zero and no Arnoldi step is taken.
        return ShiftedBlockGMRESResult(
            Y_list=[np.zeros((0, L), dtype=np.complex128) for _ in range(N_m)],
            Lambda=np.zeros((L, L), dtype=np.complex128),
            G=None if proj_ops is None else np.zeros((n_obs, L, 0), dtype=np.complex128),
            iters=np.zeros(N_m, dtype=np.int64),
            errs=np.zeros(N_m, dtype=np.float64),
            err_history=np.zeros((1, N_m), dtype=np.float64),
            X_list=[np.zeros_like(B) for _ in range(N_m)] if return_X else None,
            timings={'mvp': 0.0, 'gs': 0.0, 'lsq': 0.0, 'proj': 0.0})

    # Initial residual (X0 = 0 ⇒ R0 = B) and its thin QR: R0 = V1 · Λ
    V1, Lambda = np.linalg.qr(B, mode='reduced')   # V1: (N,L),  Lambda: (L,L)
    Vblocks = [V1]

    # Block Hessenberg matrix of A_offdiag, grown on demand (never sized by itermax).
    cap = min(itermax, 16)                          # capacity in block columns
    Hbar = np.zeros(((cap + 1) * L, cap * L), dtype=np.complex128)

    G_blocks = []                                   # each (n_obs, L, L)
    Y_list = [None] * N_m
    iters = np.zeros(N_m, dtype=np.int64)
    errs = np.full(N_m, np.inf, dtype=np.float64)
    err_history = []
    active = list(range(N_m))
    timings = {'mvp': 0.0, 'gs': 0.0, 'lsq': 0.0, 'proj': 0.0}

    for k in range(1, itermax + 1):
        Vk = Vblocks[k - 1]
        t0 = time.perf_counter()
        if proj_ops is not None:
            # G block of V_k: (n_obs, L, L), element [o, l, :] = F_o[:, l]^H V_k
            G_blocks.append(np.stack([proj_ops[o].conj().T @ Vk for o in range(n_obs)]))
        t1 = time.perf_counter()
        timings['proj'] += t1 - t0

        if k > cap:
            cap = min(itermax, 2 * cap)
            Hnew = np.zeros(((cap + 1) * L, cap * L), dtype=np.complex128)
            Hnew[:k * L, :(k - 1) * L] = Hbar[:k * L, :(k - 1) * L]
            Hbar = Hnew

        t1 = time.perf_counter()
        W = _block_mvp_offdiag(n, f, Au_til, full_idx, Vk)
        t2 = time.perf_counter()
        timings['mvp'] += t2 - t1

        # Block Gram–Schmidt, same order as bl_gmres_mvp_fft.
        col = slice((k - 1) * L, k * L)
        for i in range(1, k + 1):
            Hik = Vblocks[i - 1].conj().T @ W
            Hbar[(i - 1) * L:i * L, col] = Hik
            W = W - Vblocks[i - 1] @ Hik

        # Thin QR of the remaining block → V_{k+1}, H_{k+1,k}
        Vk1, Hk1k = np.linalg.qr(W, mode='reduced')
        Hbar[k * L:(k + 1) * L, col] = Hk1k
        Vblocks.append(Vk1)
        t3 = time.perf_counter()
        timings['gs'] += t3 - t2

        # Augmented matrix [Hbar_k | E_1 Λ], shape ((k+1)L, (k+1)L). Its QR gives
        # [[R, c_top], [0, c_bot]] with ‖c_bot‖_F the least-squares residual.
        kL = k * L
        aug = np.zeros((kL + L, kL + L), dtype=np.complex128)
        aug[:, :kL] = Hbar[:kL + L, :kL]
        aug[:L, kL:] = Lambda
        diag = np.arange(kL)

        err_row = np.full(N_m, np.nan, dtype=np.float64)
        for s in list(active):
            M = aug.copy()
            M[diag, diag] += sigmas[s]
            R_aug = np.linalg.qr(M, mode='r')
            err = np.linalg.norm(R_aug[kL:, kL:]) / B_norm
            err_row[s] = err
            if err < tol or k == itermax:
                Y_list[s] = scipy.linalg.solve_triangular(
                    R_aug[:kL, :kL], R_aug[:kL, kL:], lower=False)
                iters[s] = k
                errs[s] = err
                active.remove(s)
        err_history.append(err_row)
        timings['lsq'] += time.perf_counter() - t3

        if verbose:
            print("iter= {:}, active shifts= {:}, err range= [{:.4e}, {:.4e}]".format(
                k - 1, len(active), np.nanmin(err_row), np.nanmax(err_row)))

        if not active:
            break

    k_fin = len(err_history)
    G = None
    if proj_ops is not None:
        G = np.concatenate(G_blocks[:k_fin], axis=2)        # shape (n_obs, L, k_fin*L)

    X_list = None
    if return_X:
        X_list = []
        for s in range(N_m):
            X = np.zeros((N, L), dtype=np.complex128)
            for j in range(1, iters[s] + 1):
                X += Vblocks[j - 1] @ Y_list[s][(j - 1) * L:j * L, :]
            X_list.append(X)

    return ShiftedBlockGMRESResult(
        Y_list=Y_list, Lambda=Lambda, G=G, iters=iters, errs=errs,
        err_history=np.asarray(err_history, dtype=np.float64), X_list=X_list,
        timings=timings)
