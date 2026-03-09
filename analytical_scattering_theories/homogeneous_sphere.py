import numpy as np


def mie_compute_q_and_s(wl_0, m_m, r_p, m_p, nang=3):
    """
    Mie scattering efficiencies and complex scattering amplitudes for a
    homogeneous sphere.

    Based on Bohren & Huffman 1983 (BH83) and Fu & Sun 2001 (FS01).

    Assumptions
    -----------
    - Surrounding medium: non-absorbing, non-magnetic.
    - Particle: non-magnetic.
    - Gaussian units.

    Parameters
    ----------
    wl_0 : float   vacuum wavelength [length]
    m_m  : float   medium refractive index (real)
    r_p  : float   particle radius [length]
    m_p  : complex particle refractive index
    nang : int     number of scattering angles from 0 to 180 deg

    Returns
    -------
    Qsca : float    scattering efficiency  Csca / (π r_p²)
    Qabs : float    absorption efficiency  Cabs / (π r_p²)
    Qext : float    extinction efficiency  Cext / (π r_p²)
    S_fw : complex64  PCAS forward-scattering observable  (S11[0]+S22[0])/2
    S_bk : complex64  OCBS backscattering observable      (-S11[-1]+S22[-1])/√2
    """
    nang = int(nang)

    k0  = 2.0 * np.pi / wl_0
    k   = m_m * k0
    x   = k * r_p
    m_r = m_p / m_m

    # Number of partial-wave terms (BH83)
    nstop = int(np.floor(abs(x + 4.0 * x**0.3333 + 2.0)))
    y     = m_r * x
    nmx   = int(np.floor(max(nstop, abs(y)) + 15))

    # --- Logarithmic derivative D_n(mx) via downward recurrence (BH83 Eq.4.89) ---
    DD = np.zeros(nmx + 1, dtype=np.complex128)
    for n in range(nmx, 0, -1):
        DD[n-1] = n / y - 1.0 / (DD[n] + n / y)
    DD = DD[:nstop + 1]

    # --- Ratio R_n = psi_n / psi_{n-1} via downward recurrence ---
    R = np.zeros(nmx + 1)
    R[nmx] = x / (2 * nmx + 1)
    for n in range(nmx - 1, -1, -1):
        R[n] = 1.0 / ((2 * n + 1) / x - R[n + 1])

    # --- Riccati-Bessel psi via cumulative product of R ---
    # psi[n] = R[n]*psi[n-1]  =>  psi[1:] = psi[0] * cumprod(R[1:nstop+1])
    psi = np.empty(nstop + 1)
    psi[0]  = R[0] * np.cos(x)
    psi[1:] = psi[0] * np.cumprod(R[1:nstop + 1])

    # --- Riccati-Bessel chi via upward recurrence ---
    chi = np.zeros(nstop + 1)
    chi[0] = -np.cos(x)
    chi[1] = chi[0] / x - np.sin(x)
    for n in range(2, nstop + 1):
        chi[n] = ((2 * n - 1) / x) * chi[n - 1] - chi[n - 2]

    xi = psi + 1j * chi                        # Riccati-Bessel xi = psi + i*chi

    # --- Partial-wave coefficients a_n, b_n (BH83 Eq.4.88) - vectorised over n ---
    n_arr   = np.arange(1, nstop + 1, dtype=float)   # shape (nstop,)
    D_n     = DD[1:]                                   # D_n(mx),  n = 1..nstop
    psi_n   = psi[1:];    psi_nm1 = psi[:-1]          # psi(n),   psi(n-1)
    xi_n    = xi[1:];     xi_nm1  = xi[:-1]
    n_on_x  = n_arr / x

    a_n = ((D_n / m_r + n_on_x) * psi_n - psi_nm1) / \
          ((D_n / m_r + n_on_x) * xi_n  - xi_nm1)    # BH83 Eq.4.88
    b_n = ((m_r * D_n + n_on_x) * psi_n - psi_nm1) / \
          ((m_r * D_n + n_on_x) * xi_n  - xi_nm1)

    # --- Mie efficiencies ---
    fn1  = 2.0 * n_arr + 1.0
    Qsca = float(np.real((2.0 / x**2) * np.sum(fn1 * (np.abs(a_n)**2 + np.abs(b_n)**2))))  # BH83 Eq.4.61
    Qext = float(np.real((2.0 / x**2) * np.sum(fn1 * (a_n + b_n))))                         # BH83 Eq.4.62
    Qabs = Qext - Qsca

    # --- Angular functions pi_n(cos θ), tau_n(cos θ) via upward recurrence over n ---
    theta = np.linspace(0, np.pi, nang)
    mu    = np.cos(theta)                     # shape (nang,)

    pie = np.zeros((nstop + 1, nang))
    tau = np.zeros((nstop + 1, nang))
    pie[1] = 1.0
    pie[2] = 3.0 * mu
    tau[1] = mu
    tau[2] = 6.0 * mu**2 - 3.0               # = 2*mu*pie[2] - 3*pie[1]
    for n in range(3, nstop + 1):
        pie[n] = ((2*n - 1) / (n - 1)) * mu * pie[n-1] - (n / (n - 1)) * pie[n-2]
        tau[n] = n * mu * pie[n] - (n + 1) * pie[n-1]

    # --- Complex scattering amplitudes S1, S2 (BH83 Eq.4.74) - vectorised over angle ---
    fn2 = (2.0 * n_arr + 1.0) / (n_arr * (n_arr + 1.0))   # shape (nstop,)
    # (fn2*a_n) @ pie[1:] sums over n=1..nstop for every angle simultaneously
    S1 = (fn2 * a_n) @ pie[1:] + (fn2 * b_n) @ tau[1:]    # shape (nang,)
    S2 = (fn2 * a_n) @ tau[1:] + (fn2 * b_n) @ pie[1:]

    # Mishchenko convention: S11 = S2/(-ik), S22 = S1/(-ik)
    S11 = S2 / (-1j * k)
    S22 = S1 / (-1j * k)

    S_fw = np.complex64((S11[0]  + S22[0])  / 2.0)          # PCAS forward
    S_bk = np.complex64((-S11[-1] + S22[-1]) / np.sqrt(2))  # OCBS backward

    return Qsca, Qabs, Qext, S_fw, S_bk
