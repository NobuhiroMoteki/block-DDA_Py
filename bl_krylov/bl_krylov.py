import numpy as np
from mvp_fft.mvp_fft import MBT_fft_mvp

def bl_cocg_rq_jacobi_mvp_fft(n,f,address,Au_til,DIAG_A,B,tol,itermax):
    ''' 
    solving a block matrix equation AX=B for complex-symmetric BT matrix A,
    Gu et al 2016, arXiv, Block variants of COCG and COCR methods
    for solving complex symmetric linear systems with L right-hand sides

    *************************** Input parameters *****************************************
     n=[n1,n2,...,nM]     number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
     f                    size of final dense f*f matrix (e.g., f=3 for 3*3 matrix)
     address              array of the cuboid-coordinate index of target volume elements.  
     Au_til               fourier-transformed MBT projection of MBT matrix, non-diagonal part of the interaction matrix A
     DIAG_A               diagonal part of the interaction matrix A with size= f*len(address)
     B                    RHS block matrix with size=(f*len(address),L), column vectors of B must be linearly independent
     tol                  error tolerance of convergence (typically 1e-3 or 1e-4)
     itermax              maximum number of iteration
     fft_object           instance of pyfftw.FFTW object (direction = 'FORWARD') created in the MBT_fft_init
     ifft_object          instance of pyfftw.FFTW object (direction = 'BACKWARD') created in the MBT_fft_init
     *************************** Output ***************************************************
     X                    solution block matrix with size=(f*len(address),L) 
     iter_fin             number of iterations   
     err_fin              result vector
     **************************************************************************************
    '''
    num_element_occupy= len(address)
    num_element_cuboid= np.prod(n)
    L= B.shape[1] # number of columns of RHS vector

    jpre= 1/DIAG_A

    B_jpre= np.zeros((B.shape[0],B.shape[1]), dtype=np.complex128)

    for l in range(L):
        B_jpre[:,l]= jpre*B[:,l]

    B_jpre_norm= np.linalg.norm(B_jpre)
    X= np.zeros((B.shape[0],L), dtype=np.complex128) # Intial guess of the solution matrix X (zero matrix)

    Q, xi= np.linalg.qr(B_jpre, mode='reduced') # reduced QR decomposition of B_jpre
    S= Q

    Q1= np.zeros((B.shape[0],L), dtype=np.complex128)
    P_hat= np.zeros((f*num_element_cuboid,L), dtype=np.complex128)
    AS= np.zeros((B.shape[0],L), dtype=np.complex128)

    for k in range(itermax):
        
        P_hat.fill(0)
        ##--------- FFT accerelation of AS=A*S -----------
        for l in range(L):
            Q1[:,l]= DIAG_A*S[:,l]  # diagonal contribution
            for m in range(num_element_occupy):
                mm= address[m]
                P_hat[f*mm:f*(mm+1),l]= S[f*m:f*(m+1),l]
            P_hat[:,l]= MBT_fft_mvp(n, f, Au_til, P_hat[:,l])
            for m in range(num_element_occupy):
                mm= address[m]
                Q1[f*m:f*(m+1),l]+= P_hat[f*mm:f*(mm+1),l] # non-diagonal contribution
            AS[:,l]= Q1[:,l]*jpre
        ##------------------------------------------------

        alpha= np.linalg.solve(np.dot(np.transpose(S),AS), np.dot(np.transpose(Q),Q))
        X= X+np.dot(S,np.dot(alpha,xi))

        Qnew, tau= np.linalg.qr(Q-np.dot(AS,alpha), mode='reduced')
        xi= np.dot(tau,xi)
        err= np.linalg.norm(xi)/B_jpre_norm
        print("iter= {:}, err= {:.4f}".format(k, err))
        iter_fin= k
        err_fin= err
        if err < tol :
            break
        beta= np.linalg.solve(np.dot(np.transpose(Q),Q), np.dot(np.transpose(tau),np.dot(np.transpose(Qnew),Qnew)))
        Q= Qnew
        S= Q+np.dot(S,beta)

    return X, iter_fin, err_fin


def bl_bicgstab_jacobi_mvp_fft(n,f,address,Au_til,DIAG_A,B,tol,itermax):
    ''' 
    solving a block matrix equation AX=B for general complex BT matrix A,
    Block BiCGSTAB [Tadano etal 2009 JSIAM letters]

    *************************** Input parameters *****************************************
     n=[n1,n2,...,nM]     number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
     f                    size of final dense f*f matrix (e.g., f=3 for 3*3 matrix)
     address              array of the cuboid-coordinate index of target volume elements.  
     Au_til               fourier-transformed MBT projection of MBT matrix, non-diagonal part of the interaction matrix A
     DIAG_A               diagonal part of the interaction matrix A with size= f*len(address)
     B                    RHS block matrix with size=(f*len(address),L), column vectors of B must be linearly independent
     tol                  error tolerance of convergence (typically 1e-3 or 1e-4)
     itermax              maximum number of iteration
     fft_object           instance of pyfftw.FFTW object (direction = 'FORWARD') created in the MBT_fft_init
     ifft_object          instance of pyfftw.FFTW object (direction = 'BACKWARD') created in the MBT_fft_init
     *************************** Output ***************************************************
     X                    solution block matrix with size=(f*len(address),L) 
     iter_fin             number of iterations   
     err_fin              result vector
     **************************************************************************************
    '''
    num_element_occupy= len(address)
    num_element_cuboid= np.prod(n)
    L= B.shape[1] # number of columns of RHS vector

    jpre= 1/DIAG_A

    B_jpre= np.zeros((B.shape[0],B.shape[1]), dtype=np.complex128)

    for l in range(L):
        B_jpre[:,l]= jpre*B[:,l]

    B_jpre_norm= np.linalg.norm(B_jpre)
    X= np.zeros((B.shape[0],L), dtype=np.complex128) # Intial guess of the solution matrix X (zero matrix)

    R= B_jpre
    P= R
    R0til= R
    R0til_H= R0til.conj().T

    Q1= np.zeros((B.shape[0],L), dtype=np.complex128)
    P_hat= np.zeros((f*num_element_cuboid,L), dtype=np.complex128)
    V= np.zeros((B.shape[0],L), dtype=np.complex128)
    Z= np.zeros((B.shape[0],L), dtype=np.complex128)

    for k in range(itermax):
                
        ##--------- FFT accerelation of V=A*P -------------
        P_hat.fill(0)
        for l in range(L):
            Q1[:,l]= DIAG_A*P[:,l]  # diagonal contribution
            for m in range(num_element_occupy):
                mm= address[m]
                P_hat[f*mm:f*(mm+1),l]= P[f*m:f*(m+1),l]
            P_hat[:,l]= MBT_fft_mvp(n, f, Au_til, P_hat[:,l])
            for m in range(num_element_occupy):
                mm= address[m]
                Q1[f*m:f*(m+1),l]+= P_hat[f*mm:f*(mm+1),l] # non-diagonal contribution
            V[:,l]= Q1[:,l]*jpre
        ##-------------------------------------------------

        RV= np.dot(R0til_H,V)
        alpha= np.linalg.solve(RV,np.dot(R0til_H,R))
        T= R-np.dot(V,alpha)

        ##--------- FFT accerelation of Z=A*T -------------
        P_hat.fill(0)
        for l in range(L):
            Q1[:,l]= DIAG_A*T[:,l]  # diagonal contribution
            for m in range(num_element_occupy):
                mm= address[m]
                P_hat[f*mm:f*(mm+1),l]= T[f*m:f*(m+1),l]
            P_hat[:,l]= MBT_fft_mvp(n, f, Au_til, P_hat[:,l])
            for m in range(num_element_occupy):
                mm= address[m]
                Q1[f*m:f*(m+1),l]+= P_hat[f*mm:f*(mm+1),l] # non-diagonal contribution
            Z[:,l]= Q1[:,l]*jpre
        ##-------------------------------------------------

        qsi= np.dot(Z.conj().T,T).trace()/np.dot(Z.conj().T,Z).trace()
        X= X+np.dot(P,alpha)+qsi*T
        R= T-qsi*Z

        err= np.linalg.norm(R)/B_jpre_norm
        print("iter= {:}, err= {:.4f}".format(k, err))
        iter_fin= k
        err_fin= err
        if err < tol :
            break

        beta= np.linalg.solve(RV,np.dot(-R0til_H,Z))
        P= R+ np.dot((P-qsi*V),beta)

    return X, iter_fin, err_fin