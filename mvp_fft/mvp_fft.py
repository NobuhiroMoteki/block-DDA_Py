import numpy as np
from numba import njit

@njit
def application_function(f, n_ind, lf, k):
    '''
    ******* Input parameters ********
     # n=[n1,n2,...,nM]  number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
     f                 size of final dense f*f matrix (e.g., f=3 for 3*3 matrix)  scalar, dtype=int
     n_ind             2 x M rectangular matrix indicating the position of each element in the MBT matrix,  2D np.array with size (2,M), dtype=int
     level             current BT level (level=1,...,M).
     lf                physical length per lattice spacing (length scale factor)
     k                 wavenumber in medium
    *********************************
    ******** Output *****************
    MBT_elem           f x f general matrix of MBT level=M 
    *********************************
    '''
    #Xkm= np.zeros(3, dtype=np.complex128)
    Xkm : np.ndarray[np.complex128] = (n_ind[0,:]-n_ind[1,:])*lf
    Xkm_abs : np.complex128 = np.linalg.norm(Xkm)
    XkmXkm : np.ndarray[np.complex128,2] = np.outer(Xkm,Xkm)
    MBT_elem= np.zeros((3,3), dtype=np.complex128)
    Imat= np.identity(3, dtype=np.complex128)

    if Xkm_abs == 0 :
        MBT_elem= np.zeros((3,3), dtype=np.complex128)
    else:
        Gkm= k*k*(Imat-XkmXkm/(Xkm_abs*Xkm_abs))-((1-1j*k*Xkm_abs)/(Xkm_abs*Xkm_abs))*(Imat-3*XkmXkm/(Xkm_abs*Xkm_abs))
        Gkm *= np.exp(1j*k*Xkm_abs)/Xkm_abs
        MBT_elem= -Gkm.astype(np.complex128)
    
    return MBT_elem


def BT_fft(n,f,n_ind,level,lf,k):
    '''
    ******* Input parameters ********
     n=[n1,n2,...,nM]  number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
     f                 size of final dense f*f matrix (e.g., f=3 for 3*3 matrix) scalar, dtype=int
     n_ind             2 x M rectangular matrix indicating the position of each element in the MBT matrix 2D np.array with size (2,M), dtype=int
     level             current BT level (level=1,...,M).
     lf                physical length per lattice spacing (length scale factor)
     k                 wavenumber in medium
    *********************************
    ******** Output *****************
    MBT_elem           f x f general matrix of MBT level=M in Eigen::MatrixXcd
    *********************************
    '''
    #print("level= {:}".format(level))
    Au= np.zeros(0,dtype=np.complex128)
    if level == len(n)+1 :
        # terminate recursion
        Au.resize(f*f, refcheck=False)
        Au= (np.flipud(application_function(f,n_ind,lf,k)).flatten()).squeeze()
        #return Au
    else :
        Au.resize(f*f*np.prod(2*n[level-1:]-1),refcheck=False)
        this_n= n[level-1]
        b_edge= f*f*np.prod(2*n[level:]-1)
        #lower triangular and diagonal blocks
        for i in range(this_n,0,-1):
            n_ind[1,level-1]= 1
            n_ind[0,level-1]= i
            Au[b_edge*(this_n-i):b_edge*(this_n-i+1)]= BT_fft(n,f,n_ind,level+1,lf,k)
        #upper triangular blocks
        for i in range(2,this_n+1,1):
            n_ind[1,level-1]= i
            Au[b_edge*(i+this_n-2):b_edge*(i+this_n-1)]= BT_fft(n,f,n_ind,level+1,lf,k)
        #return Au
    return Au


def BT_pad(n,f,x):
    '''
    Generation of xz by inserting zeros into x
    ****** Input parameters **************
    n=[n1,n2,...,nM]     number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
    f                    size of final dense f*f matrix (e.g., f=3 for 3*3 matrix)
    x                    input vector before padding
    ****** Output ************************
    xz                   padded vector
    **************************************
    '''
    xz= np.zeros(0,dtype=np.complex128)
    #lenxz= 0
    if len(n) < 1 :
        lenxz= len(x)+f*f-f
        xz.resize(lenxz, refcheck=False)
        xz[0:len(x)]= x
        xz[len(x):lenxz]= np.zeros(lenxz-len(x),dtype=np.complex128)
    else :
        b_edge= f*np.prod(n[1:])
        this_n= n[0]

        for i in range(1, this_n+1):
            lenxz_old= len(xz)
            blk= BT_pad(n[1:],f,x[(i-1)*b_edge:i*b_edge])
            lenblk= len(blk)
            if len(n) > 1 :
                lenxz= lenxz_old+lenblk+(n[1]-1)*f*f*np.prod(2*n[2:]-1)
                xz.resize(lenxz, refcheck=False)
                xz[lenxz_old:]= np.zeros(len(xz)-lenxz_old, dtype=np.complex128)
                xz[lenxz_old:lenxz_old+lenblk]= blk
            else :
                lenxz= lenxz_old+lenblk
                xz.resize(lenxz, refcheck=False)
                xz[lenxz_old:]= blk
    return xz



def BT_reconstruct(n,f,bz):
    '''
    Reconstruction of b from bz
    *************************** Input parameters *****************************************
     n=[n1,n2,...,nM]     number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
     f                    size of final dense f*f matrix (e.g., f=3 for 3*3 matrix)
     bz                   input vector
     *************************** Output ***************************************************
     b                    output vector (trimmed)
     ***************************************************************************************/
     '''
    b= np.zeros(0,dtype=np.complex128)
    if len(n) < 1 :
        for i in range(f, f*f+1, f) :
            b.resize(len(b)+1, refcheck=False)
            b[-1]= bz[i-1]
    else :
        b_edge= f*f*np.prod(2*n[1:]-1)
        this_n= n[0]
        for i in range(this_n, 2*this_n) :
            lenb_old= len(b)
            blk= BT_reconstruct(n[1:],f,bz[(i-1)*b_edge:i*b_edge])
            lenblk= len(blk)
            lenb= lenb_old+lenblk
            b.resize(lenb, refcheck=False)
            b[lenb_old:lenb_old+lenblk]= blk
    return b
    

def MBT_fft_init(n,f,lf,k):
    '''
    Prepare the fourier-transformed MBT projection of MBT matrix
    *************************** Input parameters *****************************************
     n=[n1,n2,...,nM]     number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
     f                    size of final dense f*f matrix (e.g., f=3 for 3*3 matrix)
     lf                   physical length per lattice spacing (length scale factor)
     k                    wavenumber in medium
     *************************** Output ***************************************************
     Au_til                result vector
     ***************************************************************************************/
    '''
    n_ind_init= np.ones((2,len(n)), dtype=int)
    fft_length= f*f*np.prod(2*n-1)

    level_init= 1
    # MBT projection of MBT matrix A
    Au= BT_fft(n,f,n_ind_init,level_init,lf,k)

    if fft_length != len(Au) :
        raise ValueError("Au length wrong!")
    
    Au= Au[::-1] #reverse the order

    #Au_til= np.empty(fft_length, dtype=np.complex128)

    # fft using numpy
    #Au_til= np.fft.fft(Au)

    return np.fft.fft(Au)


def MBT_fft_mvp(n, f, Au_til, p_hat):
    '''
    fast matrix-vector product using FFT. matrix is MBT
    *************************** Input parameters *****************************************
     n=[n1,n2,...,nM]     number of BT_blocks at each level, where M is the number of BT levels. 1D np.array with size M, dtype=int
     f                    size of final dense f*f matrix (e.g., f=3 for 3*3 matrix)
     Au_til               fourier-transformed MBT projection of MBT matrix A
     p_hat                input vector
     *************************** Output ***************************************************
     q_hat                result vector
     **************************************************************************************
    '''
    fft_length= len(Au_til)

    # MBT projection of data vector
    p_hatz = BT_pad(n,f,p_hat)

    # zero-padding to adjust vector length to fft_length
    oldlen_p_hatz= len(p_hatz)
    p_hatz.resize(fft_length, refcheck=False)
    p_hatz[oldlen_p_hatz:]= 0

    p_hatz_til= np.fft.fft(p_hatz)
    q_hatz_til= Au_til*p_hatz_til
    q_hatz= np.fft.ifft(q_hatz_til)

    #q_hat= BT_reconstruct(n,f,q_hatz)

    return BT_reconstruct(n,f,q_hatz)
