import numpy as np
from mvp_fft.mvp_fft import MBT_fft_init
from bl_krylov.bl_krylov import bl_bicgstab_jacobi_mvp_fft
from bl_krylov.bl_krylov_ray import bl_bicgstab_jacobi_mvp_fft_ray
import psutil
from scipy.spatial.transform import Rotation as R
import time


class Target:
       
    def __init__(self, shape_name, lattice_n, lattice_lf, lattice_grid_points, lattice_grid_points_is_in_target, m_p_xyz):
        self.shape_name : str = shape_name # shape name
        self.lattice_n : np.ndarray[int] = lattice_n  # lattice size  n[0]:=n_x, n[1]:=n_y, n[2]:=n_z, 1D np.array (dtype=int) with length 3
        self.lattice_lf : np.float64 = lattice_lf #length_scale factor := physical length of lattice spacing, real scalar [um]
        self.lattice_grid_points= lattice_grid_points # lattice_grid_points: 2d numpy array of cartesian components of the lattice grid points, shape= (N,3), where N is the total number of lattice grid points= lattice_n[0]*lattice_n[1]*lattice_n[2], row index (0 to N-1) is equivalent to the lattice_address := ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
        self.lattice_grid_points_is_in_target= lattice_grid_points_is_in_target #logical flag of each lattice address that is True if inside the target volume, 1d numpy array with size= N= n[0]*n[1]*n[2], dtype=bool.
        self.m_p_xyz= m_p_xyz # m_p_xyz is the diagonal components of complex refractive index tensor np.array([m_p_x, m_p_y, m_p_z], dtype=np.complex128)

        self.lattice_address_in_target= np.where(lattice_grid_points_is_in_target)[0] # 1d numpy array (size=N, dtype= int) of lattice address:= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz of elements in target volume, where N:= number of elements in target volume
        self.lattice_pos_in_target= lattice_grid_points[self.lattice_address_in_target, :] # 2d numpy array (shape= (N,3), dtype=np.float64) of position (x,y,z) of lattice grid points in target volume, [um]
        self.num_element_occupy= self.lattice_address_in_target.size # total number of lattice grid points in target volume
        self.element_vol : np.float64 = (self.lattice_lf)**3 # physical volume of each cubical element [um3]

        # set the same m_p_xyz to all the element in target (Homogeneous, but can be anisotropic)
        self.m_p = np.ones_like(self.lattice_pos_in_target)*m_p_xyz # 2d numpy array (shape= (N,3), dtype=np.complex128) of complex refractive index (m_p_x, m_p_y, m_p_z) of lattice grid points in target volume
        self.eper_p = self.m_p**2 # permittivity, assuming nonmagnetic material

        self.total_vol= self.element_vol*self.num_element_occupy # total volume of the target [um3]
        self.ve_radius= np.cbrt(3*self.total_vol/(4*np.pi)) # volume equivalent radius of the target [um]

        

class IncidentField:

    def __init__(self, wl_0, m_m, euler_angles):
        ''' 
        ##### inputs #####
        wl_0 # vacuum wavelength [um]
        m_m # medium refractive index at this wl_0
        euler_angles: np.array of shape (L, 3), L-list of euler angle (alpha, beta, gamma) [radian] that define a 3D rotation of the particle coodinate system from the laboratory coordinate system

        '''
        
        # rotation angles (alpha, beta, gamma) [radian] are defined as follows:
        # alpha: rotation of the laboratory coordinate system (fixed in the instrument) about the z-axis through an angle (0, 2*pi) reorienting the y-axis in such a way that it coincides with the line of nodes. The original y-axis has transformed into y'-axis.
        # beta: rotation about the y'-axis through an angle (0, pi). The original z-axis has transformed into z'-axis
        # gamma: rotation about the z'-axis through an angle (0, 2*pi). 

        self.wl_0 : np.float64 = wl_0 # vacuum wavelength [um]
        self.m_m : np.float64 = m_m # medium refractive index at this wl_0
        #self.incident_polar_angles_P : np.array[np.float64,2] = incident_polar_angles_P # 2D ndarray of size Lx2, the L-list of incident polar angle (theta, phi) in particle coordinate system. The coordinate systems are defined as Mischenko's book. 
        self.eper_m : np.float64 = m_m**2 # electric permittivity of medium (medium refractive index **2)
        self.k : np.float64 = 2*np.pi*m_m/wl_0
        self.euler_angles : np.ndarray[np.float64, 2] = euler_angles  # list of euler_angle (alpha,beta,gamma) of each orientation, 2d numpy array of shape (L, 3), 
        self.L :int = euler_angles.shape[0]  # number of incident directions, this can be increased by the methods add_fixed_orientation or add_random_orientations

        #### incident field direction (inc) and observed scattering direction (sca) in the laboratory coodinate system (L) ####
        # incident field specifications
        self.u_inc_vec_L= np.array([0,0,1], dtype=np.float64) # propagating to +z direction, alighed to z_vec(L)
        self.theta_inc_vec_L= np.array([1,0,0], dtype=np.float64) # theta_inc_vec_L is aligned to x_vec(L)
        self.phi_inc_vec_L= np.array([0,1,0], dtype=np.float64) # phi_inc_vec_L is aligned to y_vec(L)
        self.e0_inc_pol_theta : np.complex128 = 1/np.sqrt(2) # theta component of e0 polarization
        self.e0_inc_pol_phi : np.complex128 = 1j/np.sqrt(2) # phi component of e0 polarization

        # scatterd field specifications (forward direction)
        self.u_sca_fw_vec_L= np.array([0,0,1], dtype=np.float64) # forward scattering propagating to +z direction
        self.theta_sca_fw_vec_L= np.array([1,0,0], dtype=np.float64) # theta_sca_fw_vec_L is aligned to x_vec
        self.phi_sca_fw_vec_L= np.array([0,1,0], dtype=np.float64) # phi_sca_fw_vec_L is aligned to y_vec

        # scatterd field specifications (backward direction)
        self.u_sca_bk_vec_L= np.array([0,0,-1], dtype=np.float64) # backward scattering propagating to -z direction
        self.theta_sca_bk_vec_L= np.array([-1,0,0], dtype=np.float64) # theta_bksca_vec_L is aligned to -x_vec (assuming theta_sca= pi, phi_sca= 0)
        self.phi_sca_bk_vec_L= np.array([0,1,0], dtype=np.float64) # phi_bksca_vec_L is aligned to y_vec (assuming theta_sca= pi, phi_sca= 0)
        #######################################################################################################################


        self.u_inc_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit vector of the direction of incident wavepropagation in particle coordinate system
        self.theta_inc_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit theta vector of incident field in particle coordinate system
        self.phi_inc_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit phi vector of incident field in particle coordinate system
        self.e0_inc_vec_P= np.zeros((self.L, 3), dtype= np.complex128) # L-list of the (x,y,z) components of the unit vector of the direction of incident wave electric polarization in particle coordinate system

        self.u_sca_fw_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit vector of direction of forward scattering in particle coordinate system
        self.theta_sca_fw_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit theta-vector of forward-scattered field in particle coordinate system
        self.phi_sca_fw_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit phi-vector of forward-scattered field in particle coordinate system

        self.u_sca_bk_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit vector of direction of backward scattering in particle coordinate system
        self.theta_sca_bk_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit theta-vector of backward-scattered field in particle coordinate system
        self.phi_sca_bk_vec_P= np.zeros((self.L, 3), dtype= np.float64) # L-list of the (x,y,z) components of the unit phi-vector of backward-scattered field in particle coordinate system

        for l in range(self.L):
            alpha= euler_angles[l,0]
            beta= euler_angles[l,1]
            gamma= euler_angles[l,2]
            r_inc = R.from_euler('ZYZ', [-gamma, -beta, -alpha])
            rotmat_inc= r_inc.as_matrix()

            ## perform rotation of incident and scattering (observation) directions
            ## add to the L-list of incident and scattering directions
            self.u_inc_vec_P[l,:] = np.dot(rotmat_inc, self.u_inc_vec_L)
            self.theta_inc_vec_P[l,:]= np.dot(rotmat_inc, self.theta_inc_vec_L)
            self.phi_inc_vec_P[l,:]= np.dot(rotmat_inc, self.phi_inc_vec_L)
            self.e0_inc_vec_P[l,:]= self.e0_inc_pol_theta*self.theta_inc_vec_P[l,:] + self.e0_inc_pol_phi*self.phi_inc_vec_P[l,:]
            

            self.u_sca_fw_vec_P[l,:]= np.dot(rotmat_inc, self.u_sca_fw_vec_L)
            self.theta_sca_fw_vec_P[l,:]= np.dot(rotmat_inc, self.theta_sca_fw_vec_L)
            self.phi_sca_fw_vec_P[l,:]= np.dot(rotmat_inc, self.phi_sca_fw_vec_L)

            self.u_sca_bk_vec_P[l,:]= np.dot(rotmat_inc, self.u_sca_bk_vec_L)
            self.theta_sca_bk_vec_P[l,:]= np.dot(rotmat_inc, self.theta_sca_bk_vec_L)
            self.phi_sca_bk_vec_P[l,:]= np.dot(rotmat_inc, self.phi_sca_bk_vec_L)



class DiscreteDipoles(Target, IncidentField):
    
    ## inherit the properties of the instances of Target and IncidentField super classes
    def __init__(self, target, incidentfield):
        Target.__init__(self, target.shape_name, target.lattice_n, target.lattice_lf, target.lattice_grid_points, target.lattice_grid_points_is_in_target, target.m_p_xyz)
        IncidentField.__init__(self,incidentfield.wl_0, incidentfield.m_m, incidentfield.euler_angles)

        ## DDA accuracy criteria
        self.dpl : np.float64 = (self.wl_0/np.abs(np.max(self.m_p_xyz)))/self.lattice_lf  # number of dipoles in particle volume

        ## define the boundery conditions (spatial distribution of relative permittivity, polarizability, internal incident field)
        self.f : np.int32 = 3 # only electric 
        self.eper_r : np.ndarray[np.complex128,2]= np.zeros((target.num_element_occupy,3), dtype=np.complex128) # relative permittivity along (x,y,z) axes in particle coordinate
        self.alpha_E : np.ndarray[np.complex128,2]= np.zeros((target.num_element_occupy,3), dtype=np.complex128) # polarizability along (x,y,z) axes in particle coordinate
        self.E_inc : np.ndarray[np.complex128,3]= np.zeros((self.L, target.num_element_occupy, 3), dtype=np.complex128) # (x,y,z) components of incident field in target in particle coordinate
        self.diag_A : np.ndarray[np.complex128]= np.zeros(3*target.num_element_occupy, dtype=np.complex128) # diagonal elements of the 3Nx3N interaction matrix A, (N:= number of filled elements)
        self.B : np.ndarray[np.complex128,2]= np.zeros((3*target.num_element_occupy, self.L), dtype=np.complex128) # 3N-length RHS vector of the matrix equation AP=B , (N:= number of filled elements)
        self.e_inc_phase : np.ndarray[np.complex128,2]= np.zeros((self.L, target.num_element_occupy),dtype=np.complex128) # phase term of incident wave in target volume

        ## define the polarization and internal field, which is the solution of Lippmann-Schwinger equtation
        self.X : np.ndarray[np.complex128,2]= np.zeros((3*target.num_element_occupy, self.L), dtype=np.complex128) # (x,y,z) components of electric polarization in particle coordinate
        self.P : np.ndarray[np.complex128,3]= np.zeros((self.L, target.num_element_occupy, 3), dtype=np.complex128) # (x,y,z) components of electric polarization in particle coordinate
        self.E : np.ndarray[np.complex128,3]= np.zeros((self.L, target.num_element_occupy, 3), dtype=np.complex128) # (x,y,z) components of internal field (total electric field) in particle coordinate

        ## DDA calculations
        self.min_num_RHS_for_parallel= 100
        self.itermax= 25
        self.tol= 1e-2
        self.converge= False

        ## DDA results
        self.C_abs= None
        self.C_ext= None
        self.S_fw_PCAS_theta= None
        self.S_fw_PCAS_phi= None
        self.S_bk_OCBS_theta= None
        self.S_bk_OCBS_phi= None
        self.S_bk_OCBS= None
        self.S_PCAS_sp_avg= None
        self.S_PCAS_sp_avg_SNR= None
        self.S_PCAS_depol= None
        self.S_PCAS_depol_SNR= None

        print("Number of dipoles per wavelength in the particle volume: dpl= {:}".format(self.dpl))


    def set_interaction_matrix(self):
        
        self.eper_r= self.eper_p/self.eper_m

        ## Clausius-Mosotti static polarizability
        alpha0_E= (3/(4*np.pi))*((self.eper_r-1)/(self.eper_r+2))*self.element_vol

        ## set Clausius-Mosotti with radiation correction (CMRR) 
        #self.alpha_E= alpha0_E/(1-(2/3)*1j*self.k**3*alpha0_E)

        ## set CR2009 polarizability ref. Chaumet and Rahmani 2009 JQSRT, which shows better accuracy than CMRR
        a= (3*self.element_vol/(4*np.pi))**(1/3) # volume equivalent radius of each element
        M_term= (8*np.pi/3)*((1-1j*self.k*a)*np.exp(1j*self.k*a)-1)
        self.alpha_E= alpha0_E/(1-M_term*alpha0_E/self.element_vol)

        for m in range(self.num_element_occupy):
            self.diag_A[3*m:3*(m+1)]= 1/self.alpha_E[m,:] 

        ## MBT projection of the non-diagonal block of DDA matrix  
        self.Au_til= MBT_fft_init(self.lattice_n, self.f, self.lattice_lf, self.k)

        ## set incident field inside the filled elements
        for l in range(self.L):
            self.e_inc_phase[l,:]= np.exp(1j*self.k*np.dot(self.lattice_pos_in_target[:,:],self.u_inc_vec_P[l,:]))
            self.E_inc[l,:,:]= np.outer(self.e_inc_phase[l,:], self.e0_inc_vec_P[l,:])
            self.B[:,l]= self.E_inc[l,:,:].flatten()

    def solve_matrix_equation(self):

        start_time= time.time()

        if(self.L >= self.min_num_RHS_for_parallel):
            # use parallel solver
            print("Concurrent computing of block-Matrix solver using multiple CPU cores")
            num_phys_cores= psutil.cpu_count(logical=False)
            max_num_procs= num_phys_cores//2
            print("num_phys_cores={:}, max_num_procs={}".format(num_phys_cores, max_num_procs))
            for n in range(max_num_procs,1,-1):
                if self.L % n == 0 :
                    num_procs= n
                    break
            print("Starting block-BiCGStab iterative solver (num_procs={:})".format(num_procs))
            self.X, iter_fin, err_fin= bl_bicgstab_jacobi_mvp_fft_ray(self.lattice_n, self.f, self.lattice_address_in_target, self.Au_til, self.diag_A, self.B, self.tol, self.itermax, num_procs)
        else:
            # use serial solver
            print("Serial computing of block-Matrix solver (single CPU core)")
            print("Starting block-BiCGStab iterative solver...")
            self.X, iter_fin, err_fin= bl_bicgstab_jacobi_mvp_fft(self.lattice_n, self.f, self.lattice_address_in_target, self.Au_til, self.diag_A, self.B, self.tol, self.itermax)

        end_time= time.time()
        elapsed_time= end_time-start_time

        if err_fin < self.tol :
            self.converge = True
            print("block-BiCGStab converged! (iter_fin={:}, err_fin={:.4f}, solver time={:.1f}s)".format(iter_fin,err_fin,elapsed_time))
            for l in range(self.L):
                self.P[l,:,:]= self.X[:,l].reshape(self.num_element_occupy,3)
                self.E[l,:,:]= self.P[l,:,:]*(4*np.pi)/((self.eper_r[:,:]-1)*self.element_vol)
        


    def compute_C_abs(self):
        self.C_abs= np.zeros(self.L, dtype=np.float64) # absorption cross sections
        if np.mean((self.eper_r).imag) < 1e-12:
            pass
        else:
            for l in range(self.L):
                self.C_abs[l] = 4*np.pi*self.k*np.sum(np.imag(np.tensordot(self.P[l,:,:],np.conj(self.E[l,:,:]), axes=2)))
        return self.C_abs
    
   
    def compute_C_ext(self):
        self.C_ext= np.zeros(self.L, dtype=np.float64) # extinction cross section for each incident beam
        for l in range(self.L):
            self.C_ext[l]=  4*np.pi*self.k*np.sum(np.imag(np.tensordot(self.P[l,:,:],np.conj(self.E_inc[l,:,:]), axes=2)))
        return self.C_ext
    
  
    def compute_PCAS_observable_S_fw(self):
        self.S_fw_PCAS_theta= np.zeros(self.L, dtype=np.complex128) # observed S(0)p in PCAS (s-pol, vertical e-polarization in L coordinate system)
        self.S_fw_PCAS_phi= np.zeros(self.L, dtype=np.complex128) # observed S(0)s in PCAS (p-pol, horizontal e-polarization in L coordinate system)
        for l in range(self.L):
            e_sca_fw_phase= np.exp(1j*self.k*np.dot(self.lattice_pos_in_target[:,:],self.u_sca_fw_vec_P[l,:]))
            self.S_fw_PCAS_theta[l] = self.k**2*np.dot(np.dot(self.P[l,:,:], self.theta_sca_fw_vec_P[l,:]), np.conj(np.sqrt(2)*e_sca_fw_phase[:]))
            self.S_fw_PCAS_phi[l] = self.k**2*np.dot(np.dot(self.P[l,:,:], self.phi_sca_fw_vec_P[l,:]), np.conj(np.sqrt(2)*1j*e_sca_fw_phase[:]))
        return self.S_fw_PCAS_theta, self.S_fw_PCAS_phi
    

    def compute_OCBS_observable_S_bk(self):
        self.S_bk_OCBS_theta= np.zeros(self.L, dtype=np.complex128)
        self.S_bk_OCBS_phi= np.zeros(self.L, dtype=np.complex128)
        self.S_bk_OCBS= np.zeros(self.L, dtype=np.complex128) # observed S(180) in Optical Coherence Backscattering Sensor (OCBS)
        for l in range(self.L):
            e_sca_bk_phase= np.exp(1j*self.k*np.dot(self.lattice_pos_in_target[:,:],self.u_sca_bk_vec_P[l,:]))
            self.S_bk_OCBS_theta[l] = self.k**2*np.dot(np.dot(self.P[l,:,:], self.theta_sca_bk_vec_P[l,:]), np.conj(np.sqrt(2)*e_sca_bk_phase[:])) # sign of the x-component of Polarization has changed because of coordinate transformation (x -> -x)  
            self.S_bk_OCBS_phi[l] = self.k**2*np.dot(np.dot(self.P[l,:,:], self.phi_sca_bk_vec_P[l,:]), np.conj(np.sqrt(2)*1j*e_sca_bk_phase[:]))
            #self.S_bk_OCBS[l] = self.k**2*np.dot(np.dot(self.P[l,:,:], -1*self.theta_sca_bk_vec_P[l,:]-1j*self.phi_sca_bk_vec_P[l,:]), np.conj(e_sca_bk_phase[:]))
            self.S_bk_OCBS= (-self.S_bk_OCBS_theta+self.S_bk_OCBS_phi)/np.sqrt(2)
        return self.S_bk_OCBS
    




    






        

    

        

        

    