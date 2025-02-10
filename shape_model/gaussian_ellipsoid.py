import numpy as np
import scipy
from sklearn.neighbors import KDTree
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)

def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)


class gaussian_ellipsoid_shape_model:
    ''' 
    generate a lattice grid points of the volume of a Gaussian Random Ellipsoid (GRE) that is used for DDA calculations.
    Key Features:

    1. Theoretical formulae of gaussian random ellipsoid was defined in Muinonen and Pieniluoma 2011 JQSRT.
    2. lattice spacing lf is automatically determied according to an empirical formulae that considers the compromize between shape accuracy and computation cost.
    3. the allowed range of each of the input parameters were determined from the compromize between the generality and accuracy of the shape representation

    ''' 

    def __init__(self, r_v_base, bc_ratio, ab_ratio, beta):

        '''
        #### input  ####
        c: semiradius of base ellipsoid along z axis (z-axis coincide with beam propagation direction in laboratory coordinate in default particle orientation)
        bc_ratio: b to c ratio, where b is semiradius of base ellipsoid along y axis
        ab_ratio: a to b ratio, where a is semiradius of base ellipsoid along x axis
        beta: standard deviation of the Gaussian ellipsoid

        '''

        # Define the base ellipsoid a >= b >= c
        #### length_unit= um ###
        self.r_v_base= r_v_base  # allowed range = [0.1, 0.5], volume equivalent radius of base ellipsoid
        self.bc_ratio= bc_ratio # allowed range = [1.0, 5.0], b/c ratio of base ellipsoid
        self.ab_ratio= ab_ratio # allowed range = [1.0, 2.0], a/b ratio of base ellipsoid
        self.beta= beta # standard deviation of GE surface deformation, allowed range= [0, 0.3]
        self.name= None

        c= np.cbrt(self.r_v_base**3/(self.ab_ratio*self.bc_ratio**2)) # semiradius of base ellipsoid along z axis

        #self.lattice_lf= (1/100)*np.sqrt(c/0.05)*(self.ab_ratio*self.bc_ratio)**(1/5) # [um] grid point interval of the cuboid lattice
        self.lattice_lf= (1/75)*np.sqrt(c/0.05)*(self.ab_ratio*self.bc_ratio)**(1/5) # [um] grid point interval of the cuboid lattice 
        self.distance_factor= 3**0.5/2*(bc_ratio*ab_ratio*(c/0.1))**(1/6)


        
    def compute_r_points_on_GRE(self, rng):
        '''
        #### input  ####
        random_seed: seed for random number generator created using the np.random.default_rng()

        ### output ###
        r_points_on_GE_surf: r samples on the computed GE surface, 2d numpy array with shape (N,3), each row is the 3 cartesian components (x,y,z) of each sample, N= N_theta*N_phi*surface_interpolation_factor**2
        xyz_meshes_GRE_surf: 3-tuple of meshgrid data of each cartesian coordinate as a function of (theta,phi), 2d numpy array with shape=(N_phi*surface_interpolation_factor, N_theta*surface_interpolation_factor)
        '''
       
        self.name= "GRE_shape__r_v_base={:.2f}um__ab_ratio={:.1f}__bc_ratio={:.1f}__beta={:.2f}".format(self.r_v_base,self.ab_ratio,self.bc_ratio,self.beta)
        
        c= np.cbrt(self.r_v_base**3/(self.ab_ratio*self.bc_ratio**2)) # semiradius of base ellipsoid along z axis
        b= c*self.bc_ratio
        a= b*self.ab_ratio

        h0= c**2/a # minium radius of curvature

        lc= c*0.3 # correlation length (fixed to c*0.3)

        # Generate grid points of polar coordinates (theta, phi)
        N_theta= 25
        N_phi= 100
        theta = np.linspace(0, np.pi, N_theta)
        phi = np.linspace(0, 2*np.pi, N_phi)
        theta_mesh, phi_mesh = np.meshgrid(theta, phi, indexing='ij')

        # Cartesian coorinates x0(theta,phi), y0(theta,phi), z0(theta,phi) on the surface of the base ellipsold (x/a)**2 + (y/b)**2 + (z/c)**2 =1
        x0 = a*np.sin(theta_mesh)*np.cos(phi_mesh)
        y0 = b*np.sin(theta_mesh)*np.sin(phi_mesh)
        z0 = c*np.cos(theta_mesh)
        # index rule: x0[i_theta, i_phi], y0[i_theta, i_phi], z0[i_theta, i_phi]


        @njit
        def compute_s_vec_cov_matrix(x0,y0,z0,lc,beta,N_theta,N_phi):
            N= N_theta*N_phi # total number of surface grid points
            s_vec_cov_matrix= np.zeros((N,N))
            #index = i_theta*N_phi + i_phi
            for i_dim0 in range(N):
                #if i_dim0 % 100 == 0:
                    #print("i_dim0/N= {:.1f} % processed".format(i_dim0/N*100))
                i_theta_dim0= int(i_dim0 // N_phi)
                i_phi_dim0= int(i_dim0 -i_theta_dim0*N_phi)
                for i_dim1 in range(N):
                    i_theta_dim1= int(i_dim1 // N_phi)
                    i_phi_dim1= int(i_dim1 -i_theta_dim1*N_phi)
                    if(i_dim1 > i_dim0):
                        d2= (x0[i_theta_dim0,i_phi_dim0]- x0[i_theta_dim1,i_phi_dim1])**2 + (y0[i_theta_dim0,i_phi_dim0]- y0[i_theta_dim1,i_phi_dim1])**2 + (z0[i_theta_dim0,i_phi_dim0]- z0[i_theta_dim1,i_phi_dim1])**2
                        Cs= np.exp(-0.5*d2/lc**2)
                        s_vec_cov_matrix[i_dim0,i_dim1]= beta**2*Cs
                    elif(i_dim1 == i_dim0):
                        s_vec_cov_matrix[i_dim0,i_dim1]= beta**2
                    else:
                        s_vec_cov_matrix[i_dim0,i_dim1]=s_vec_cov_matrix[i_dim1,i_dim0]
            return s_vec_cov_matrix
        
        s_vec_cov_matrix= compute_s_vec_cov_matrix(x0,y0,z0,lc,self.beta,N_theta,N_phi)
        s_vec_mean= np.zeros(N_theta*N_phi)

        s_samples= rng.multivariate_normal(s_vec_mean, s_vec_cov_matrix, size=1, method='eigh').squeeze()
        
        s= np.zeros_like(x0) 
        for i_theta in range(N_theta):
            for i_phi in range(N_phi):
                index= i_theta*N_phi + i_phi
                s[i_theta,i_phi]= s_samples[index]

        f_interp= scipy.interpolate.RectBivariateSpline(theta, phi, s) # interpolation fnction of s(theta,phi)

        # multiply the number of grid points of polar coordinates (theta, phi) by the surface_interpolation_factor
        surface_interpolation_factor= int(4*np.cbrt(self.bc_ratio*self.ab_ratio))
        theta_new = np.linspace(0, np.pi, surface_interpolation_factor*N_theta)
        phi_new = np.linspace(0, 2*np.pi, surface_interpolation_factor*N_phi)
        theta_new_mesh, phi_new_mesh = np.meshgrid(theta_new, phi_new, indexing='ij')

        s_intp= f_interp(theta_new, phi_new)

        # Cartesian coorinates x0(theta,phi), y0(theta,phi), z0(theta,phi) on the surface of the base ellipsold (x/a)**2 + (y/b)**2 + (z/c)**2 =1
        x0 = a*np.sin(theta_new_mesh)*np.cos(phi_new_mesh)
        y0 = b*np.sin(theta_new_mesh)*np.sin(phi_new_mesh)
        z0 = c*np.cos(theta_new_mesh)

        n0_vec= np.array([x0/a**2, y0/b**2, z0/c**2])/np.sqrt((x0/a**2)**2 + (y0/b**2)**2 + (z0/c**2)**2) # unit outward normal on the surface point of the base ellipsoid, n0_vec(theta,phi)4

        x_GRE_surf= x0+h0*((np.exp(s_intp)-(1/2)*self.beta**2)-1)*n0_vec[0,:,:]
        y_GRE_surf= y0+h0*((np.exp(s_intp)-(1/2)*self.beta**2)-1)*n0_vec[1,:,:]
        z_GRE_surf= z0+h0*((np.exp(s_intp)-(1/2)*self.beta**2)-1)*n0_vec[2,:,:]

        xyz_meshes_GRE_surf= (x_GRE_surf,y_GRE_surf,z_GRE_surf)

        x_flat= x_GRE_surf.flatten().squeeze()
        y_flat= y_GRE_surf.flatten().squeeze()
        z_flat= z_GRE_surf.flatten().squeeze()

        r_points_on_GRE_surf=  np.column_stack((x_flat,y_flat,z_flat)) # (N,3) numpy array

        return r_points_on_GRE_surf, xyz_meshes_GRE_surf
    
    

    def create_cuboid_lattice_that_encloses_GRE_shape(self, r_points_on_GRE_surf):
        ''' 
        Create a cuboid lattice with grid spacing lf from the given GRE surface points

        #### input ####
        r_points_on_GE_surf: one of the return values of the function self.compute_r_points_on_GRE()

        #### output ####
        lattice_domain: boundaries of the cuboid lattice domain that enclose the GRE, a tuple of three 2-lists ([-x_lim, +x_lim],[-y_lim, +y_lim],[-z_lim, +z_lim]), in um unit
        lattice_n: number of lattice grid points along each axis, 1d numpy array of 3 integers: n[0]:=n_x, n[1]:=n_y, n[2]:=n_z
        lattice_grid_points: 2d numpy array of cartesian components of the lattice grid points, shape= (N,3), where N is the total number of lattice grid points= lattice_n[0]*lattice_n[1]*lattice_n[2], row index (0 to N-1) is equivalent to the lattice_address := ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
        '''

        x_range_max= np.max(np.abs(r_points_on_GRE_surf[:,0]))
        y_range_max= np.max(np.abs(r_points_on_GRE_surf[:,1]))
        z_range_max= np.max(np.abs(r_points_on_GRE_surf[:,2]))

        lattice_x_range= x_range_max+ self.lattice_lf
        lattice_y_range= y_range_max+ self.lattice_lf
        lattice_z_range= z_range_max+ self.lattice_lf
        lattice_domain= ([-lattice_x_range, lattice_x_range], [-lattice_y_range, lattice_y_range], [-lattice_z_range, lattice_z_range])

        x_grid, y_grid, z_grid = np.meshgrid(np.arange(-lattice_x_range,lattice_x_range, self.lattice_lf),np.arange(-lattice_y_range,lattice_y_range, self.lattice_lf),np.arange(-lattice_z_range,lattice_z_range, self.lattice_lf), indexing='ij') # this order is correct given the definition of the lattice_address := ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
        lattice_grid_points= np.column_stack((x_grid.ravel(),y_grid.ravel(),z_grid.ravel())) # (N,3) numpy array
        
        n_x= len(np.arange(-lattice_x_range,lattice_x_range, self.lattice_lf))
        n_y= len(np.arange(-lattice_y_range,lattice_y_range, self.lattice_lf))
        n_z= len(np.arange(-lattice_z_range,lattice_z_range, self.lattice_lf))
        lattice_n= np.array([n_x,n_y,n_z], dtype=np.int32) # number of grid points along (x,y,z) axes
        # lattice_address = ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz, where ix=0,...,lattice_n[0] ; iy=0,...,lattice_n[1] ; iz=0,...,lattice_n[2]

        return lattice_domain, lattice_n, lattice_grid_points
    


    def find_nearest_distance_from_the_GRE_surf(self, lattice_grid_points, r_points_on_GRE_surf):
        ''' 
        Find the minimal distance of each lattice grid point from the set of discrete sampling points over the GRE surface, that approximate the distance of each lattice grid point from the continuous GRE surface

        #### input ####
        lattice_grid_points: one of the return values of the function self.create_cuboid_lattice_that_encloses_GRE_shape()
        r_points_on_GRE_surf: one of the return values of the function self.compute_r_points_on_GRE()

        #### output ####
        dist_from_GRE: 1d numpy array of size N, where N is the total number of lattice grid points. The index of this array = lattice_address
        
        '''

        # KDTree for the points cloud distributed over the surface of Gaussian Ellipsoid
        tree = KDTree(r_points_on_GRE_surf)
        query_point = lattice_grid_points # この点から最も近い点を探す
        dist, ind = tree.query(query_point, k=1)  # k=1 は最も近い1点を求めることを意味する
        dist_from_GRE= dist[:,0]

        return dist_from_GRE
    

    @staticmethod
    @njit
    def extract_lattice_address_in_GRE_volume(lattice_lf, distance_factor, lattice_n, dist_from_GRE):
        
        '''
        #### input  ####
        lattice_n: number of lattice grid points of cuboid along x, y, z axes, 1d numpy array of size 3
        lf: grid point interval of the cuboid lattice in um
        dist_from_GRE: distance from the closest sample point on the GE surface, 1d numpy array of size N, where N= n[0]*n[1]*n[2] is the total number of grid points in the cuboid lattice 
        

        ### output ###
        lattice_grid_points_is_in_GREvol: logical flag of each lattice address that is True if inside the GE volume, 1d numpy array with size= N= n[0]*n[1]*n[2], dtype=bool.

        '''

        # mask along z
        lattice_grid_points_is_in_GREvol_along_z= np.full(np.prod(lattice_n), False, dtype=bool)
        for ix in range(lattice_n[0]):
            for iy in range(lattice_n[1]):
                lattice_address_xy= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]
                is_in_vol_along_z= np.full((2,lattice_n[2]),False,dtype=bool)
                for iter in range(2):
                    crossed_1st_boundary= False
                    
                    if iter == 0:
                        for iz in range(0,lattice_n[2],1):
                            lattice_address= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
                            if (dist_from_GRE[lattice_address] < distance_factor*lattice_lf) and (crossed_1st_boundary == False) :
                                crossed_1st_boundary= True
                                continue
                            if crossed_1st_boundary == True:
                                is_in_vol_along_z[iter,iz]= True
                    if iter == 1:
                        for iz in range(lattice_n[2]-1,-1,-1):
                            lattice_address= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
                            if (dist_from_GRE[lattice_address] < distance_factor*lattice_lf) and (crossed_1st_boundary == False) :
                                crossed_1st_boundary= True
                                continue
                            if crossed_1st_boundary == True:
                                is_in_vol_along_z[iter,iz]= True

                lattice_grid_points_is_in_GREvol_along_z[lattice_address_xy:lattice_address_xy+lattice_n[2]:1] = is_in_vol_along_z[0,:]*is_in_vol_along_z[1,:]


        # mask along y
        lattice_grid_points_is_in_GREvol_along_y= np.full(np.prod(lattice_n), False, dtype=bool)
        for iz in range(lattice_n[2]):
            for ix in range(lattice_n[0]):
                lattice_address_zx= ix*lattice_n[1]*lattice_n[2]+iz
                is_in_vol_along_y= np.full((2,lattice_n[1]),False,dtype=bool)
                for iter in range(2):
                    crossed_1st_boundary= False
                    
                    if iter == 0:
                        for iy in range(0,lattice_n[1],1):
                            lattice_address= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
                            if (dist_from_GRE[lattice_address] < distance_factor*lattice_lf) and (crossed_1st_boundary == False) :
                                crossed_1st_boundary= True
                                continue
                            if crossed_1st_boundary == True:
                                is_in_vol_along_y[iter,iy]= True
                    if iter == 1:
                        for iy in range(lattice_n[1]-1,-1,-1):
                            lattice_address= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
                            if (dist_from_GRE[lattice_address] < distance_factor*lattice_lf) and (crossed_1st_boundary == False) :
                                crossed_1st_boundary= True
                                continue
                            if crossed_1st_boundary == True:
                                is_in_vol_along_y[iter,iy]= True

                lattice_grid_points_is_in_GREvol_along_y[lattice_address_zx:lattice_address_zx+lattice_n[1]*lattice_n[2]:lattice_n[2]] = is_in_vol_along_y[0,:]*is_in_vol_along_y[1,:]


        # mask along x
        lattice_grid_points_is_in_GREvol_along_x= np.full(np.prod(lattice_n), False, dtype=bool)
        for iy in range(lattice_n[1]):
            for iz in range(lattice_n[2]):
                lattice_address_yz= iy*lattice_n[2]+iz
                is_in_vol_along_x= np.full((2,lattice_n[0]),False,dtype=bool)
                for iter in range(2):
                    crossed_1st_boundary= False
                    
                    if iter == 0:
                        for ix in range(0,lattice_n[0],1):
                            lattice_address= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
                            if (dist_from_GRE[lattice_address] < distance_factor*lattice_lf) and (crossed_1st_boundary == False) :
                                crossed_1st_boundary= True
                                continue
                            if crossed_1st_boundary == True:
                                is_in_vol_along_x[iter,ix]= True
                    if iter == 1:
                        for ix in range(lattice_n[0]-1,-1,-1):
                            lattice_address= ix*lattice_n[1]*lattice_n[2]+iy*lattice_n[2]+iz
                            if (dist_from_GRE[lattice_address] < distance_factor*lattice_lf) and (crossed_1st_boundary == False) :
                                crossed_1st_boundary= True
                                continue
                            if crossed_1st_boundary == True:
                                is_in_vol_along_x[iter,ix]= True

                lattice_grid_points_is_in_GREvol_along_x[lattice_address_yz::lattice_n[1]*lattice_n[2]] = is_in_vol_along_x[0,:]*is_in_vol_along_x[1,:]

        lattice_grid_points_is_in_GREvol= np.logical_and(np.logical_and(lattice_grid_points_is_in_GREvol_along_z, lattice_grid_points_is_in_GREvol_along_x), lattice_grid_points_is_in_GREvol_along_y)

        return lattice_grid_points_is_in_GREvol
    


    def visualize_the_generated_GRE_shape_and_incindent_beam(self, xyz_meshes_GRE_surf, lattice_grid_points, lattice_grid_points_is_in_GREvol, euler_angles_deg):
        ''' 
        visualize GRE shape and DDA dipoles in laboratory coodinate system

        ### inputs ###
        xyz_meshes_GRE_surf: one of the output from the member function "compute_r_points_on_GRE()"
        lattice_grid_points: one of the output from the member function "create_cuboid_lattice_that_encloses_GRE_shape()"
        lattice_grid_points_is_in_GREvol: output from the member function "extract_lattice_address_in_GRE_volume()"
        euler_angles_deg: Euler angles (alpha, beta, gamma) [degree] of rotation of particle coordinate systemfrom the laboratory coordinate system, ndarray of size 3.
        
        '''

        lattice_grid_points_in_GREvol= lattice_grid_points[lattice_grid_points_is_in_GREvol]
        num_elements_occupy= np.sum(lattice_grid_points_is_in_GREvol)

        euler_alpha_rad= np.radians(euler_angles_deg[0])
        euler_beta_rad= np.radians(euler_angles_deg[1])
        euler_gamma_rad= np.radians(euler_angles_deg[2])
        r = R.from_euler('ZYZ', [euler_alpha_rad, euler_beta_rad, euler_gamma_rad])
        rotmat= r.as_matrix()
        xyz_meshes_GRE_surf_x_in_L_coordinate= xyz_meshes_GRE_surf[0]*rotmat[0,0]+xyz_meshes_GRE_surf[1]*rotmat[0,1]+xyz_meshes_GRE_surf[2]*rotmat[0,2]
        xyz_meshes_GRE_surf_y_in_L_coordinate= xyz_meshes_GRE_surf[0]*rotmat[1,0]+xyz_meshes_GRE_surf[1]*rotmat[1,1]+xyz_meshes_GRE_surf[2]*rotmat[1,2]
        xyz_meshes_GRE_surf_z_in_L_coordinate= xyz_meshes_GRE_surf[0]*rotmat[2,0]+xyz_meshes_GRE_surf[1]*rotmat[2,1]+xyz_meshes_GRE_surf[2]*rotmat[2,2]
        lattice_grid_points_in_GREvol_in_L_coordinate= (np.matmul(rotmat, lattice_grid_points_in_GREvol.T)).T


        fig = plt.figure(figsize=[8,8])
        ax = fig.add_subplot(111, projection='3d')

        plot_alpha= np.min([30/(np.log(num_elements_occupy)+5*self.bc_ratio*self.ab_ratio),0.6])
        plot_s= np.min([10/(np.log(num_elements_occupy)+self.bc_ratio*self.ab_ratio),8])

        ax.plot_wireframe(xyz_meshes_GRE_surf_x_in_L_coordinate,xyz_meshes_GRE_surf_y_in_L_coordinate,xyz_meshes_GRE_surf_z_in_L_coordinate, color='black', linewidth= 0.3, alpha=0.6)
        ax.scatter(lattice_grid_points_in_GREvol_in_L_coordinate[:,0], lattice_grid_points_in_GREvol_in_L_coordinate[:,1], lattice_grid_points_in_GREvol_in_L_coordinate[:,2], marker='.' , s=plot_s, color='r', alpha=plot_alpha)
        
        axis_range= np.max(np.abs(lattice_grid_points[:,0]))
        ax.set_xlim([-axis_range,axis_range])
        ax.set_ylim([-axis_range,axis_range])
        ax.set_zlim([-axis_range,axis_range])

        particle_name= self.name
        euler_angles_string= "euler_angles=({:.0f},{:.0f},{:.0f})".format(euler_angles_deg[0],euler_angles_deg[1],euler_angles_deg[2])
        ax.set_title(particle_name + " " + euler_angles_string, fontsize=10)
        ax.set_xlabel("x [$\\mu$m]", fontsize=13)
        ax.set_ylabel("y [$\\mu$m]", fontsize=13)
        ax.set_zlabel("z [$\\mu$m]", fontsize=13)
        ax.tick_params(labelsize=11)

        plt.tight_layout()

        default_u_inc= np.array([0,0,1],dtype=np.float32)
        default_theta_inc= np.array([1,0,0],dtype=np.float32)
        default_phi_inc= np.array([0,1,0],dtype=np.float32)

        u_inc= default_u_inc*axis_range 
        theta_inc= default_theta_inc*axis_range 
        phi_inc= default_phi_inc*axis_range 

        start_point= np.array([0,0,0],dtype=np.float32)

        ax.arrow3D(start_point[0],start_point[1],start_point[2], u_inc[0],u_inc[1],u_inc[2], mutation_scale=20, ec ='black', fc='black')
        ax.arrow3D(start_point[0],start_point[1],start_point[2], theta_inc[0],theta_inc[1],theta_inc[2], mutation_scale=20, arrowstyle="-|>", linestyle='dashed', ec ='red',fc='red')
        ax.arrow3D(start_point[0],start_point[1],start_point[2], phi_inc[0],phi_inc[1],phi_inc[2], mutation_scale=20, arrowstyle="-|>", linestyle='dashed', ec ='blue',fc='blue')

        plt.show()



    def compute_GRE_volume_and_ve_radius(self, lattice_grid_points_is_in_GREvol):

        num_elements_in_GRE= np.sum(lattice_grid_points_is_in_GREvol)
        element_vol= self.lattice_lf**3
        volume_GRE= element_vol*num_elements_in_GRE
        ve_radius_GRE= np.cbrt(3*volume_GRE/(4*np.pi))

        return volume_GRE, ve_radius_GRE




    

