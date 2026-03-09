import numpy as np
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles

# Define the ellipsoid parameters a >= b >= c
#### length_unit= um ###
r_v_base= 0.5 # allowed range = [0.1, 0.5], volume equivalent radius of base ellipsoid
bc_ratio= 1 # allowed range = [1.0, 5], b/c ratio of base ellipsoid
ab_ratio= 1 # allowed range = [1.0, 2.0], a/b ratio of base ellipsoid
beta= 0.0 # standard deviation of GE surface deformation, allowed range= [0, 0.3]

gre_shape_model= gaussian_ellipsoid_shape_model(r_v_base, ab_ratio, bc_ratio, beta)

random_seed= 0
r_points_on_GRE_surf, xyz_meshes_GRE_surf = gre_shape_model.compute_r_points_on_GRE(random_seed)
lattice_domain, lattice_n, lattice_grid_points= gre_shape_model.create_cuboid_lattice_that_encloses_GRE_shape(r_points_on_GRE_surf)
dist_from_GRE= gre_shape_model.find_nearest_distance_from_the_GRE_surf(lattice_grid_points, r_points_on_GRE_surf)
lattice_grid_points_is_in_target= gre_shape_model.extract_lattice_address_in_GRE_volume(gre_shape_model.lattice_lf, gre_shape_model.distance_factor, lattice_n, dist_from_GRE)

wl_0= 0.8337
m_m=1.329
m_m=1.0

#Organics
m_p_x= 1.5+0.j
m_p_y= 1.5+0.j
m_p_z= 1.5+0.j

m_p_xyz= np.array([m_p_x, m_p_y, m_p_z], dtype=np.complex64)

m_p_avg= (m_p_x+m_p_y+m_p_z)/3
#m_p_xyz= np.array([m_p_avg, m_p_avg, m_p_avg], dtype=np.complex64)
m_p_avg

target= Target(gre_shape_model.name, lattice_n, gre_shape_model.lattice_lf, lattice_grid_points, lattice_grid_points_is_in_target, m_p_xyz)

rng= np.random.default_rng(20)
number_of_orientations= 3
euler_alpha_deg= rng.uniform(0,360, number_of_orientations).reshape(number_of_orientations,1)
euler_beta_deg= rng.uniform(0,180, number_of_orientations).reshape(number_of_orientations,1)
euler_gamma_deg= rng.uniform(0,360, number_of_orientations).reshape(number_of_orientations,1)

# euler_alpha_deg= np.array([0,0,90]).reshape(number_of_orientations,1)
# euler_beta_deg= np.array([0,45,45]).reshape(number_of_orientations,1)
# euler_gamma_deg= np.array([0,0,0]).reshape(number_of_orientations,1)

euler_angles_radian= np.radians(np.hstack((euler_alpha_deg,euler_beta_deg,euler_gamma_deg)))

incident_field= IncidentField(wl_0, m_m, euler_angles_radian)

discrete_dipoles= DiscreteDipoles(target, incident_field)

discrete_dipoles.set_interaction_matrix()

discrete_dipoles.solve_matrix_equation()

r_p= discrete_dipoles.ve_radius
m_p_avg= (m_p_x+m_p_y+m_p_z)/3
Q_sca_mie, Q_abs_mie, Q_ext_mie, S_fw_mie, S_bk_mie = mie_compute_q_and_s(wl_0,m_m,r_p,m_p_avg,nang=3)

C_abs= discrete_dipoles.compute_C_abs_in_P_coodinate_system()
Q_abs= C_abs/(np.pi*discrete_dipoles.ve_radius**2)

C_ext= discrete_dipoles.compute_C_ext_in_P_coodinate_system()
Q_ext= C_ext/(np.pi*discrete_dipoles.ve_radius**2)

S_fw_theta_P, S_fw_phi_P= discrete_dipoles.compute_PCAS_observable_S_fw()

S_bk_P= discrete_dipoles.compute_OCBS_observable_S_bk()

S_sp_avg, S_sp_avg_SNR, S_depol, S_depol_SNR= discrete_dipoles.compute_PCAS_observable_S_fw_and_its_SNR()