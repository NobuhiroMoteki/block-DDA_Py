import numpy as np
from shape_model.gaussian_ellipsoid import gaussian_ellipsoid_shape_model
from analytical_scattering_theories.homogeneous_sphere import mie_compute_q_and_s
from bl_dda.scatterer import Target, IncidentField, DiscreteDipoles
import h5py
import datetime

rng= np.random.default_rng(12345)
max_try_DDA_converge= 4
num_division_RHS_blocks= 1

output_dda_hdf5_file= "dda_results\\pcas_ocbs_simulated_data.hdf5"



with h5py.File(output_dda_hdf5_file, "r+") as h5:

    num_orientations= h5['target'].attrs['num_orientations']
    num_RHS_for_block_krylov= int(num_orientations/num_division_RHS_blocks)

    wl_0= h5['target'].attrs['wl_0_um']
   
    m_p_xyz= h5['target']['m_p_xyz'][:]
    m_m_list= h5['target']['m_m_list']
    r_v_base_list= h5['target']['r_v_base_list']
    bc_ratio_list= h5['target']['bc_ratio_list']
    ab_ratio_list= h5['target']['ab_ratio_list']
    gre_beta_list= h5['target']['gre_beta_list']

    for ind_m_m_list, m_m in enumerate(m_m_list):
        for ind_r_v_base_list, r_v_base in enumerate(r_v_base_list):
            for ind_bc_ratio_list, bc_ratio in enumerate(bc_ratio_list):
                for ind_ab_ratio_list, ab_ratio in enumerate(ab_ratio_list):
                    for ind_gre_beta_list, gre_beta in enumerate(gre_beta_list):

                        try:

                            current_time = datetime.datetime.now().time()
                            time_string = current_time.strftime('%H:%M:%S')
                            print("-----------------------------------------")
                            print("Current time:", time_string)
                            print("wl_0={:.4f}, m_m={:.4f} ".format(wl_0, m_m))
                            print("m_p_xyz={:}, r_v_base={:.4f}, bc_ratio={:.4f}, ab_ratio={:.4f}, gre_beta={:.4f} ".format(m_p_xyz,r_v_base,bc_ratio,ab_ratio,gre_beta))
                            print("num_orientations= {:}".format(num_orientations))
                            
                            #Skip this consition if alreadly computed
                            if ( h5['target']['simulated_data']['S_fw_PCAS_mie'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list].imag > 0):
                                continue

                            gre_shape_model= gaussian_ellipsoid_shape_model(r_v_base, bc_ratio, ab_ratio, gre_beta)
                            rng= np.random.default_rng(12345)
                            r_points_on_GRE_surf, xyz_meshes_GRE_surf = gre_shape_model.compute_r_points_on_GRE(rng)
                            lattice_domain, lattice_n, lattice_grid_points= gre_shape_model.create_cuboid_lattice_that_encloses_GRE_shape(r_points_on_GRE_surf)
                                
                            dist_from_GRE= gre_shape_model.find_nearest_distance_from_the_GRE_surf(lattice_grid_points, r_points_on_GRE_surf)
                            lattice_grid_points_is_in_target= gre_shape_model.extract_lattice_address_in_GRE_volume(gre_shape_model.lattice_lf, gre_shape_model.distance_factor, lattice_n, dist_from_GRE)

                            target= Target(gre_shape_model.name, lattice_n, gre_shape_model.lattice_lf, lattice_grid_points, lattice_grid_points_is_in_target, m_p_xyz)

                            for i_repeat in range(num_division_RHS_blocks):
                                print("num_RHS_for_block_krylov= {:}".format(num_RHS_for_block_krylov))
                                print("repeating block DDA for {:} random orientations= {:} / {:}".format(num_orientations, i_repeat, num_division_RHS_blocks))
                                i_start= i_repeat*num_RHS_for_block_krylov
                                i_end= (i_repeat+1)*num_RHS_for_block_krylov

                                i_try_DDA_converge= 0
                                while i_try_DDA_converge < max_try_DDA_converge :
                                    i_try_DDA_converge += 1
                                    print("i_try_DDA_converge = {:}".format(i_try_DDA_converge))
                                    euler_alpha_deg= rng.uniform(0,360, num_RHS_for_block_krylov).reshape(num_RHS_for_block_krylov,1)
                                    euler_beta_deg= rng.uniform(0,180, num_RHS_for_block_krylov).reshape(num_RHS_for_block_krylov,1)
                                    euler_gamma_deg= rng.uniform(0,360, num_RHS_for_block_krylov).reshape(num_RHS_for_block_krylov,1)
                                    euler_angles_radian= np.radians(np.hstack((euler_alpha_deg,euler_beta_deg,euler_gamma_deg)))

                                    incident_field= IncidentField(wl_0, m_m, euler_angles_radian)
                                    discrete_dipoles= DiscreteDipoles(target, incident_field)
                                    discrete_dipoles.set_interaction_matrix()
                                    discrete_dipoles.solve_matrix_equation()

                                    if discrete_dipoles.converge == True :
                                        C_abs= discrete_dipoles.compute_C_abs()
                                        C_ext= discrete_dipoles.compute_C_ext()
                                        S_fw_PCAS_theta, S_fw_PCAS_phi= discrete_dipoles.compute_PCAS_observable_S_fw()
                                        S_bk_OCBS= discrete_dipoles.compute_OCBS_observable_S_bk()
                                        break
                                    else:
                                        C_abs= np.nan
                                        C_ext= np.nan
                                        S_fw_PCAS_theta = np.nan
                                        S_fw_PCAS_phi= np.nan
                                        S_bk_OCBS= np.nan                   
                                    
                                h5['target']['simulated_data']['C_abs'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list, i_start:i_end] = C_abs
                                h5['target']['simulated_data']['C_ext'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list, i_start:i_end] = C_ext
                                h5['target']['simulated_data']['S_fw_PCAS_theta'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list, i_start:i_end] = S_fw_PCAS_theta
                                h5['target']['simulated_data']['S_fw_PCAS_phi'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list, i_start:i_end] = S_fw_PCAS_phi
                                h5['target']['simulated_data']['S_bk_OCBS'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list, i_start:i_end] = S_bk_OCBS

                            r_ve= discrete_dipoles.ve_radius
                            h5['target']['simulated_data']['r_ve'][ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list] = r_ve
                            print("r_ve={:.4f}, mean C_ext={:.4f}, mean S_fw_PCAS_theta={:.4g}, mean S_fw_PCAS_phi={:.4g}, mean S_bk_OCBS={:.4g}".format(r_ve, np.mean(C_ext), np.mean(S_fw_PCAS_theta), np.mean(S_fw_PCAS_phi), np.mean(S_bk_OCBS)))

                            m_p_avg= np.mean(m_p_xyz)
                            Q_sca_mie, Q_abs_mie, Q_ext_mie, S_fw_PCAS_mie, S_bk_OCBS_mie = mie_compute_q_and_s(wl_0,m_m,r_ve,m_p_avg,nang=3)
                            C_abs_mie= Q_abs_mie*np.pi*r_ve**2
                            C_ext_mie= Q_ext_mie*np.pi*r_ve**2
                            print("C_ext_mie={:.4f}, S_fw_PCAS_mie={:.4g}, S_bk_OCBS_mie={:.4g}".format(C_ext_mie, S_fw_PCAS_mie, S_bk_OCBS_mie))
                                
                            h5['target']['simulated_data']['C_abs_mie'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list] =  C_abs_mie
                            h5['target']['simulated_data']['C_ext_mie'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list] =   C_ext_mie
                            h5['target']['simulated_data']['S_fw_PCAS_mie'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list] =  S_fw_PCAS_mie
                            h5['target']['simulated_data']['S_bk_OCBS_mie'][ind_m_m_list, ind_r_v_base_list, ind_bc_ratio_list, ind_ab_ratio_list, ind_gre_beta_list] =  S_bk_OCBS_mie
                        
                        except KeyboardInterrupt:
                            print("Keyboard interrupt exception caught")
                            h5.close()
                            exit()
                                


