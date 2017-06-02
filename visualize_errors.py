import numpy as np
import Zernike as Zn
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter

##folder_extensions = ["5_1/", "5_5/", "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/", "sub_zerns_1/", "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/"]
##folder_names = ["SLM_codes_matlab/20170504_" + folder_extensions[i] for i in range(len(folder_extensions))]
##save_strings = ["5_1", "5_5", "6_2", "6_4", "6_6", "3_zerns_1", "3_zerns_2", "3_zerns_3", "sub_zerns_1", "sub_zerns_2", "sub_zerns_3", "sub_zerns_4"]
##save_strings = [save_strings[i] + "_optimized_8" for i in range(len(save_strings))]
##save_fold = "SLM_codes_matlab/reconstructions/20170504_measurement/"
##a_ref_ordering = 'fringe'
##code = 'single_zerns'
##inset = True

#### 20170512 random surface measurements
##folder_names = ["SLM_codes_matlab/20170512_rand_surf_" + str(i)+"/" for i in range(4)]
##save_strings = ["low", "medium", "high", "extreme"]
##save_strings = [save_strings[i] + "_optimized_8" for i in range(len(save_strings))]
##a_ref_ordering = 'brug'
##order = a_ref_ordering
##save_fold = "SLM_codes_matlab/reconstructions/20170512_measurement/more_orders/"
##code = 'random_surfs'
##inset = False

##
###### 20170515 zernike measurements
folder_extensions = ["365_11/", "365_22/", "544_04/", "544_08/","544_17/","544_33/", "365_03/", "365_06/",]#, "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/", "sub_zerns_1/", "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/", "5_1/"]
folder_names = ["SLM_codes_matlab/20170515_leica_" + folder_extensions[i] for i in range(len(folder_extensions))]
a_ref_ordering = 'brug' #for these measurements, fringe ordering was still used
save_strings = ["365_11", "365_22", "544_04", "544_08", "544_17", "544_33", "365_03", "365_06"]
save_strings = [save_strings[i] + "_optimized_8" for i in range(len(save_strings))]
end_string = save_strings##end_string = [save_strings[i] + "_vars_lsq_only" for i in range(len(save_strings))]
save_fold = "SLM_codes_matlab/reconstructions/20170515_measurement/"
code = 'lenses'
inset = True

only_use_lsq_vars = False
if only_use_lsq_vars:
    save_strings = [save_strings[i] + "_only_use_lsq" for i in range(len(save_strings))]


order = 'brug'

radius_fold = "SLM_codes_matlab/reconstructions/"

n_maxes = np.load(folder_names[0] + save_strings[0] + "_n_vector.npy")
j_maxes = np.copy(n_maxes)#np.insert(np.rint(np.logspace(1.1, 2.1, num = 15)),0, 47)
phase_rms_vec_inter = np.zeros((len(j_maxes), len(folder_names)))
phase_rms_vec_lsq = np.zeros((len(j_maxes), len(folder_names)))
phase_rms_vec_janss = np.zeros((len(j_maxes), len(folder_names)))
phase_rms_vec_int_corrected = np.zeros((len(j_maxes), len(folder_names)))
##
##a_correction = np.load(save_fold + "average_to_6.npy")
convert_a = (a_ref_ordering == 'fringe')
response = raw_input("recalculate? y/n")
if response == 'y':
    interf_rms_int = np.zeros((len(j_maxes), len(folder_names)))
    interf_rms_lsq = np.zeros((len(j_maxes), len(folder_names)))
    interf_rms_janss = np.zeros((len(j_maxes), len(folder_names)))
    
    N_int = 600
    xi, yi = np.linspace(-1, 1, N_int), np.linspace(-1, 1, N_int)
    xi, yi = np.meshgrid(xi, yi)
    mask = [np.sqrt((xi) ** 2 + (yi) ** 2) >= 1]

    if convert_a:
        print("conversion is true, making Z matrices")
        try:
            print("found the SLM matrix!")
            Z_mat_30 = np.load("Z_mat_30.npy")
            Z_mat_50 = np.load("Z_mat_50.npy")
            power_mat_30 = np.load("power_mat_30.npy")
            power_mat_50 = np.load("power_mat_50.npy")
        except:
            j_30 = np.arange(2, 32)
            n_30, m_30 = Zn.Zernike_j_2_nm(j_30, ordering = 'fringe')
            j_brug_30 = Zn.Zernike_nm_2_j(n_30, m_30, ordering = 'brug')
            n_max_30 = np.max(n_30).astype(int)
            print("n_max 30 = " + str(n_max_30))
            j_brug_30 = Zn.max_n_to_j(n_max_30, order = 'brug')[str(n_max_30)]
            n_max_30 = np.max(n_30).astype(int)
            power_mat_30 = Zn.Zernike_power_mat(n_max_30, order = 'brug')
            Z_mat_30 = Zn.Zernike_xy(xi, yi, power_mat_30, j_brug_30)

            j_50 = np.arange(2, 52)
            n_50, m_50 = Zn.Zernike_j_2_nm(j_50, ordering = 'fringe')
            j_brug_50 = Zn.Zernike_nm_2_j(n_50, m_50, ordering = 'brug')
            n_max_50 = np.max(n_50).astype(int)
            print("n_max 50 = " + str(n_max_50))
            j_brug_50 = Zn.max_n_to_j(n_max_50, order = 'brug')[str(n_max_50)]
            n_max_50 = np.max(n_50).astype(int)
            power_mat_50 = Zn.Zernike_power_mat(n_max_50, order = 'brug')
            Z_mat_50 = Zn.Zernike_xy(xi, yi, power_mat_50, j_brug_50)
            
            np.save("Z_mat_30.npy", Z_mat_30)
            np.save("Z_mat_50.npy", Z_mat_50)
            np.save("power_mat_30.npy", power_mat_30)
            np.save("power_mat_50.npy", power_mat_50)

    else:
        n_max_fit = 10
        j_ref = Zn.max_n_to_j(n_max_fit, order = order)[str(n_max_fit)]
        a_ref = np.load(folder_names[0] + "reference_slm_vector.npy")
        assert(len(j_ref) == len(a_ref))
        print("it is " + str(len(j_ref) == len(a_ref)) + " that a_ref is up to order " + str(n_max_fit))
        try:
            Z_mat_ref = np.load("Z_mat_ref_" + str(n_max_fit) + ".npy")
            power_mat_ref = np.load("power_mat_ref_" + str(n_max_fit) + ".npy")
        except:
            power_mat_ref = Zn.Zernike_power_mat(n_max_fit, order = order)
            Z_mat_ref = Zn.Zernike_xy(xi, yi, power_mat_ref, j_ref)
            np.save("Z_mat_ref_" + str(n_max_fit) + ".npy", Z_mat_ref)
            np.save("power_mat_ref_" + str(n_max_fit) + ".npy", power_mat_ref)

    for ii in range(len(j_maxes)):
        print(ii)
        n_max = n_maxes[ii]
        j = Zn.max_n_to_j(n_max, order = 'brug')[str(n_max)]
        j_janss = Zn.max_n_to_j(n_max-1, order = 'brug')[str(n_max-1)]

        j_max = np.max(j)
        try:
            power_mat = np.load("matrices_600_radius/power_mat_" + str(n_max) + ".npy")
            Z_mat = np.load("matrices_600_radius/Z_mat_" + str(n_max) + ".npy")
            power_mat_janss = np.load("matrices_600_radius/power_mat_" + str(n_max-1) + ".npy")
            Z_mat_janss = np.load("matrices_600_radius/Z_mat_" + str(n_max-1) + ".npy")
        except:
            power_mat = Zn.Zernike_power_mat(n_max, order = 'brug')
            Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j)
            power_mat_janss = Zn.Zernike_power_mat(n_max-1, order = 'brug')
            Z_mat_janss = Zn.Zernike_xy(xi, yi, power_mat_janss, j_janss)

        for jj in range(len(folder_names)):
            folder_name = folder_names[jj]
            save_string = save_strings[jj]
            
##            with open(folder_name + "rms_dict_j_" + save_string + "_" + str(int(j_max)) + ".txt") as data:
##                    rms_dict = json.load(data)
##
##            interf_rms_int[ii, jj] = rms_dict["rms_inter"]
##            interf_rms_lsq[ii, jj] = rms_dict["rms_lsq"]
##            interf_rms_janss[ii, jj] = rms_dict["rms_janss"]

            a_ref = np.load(folder_name + "reference_slm_vector.npy")

            if convert_a:
                a_ref = Zn.convert_fringe_2_brug(a_ref)
                try:
                    if a_ref.shape[0] == power_mat_30.shape[-1]:
                        phi_ref = np.ma.array(np.dot(Z_mat_30, a_ref), mask = mask)
                    elif a_ref.shape[0] == power_mat_50.shape[-1]:
                        phi_ref = np.ma.array(np.dot(Z_mat_50, a_ref), mask = mask)
                    else:
                        raise ValueError('Z_mat_ref and a_ref are not the same size')
                except ValueError as err:
                    print(err.args)
            else:
                phi_ref = np.ma.array(np.dot(Z_mat_ref, a_ref), mask = mask)

            a_inter_corrected = np.copy(a_inter)
##            max_index = np.min([len(a_correction), len(a_inter_corrected)])
##            a_inter_corrected[:max_index] -= a_correction[:max_index]
            
            phi_lsq = np.ma.array(np.dot(Z_mat, a_lsq_opt), mask = mask)
            phi_int = np.ma.array(np.dot(Z_mat, a_inter), mask = mask)
            phi_int_corrected = np.ma.array(np.dot(Z_mat, a_inter_corrected), mask = mask)
            phi_janss = np.ma.array(np.dot(Z_mat_janss, a_janss_opt), mask = mask)
            phi_diff_int, phi_diff_lsq, phi_diff_janss, phi_diff_int_corrected = np.ma.array(phi_ref - phi_int, mask = mask), np.ma.array(phi_ref - phi_lsq, mask = mask), np.ma.array(phi_ref - phi_janss, mask = mask), np.ma.array(phi_ref - phi_int_corrected, mask = mask)
            eps_vec = np.stack((np.sqrt((phi_diff_int**2).sum())/N_int, np.sqrt((phi_diff_lsq**2).sum())/N_int, np.sqrt((phi_diff_janss**2).sum())/N_int, np.sqrt((phi_diff_int_corrected**2).sum())/N_int))
                        

            phase_rms_vec_inter[ii, jj] = eps_vec[0]
            phase_rms_vec_lsq[ii, jj] = eps_vec[1]
            phase_rms_vec_janss[ii, jj] = eps_vec[2]
            phase_rms_vec_int_corrected[ii, jj] = eps_vec[3]
                               
            
    for jj in range(len(folder_names)):
##        np.save(folder_names[jj] + save_strings[jj] + "_phase_rms_inter.npy", phase_rms_vec_inter[:, jj])
        np.save(folder_names[jj] + save_strings[jj] + "_phase_rms_lsq.npy", phase_rms_vec_lsq[:, jj])
        np.save(folder_names[jj] + save_strings[jj] + "_phase_rms_janss.npy", phase_rms_vec_janss[:, jj])
##        np.save(folder_names[jj] + save_strings[jj] + "_phase_rms_inter_corrected.npy", phase_rms_vec_int_corrected[:, jj])                      
                               
else:
    for jj in range(len(folder_names)):
##        phase_rms_vec_inter[:, jj] = np.load(folder_names[jj] + save_strings[jj] + "_phase_rms_inter.npy")
        phase_rms_vec_lsq[:, jj] = np.load(folder_names[jj] + save_strings[jj] + "_phase_rms_lsq.npy")
        phase_rms_vec_janss[:, jj] = np.load(folder_names[jj] + save_strings[jj] + "_phase_rms_janss.npy")
##        phase_rms_vec_int_corrected[:, jj] = np.load(folder_names[jj] + save_strings[jj] + "_phase_rms_inter_corrected.npy")

ordering_j = np.argsort(j_maxes)
for jj in range(len(folder_names)):
    folder_name = folder_names[jj]
    save_string = save_strings[jj]
    a_ref = np.load(folder_name + "reference_slm_vector.npy")
    if convert_a:
        a_ref = Zn.convert_fringe_2_brug(a_ref)
    nonzero_a = np.nonzero(a_ref)[0] + 2
    n_nnz, m_nnz = Zn.Zernike_j_2_nm(nonzero_a, ordering = 'brug')
    n_nnz = n_nnz.astype(int)
    m_nnz = m_nnz.astype(int)
##    phase_rms_vec_inter[:, jj] = phase_rms_vec_inter[ordering_j, jj]
    phase_rms_vec_lsq[:, jj] = phase_rms_vec_lsq[ordering_j, jj]
    phase_rms_vec_janss[:, jj] = phase_rms_vec_janss[ordering_j, jj]
##    phase_rms_vec_int_corrected[:, jj] = phase_rms_vec_int_corrected[ordering_j, jj]
    j_plot = n_maxes[ordering_j]
    f, ax = plt.subplots(figsize = (4.98, 4.98/2.2))
    ax.plot([0, np.max(j_plot)+2],[np.min(phase_rms_vec_janss[:,jj]), np.min(phase_rms_vec_janss[:,jj])], 'g--')
    ax.plot([0, np.max(j_plot)+2],[np.min(phase_rms_vec_lsq[:,jj]), np.min(phase_rms_vec_lsq[:,jj])], 'b--')
    #ax.plot([0, np.max(j_plot)+2],[np.min(phase_rms_vec_inter[:,jj]), np.min(phase_rms_vec_inter[:,jj])], 'k--')
##    ax.plot([0, np.max(j_plot)+2],[np.min(phase_rms_vec_int_corrected[:, jj]), np.min(phase_rms_vec_int_corrected[:, jj])], 'k--')

    #ax.plot(j_plot, phase_rms_vec_inter[:, jj], 'sk-', label = 'Interferogram')
    ax.plot(j_plot, phase_rms_vec_lsq[:, jj], '2b-', label = 'LSQ')
    ax.plot(j_plot, phase_rms_vec_janss[:, jj], 'og-', label = 'Janssen')
##    ax.plot(j_plot, phase_rms_vec_int_corrected[:, jj], 'dk-', label = 'Interferogram')
    #ax.set_xscale('log')
##    ax.set_yscale('log')
    ax.set_ylim([0, 2])
    ax.set_xticks(j_plot)
    ax.set_xlim([1, n_maxes[-1]+1])
    #ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r'Highest fitted Zernike order')
    ax.set_ylabel(r'$\varepsilon$')
    #ax.minorticks_off()
    title_label = save_string
    ax.set_title(title_label)
##    title_label = ''
    if code == 'single_zerns':
        if jj < 2:
            xlimits = [4, 9]
            xticks = np.arange(5, 11, 2)
        else:
            xlimits = [5, 10]
            xticks = np.arange(6, 12, 2)
    else:
        xlimits = [5, 10]
        xticks = np.arange(6, 12, 2)

    if code == 'lenses':
        left, bottom, width, height = [0.35, 0.5, 0.25, 0.3]
    else:
        left, bottom, width, height = [0.55, 0.55, 0.25, 0.3]
    
    if inset:
        
        ax2 = f.add_axes([left, bottom, width, height])
        ax2.set_xlim(xlimits)
        ax2.plot(j_plot, phase_rms_vec_lsq[:, jj], '2b-', markersize = 3, linewidth = 1)
        ax2.plot(j_plot, phase_rms_vec_janss[:, jj], 'og-', markersize = 3, linewidth = 1)
        minimum = np.min(np.stack((phase_rms_vec_lsq[:, jj], phase_rms_vec_janss[:, jj])))
        maximum = np.max([1.25*minimum, 1.25*np.min(phase_rms_vec_lsq[xlimits, jj])])
        ax2.set_ylim([0.9 * minimum, maximum])
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticks)
        ax2.tick_params(labelsize = 6)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.tick_params(labelsize = 8)
    ax.legend(prop={'size':7}, loc = 2)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(loc = 2, bbox_to_anchor=(1.05, 1.0), ncol = 1, fontsize = 6)

##    for i in range(len(n_nnz)):
##        title_label += r'$Z_{' + str(n_nnz[i]) + '}^{' + str(m_nnz[i]) + '}$ +'
##    ax.set_title(title_label[:-1])

    f.savefig(save_fold + "linear_epsilon_var_j" + save_string + '.png', bbox_inches = 'tight', dpi = 300)
    plt.close(f)

plt.show()
