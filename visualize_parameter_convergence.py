import numpy as np
import Zernike as Zn
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import ScalarFormatter

##folder_extensions = ["5_1/", "5_5/", "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/", "sub_zerns_1/", "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/"]
##folder_names = ["SLM_codes_matlab/20170504_" + folder_extensions[i] for i in range(len(folder_extensions))]
##save_strings = ["5_1", "5_5", "6_2", "6_4", "6_6", "3_zerns_1", "3_zerns_2", "3_zerns_3", "sub_zerns_1", "sub_zerns_2", "sub_zerns_3", "sub_zerns_4"]
####save_strings = [save_strings[i] + "_fix_janss" for i in range(len(save_strings))]
##save_fold = "SLM_codes_matlab/reconstructions/20170504_measurement/"
##a_ref_ordering = 'fringe'
##order = 'brug'
##show_all_nonzero = True

#### 20170515 zernike measurements
folder_extensions = ["365_22/", "544_04/", "544_08/","544_17/","544_33/", "365_03/", "365_06/",]#"365_11/",
folder_names = ["SLM_codes_matlab/20170515_leica_" + folder_extensions[i] for i in range(len(folder_extensions))]
a_ref_ordering = 'brug' #for these measurements, fringe ordering was still used
save_strings = ["365_22", "544_04", "544_08", "544_17", "544_33", "365_03", "365_06"]#"365_11"
load_string = ["365_22", "544_04", "544_08", "544_17", "544_33", "365_03", "365_06"]#"365_11", 
save_fold = "SLM_codes_matlab/reconstructions/20170515_measurement/"
show_all_nonzero = False
order = 'brug'

only_use_lsq_vars = True
if only_use_lsq_vars:
    save_strings = [save_strings[i] + "_only_use_lsq" for i in range(len(save_strings))]

convert_a = (a_ref_ordering == 'fringe' or a_ref_ordering == 'Fringe')

n_maxes = np.load(folder_names[0] + save_strings[0] + "_n_vector.npy")
j_maxes = np.copy(n_maxes)#np.insert(np.rint(np.logspace(1.1, 2.1, num = 15)),0, 47)

for jj in range(len(folder_names)):
    folder_name = folder_names[jj]
    save_string = save_strings[jj]
    a_ref = np.load(folder_name + "reference_slm_vector.npy")
    if convert_a:
        a_ref = Zn.convert_fringe_2_brug(a_ref)    

    order_a_ref = np.argsort(n_maxes)
##    a_ref = a_ref[order_a_ref]
    if show_all_nonzero:
        elements = np.nonzero(a_ref)[0]
        print(elements)
    else:
        elements = Zn.Zernike_nm_2_j(np.array([2, 4, 6]), np.zeros(3), ordering = order).astype(int) - 2
        print(elements)
    convergence = np.zeros((len(elements), len(n_maxes), 3))
    roi = np.arange(2, 11)
    for ii in range(1,len(j_maxes)):
        
        
        n_max = n_maxes[ii]
        j = Zn.max_n_to_j(n_max, order = 'brug')[str(n_max)]
        j_max = np.max(j)
        
        j_janss = Zn.max_n_to_j(n_max-1, order = 'brug')[str(n_max-1)]
        j_janss_max = np.max(j_janss)

        coefficient_dict = np.load(folder_name + "coeff_dictionary_j_" + save_string + "_" + str(int(j_max)) + ".npy").item()
        a_lsq_opt = coefficient_dict["coeff_lsq"]
        a_inter = coefficient_dict["coeff_inter"]
        a_janss_opt = coefficient_dict["coeff_janss"]
        for kk in range(len(elements)):
            if elements[kk] > len(a_lsq_opt) - 1:
                convergence[kk, ii, 0:2] = np.nan
            else:
                convergence[kk, ii, 0] = (a_inter[elements[kk]] - a_ref[elements[kk]])/np.abs(a_ref[elements[kk]])
                convergence[kk, ii, 1] = (a_lsq_opt[elements[kk]] - a_ref[elements[kk]])/np.abs(a_ref[elements[kk]])
            if elements[kk] > len(a_janss_opt) - 1:
                convergence[kk, ii, 2] = np.nan
            else:
                convergence[kk, ii, 2] = (a_janss_opt[elements[kk]] - a_ref[elements[kk]])/np.abs(a_ref[elements[kk]])
        
        
    if len(elements) == 1:
        f, ax = plt.subplots(1,1)
        ax.plot(n_maxes[roi], convergence[0, roi, 1], '2-', label = 'LSQ')
        ax.plot(n_maxes[roi], convergence[0, roi, 2], 'o-', label = 'Janssen')
        ax.plot([0, n_maxes[-1]+1], [0, 0], 'k-')
        
        j_nnz = np.nonzero(a_ref)[0] + 2
        n_nnz, m_nnz = Zn.Zernike_j_2_nm(j_nnz, ordering = order)
        ax.set_title(r'$a_{ ' + str(n_nnz[0]) + '}^{' + str(m_nnz[0]) + '}$')
        ax.set_xlim(np.min(n_nnz), roi[-1])
##        ax.set_yscale('log')
        ax.legend(loc = 'lower left')
##        ax.set_ylimits(-2, 2)
    else:
##        f, ax = plt.subplots(len(elements), 1)
##        j_nnz = elements + 2
##        n_nnz, m_nnz = Zn.Zernike_j_2_nm(j_nnz, ordering = order)
##        for ii in range(len(elements)):
####            ax[ii].set_xlim(np.min(n_nnz), roi[-1])
##            coefficient = r'$a_{ ' + str(n_nnz[ii]) + '}^{' + str(m_nnz[ii]) + '}$'
##            ax[ii].set_title(coefficient)
##            ax[ii].plot(n_maxes[roi], convergence[ii, roi, 1], label = 'LSQ' + coefficient)
##            ax[ii].plot(n_maxes[roi], convergence[ii, roi, 2], label = 'Janssen' + coefficient)
##            ax[ii].plot([0, n_maxes[-1]+1], [0, 0], 'k-')
##            ax[ii].set_yscale('log')
        j_nnz = elements + 2
        n_nnz, m_nnz = Zn.Zernike_j_2_nm(j_nnz, ordering = order)
        
##            ax[ii].set_xlim(np.min(n_nnz), roi[-1])
        if show_all_nonzero:
            f, ax = plt.subplots(len(elements),1, sharex = True)
            for ii in range(len(elements)):
                coefficient = r'$a_{ ' + str(n_nnz[ii]) + '}^{' + str(m_nnz[ii]) + '}$'
                ax[ii].plot(n_maxes[roi], convergence[ii, roi, 1], '2-', label = 'LSQ ' + coefficient)
                ax[ii].plot(n_maxes[roi], convergence[ii, roi, 2], 'o-', label = 'Janssen ' + coefficient)
                ax[ii].plot([0, n_maxes[-1]+1], [0, 0], 'k-')
##                ax[ii].set_yscale('log')
                ax[ii].legend(loc = 'lower left')
##                ax[ii].set_ylim(-2, 2)
        else:
            f, ax = plt.subplots(2, 1, sharex = True)
            for ii in range(len(elements)):
                if ii < 1:
                    coefficient = r'$a_{ ' + str(n_nnz[ii]) + '}^{' + str(m_nnz[ii]) + '}$'
                    ax[0].set_title('Radial low')
                    ax[0].plot(n_maxes[roi], convergence[ii, roi, 1], '2-', label = 'LSQ ' + coefficient)
                    ax[0].plot(n_maxes[roi], convergence[ii, roi, 2], 'o-', label = 'Janssen ' + coefficient)
                    ax[0].plot([0, n_maxes[-1]+1], [0, 0], 'k-')
##                    ax[0].set_yscale('log')
                    ax[0].legend(loc = 'lower left')
##                    ax[0].set_ylim(-2, 2)
                else:
                    coefficient = r'$a_{ ' + str(n_nnz[ii]) + '}^{' + str(m_nnz[ii]) + '}$'
                    ax[1].set_title('Radial high')
                    ax[1].plot(n_maxes[roi], convergence[ii, roi, 1], '2-', label = 'LSQ ' + coefficient)
                    ax[1].plot(n_maxes[roi], convergence[ii, roi, 2], 'o-', label = 'Janssen ' + coefficient)
                    ax[1].plot([0, n_maxes[-1]+1], [0, 0], 'k-')
##                    ax[1].set_yscale('log')
                    ax[1].legend(loc = 'lower left')
##                    ax[1].set_ylim(-2, 2)
    plt.show()
