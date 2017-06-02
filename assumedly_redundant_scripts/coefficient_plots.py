import sys
##import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
from matplotlib import rc
import Hartmann as Hm
import displacement_matrix as Dm
import Zernike as Zn
import mirror_control as mc
import LSQ_method as LSQ
import matplotlib.pyplot as plt
import janssen
import phase_extraction_slm as PE
import phase_unwrapping_test as pw
import scipy.optimize as opt
import scipy.ndimage as ndimage
import json #to write editable txt files

gold = (1 + np.sqrt(5))/2

folder_extensions = ["Z_5_1/", "Z_5_3/", "Z_5_5/", "Z_4_m2_2_0/", "Z_4_m4_3_3/", "Z_5_m3_4_0/", "Z_6_0_4_0_2_0/", "Z_7_1_5_1_3_1/"]#["coma/", "defocus/", "astigmatism/", "spherical/"]
#folder_extensions = ["coma/", "defocus/", "astigmatism/", "spherical/"]
folder_names = ["SLM_codes_matlab/20170405_" + folder_extensions[i] for i in range(len(folder_extensions))]

#save_string = ["coma", "defocus", "astigmatism", "spherical"]
save_string = ["5_coma", "5_trifoil", "5_pentafoil", "mix_defocus", "mix_trifoil", "mix_spherical", "all_spherical", "all_coma"]#["x_coma", "y_coma", "mix", "x_coma_high_order", "asymetric_coma", "asymetric_astigmatism"]
save_fold = "SLM_codes_matlab/reconstructions/log_diff/"
radius_fold = "SLM_codes_matlab/reconstructions/"
dpi_num = 300
N = 30

ind_dict = {'0': {'ind':[12], 'mag':[2]},
            '1': {'ind':[17], 'mag':[1.5]},
            '2': {'ind':[24], 'mag':[1.5]},
            '3': {'ind':[11, 2], 'mag':[1.5, 3.0]},
            '4': {'ind':[16, 8], 'mag':[2.5, 3.0]},
            '5': {'ind':[18, 7], 'mag':[1.5, -1.0]},
            '6': {'ind':[2, 7, 14], 'mag':[3.0, -1.0, 1.0]},
            '7': {'ind':[5, 12, 21], 'mag':[2.0, -1.0, 0.8]}}

params = {'legend.fontsize': 6,
         'axes.labelsize': 6,
         'axes.titlesize':7,
         'xtick.labelsize':6,
         'ytick.labelsize':6}
plt.rcParams.update(params)
#plot 30
for ii in range(len(folder_names)):
    a_slm_dict = ind_dict[str(ii)]
    a_slm = np.zeros(30)
    for jj in range(len(a_slm_dict['ind'])):
        a_slm[a_slm_dict['ind'][jj]] = a_slm_dict['mag'][jj]
        
    a_dict = np.load(folder_names[ii] + "coeff_dictionary_j_60.npy").item()
    a_inter = a_dict["coeff_inter"]
    a_janss = a_dict["coeff_janss"]
    a_lsq = a_dict["coeff_lsq"]
    j_max = len(a_inter)
    a_axis = np.arange(2, j_max+2)
    #plot values
    for i in range(len(a_inter)/N):
        f2, ax2 = plt.subplots(2,1, figsize = (4.98, 4.98/2.2), sharex = True)
        a_x = a_axis[i*N : (i+1)*N]
        ax2[0].plot([i*N, (i+1)*N+2], [0, 0], 'k-', linewidth = 0.8)

        ax2[0].plot(a_x, a_inter[i*N : (i+1)*N], 'sk', label = 'Interferogram')
        ax2[0].plot(a_x, a_janss[i*N : (i+1)*N], 'og', label = 'Janssen')
        ax2[0].plot(a_x, a_lsq[i*N : (i+1)*N], '2b', label = 'LSQ')
        #ax2[0].plot(a_x, a_slm[i*N : (i+1)*N], 'rx', label = 'SLM input', markersize = 1)
        ax2[0].set_xlim([i*N , (i+1)*N+1.5])
        min_a = np.min([np.min(a_inter), np.min(a_janss), np.min(a_lsq)])
        max_a = np.max([np.max(a_inter), np.max(a_janss), np.max(a_lsq)])
        b_gold = max_a - min_a / (gold)
        ax2[0].set_ylim([min_a - b_gold, max_a + b_gold])
        ax2[0].set_ylabel(r'$a_j$')
##        ax2[0].set_xlabel(r'$j$')
        ax2[0].legend(prop={'size':7}, loc = 'upper right')
        box = ax2[0].get_position()
        ax2[0].set_position([box.x0, box.y0, box.width, box.height * 0.8])
        ax2[0].legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.35), ncol = 4, fontsize = 6)
    ##    f2.savefig(save_fold + "zn_coeff_"+save_string[ii]+"_j_30.png", bbox_inches = 'tight', dpi = dpi_num)
    ##    plt.close(f2)

        #plot diffs
        diff_janss = np.abs(a_inter - a_janss)
        diff_lsq = np.abs(a_inter - a_lsq)
        diff_rms_janss = np.sum(diff_janss)
        diff_rms_lsq = np.sum(diff_lsq)
        print("lsq sum of diff: " + str(diff_rms_lsq) + ", janssen sum of diff: " + str(diff_rms_janss))
        x_axis = np.arange(2, j_max+2)
        
    ##    f, ax = plt.subplots(1,1, figsize = (4.98, 4.98/4.4))
        ax2[1].semilogy(a_x, diff_janss[i*N : (i+1)*N], 'go', label = 'Janssen')
        ax2[1].semilogy(a_x, diff_lsq[i*N : (i+1)*N], '2b', label = 'LSQ')
        ax2[1].set_ylim(bottom = 1e-4, top = 1e0)

        ax2[1].set_xlabel(r'$j$')
        ax2[1].set_ylabel(r'$|a_j^{{inter}} - a_j^{SH}|$')

        #ax2[1].legend(prop={'size':7}, loc = 'upper right')
        box = ax2[1].get_position()
        ax2[1].set_position([box.x0, box.y0, box.width, box.height * 0.8])
        #ax2[1].legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.35), ncol = 2, fontsize = 6)
        
        f2.savefig(save_fold + "zn_log_"+save_string[ii]+"_j_" +str(i*N) + "_" + str((i+1)*N) + ".png", bbox_inches = 'tight', dpi = dpi_num)
        plt.close(f2)

##    #plot 60
##    a_dict = np.load(folder_names[ii] + "coeff_dictionary_j_60.npy").item()
##    a_inter = a_dict["coeff_inter"]
##    a_janss = a_dict["coeff_janss"]
##    a_lsq = a_dict["coeff_lsq"]
##    j_max = len(a_inter)
##
##    diff_janss = np.abs(a_inter - a_janss)
##    diff_lsq = np.abs(a_inter - a_lsq)
##    diff_rms_janss = np.sum(diff_janss)
##    diff_rms_lsq = np.sum(diff_lsq)
##    print("lsq sum of diff: " + str(diff_rms_lsq) + ", janssen sum of diff: " + str(diff_rms_janss))
##    x_axis = np.arange(2, j_max+2)
##    N = 30
##    f2, ax2 = plt.subplots(2,1, figsize = (4.98, 4.98/2.2), sharey= True, gridspec_kw = {'height_ratios': [1,1]})
##    ax2[0].semilogy(x_axis[:N], diff_janss[:N], 'go', label = 'Janssen')
##    ax2[0].semilogy(x_axis[:N], diff_lsq[:N], '2b', label = 'LSQ')
##    ax2[1].semilogy(x_axis[N:], diff_janss[N:], 'go')
##    ax2[1].semilogy(x_axis[N:], diff_lsq[N:], '2b')
##    ax2[0].set_ylim(bottom = 1e-4, top = 1e0)
##    ax2[1].set_xlabel(r'$j$')
##    ax2[1].set_ylabel(r'$|a_j^{{inter}} - a_j^{SH}|$')
##
##    ax2[0].legend(prop={'size':7}, loc = 'upper right')
##    box = ax2[0].get_position()
##    ax2[0].set_position([box.x0, box.y0, box.width, box.height * 0.8])
##    box = ax2[1].get_position()
##    ax2[1].set_position([box.x0, box.y0, box.width, box.height * 0.8])
##    ax2[0].legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.35), ncol = 2, fontsize = 6)
##    f2.savefig(save_fold + "zn_log_" + save_string[ii] + "_j_60.png", bbox_inches = 'tight', dpi = dpi_num)
##    plt.close(f2)
