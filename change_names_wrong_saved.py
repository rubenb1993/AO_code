import numpy as np
import Zernike as Zn

#### 20170504 zernike measurements
folder_extensions = ["5_5/", "6_2/",]#,"5_5/", "6_2/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/", "sub_zerns_1/", "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/", "5_1/",]
folder_names = ["SLM_codes_matlab/20170504_" + folder_extensions[i] for i in range(len(folder_extensions))]
a_ref_ordering = 'fringe' #for these measurements, fringe ordering was still used
save_string = ["5_1", "5_5", "6_2", "6_4", "6_6", "3_zerns_1", "3_zerns_2", "3_zerns_3", "sub_zerns_1", "sub_zerns_2", "sub_zerns_3", "sub_zerns_4"]# "5_5", "6_2",
save_string = [save_string[i] + "_new_ordering" for i in range(len(save_string))] #string to save different plots with
# x0, y0 of 20170504 measurements: 332, 236, 600
##x0 = 332 + r_int_px         #middle x-position in [px] of the spot on the interferogram camera. Measured by making a box around the spot using ImageJ
##y0 = 236 + r_int_px         #middle y-position in [px] of the spot on the interferogram camera. Measured by making a box around the spot using ImageJ
##radius = int(r_int_px)      #radius in px of the spot on the interferogram camera
save_fold = "SLM_codes_matlab/reconstructions/20170504_measurement/"
order = 'brug'

n_maxes = np.insert(np.arange(2, 14), 0, 10)
j = Zn.max_n_to_j(n_maxes, order = order)
j_maxes = np.zeros(len(n_maxes))
for i in range(len(n_maxes)):
    j_maxes[i] = np.max(j[str(n_maxes[i])])
    j_max = int(j_maxes[i])
    for jj in range(len(folder_names)):
        folder = folder_names[jj]
        coeff_dict = np.load(folder + "coeff_dictionary_j_" + save_string[jj] + "_" + str(j_max) + ".npy")
        vars_dict = np.load(folder + "vars_dictionary_j_" + save_string[jj] + "_" + str(j_max) + ".npy")
        np.save(folder + "coeff_dictionary_j_" + save_string[jj+1] + "_" + str(j_max) + ".npy", coeff_dict)
        np.save(folder + "vars_dictionary_j_" + save_string[jj+1] + "_" + str(j_max) + ".npy", vars_dict)

for jj in range(len(folder_names)):
    folder = folder_names[jj]
    inter_rms = np.load(folder + save_string[jj] + "_inter_rms_var_j.npy")
    janss_rms = np.load(folder + save_string[jj] + "_janss_rms_var_j.npy")
    lsq_rms = np.load(folder + save_string[jj] + "_lsq_rms_var_j.npy")
    n_vector = np.load(folder + save_string[jj] + "_n_vector.npy")
    phase_int = np.load(folder + save_string[jj] + "_phase_rms_inter.npy")
    phase_janss = np.load(folder + save_string[jj] + "_phase_rms_janss.npy")
    phase_lsq = np.load(folder + save_string[jj] + "_phase_rms_lsq.npy")
    

    np.save(folder + save_string[jj+1] + "_inter_rms_var_j.npy", inter_rms)
    np.save(folder + save_string[jj+1] + "_janss_rms_var_j.npy", janss_rms)
    np.save(folder + save_string[jj+1] + "_lsq_rms_var_j.npy", lsq_rms)
    np.save(folder + save_string[jj+1] + "_n_vector.npy", n_vector)
    np.save(folder + save_string[jj+1] + "_phase_rms_inter.npy", phase_int)
    np.save(folder + save_string[jj+1] + "_phase_rms_janss.npy", phase_janss)
    np.save(folder + save_string[jj+1] + "_phase_rms_lsq.npy", phase_lsq)
