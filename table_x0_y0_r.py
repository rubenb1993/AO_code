import numpy as np
import json

#### 20170504 zernike measurements
##folder_extensions = ["5_1/", "5_5/", "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/","sub_zerns_1/",  "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/"]
##folder_names = ["SLM_codes_matlab/20170504_" + folder_extensions[i] for i in range(len(folder_extensions))]
##a_ref_ordering = 'fringe' #for these measurements, fringe ordering was still used
##save_strings = ["5_1", "5_5", "6_2", "6_4", "6_6", "3_zerns_1", "3_zerns_2", "3_zerns_3", "sub_zerns_1", "sub_zerns_2", "sub_zerns_3", "sub_zerns_4", ]
##old_vars_string = [ "sub_zerns_2"]#"sub_zerns_1",
##save_strings = [save_strings[i] + "_fix_janss" for i in range(len(save_strings))] #string to save different plots with
##print_string = save_strings
##save_fold = "SLM_codes_matlab/reconstructions/20170504_measurement/"

###### 20170512 random surface measurements
##folder_names = ["SLM_codes_matlab/20170512_rand_surf_" + str(i)+"/" for i in range(4)]
##save_strings = ["low", "medium", "high", "extreme"]
##save_strings = [save_strings[i] + "_optimized_8" for i in range(len(save_strings))] #string to save different plots with
##print_string = save_strings
##a_ref_ordering = 'brug'
##save_fold = "SLM_codes_matlab/reconstructions/20170512_measurement/"

###### 20170515 zernike measurements
folder_extensions = ["544_04/", "544_08/","544_17/","544_33/", "365_03/", "365_06/", "365_11/", "365_22/"]#, "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/", "sub_zerns_1/", "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/", "5_1/"]
folder_names = ["SLM_codes_matlab/20170515_leica_" + folder_extensions[i] for i in range(len(folder_extensions))]
a_ref_ordering = 'brug' #for these measurements, fringe ordering was still used
save_strings = [ "544_04", "544_08", "544_17", "544_33", "365_03", "365_06", "365_11", "365_22",]
print_string = save_strings
save_fold = "SLM_codes_matlab/reconstructions/20170515_measurement/"


convert_a = (a_ref_ordering == 'fringe' or a_ref_ordering == 'Fringe')

n_maxes = np.load(folder_names[0] + save_strings[0] + "_n_vector.npy")
j_maxes = np.copy(n_maxes)#np.insert(np.rint(np.logspace(1.1, 2.1, num = 15)),0, 47)

data = np.zeros((len(folder_names), 6))
##data_names = ["x_lsq", "y_lsq", "r_LSQ", "x_janss", "y_janss", "r_janss"]
data_names = ["x_lsq", "x_janss", "y_lsq", "y_janss", "r_LSQ", "r_janss"]

opt_8 = False
if opt_8:
    load_string = "optimized_8"
    txt_string = "n_opt_8"
else:
    load_string = "optimized"
    txt_string = "n_opt_10"

for ii in range(len(folder_names)):
    folder = folder_names[ii]
    vars_janss = np.load(folder + load_string + "_janss_vars.npy")
    vars_lsq = np.load(folder + load_string + "_lsq_vars.npy")
    data[ii, 0] = vars_lsq[0]
    data[ii, 2] = vars_lsq[1]
    data[ii, 4] = vars_lsq[2]
    data[ii, 1] = vars_janss[0]
    data[ii, 3] = vars_janss[1]
    data[ii, 5] = vars_janss[2]

row_format ="{:>15}" * (len(data_names) + 1)
print row_format.format("", *data_names)
for team, row in zip(print_string, data):
    print row_format.format(team, *row)

np.savetxt(save_fold + "x_y_r_table_" + txt_string + ".csv", data, delimiter=' & ', fmt='%2.3f', newline=' \\\\\n')

