"""Algorithm for finding the assumed tip/tilt/astigmatism ever present in the interferogram method
amount: the highest zernike mode that should be corrected
returns the array of average differences between the intended phase and the interferogram phase
"""

import numpy as np
import Zernike as Zn

##folder_extensions = ["5_1/", "5_5/", "6_2/", "6_4/", "6_6/", "3_zerns_1/", "3_zerns_2/", "3_zerns_3/", "sub_zerns_1/", "sub_zerns_2/", "sub_zerns_3/", "sub_zerns_4/"]
##folder_names = ["SLM_codes_matlab/20170504_" + folder_extensions[i] for i in range(len(folder_extensions))]
##save_string = ["5_1", "5_5", "6_2", "6_4", "6_6", "3_zerns_1", "3_zerns_2", "3_zerns_3", "sub_zerns_1", "sub_zerns_2", "sub_zerns_3", "sub_zerns_4"]
##save_fold = "SLM_codes_matlab/reconstructions/20170504_measurement/"
##a_ref_ordering = 'fringe'

folder_names = ["SLM_codes_matlab/20170512_rand_surf_" + str(i)+"/" for i in range(4)]
save_string = ["low", "medium", "high", "extreme"]
a_ref_ordering = 'brug'
save_fold = "SLM_codes_matlab/reconstructions/20170512_measurement/"

convert_a = (a_ref_ordering == 'fringe')

amount = 1
j = Zn.max_n_to_j(amount, order = 'brug')[str(amount)]
amount = len(j)
a_avg = np.zeros((amount, len(folder_names)))

for ite in range(len(folder_names)):
    folder_name = folder_names[ite]
    a_dict = np.load(folder_name + "coeff_dictionary_j_"+save_string[ite] + "_55.npy").item()
    a_ite = a_dict["coeff_inter"]
    a_ref = np.load(folder_name + "reference_slm_vector.npy")
    if convert_a:
        print("converting a")
        a_ite = Zn.convert_fringe_2_brug(a_ite)
        a_ref = Zn.convert_fringe_2_brug(a_ref)
    a_avg[:, ite] = a_ite[:amount] - a_ref[:amount]

a_avg = np.mean(a_avg, axis = 1)
print(a_avg)
raw_input("fo sho do!")
np.save(save_fold + "average_to_" + str(amount) + ".npy", a_avg)
