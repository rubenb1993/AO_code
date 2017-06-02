"""Script to show all radial parts of the first n_max zernike orders.
Blue dots represent the sampling on the SH sensor.
Nyquist dictates that 2* the amount of 0's in the system should be sampled in order to not aliase.
The n'th order zernike has n radial 0's (including algebriac multiplicity)
Therefore the 6th order should be the highest measureable Zernike without aliasing
"""

import numpy as np
import Zernike as Zn
import matplotlib.pyplot as plt

N_spots = 13
x_sample = np.linspace(-1, 1, N_spots)
y_sample = np.array([0]*len(x_sample))
xx_s, yy_s = np.meshgrid(x_sample, y_sample)
x_tot = np.linspace(-1, 1, 600)
y_tot = np.array([0]*len(x_tot))
xx, yy = np.meshgrid(x_tot, y_tot)

n_max = 12
n_vec = np.arange(2, n_max+1, 1)
##m = np.zeros((len(n_vec), max(n_vec)+1))
n_dict = {}
for i in range(len(n_vec)):
    n_dict[str(n_vec[i])] = np.arange(n_vec[i]%2, n_vec[i]+2, 2)
##
##n_max, m_max = n_vec[-1], n_vec[-1]
j = Zn.max_n_to_j(n_max, order = 'brug')[str(n_max)]
##j_max = Zn.Zernike_nm_2_j(n_max, m_max, ordering = 'brug')
##j_vec = np.arange(2, int(j_max+2))
##power_mat = Zn.Zernike_power_mat(int(j_max+2))
power_mat = np.load("matrices_600_radius/power_mat_" + str(n_max) + ".npy")
R_n_m = Zn.Zernike_xy(xx, yy, power_mat, j)
R_sample = Zn.Zernike_xy(xx_s, yy_s, power_mat, j)

                                 
f, ax = plt.subplots(len(n_vec), len(n_dict[str(n_vec[-1])]), sharex = True, sharey = True)
for ii in range(len(n_vec)):
    dict_entry = n_dict[str(ii+2)]
    for jj in range(len(dict_entry)):
        j = Zn.Zernike_nm_2_j(n_vec[ii], dict_entry[jj], ordering = 'brug')
        ax[ii, jj].plot(x_tot, R_n_m[300, :, int(j)-2], 'b-')
        ax[ii, jj].plot(x_sample, R_sample[6, :, int(j)-2], 'ob')
        if jj == 0:
            ax[ii, jj].set_ylabel(r'$R_{' + str(n_vec[ii]) + '}^{ '+ str(dict_entry[jj]) + '}(r)$', labelpad = -10)
        else:
            ax[ii, jj].set_ylabel(r'$R_{' + str(n_vec[ii]) + '}^{ '+ str(dict_entry[jj]) + '}(r)$')
    for jj in range(len(dict_entry), len(n_dict[str(n_vec[-1])])):
        ax[ii, jj].axis('off')

f.savefig('sampling_sh_sensor.png', bbox_inches = 'tight')
plt.show()
    
