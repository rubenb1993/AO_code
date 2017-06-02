"""Quick test script to see if the units of LEICA provided by Silvania
seem to give a correct wavefront. Save them in different orders
"""

import numpy as np
import Zernike as Zn
import matplotlib.pyplot as plt

j = np.arange(2, 37)
n, m = Zn.Zernike_j_2_nm(j, ordering = 'fringe')
a_544 = np.array([-0.0081, 0.0080, -0.0002, -0.0143, 0.0211, -0.0474, 0.0478, -0.0159, -0.0014, 0.0136, 0.0003, -0.0017, -0.0029, 0.0053, -0.0093, -0.0032, -0.0035, -0.0011, -0.0023, 0.0096, -0.0089, -0.0033, -0.0069, -0.0135, 0.0052, 0.0002, -0.0027, -0.0065, -0.0002, -0.0088, -0.0020, 0.0189, -0.0090, 0.0109, -0.0090])
norm = np.sqrt((2*(n+1)/(1 + (m==0))))

a_544 *= 2

a_365 = np.array([-0.0025, -0.0026, -0.001, -0.01, 0.0274, -0.0331, -0.0121, -0.0179, -0.0024, 0.0225, -0.0068, 0.0045, -0.0035, 0.0, -0.0151, 0.0050, 0.0019, -0.0102, -0.0004, 0.0098, -0.0157, 0.0067, -0.0134, 0.0013, 0.0060, 0.0048, 0.0021, -0.0026, -0.0058, -0.0068, -0.0116, 0.0177, 0.0015, -0.0013, -0.0044])
a_365 *= 2

wl_544 = 544e-9
wl_365 = 365e-9

a_544 = Zn.convert_fringe_2_brug(a_544)
a_365 = Zn.convert_fringe_2_brug(a_365)
np.save("SLM_codes_matlab/objective_544_nm_0.4_pi_pv.npy", a_544)
np.save("SLM_codes_matlab/objective_365_nm_0.3_pi_pv.npy", a_365)

a_544 *= 2
a_365 *= 2

np.save("SLM_codes_matlab/objective_544_nm_0.8_pi_pv.npy", a_544)
np.save("SLM_codes_matlab/objective_365_nm_0.6_pi_pv.npy", a_365)

a_544 *= 2
a_365 *= 2

np.save("SLM_codes_matlab/objective_544_nm_1.7_pi_pv.npy", a_544)
np.save("SLM_codes_matlab/objective_365_nm_1.1_pi_pv.npy", a_365)

a_544 *= 2
a_365 *= 2

np.save("SLM_codes_matlab/objective_544_nm_3.3_pi_pv.npy", a_544)
np.save("SLM_codes_matlab/objective_365_nm_2.2_pi_pv.npy", a_365)

xi, yi = np.linspace(-1, 1, 200), np.linspace(-1, 1, 200)
xi, yi = np.meshgrid(xi, yi)
mask = [np.sqrt(xi**2 + yi**2) >= 1]

j_tot = Zn.max_n_to_j(int(n[-1]), order ='brug')[str(int(n[-1]))]
power_mat = Zn.Zernike_power_mat(int(n[-1]), order = 'brug')
Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j_tot)
abberations_544 = np.dot(Z_mat, a_544)
abberations_544 = np.ma.array(abberations_544, mask = mask)
abberations_365 = np.dot(Z_mat, a_365)
abberations_365 = np.ma.array(abberations_365, mask = mask)
pv_544 = (abberations_544.max() - abberations_544.min())/(np.pi)
pv_365 = (abberations_365.max() - abberations_365.min())/(np.pi)
print(pv_365, pv_544)

f, ax = plt.subplots(2, 1)
ax[0].imshow(abberations_365, vmin = -np.pi, vmax = np.pi)
ax[1].imshow(abberations_544, vmin = -np.pi, vmax = np.pi)
plt.show()
