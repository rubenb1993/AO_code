"""Test script for seeing if fitting 10 zernike orders
could create a random surface. Later implemented in interface for SLM
"""

import numpy as np
import Zernike as Zn
import matplotlib.pyplot as plt
import phase_unwrapping_test as pw
import Zernike as Zn

n_max = 9
j = Zn.max_n_to_j(n_max, order = 'Brug')[str(n_max)]
N_int = 600
x, y = np.linspace(-3.75, 3.75, 1920), np.linspace(-1080.0/(2*256), 1080.0/(2*256), 1080)
xx, yy = np.meshgrid(x, y)

np.random.seed(0)
surf = 2*np.pi* np.random.randn(y.shape[0], x.shape[0])

mask = [np.sqrt(xx**2 + yy**2) > 1]
inside = np.where(np.sqrt(xx**2 + yy**2) < 1)
surf_phase = pw.butter_filter_unwrapped(surf, 5, 1, pad = True)
surf_phase *= 3*np.pi/np.max(surf_phase)

power_mat = Zn.Zernike_power_mat(n_max, order = 'brug')
Z_mat = Zn.Zernike_xy(xx, yy, power_mat, j)
Zernike_2d = np.zeros((len(inside[0]), len(j))) ## flatten to perform least squares fit
for i in range(len(j)):
    Zernike_2d[:, i] = Z_mat[inside[0],inside[1],i].flatten()

a = np.linalg.lstsq(Zernike_2d, surf_phase[inside])[0]
print(a)

f, ax = plt.subplots(3,1)
ax[0].imshow(np.ma.array(surf, mask = mask))
ax[1].imshow(np.ma.array(surf_phase, mask = mask))
ax[2].imshow(np.ma.array(np.dot(Z_mat, a), mask = mask))
plt.show()
