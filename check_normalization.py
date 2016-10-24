import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import Zernike as Zn
import scipy.special as spec
import math
#import sympy.mpmath as mp

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y



x, y = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
dx = x[1] - x[0]
xx, yy = np.meshgrid(x, y)
rho, theta = cart2pol(xx, yy)
j_max = 7
n, m = Zn.Zernike_j_2_nm(np.arange(2,j_max+2))
Cnm = Zn.complex_zernike(j_max, xx, yy)
mask = [xx**2 + yy**2 >= 1]
xn, yn = np.ma.array(xx, mask = mask), np.ma.array(yy, mask = mask)
tiled_mask = np.tile(mask, (Cnm.shape[2], 1,1)).T
Cnm_mask = np.ma.array(Cnm, mask = tiled_mask)
A = dx**2 * np.sum(Cnm_mask * np.conj(Cnm_mask), axis = (0,1))/np.pi

power_mat = Zn.Zernike_power_mat(j_max+1)
j = np.arange(2, j_max+2)
Znm = Zn.Zernike_xy(xx, yy, power_mat, j)
Znm_mask = np.ma.array(Znm, mask = tiled_mask)
A2 = dx**2 * np.sum(Znm_mask * Znm_mask, axis = (0,1)) / np.pi

fig, axarr = plt.subplots(2,2, sharex = True, sharey = True, figsize = plt.figaspect(1.))
axarr[0,0].contourf(xn, yn, Znm_mask[...,3], rstride=1, cstride=1, cmap=cm.YlOrBr, linewidth = 0)
axarr[0,0].set_title('Z_5')
axarr[1,0].contourf(xn, yn, Znm_mask[...,4], rstride=1, cstride=1, cmap=cm.YlOrBr, linewidth = 0)
axarr[1,0].set_title('Z_6')
axarr[0,1].contourf(xn, yn, np.sqrt(3)*np.real(Cnm_mask[...,3]), rstride=1, cstride=1, cmap=cm.YlOrBr, linewidth = 0)
axarr[0,1].set_title('Re(C_5)')
plot = axarr[1,1].contourf(xn, yn, np.sqrt(3)*np.imag(Cnm_mask[...,3]), rstride=1, cstride=1, cmap=cm.YlOrBr, linewidth = 0)
axarr[1,1].set_title('Im(C_5)')
print(n, m)
fig.subplots_adjust(right = 0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(plot, cax=cbar_ax)
plt.show()

print(A * (n+1), A2)

