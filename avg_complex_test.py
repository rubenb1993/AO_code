import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import mirror_control as mc
import edac40
import MMCorePy
import PIL.Image
import Hartmann as Hm
import Zernike as Zn
import numpy as np
import PIL.Image
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import cm
import LSQ_method as LSQ
from scipy.interpolate import griddata


# Test figures
impath_zero = os.path.abspath("dm_sh.tif")
impath_dist = os.path.abspath("ref_sh.tif")
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)

[ny,nx] = zero_image.shape
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 370
r_sh_m = r_int_px * px_size_int
r_sh_px = r_sh_m / px_size_sh
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max = 11
j_range = np.arange(2, j_max+2)
len_box = 70/r_sh_px
half_len_box = 35/r_sh_px


x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, dist_image, xx, yy)
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)

x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px

xi, yi = np.linspace(-1, 1, 2*r_int_px), np.linspace(-1, 1, 2*r_int_px)
xi, yi = np.meshgrid(xi, yi)
xn, yn = np.ma.masked_where(xi**2 + yi**2 >= 1, xi), np.ma.masked_where(xi**2 + yi**2 >= 1, yi)
power_mat = Zn.Zernike_power_mat(j_max+2)


C_nm = Zn.complex_zernike(j_max, x_pos_norm, y_pos_norm)
mask = [x_pos_norm**2 + y_pos_norm**2 >= 1]
tiled_mask = np.tile(mask, (C_nm.shape[1],1)).T

C_nm_mask = np.ma.array(C_nm, mask=tiled_mask)
##plt.contourf(xn, yn, Zxn[...,4], rstride=1, cstride=1, cmap=cm.gray, linewidth = 0)
##cbar = plt.colorbar()
##plt.show()
spot_size = 35.0
half_len_box = spot_size/r_int_px
box_px = 2*spot_size
x_left = x_pos_norm - half_len_box
x_right = x_pos_norm + half_len_box
y_up = y_pos_norm + half_len_box
y_down = y_pos_norm - half_len_box
j_range = np.arange(1, j_max+2)
Cnm_avg = np.zeros((len(x_pos_norm), len(j_range)), dtype = np.complex_)
for ii in range(len(x_pos_norm)):
    x, y = np.linspace(x_left[ii], x_right[ii], box_px), np.linspace(y_down[ii], y_up[ii], box_px)
    xx, yy = np.meshgrid(x, y)
    mask = [xx**2 + yy**2 >= 1]
    Cnm_xy = Zn.complex_zernike(j_max, xx, yy)
    tiled_mask = np.tile(mask, (Cnm_xy.shape[2],1,1)).T
    Cnm_mask = np.ma.array(Cnm_xy, mask = tiled_mask)
    Cnm_avg[ii,:] = np.sum(Cnm_mask, axis = (0,1))/box_px**2
    
G_2 = LSQ.geometry_matrix_2(x_pos_norm, y_pos_norm, j_max, r_sh_px)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize = plt.figaspect(0.5), sharex = True, sharey = True)
curAx = plt.gca()
ax1.scatter(x_pos_norm, y_pos_norm)

circle1 = plt.Circle([0,0] , 1, color = 'k', fill=False)
circle2 = plt.Circle([0,0] , 1, color = 'k', fill=False)
for i in range(len(x_pos_flat)):
    ax1.add_patch(Rectangle((x_pos_norm[i] - half_len_box, y_pos_norm[i] - half_len_box), len_box, len_box, alpha=1, facecolor='none'))
    ax2.add_patch(Rectangle((x_pos_norm[i] - half_len_box, y_pos_norm[i] - half_len_box), len_box, len_box, alpha=1, facecolor='none'))

ax1.add_artist(circle1)
zi = griddata((x_pos_norm, y_pos_norm), np.real(Cnm_avg[:,5]), (xi, yi), method='linear')
zj = griddata((x_pos_norm, y_pos_norm), np.real(C_nm_mask[:, 5]), (xi, yi), method='linear')
ax1.imshow(zi, vmin=np.real(Cnm_avg[:,5]).min(), vmax=np.real(Cnm_avg[:,5]).max(), origin='lower',
           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
curAx.set_xlim([-1.2, 1.2])
curAx.set_ylim([-1.2, 1.2])

ax2.scatter(x_pos_norm, y_pos_norm)
ax2.add_artist(circle2)
im = ax2.imshow(zj, vmin=np.real(Cnm_avg[:,5]).min(), vmax=np.real(Cnm_avg[:,5]).max(), origin='lower',
           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
cbar_ax = fig.add_axes([0.91, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
ax1.set_title('with edge correction')
ax2.set_title('without edge correction')
plt.show()
