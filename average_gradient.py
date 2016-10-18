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


Zx, Zy = Zn.xder_brug(xi, yi, power_mat, j_range), Zn.yder_brug(xi, yi, power_mat, j_range)
mask = [xi**2 + yi**2 >= 1]
tiled_mask = np.tile(mask, (Zx.shape[2],1,1)).T

Zxn, Zyn = np.ma.array(Zx, mask=tiled_mask), np.ma.array(Zy, mask=tiled_mask)
##plt.contourf(xn, yn, Zxn[...,4], rstride=1, cstride=1, cmap=cm.gray, linewidth = 0)
##cbar = plt.colorbar()
##plt.show()
x_left = x_pos_norm - half_len_box
x_right = x_pos_norm + half_len_box
y_up = y_pos_norm + half_len_box
y_down = y_pos_norm - half_len_box
G = np.zeros((2*len(x_pos_zero), len(j_range)))
G_test = np.zeros((2*len(x_pos_zero), len(j_range)))
for ii in range(len(x_pos_zero)):
    x = np.linspace(x_left[ii], x_right[ii], 70)
    y = np.linspace(y_down[ii], y_up[ii], 70)
    xx, yy = np.meshgrid(x, y)
    mask = [xx**2 + yy**2 >= 1]
    Zx, Zy = Zn.xder_brug(xx, yy, power_mat, j_range), Zn.yder_brug(xx, yy, power_mat, j_range)
    tiled_mask = np.tile(mask, (Zx.shape[2],1,1)).T
    Zxn, Zyn = np.ma.array(Zx, mask=tiled_mask), np.ma.array(Zy, mask=tiled_mask)
    G[ii, :] = np.sum(Zxn, axis = (0,1))/70**2
    G[ii+len(x_pos_zero), :] = np.sum(Zyn, axis = (0,1))/70**2
    G_test[ii, :] = np.sum(Zx, axis=(0,1))/70**2
    G_test[ii+len(x_pos_zero), :] = np.sum(Zy, axis=(0,1))/70**2
    
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
zi = griddata((x_pos_norm, y_pos_norm), G[:len(x_pos_norm),3], (xi, yi), method='linear')
zj = griddata((x_pos_norm, y_pos_norm), G_test[:len(x_pos_norm), 3], (xi, yi), method='linear')
ax1.imshow(zi, vmin=G[:len(x_pos_norm),3].min(), vmax=G[:len(x_pos_norm),3].max(), origin='lower',
           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
curAx.set_xlim([-1.2, 1.2])
curAx.set_ylim([-1.2, 1.2])

ax2.scatter(x_pos_norm, y_pos_norm)
ax2.add_artist(circle2)
im = ax2.imshow(zj, vmin=G_test[:len(x_pos_norm),3].min(), vmax=G_test[:len(x_pos_norm),3].max(), origin='lower',
           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
cbar_ax = fig.add_axes([0.91, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
ax1.set_title('with edge correction')
ax2.set_title('without edge correction')
plt.show()
