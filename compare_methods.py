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
import phase_extraction as PE
import phase_unwrapping_test as pw

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=True)

## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 340
r_sh_px = 370
r_sh_m = r_sh_px * px_size_int
j_max= 20          # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px

## image making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)

## centre and radius of interferogam. Done by eye, with the help of define_radius.py
x0 = 550
y0 = 484
radius = int(340)

## pack everything neatly in 1 vector against clutter
constants = np.stack((px_size_sh, px_size_int, f_sh, r_int_px, r_sh_px, r_sh_m, j_max, wavelength, box_len, x0, y0, radius))

x_pcg, y_pcg = np.linspace(-1, 1, 2*radius), np.linspace(-1, 1, 2*radius)
N = len(x_pcg)
i, j = np.linspace(0, N-1, N), np.linspace(0, N-1, N)
xx_alg, yy_alg = np.meshgrid(x_pcg, y_pcg)
ii, jj = np.meshgrid(i, j)


### gather phase from interferograms and sh spots
org_phase, delta_i, sh_spots, inter_0 = PE.phase_extraction(constants)
Un_sol = np.triu_indices(delta_i.shape[-1], k = 1) ## delta i has the shape of the amount of difference interferograms, while org phase has all possible combinations
org_unwr = np.zeros(org_phase.shape)

## unwrap phases with dct method
for k in range(org_phase.shape[-1]):
    org_unwr[...,k] = pw.unwrap_phase_dct(org_phase[..., k], xx_alg, yy_alg, ii, jj, N, N)
    org_unwr[...,k] -= delta_i[..., Un_sol[0][k]]

## make mask and find mean within mask
mask = [np.sqrt((xx_alg) ** 2 + (yy_alg) ** 2) >= 1]
mask_tile = np.tile(mask, (org_phase.shape[-1],1,1)).T
org_mask = np.ma.array(org_unwr, mask = mask_tile)
mean_unwr = org_mask.mean(axis=(0,1))
org_unwr -= mean_unwr

## smoothing interferogram due to median
org_med = np.median(org_unwr, axis = 2)
xy_inside = np.where(np.sqrt(xx_alg**2 + yy_alg**2) <= 1)
org_med_flat = org_med[xy_inside]
x_in, y_in = xx_alg[xy_inside], yy_alg[xy_inside]

j_max = 30
j = np.arange(2, j_max) #start at 2, end at j_max-1
power_mat = Zn.Zernike_power_mat(j_max)
Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j)
Zernike_2d = np.zeros((len(x_in), j_max)) 
for i in range(len(j)):
    Zernike_2d[:, i] = Zernike_3d[...,i].flatten()

print("fitting!")
a_inter, a_inter_res = np.linalg.lstsq(Zernike_2d, org_med_flat)[0:2]
a_inter *= wavelength
a_inter_res *= wavelength

f, ax = plt.subplots(2,2)
ax[0,0].imshow(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], cmap = 'bone', origin = 'lower')
ax[1,0].imshow(org_med, cmap = 'jet', origin = 'lower')
Zn.plot_zernike(j_max, a_inter, ax = ax[1,1])
Zn.plot_interferogram(j_max, a_inter, ax = ax[0,1])
plt.show()
