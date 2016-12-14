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
import scipy.optimize as opt

def rms_piston(piston, *args):
    """function for rms value to be minimized. args should contain j_max, a_filt, N, Z_mat, orig, mask in that order"""
    if args:
        j_max, a_filt, N, Z_mat, orig, mask = args
    else:
        print("you should include the right arguments!")
        return
    orig = np.ma.array(orig, mask = mask)
    inter = np.ma.array(Zn.int_for_comp(j_max, a_filt, N, piston, Z_mat), mask = mask)
    rms = np.sqrt(np.sum((inter - orig)**2)/N**2)
    return rms


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
j_max= 30         # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px

## image making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)
fold_name = "20161213_new_inters/"

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
new_img = raw_input('take new image?')
if new_img == 'y':
    new_img = True
else:
    new_img = False
org_phase, delta_i, sh_spots, inter_0, flat_wf = PE.phase_extraction(constants, take_new_img = new_img, folder_name = fold_name, show_id_hat = False, show_hough_peaks = False)
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

##f, ax = plt.subplots(1,6)
##for i in range(6):
##    ax[i].imshow(np.ma.array(org_unwr[...,i], mask = mask), vmin = -15, vmax = 15)
##    
##plt.show()

## smoothing interferogram due to median
org_med = np.median(org_unwr, axis = 2)
xy_inside = np.where(np.sqrt(xx_alg**2 + yy_alg**2) <= 1)
org_med_flat = org_med[xy_inside]
x_in, y_in = xx_alg[xy_inside], yy_alg[xy_inside]

j = np.arange(2, j_max) #start at 2, end at j_max-1
power_mat = Zn.Zernike_power_mat(j_max+2)
Zernike_3d = Zn.Zernike_xy(x_in, y_in, power_mat, j) ## create a 3d matrix with values of zernike polynomials at x_in and y_in 
Zernike_2d = np.zeros((len(x_in), j_max)) ## flatten to perform least squares fit
for i in range(len(j)):
    Zernike_2d[:, i] = Zernike_3d[...,i].flatten()

a_inter = np.linalg.lstsq(Zernike_2d, org_med_flat)[0]
a_inter *= wavelength
orig = np.ma.array(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask)


f, ax = plt.subplots(2,2)
ax[0,0].imshow(np.ma.array(orig, mask = mask), cmap = 'bone', origin = 'lower')
ax[0,0].set_title('original')
ax[1,0].set_title('median')
ax[1,1].set_title('fitted')
ax[0,1].set_title('from fit')
ax[1,0].imshow(np.ma.array(org_med, mask = mask), cmap = 'jet', origin = 'lower')
Zn.plot_zernike(j_max, a_inter, ax = ax[1,1])
Zn.plot_interferogram(j_max, a_inter, ax = ax[0,1])

for axes in ax.reshape(-1):
    axes.set(adjustable='box-forced', aspect='equal')

### using only 1 phase, but smoothing
phase = org_phase[..., 0]
shape = list(org_phase.shape)
shape[2] = 10
filt_phase = np.zeros(org_phase.shape)
save_phase = np.zeros(shape)
unwr_phase = np.zeros(org_phase.shape)
a_filt = np.zeros((j_max, shape[2]))

org_mask = np.ma.array(org_unwr, mask = mask_tile)
mean_unwr = org_mask.mean(axis=(0,1))

filt_yn = raw_input("do you want to filter the phase?")
if filt_yn == 'y':
    f, axarr = plt.subplots(3,shape[2])
    v = np.linspace(-10, 20)

    for i in range(shape[2]):
        print("filtering with a " + str(3+ 2*i) + " x " + str(3 + 2*i) + " filter")
        for k in range(org_phase.shape[-1]):
            print(k)
            filt_phase[..., k] = pw.filter_wrapped_phase(org_phase[..., k], 3 + 2*i)
            unwr_phase[..., k] = pw.unwrap_phase_dct(filt_phase[..., k], xx_alg, yy_alg, ii, jj, N, N)
            unwr_phase[..., k] -= delta_i[..., Un_sol[0][k]]

        unwr_mask = np.ma.array(unwr_phase, mask = mask_tile)
        mean_unwr = unwr_mask.mean(axis = (0, 1))
        unwr_phase -= mean_unwr
        
        fit_phase = np.median(unwr_phase, axis=2)
        save_phase[...,i] = fit_phase
        fit_phase = save_phase[..., i]
        
        a_filt[:, i] = np.linalg.lstsq(Zernike_2d, fit_phase[xy_inside])[0]
        a_filt[:, i] *= wavelength
    np.save(fold_name + "coefficients", a_filt)
    np.save(fold_name + "filtered_phases", save_phase)
else:
    save_phase = np.load(fold_name + "filtered_phases.npy")
    a_filt = np.load(fold_name + "coefficients.npy")
##
##for i in range(shape[2]):
##    fit_phase = save_phase[...,i]
##    a_filt[:, i] = np.linalg.lstsq(Zernike_2d, fit_phase[xy_inside])[0]
##    a_filt[:, i] *= wavelength
##    f, ax = plt.subplots(2,2)
##    ax[0,0].imshow(np.ma.array(inter_0[y0-radius:y0+radius, x0-radius:x0+radius], mask = mask), cmap = 'bone', origin = 'lower')
##    ax[0,0].set_title('original')
##    ax[1,0].set_title('median ' + str(3 + 2*i) + " x " + str(3  + 2*i))
##    ax[1,1].set_title('fitted')
##    ax[0,1].set_title('from fit')
##    ax[1,0].imshow(np.ma.array(save_phase[..., i], mask = mask), cmap = 'jet', origin = 'lower', vmin = -10, vmax = 20)
##    Zn.plot_zernike(j_max, a_filt[:, i], ax = ax[1,1], v = v)
##    Zn.plot_interferogram(j_max, a_filt[:, i], ax = ax[0,1])
##    for axes in ax.reshape(-1):
##        axes.set(adjustable='box-forced', aspect='equal')


    
    #axarr[0, i].imshow(np.ma.array(fit_phase, mask= mask), vmin = -15, vmax = 15, origin = 'lower')
    #Zn.plot_zernike(j_max, a_filt[:,i], ax = axarr[1,i], v=v)
    #Zn.plot_interferogram(j_max, a_filt[:,i], ax= axarr[2,i])


    
#indices_max = np.argsort(np.abs(a_filt[:,-1]))[-5:-1]
pistons = np.linspace(0, 2*np.pi, num = 30)
inters = np.zeros((N, N, len(pistons), a_filt.shape[-1]))
rms = np.zeros(len(pistons))
xi, yi = np.linspace(-1, 1, N), np.linspace(-1, 1, N)
xi, yi = np.meshgrid(xi, yi)
#power_mat = Zn.Zernike_power_mat(j_max+2)
j_range = np.arange(2, j_max+2)
Z_mat = Zn.Zernike_xy(xi, yi, power_mat, j_range)
Z = np.zeros((N, N, len(pistons), a_filt.shape[-1]))
orig /= np.max(orig) * 0.9
orig[orig > 1] = 1

mins = np.zeros(a_filt.shape[-1])
f, ax = plt.subplots(2, 5)
f2, ax2 = plt.subplots(1,1)
indexes = np.unravel_index(np.arange(10), (2, 5))
print("optimizinggg")
for i in range(len(mins)):
    mins[i] = opt.fmin(rms_piston, 0, args = (j_max, a_filt[:,i], N, Z_mat, orig, mask))
    ax[indexes[0][i], indexes[1][i]].imshow(orig - np.ma.array(Zn.int_for_comp(j_max, a_filt[:,i], N, mins[i], Z_mat), mask = mask), vmin = -1, vmax = 0.25, origin = 'lower')

rms_vec = np.zeros(len(mins))
for i in range(len(mins)):
    rms_vec[i] = rms_piston(mins[i], j_max, a_filt[:,i], N, Z_mat, orig, mask)

### Gather with SH patterns
zero_image = sh_spots[..., 1]
zero_image_zeros = np.copy(sh_spots[..., 1])
dist_image = sh_spots[..., 2]
image_control = sh_spots[..., 0]
[ny,nx] = zero_image.shape
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)

x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image_zeros)
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (box_len/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = mc.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)
G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)

a_lsq = LSQ.LSQ_coeff(x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, G, image_control, dist_image, px_size_sh, r_sh_px, f_sh, j_max) 
a_janss = janssen.coeff(x_pos_zero_f, y_pos_zero_f, image_control, dist_image, px_size_sh, f_sh, r_sh_m, j_max)
a_janss = np.real(a_janss)

pist_lsq = opt.fmin(rms_piston, 0, args = (j_max, a_lsq, N, Z_mat, orig, mask))
pist_janss = opt.fmin(rms_piston, 0, args = (j_max, np.real(a_janss), N, Z_mat, orig, mask))


f, ax = plt.subplots(2,2)
ax[0,0].imshow(np.ma.array(orig, mask = mask), vmin = 0, vmax = 1, cmap = 'bone', origin = 'lower')
Zn.plot_interferogram(j_max, a_filt[:,9], piston = mins[9], ax = ax[0,1])
Zn.plot_interferogram(j_max, a_lsq, piston = pist_lsq, ax = ax[1,0], fliplr = True)
Zn.plot_interferogram(j_max, a_janss, piston = pist_janss, ax = ax[1,1], fliplr = True)
ax[0,0].set_title('original')
ax[0,1].set_title('from int')
ax[1,0].set_title('from lsq')
ax[1,1].set_title('from janss')




ax2.plot(rms_vec)
plt.show()
##for j in range(a_filt.shape[-1]):
##    print(j)
##    for i in range(len(pistons)):
##        piston = pistons[i]
##        inters[...,i, j] = Zn.int_for_comp(j_max, a_filt[:,j], N, piston, Z_mat)
##
##
##mask_inter = np.tile(mask, (a_filt.shape[-1], len(pistons), 1, 1)).T
##inters = np.ma.array(inters, mask = mask_inter)
##orig_tile = np.tile(orig, (a_filt.shape[-1], len(pistons), 1, 1)).transpose(2, 3, 1, 0)
##print('cruncing rms')
##rms = np.sqrt(np.sum((inters - orig_tile)**2, axis = (0,1))/N**2)
##print('done cruncing')

##f, ax = plt.subplots(1, 2)
##ax[0].imshow(orig_tile[...,0], cmap = 'bone', origin = 'lower')
##Zn.plot_interferogram(j_max, a_filt[:,9], piston = pistons[np.argmin(rms)], ax = ax[1])
##f, ax = plt.subplots(1,1)
##
##for i in range(a_filt.shape[-1]):
##    ax.plot(rms[:, i])

##diff = np.linalg.norm(np.diff(a_filt, axis = 1), axis = 0)/np.linalg.norm(a_filt[:,:-1], axis = 0)
##f, ax = plt.subplots(1,1)
##k = np.arange(1,shape[2])
##ax.scatter(3 + 2*k, diff)
#plt.show()
