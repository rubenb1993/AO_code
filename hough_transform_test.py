# -*- coding: utf-8 -*-
import sys
import os
import time

if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
if "H:\Desktop\numba" not in sys.path:
    sys.path.append("H:\Desktop\\numba")
##import MMCorePy
import PIL.Image
import numpy as np
import Hartmann as Hm
import displacement_matrix as Dm
import LSQ_method as LSQ
import mirror_control as mc
import matplotlib.pyplot as plt
import edac40
import math
import peakdetect
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from numba import jit


# Set up ranges
def make_jit_arrays(img):
    width, height = img.shape
    diag_len = math.ceil(math.sqrt(width * width + height * height))  # diagonal length of image
    rhos = np.linspace(-diag_len, diag_len, 2.0 * diag_len)  # x-axis for plot with hough transform
    thetas = np.deg2rad(np.arange(-180, 180))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    Acc = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_id, x_id = np.nonzero(img)
    return cos_t, sin_t, x_id, y_id, diag_len, Acc, num_thetas

def hough_jit(img, (cos_t, sin_t, x_id, y_id, diag_len, Acc, num_thetas)):
    # binning
    for i in range(len(x_id)):
        x = x_id[i]
        y = y_id[i]
        for j in range(num_thetas):
            rho = round(x * cos_t[j] + y * sin_t[j]) + diag_len
            Acc[rho, j] += 1
    return Acc

def hough_numpy(img, x, y):
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height)).astype(int)  # diagonal length of image
    rhos = np.linspace(-diag_len, diag_len, 2.0 * diag_len)  # x-axis for plot with hough transform
    
    # pre-compute angles
    thetas = np.linspace(0, np.pi, 360)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    Acc = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_id, x_id = np.nonzero(img)
    y_tr, x_tr = y[y_id], x[x_id]
    cos_tile, sin_tile = np.tile(cos_t, (len(x_id), 1)), np.tile(sin_t, (len(x_id), 1))
    x_tr_tile, y_tr_tile = np.tile(x_tr, (len(thetas), 1)).T, np.tile(y_tr, (len(thetas), 1)).T
    rho = np.round(x_tr_tile * cos_tile - y_tr_tile * sin_tile) + diag_len  # precompute rho
    rho = rho.astype(int)
    # binning more efficiently
    for j in range(len(x_id)):
        for i in range(num_thetas):
            Acc[rho[j, i], i] += 1

    return Acc, rhos, thetas, diag_len

def max_finding(Acc, rhos, thetas, lookahead = 15, delta = 30):
    idmax = np.argmax(Acc)
    rho_max_index, theta_max_index = np.unravel_index(idmax, Acc.shape)
    [max_peaks, min_peaks] = peakdetect.peakdetect(Acc[:, theta_max_index], x_axis = rhos, lookahead = lookahead, delta = delta)
    rho_max_i, y_max_i = zip(*max_peaks)
    s_index = np.argmax(y_max_i)
    s_max = rho_max_i[s_index]
    theta_max = thetas[theta_max_index]
    return rho_max_i, theta_max, s_max

def set_subplot_interferograms(*args):
    for arg in args:
        arg.get_xaxis().set_ticks([])
        arg.get_yaxis().set_ticks([])
        arg.axis('off')
        arg.set_title(titles[i], fontsize = 9, loc = 'left')

def make_colorbar(f, image):
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.3, 0.05, 0.4])
    cbar = f.colorbar(image, cax = cbar_ax)
    tick_locator = ticker.MaxNLocator(nbins = 5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_major_locator(ticker.AutoLocator())
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize = 6)
    cbar.set_label('Intensity level', size = 7)

### set up cameras and mirror
sh, int_cam = mc.set_up_cameras()
global mirror
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

## Given paramters for centroid gathering and displacing
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 375
r_sh_px = 410
r_sh_m = r_sh_px * px_size_int
j_max= 20          # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px

### images for making wavefront flat
actuators = 19
u_dm = np.zeros(actuators)
mc.set_displacement(u_dm, mirror)
time.sleep(0.2)

raw_input("Did you calibrate the reference mirror? Block DM")
sh.snapImage()
image_control = sh.getImage().astype(float)

raw_input("block reference mirror!")
sh.snapImage()
zero_image = sh.getImage().astype(float)

### make actual wf flat
u_dm, x_pos_norm_f, y_pos_norm_f, x_pos_zero_f, y_pos_zero_f, V2D = mc.flat_wavefront(u_dm, zero_image, image_control, r_sh_px, r_int_px, sh, mirror, show_accepted_spots = False)


### Choose abberations
a = np.zeros(j_max)
ind = np.array([0])
a[3] = 0.1 * wavelength
a[2] = 0.5 * wavelength
a[5] = -0.3 * wavelength
#a[4] = -0.25 * wavelength
#a[6] = -0.5 * wavelength
#a[3] = 0.5 * wavelength
#a[6] = 0.7 * wavelength

V2D_inv = np.linalg.pinv(V2D)
G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
v_abb = (f_sh/(r_sh_m * px_size_sh)) * np.dot(V2D_inv, np.dot(G, a))
u_dm -= v_abb
mc.set_displacement(u_dm, mirror)

raw_input("remove piece of paper")
time.sleep(0.2)
int_cam.snapImage()
image_i0 = int_cam.getImage().astype(float)
PIL.Image.fromarray(image_i0).save("20161125_interferograms_for_theory/interferogram_0.tif")

raw_input("tip and tilt 1")
time.sleep(1)
int_cam.snapImage()
image_i1 = int_cam.getImage().astype(float)
PIL.Image.fromarray(image_i1).save("20161125_interferograms_for_theory/interferogram_1.tif")

raw_input("tip and tilt 2")
time.sleep(1)
int_cam.snapImage()
image_i2 = int_cam.getImage().astype(float)
PIL.Image.fromarray(image_i2).save("20161125_interferograms_for_theory/interferogram_2.tif")


Id1 = image_i1 - image_i0
Id2 = image_i2 - image_i0

x0 = 550
y0 = 484
radius = 340
ny, nx = image_i0.shape
x, y = np.linspace(-1.0 * radius, 1.0 * radius, 2*radius), np.linspace(-1.0 * radius, 1.0 * radius, 2*radius)
xx, yy = np.meshgrid(x, y)
ss = np.sqrt(xx**2 + yy**2)


Id_int = np.zeros((2*radius, 2*radius, 3))
Im_i0 = np.zeros((2*radius, 2*radius, 3))
Im_i0[..., 0] = image_i0[y0-radius:y0+radius, x0-radius:x0+radius]
Im_i0[..., 1] = image_i1[y0-radius:y0+radius, x0-radius:x0+radius]
Im_i0[..., 2] = image_i2[y0-radius:y0+radius, x0-radius:x0+radius]

Id_int[..., 0] = Id1[y0-radius:y0+radius, x0-radius:x0+radius]
Id_int[..., 1] = Id2[y0-radius:y0+radius, x0-radius:x0+radius]

zeros_1, zeros_2 = np.abs(Id_int[..., 0]) <= 1, np.abs(Id_int[..., 1]) <= 1

Id_zeros = np.zeros(Id_int.shape, dtype = float)
Id_zeros[zeros_1, 0] = 1
Id_zeros[zeros_2, 1] = 1

mask = [np.sqrt((xx) ** 2 + (yy) ** 2) >= radius]
mask_tile = np.tile(mask, (Id_zeros.shape[-1],1,1)).T
Id_zeros_mask = np.ma.array(Id_zeros, mask=mask_tile)


### make Hough transform of all points
#Initialize constants
width, height = Id_int[...,0].shape
num_thetas = 360
diag_len = np.ceil(np.sqrt(width * width + height * height)).astype(int)
Acc = np.zeros((2 * diag_len, num_thetas, 2), dtype=np.uint64)
Lambda = np.zeros(2)
ti = np.zeros((2,2))
sigma = np.zeros(2)
x_shape = list(xx.shape)
x_shape.append(2)
tau_i = np.zeros(x_shape)
Id_hat = np.zeros(Id_int.shape)
theta_max = np.zeros(2)
print("making hough transforms... crunch crunch")

for jj in range(2):
    Acc[...,jj], rhos, thetas, diag_len = hough_numpy(Id_zeros_mask[..., jj], x, y)     
    rho_max_i, theta_max[jj], s_max = max_finding(Acc[...,jj], rhos, thetas)
    Lambda[jj] = np.sum(np.diff(rho_max_i), dtype=float) / (len(rho_max_i)-1.0)
    ti[...,jj] = (2 * np.pi/ Lambda[jj]) * np.array([np.cos(theta_max[jj]), np.sin(theta_max[jj])])
    sigma[jj] = 2 * np.pi * s_max / Lambda[jj]
    tau_i[...,jj] = -ti[0, jj] * xx + ti[1, jj] * yy
    
    sin_shift = np.zeros(tau_i.shape)
    sin_shift[...,jj] = np.sin((tau_i[...,jj] + sigma[...,jj])/2.0)
    Id_hat[..., jj] = Id_int[..., jj]/(-2.0 * sin_shift[...,jj])



yn = raw_input("Plot? y/n")
if (yn == 'y'):
    mask = [np.sqrt((xx) ** 2 + (yy) ** 2) >= radius]
    titles = [r'a)', r'b)', r'c)']
    f, axarr = plt.subplots(1,3, figsize=(4.98,3.07), frameon = False)
    for i in range(3):
        im1 = axarr[i].imshow(np.ma.array(Im_i0[..., i], mask = mask), cmap = 'bone', vmin = 0, vmax = 170)
        set_subplot_interferograms(axarr[i])
    make_colorbar(f, im1)
    f2, axarr2 = plt.subplots(1,2, figsize = (4.98 * 0.66, 3.07))
    f3, axarr3 = plt.subplots(1,2, figsize = (4.98 * 0.66, 3.07))
    f4, axarr4 = plt.subplots(2, 1, figsize = (4.98, 3.07))
    f5, axarr5 = plt.subplots(1,2, figsize = (4.98 * 0.66, 3.07))
    for i in range(2):
        #differences
        im2 = axarr2[i].imshow(np.ma.array(Id_int[..., i], mask = mask), cmap = 'bone', vmin = -170, vmax = 170)
        set_subplot_interferograms(axarr2[i])
        #zeros
        im3 = axarr3[i].imshow(np.ma.array(Id_zeros[..., i], mask = mask), cmap = 'bone_r')
        set_subplot_interferograms(axarr3[i])
        ## Hough transform
        no_ticks_x = np.arange(0,5)
        no_ticks_y = np.arange(0,5,2)
        tick_size_x, tick_size_y = (np.array(Acc[...,i].shape) -1)/4.0 #first x than y because it is already transposed
        x_ticks_orig, y_ticks_orig = np.rint(tick_size_x * no_ticks_x).astype(int), np.rint(tick_size_y * no_ticks_y).astype(int)
        x_ticks_new = np.round(rhos[x_ticks_orig]).astype(int)
        y_ticks_new = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']
        x_ticks_new[2] = 0.0
        im4 = axarr4[i].imshow(Acc[...,i].T, interpolation='none', cmap = 'bone', origin = 'lower')
        axarr4[i].set_xticks(x_ticks_orig)
        axarr4[i].set_xticklabels(x_ticks_new, fontsize = 9)
        axarr4[i].set_yticks(y_ticks_orig)
        axarr4[i].set_yticklabels(y_ticks_new, fontsize = 9)
        axarr4[i].set_title(titles[i], fontsize = 9, loc = 'left')
        ## sinuses
        im5 = axarr5[i].imshow(np.ma.array(Id_hat[..., i], mask = mask), cmap = 'bone', vmin = -170, vmax = 170)
        set_subplot_interferograms(axarr5[i])
    make_colorbar(f2, im2)
    make_colorbar(f5, im5)
    axarr4[1].set_xlabel(r'$\rho$')
    axarr4[1].set_ylabel(r'$\theta$')
    f.savefig('original_interferograms.png', bbox_inches='tight', dpi = 1200)
    f2.savefig('difference_interferograms.png', bbox_inches = 'tight', dpi = 1200)
    f3.savefig('zeros_diff_interferograms.png', bbox_inches = 'tight', dpi = 1200)
    f4.savefig('hough_accumulator.png', bbox_inches = 'tight', dpi = 1200)
    f5.savefig('I_hat_interferograms.png', bbox_inches = 'tight', dpi = 1200)
    plt.show() 
else:
    print("input not y, assuming n")
    #break

    
d1 = (tau_i[..., 0] + sigma[0])/2.0
d2 = (tau_i[..., 1] + sigma[1])/2.0
sin_d1_d2 = np.sin(d1-d2)
sin_d1 = np.sin(d1)
sin_d2 = np.sin(d2)
zero_line1 = np.abs(sin_d1) <= 5e-2
zero_line2 = np.abs(sin_d2) <= 5e-2
zero_line3 = np.abs(sin_d1_d2) <= 5e-2
#Id_zeros[zeros_3, 2] = 1
Id_zeros_all = np.zeros((2*radius, 2*radius))
Id_zeros_all[zero_line1] = 1
Id_zeros_all[zero_line2] = 1
Id_zeros_all[zero_line3] = 1
stack_mask = np.dstack((zero_line1, zero_line2, zero_line3, np.squeeze(mask)))
complete_mask = np.any(stack_mask, axis = 2)
assert complete_mask.shape == np.squeeze(mask).shape

atanshape = list(Id_hat[...,0].shape)
atany = np.zeros(atanshape)
#atany2 = np.zeros(atanshape)
angfact = np.zeros(d1.shape)
angfact = d1 - d2
atany = Id_hat[...,1]
org_phase = np.zeros(d1.shape)
#f, axarr = plt.subplots(3,4)
f, axarr = plt.subplots(2,3, figsize=(9.31,5.91))
axarr[0, 0].imshow(np.ma.array(Im_i0[...,2], mask = mask), cmap = 'bone')
axarr[0, 0].set_title('Original 2nd shifted', fontsize = 9)
atanx =(Id_hat[...,0] - np.cos(angfact) * Id_hat[...,1])/ np.sin(angfact)
org_phase = np.arctan2(atany, atanx)
    
axarr[0, 1].imshow(np.ma.array(np.cos(org_phase), mask = mask), cmap = 'bone')
axarr[0, 1].set_title('recovered interferogram', fontsize = 9)
axarr[0, 2].imshow(np.ma.array(org_phase, mask = mask), cmap = 'bone')
axarr[0, 2].set_title('recovered (wrapped) phase', fontsize = 9)
axarr[1, 0].imshow(np.ma.array(Id_zeros_all, mask = mask), cmap = 'bone')
axarr[1, 0].set_title('mask', fontsize = 9)
axarr[1, 1].imshow(np.ma.array(org_phase, mask = complete_mask), cmap = 'bone')
axarr[1, 1].set_title('masked phase', fontsize = 9)
axarr[1, 2].imshow(np.cos(np.ma.array(org_phase, mask = complete_mask)), cmap = 'bone')
axarr[1, 2].set_title('int masked phase', fontsize = 9)
for i in range(6):
    j = np.unravel_index(i, axarr.shape)
    axarr[j].get_xaxis().set_ticks([])
    axarr[j].get_yaxis().set_ticks([])

plt.show()#block=False)

#raw_input('savefig?')
#plt.savefig('wrong_phase_interferogram.png', bbox_inches='tight')

