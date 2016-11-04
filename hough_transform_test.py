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

#from numba import jit


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
    diag_len = np.ceil(np.sqrt(width * width + height * height))  # diagonal length of image
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

##impath_i0 = os.path.abspath("int_test_i0.tif")
##image_i0 = np.asarray(PIL.Image.open(impath_i0)).astype(float)
##impath_i1 = os.path.abspath("int_test_i1.tif")
##image_i1 = np.asarray(PIL.Image.open(impath_i1)).astype(float)
##impath_i2 = os.path.abspath("int_test_i2.tif")
##image_i2 = np.asarray(PIL.Image.open(impath_i2)).astype(float)

### set up cameras and mirror
sh, int_cam = mc.set_up_cameras()
global mirror
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

## Given paramters for centroid gathering and displacing
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 340
r_sh_px = 370
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
a[8] = 0.4 * wavelength
#a[2] = 2 * wavelength
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

raw_input("first tip tilt")
u_dm[4] -= 0.25
u_dm[7] -= 0.45
mc.set_displacement(u_dm, mirror)
time.sleep(2)
int_cam.snapImage()
image_i1 = int_cam.getImage().astype(float)

raw_input("second tip tilt")
u_dm[4] -= 0.2
u_dm[7] += 0.15
mc.set_displacement(u_dm, mirror)
time.sleep(2)
int_cam.snapImage()
image_i2 = int_cam.getImage().astype(float)

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

width, height = Id_int[...,0].shape
num_thetas = 360
diag_len = np.ceil(np.sqrt(width * width + height * height))
Acc = np.zeros((2 * diag_len, num_thetas, 2), dtype=np.uint64)
##Acc1, rhos, thetas, diag_len = hough_numpy(Id1_zeros_mask, x, y)  # make_jit_arrays(Id1_zeros_mask))
##Acc2, rhos, thetas, diag_len = hough_numpy(Id2_zeros_mask, x, y) 

#f, ax = plt.subplots(2, 1)
Lambda = np.zeros(2)
ti = np.zeros((2,2))
sigma = np.zeros(2)
x_shape = list(xx.shape)
x_shape.append(2)
tau_i = np.zeros(x_shape)
#tau_i2 = np.zeros(x_shape)
Id_hat = np.zeros(Id_int.shape)
#Id_hat2 = np.zeros(Id_int.shape)

##f2, ax2 = plt.subplots(3,3)
##ax2[0,0].imshow(np.ma.array(Im_i0[...,0], mask=mask), cmap = 'bone')
for jj in range(2):
    Acc[...,jj], rhos, thetas, diag_len = hough_numpy(Id_zeros_mask[..., jj], x, y) 
    #ax[jj].imshow(Acc[...,jj].T, extent=[np.min(rhos), np.max(rhos), np.max(thetas), np.min(thetas)], interpolation='none',
          #aspect='auto')
##    ax2[0,jj+1].imshow(np.ma.array(Im_i0[..., jj+1], mask = mask), cmap = 'bone' )
    
    rho_max_i, theta_max, s_max = max_finding(Acc[...,jj], rhos, thetas)
    Lambda[jj] = np.sum(np.diff(rho_max_i), dtype=float) / (len(rho_max_i)-1.0)
    ti[...,jj] = (2 * np.pi/ Lambda[jj]) * np.array([np.cos(theta_max), np.sin(theta_max)])
    sigma[jj] = 2 * np.pi * s_max / Lambda[jj]
    tau_i[...,jj] = -ti[0, jj] * xx + ti[1, jj] * yy
    #tau_i2[..., jj] = ti[0, jj] * xx - ti[1, jj] * yy
    
    sin_ud_lr = np.zeros(tau_i.shape)
    sin_ud_lr[...,0] = np.sin((tau_i[...,jj] + sigma[...,jj])/2.0)
    #sin_ud_lr[...,1] = np.sin((tau_i2[...,jj] + sigma[...,jj])/2.0)
    Id_hat[..., jj] = Id_int[..., jj]/(-2.0 * sin_ud_lr[...,0])
    #Id_hat2[..., jj] = Id_int[..., jj]/(-2.0 * sin_ud_lr[..., 1])
##    ax2[1, jj+1].imshow(np.ma.array(Id_hat[..., jj], mask = mask), vmin = -100, vmax = 100, cmap = 'bone')
##    ax2[2, jj+1].imshow(np.ma.array(Id_hat2[..., jj], mask = mask), vmin = -100, vmax = 100, cmap = 'bone')
    
d1 = (tau_i[..., 0] + sigma[0])/2.0
d2 = (tau_i[..., 1] + sigma[1])/2.0
#d3 = (tau_i2[..., 0] + sigma[0])/2.0
#d4 = (tau_i2[..., 1] + sigma[1])/2.0
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

dshape = list(d1.shape)
dshape.append(4)
atanshape = list(Id_hat[...,0].shape)
atanshape.append(4)
atany = np.zeros(atanshape)
#atany2 = np.zeros(atanshape)
angfact = np.zeros(dshape)
angfact[...,0] = d1 - d2
angfact[...,1] = d1 + d2
angfact[...,2] = -d1 + d2
angfact[...,3] = -(d1 + d2)
#angfact2 = np.zeros(dshape)
#angfact2[...,0] = d3 - d4
#angfact2[...,1] = d3 + d4
#angfact2[...,2] = -d3 + d4
#angfact2[...,3] = -(d3 + d4)
atanx = Id_hat[...,1]
org_phase = np.zeros(dshape)
#f, axarr = plt.subplots(3,4)
f, axarr = plt.subplots(2,3, figsize=(9.31,5.91))
axarr[0, 0].imshow(np.ma.array(Im_i0[...,0], mask = mask), cmap = 'bone')
axarr[0, 0].set_title('Original interferogram', fontsize = 9)
for i in range(4):
    atany[...,i] =(Id_hat[...,0] - np.cos(angfact[...,i]) * Id_hat[...,1])/ np.sin(angfact[...,i])
    #atany2[...,i] = (Id_hat2[...,0] - np.cos(angfact2[...,i]) * Id_hat2[...,1])/ np.sin(angfact2[...,i])
##    axarr[0,i].imshow(np.ma.array(atany[...,i], mask = mask), cmap = 'bone', vmin = -100, vmax = 100)
##    axarr[1,i].imshow(np.sin(angfact[...,i]), cmap = 'bone')
    org_phase[...,i] = np.arctan2(atanx, atany[...,i]) - d2
##    axarr[2,i].imshow(np.ma.array(np.cos(org_phase[..., i]), mask=mask), cmap = 'bone')
    
axarr[0, 1].imshow(np.ma.array(np.cos(org_phase[..., 0]), mask = mask), cmap = 'bone')
axarr[0, 1].set_title('recovered interferogram', fontsize = 9)
axarr[0, 2].imshow(np.ma.array(org_phase[...,0], mask = mask), cmap = 'bone')
axarr[0, 2].set_title('recovered (wrapped) phase', fontsize = 9)
axarr[1, 0].imshow(np.ma.array(Id_zeros_all, mask = mask), cmap = 'bone')
axarr[1, 0].set_title('mask', fontsize = 9)
axarr[1, 1].imshow(np.ma.array(org_phase[..., 0], mask = complete_mask), cmap = 'bone')
axarr[1, 1].set_title('masked phase', fontsize = 9)
axarr[1, 2].imshow(np.cos(np.ma.array(org_phase[..., 0], mask = complete_mask)), cmap = 'bone')
axarr[1, 2].set_title('int masked phase', fontsize = 9)
for i in range(6):
    j = np.unravel_index(i, axarr.shape)
    axarr[j].get_xaxis().set_ticks([])
    axarr[j].get_yaxis().set_ticks([])

plt.show(block=False)

raw_input('savefig?')
plt.savefig('phase_retrieval_interferogram.png', bbox_inches='tight')

