import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
#from matplotlib import rc
import Hartmann as Hm
import displacement_matrix as Dm
import LSQ_method as LSQ
import mirror_control as mc
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker
import math
from scipy import signal
import peakdetect
#from numba import jit

def make_jit_arrays(img):
    width, height = img.shape
    diag_len = math.ceil(math.sqrt(width*width + height*height)) #diagonal length of image
    rhos = np.linspace(-diag_len, diag_len, 2.0*diag_len) #x-axis for plot with hough transform
    thetas = np.deg2rad(np.arange(-180,180))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    Acc = np.zeros((2 * diag_len, num_thetas), dtype = np.uint64)
    y_id, x_id = np.nonzero(img)
    return cos_t, sin_t, x_id, y_id, diag_len, Acc, num_thetas


def hough_jit(img, (cos_t, sin_t, x_id, y_id, diag_len, Acc, num_thetas)):
    #binning
    for i in range(len(x_id)):
        x = x_id[i]
        y = y_id[i]
        for j in range(num_thetas):
            rho = round(x * cos_t[j] + y * sin_t[j]) + diag_len
            Acc[rho, j] += 1
    return Acc

def hough_numpy(img):
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width*width + height*height)) #diagonal length of image 
    rhos = np.linspace(-diag_len, diag_len, 2.0*diag_len) #x-axis for plot with hough transform

    #pre-compute angles
    thetas = np.deg2rad(np.linspace(-180,180, 360))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    Acc = np.zeros((2 * diag_len, num_thetas), dtype = np.uint64)
    y_id, x_id = np.nonzero(img)
    cos_tile, sin_tile = np.tile(cos_t, (len(x_id),1)), np.tile(sin_t, (len(x_id),1))
    x_id_tile, y_id_tile = np.tile(x_id, (len(thetas),1)).T, np.tile(y_id, (len(thetas),1)).T
    rho = np.round(x_id_tile * cos_tile + y_id_tile * sin_tile) + diag_len #precompute rho
    rho = rho.astype(int)
    #binning more efficiently
    for j in range(len(x_id)):
        for i in range(num_thetas):
            Acc[rho[j,i], i] += 1

    return Acc, rhos, thetas

#Acc = hough_jit(img, make_jit_arrays(img)) 
##f, (ax1, ax2, ax3) = plt.subplots(1, 3)

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
a[2] = 1 * wavelength
a[3] = 0.5 * wavelength
a[6] = 0.7 * wavelength
V2D_inv = np.linalg.pinv(V2D)
G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
v_abb = (f_sh/(r_sh_m * px_size_sh)) * np.dot(V2D_inv, np.dot(G, a))
u_dm -= v_abb
mc.set_displacement(u_dm, mirror)
raw_input("remove piece of paper")
time.sleep(0.2)
int_cam.snapImage()
image_i0 = int_cam.getImage().astype(float)

print("first tip tilt")
u_dm[4] += 0.6
u_dm[7] -= 0.45
mc.set_displacement(u_dm, mirror)
time.sleep(2)
int_cam.snapImage()
image_i1 = int_cam.getImage().astype(float)

print("second tip tilt")
u_dm[4] -= 1.3
u_dm[7] += 0.7
mc.set_displacement(u_dm, mirror)
time.sleep(2)
int_cam.snapImage()
image_i2 = int_cam.getImage().astype(float)

Id1 = image_i1 - image_i0
Id2 = image_i2 - image_i0

f, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.imshow(image_i0)
ax2.imshow(image_i1)
ax3.imshow(image_i2)
plt.show()

 ##f2, (ax1, ax2) = plt.subplots(1, 2)
x0 = 550
y0 = 484
radius = 340
ny, nx = image_i0.shape
x, y = np.linspace(1, nx, nx), np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
zeros_1, zeros_2 = np.abs(Id1) <= 1, np.abs(Id2) <= 1
Id1_zeros, Id2_zeros = np.zeros((ny, nx)), np.zeros((ny, nx))
Id1_zeros[zeros_1] = 1
Id2_zeros[zeros_2] = 1

mask = [np.sqrt( (xx-x0)**2 + (yy-y0)**2) >= radius]
Id1_zeros_mask = np.ma.array(Id1_zeros, mask = mask)
Id2_zeros_mask = np.ma.array(Id2_zeros, mask=mask)
Acc1, rhos, thetas = hough_numpy(Id1_zeros_mask) #, make_jit_arrays(Id1_zeros_mask)) 

minus = ax1.imshow(Id1_zeros_mask)
ax2.imshow(Id2_zeros_mask)

f, ax = plt.subplots(1,1)
ax.imshow(Acc1.T, extent = [np.min(rhos), np.max(rhos), np.max(thetas), np.min(thetas)], interpolation = 'none', aspect = 'auto')

idmax = np.argmax(Acc1)
rho_max, theta_max = np.unravel_index(idmax, Acc1.shape)
f, ax = plt.subplots(1,1)
ax.plot(Acc1[:,theta_max])

[max_peaks, min_peaks] = peakdetect.peakdetect(Acc1[:,theta_max], lookahead = 20, delta = 100)
x_max_i, y_max_i = zip(*max_peaks)
Lambda = np.sum(np.diff(x_max_i), dtype = float)/len(x_max_i)
ax.scatter(x_max_i, y_max_i)
plt.show()

