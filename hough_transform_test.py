import sys
import os
#import time

if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
##import MMCorePy
import PIL.Image
import numpy as np
# from matplotlib import rc
#import Hartmann as Hm
#import displacement_matrix as Dm
#import mirror_control as mc
import matplotlib.pyplot as plt
#from matplotlib import cm
import edac40
#import matplotlib.ticker as ticker
import math
#from scipy import signal
import peakdetect

# from numba import jit

img = np.zeros((121, 121))
img[30:80, 30:80] = np.eye(50)
img[:, 50] = 1
img[80, :] = 1
img[10, :] = 1
img[:, 10] = 1


# Set up ranges
def hough_lines(img):
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height))  # diagonal length of image
    rhos = np.linspace(-diag_len, diag_len, 2.0 * diag_len)  # x-axis for plot with hough transform

    # pre-compute angles
    thetas = np.deg2rad(np.linspace(-180, 180, 360))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    Acc = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_id, x_id = np.nonzero(img)

    # binning
    for i in range(len(x_id)):
        x = x_id[i]
        y = y_id[i]

        for j in range(num_thetas):
            rho = round(x * cos_t[j] + y * sin_t[j]) + diag_len
            Acc[rho, j] += 1

    return Acc


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

Acc = hough_jit(img, make_jit_arrays(img))
#f, (ax1, ax2, ax3) = plt.subplots(1, 3)

impath_i0 = os.path.abspath("AO_code/int_test_i0.tif")
image_i0 = np.asarray(PIL.Image.open(impath_i0)).astype(float)
impath_i1 = os.path.abspath("AO_code/int_test_i1.tif")
image_i1 = np.asarray(PIL.Image.open(impath_i1)).astype(float)
impath_i2 = os.path.abspath("AO_code/int_test_i2.tif")
image_i2 = np.asarray(PIL.Image.open(impath_i2)).astype(float)

Id1 = image_i1 - image_i0
Id2 = image_i2 - image_i0

#ax1.imshow(image_i0)
#ax2.imshow(image_i1)
#ax3.imshow(image_i2)

f2, (ax1, ax2) = plt.subplots(1, 2)
x0 = 550
y0 = 484
radius = 340
ny, nx = image_i0.shape
x, y = np.linspace(-1.0 * radius, 1.0 * radius, 2*radius), np.linspace(-1.0 * radius, 1.0 * radius, 2*radius)
xx, yy = np.meshgrid(x, y)
ss = np.sqrt(xx**2 + yy**2)
Id1_int = Id1[y0-radius:y0+radius, x0-radius:x0+radius]
Id2_int = Id2[y0-radius:y0+radius, x0-radius:x0+radius]
zeros_1, zeros_2 = np.abs(Id1_int) <= 1, np.abs(Id2_int) <= 1
Id1_zeros, Id2_zeros = np.zeros(Id1_int.shape, dtype = float), np.zeros(Id2_int.shape, dtype = float)
Id1_zeros[zeros_1] = 1
Id2_zeros[zeros_2] = 1

mask = [np.sqrt((xx) ** 2 + (yy) ** 2) >= radius]
Id1_zeros_mask = np.ma.array(Id1_zeros, mask=mask)
Id2_zeros_mask = np.ma.array(Id2_zeros, mask=mask)
Acc1, rhos, thetas, diag_len = hough_numpy(Id2_zeros_mask, x, y)  # make_jit_arrays(Id1_zeros_mask))

minus = ax1.imshow(Id1_zeros_mask)
ax2.imshow(Id2_zeros_mask)

f, ax = plt.subplots(1, 1)
ax.imshow(Acc1.T, extent=[np.min(rhos), np.max(rhos), np.max(thetas), np.min(thetas)], interpolation='none',
          aspect='auto')


rho_max_i, theta_max, s_max = max_finding(Acc1, rhos, thetas)
Lambda = np.sum(np.diff(rho_max_i), dtype=float) / (len(rho_max_i))
ti = 2 * np.pi/ Lambda * np.array([np.cos(theta_max), np.sin(theta_max)])
sigma = 2 * np.pi * s_max / Lambda
tau_i = ti[0] * xx + ti[1] * yy

f, ax = plt.subplots(1,1)
sin_ud_lr = np.flipud(np.sin((tau_i + sigma)/2.0))
ax.imshow( Id2_int/(-2.0 * sin_ud_lr), vmin = -10, vmax = 10)

y = np.zeros((len(x)))
f, ax = plt.subplots(1,1)
ax.imshow(Id2_zeros_mask, extent=[-radius, radius, -radius, radius], aspect = 'auto')
for i in range(len(rho_max_i)):
    y= -x / np.tan(theta_max) + (rho_max_i[i]) / np.sin(theta_max)
    mask = [np.sqrt(x**2 + y**2) >= radius]
    y_ma, x_ma = np.ma.array(y, mask = mask), np.ma.array(x, mask=mask)
    ind_true= np.where(mask[0] == 0)
    ax.plot([x_ma[ind_true[0][0]], x_ma[ind_true[0][-1]]],[y_ma[ind_true[0][0]], y_ma[ind_true[0][-1]]], 'k--', lw = 4)

plt.show()

