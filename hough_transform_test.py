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
import mirror_control as mc
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker
import math
#from numba import jit

img = np.zeros((121,121))
img[30:80, 30:80] = np.eye(50)
img[:, 50] = 1
img[80, :] = 1
img[10,:] = 1
img[:,10] = 1
#Set up ranges
def hough_lines(img):
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width*width + height*height)) #diagonal length of image 
    rhos = np.linspace(-diag_len, diag_len, 2.0*diag_len) #x-axis for plot with hough transform

    #pre-compute angles
    thetas = np.deg2rad(np.linspace(-90,90, 2.0*diag_len))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    Acc = np.zeros((2 * diag_len, num_thetas), dtype = np.uint64)
    y_id, x_id = np.nonzero(img)

    #binning
    for i in range(len(x_id)):
        x = x_id[i]
        y = y_id[i]

        for j in range(num_thetas):
            rho = round(x * cos_t[j] + y * sin_t[j]) + diag_len
            Acc[rho, j] += 1

    return Acc, rhos, thetas

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

Acc = hough_jit(img, make_jit_arrays(img)) 
#f, (ax1, ax2, ax3) = plt.subplots(1, 3)

impath_i0 = os.path.abspath("int_test_i0.tif")
image_i0 = np.asarray(PIL.Image.open(impath_i0)).astype(float)
impath_i1 = os.path.abspath("int_test_i1.tif")
image_i1 = np.asarray(PIL.Image.open(impath_i1)).astype(float)
impath_i2 = os.path.abspath("int_test_i2.tif")
image_i2 = np.asarray(PIL.Image.open(impath_i2)).astype(float)

Id1 = image_i1 - image_i0
Id2 = image_i2 - image_i0

##ax1.imshow(image_i0)
##ax2.imshow(image_i1)
##ax3.imshow(image_i2)

f2, (ax1, ax2) = plt.subplots(1, 2)
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
Acc1 = hough_jit(img, make_jit_arrays(Id1_zeros_mask)) 

minus = ax1.imshow(Id1_zeros_mask)
ax2.imshow(Id2_zeros_mask)
plt.colorbar(minus)

f, ax = plt.subplots(1,1)
ax.imshow(Acc1.T)
plt.show()
