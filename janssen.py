## Find the real Zernike coefficients using the derivative of the wavefront

## Part of the Adaptive Optics code by Ruben Biesheuvel (ruben.biesheuvel@gmail.com)

## Will use [1] A. J. E. M. Janssen, Zernike expansion of derivatives and Laplacians of the Zernike circle polynomials,
##                J. Opt. Soc. Am. A, Vol 31, No. 7, 2014

## Code will go through:
#Gathering Centroids
#Centroids -> dWx and dWy
#Find complex coefficients
#Convert complex coefficients to real

# Note that Janssen normalizes according to int(|Z_n^m|^2) = pi/(n+1)

import Hartmann as Hm
#import Zernike as Zn
import Zernike as Zn
import numpy as np
import math
import scipy.special


### Gather centroids

#Get 0 measurement saved as image
#x_pos_flat, y_pos_flat = Hm.zero_positions(zero_image)

## Given paramters for centroid gathering
nx = 128            #number of pixels x direction
ny = 128            #                 y direction
px_size = 1e-6      # width of pixels 
f = 1e-3            # focal length
r_sh = 10e-3        # radius of shack hartmann sensor
x = np.linspace(0, nx, nx+1)
y = np.linspace(0, ny, ny+1)
xx, yy = np.meshgrid(x, y)

# Gather centroids and slope
#x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy)
#dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh)

# Make Zernike matrix
# Chosen is not to include piston, Zernikes evaluated in flat position
#make test x and y
x = np.array([0.0, 0.5])
y = np.array([1.0, 0.5])
j_max = 5           #Maximum fringe number for zernike evaluation
power_mat = Zn.Zernike_power_mat(j_max, Janssen = True)
Z_mat = np.zeros((len(x), j_max-1))
for jj in range(2, j_max+1):
    Z_mat[:, jj-2] = Zn.Zernike_xy(x, y, power_mat, jj)

#Invert and solve for beta
Z_mat_inv = np.linalg.pinv(Z_mat)
dW_plus = dWdx + 1j * dWdy
dW_min = dWdx - 1j * dWdy
beta_plus = np.dot(Z_mat_inv, dW_plus)
beta_min = np.dot(Z_mat_inv, dW_min)

