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

import os
import Hartmann as Hm
import Zernike as Zn
import numpy as np
import PIL.Image



### Test figures
impath_zero = os.path.abspath("AO_code/test_images/shref.tif")
impath_dist = os.path.abspath("AO_code/test_images/sh_pattern_no_gain89.tif")
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)

#Get 0 measurement saved as image
x_pos_flat, y_pos_flat = Hm.zero_positions(zero_image)

## Given paramters for centroid gathering
[ny,nx] = zero_image.shape
px_size = 5.2e-6     # width of pixels 
f = 17.6e-3            # focal length
r_sh = nx*px_size/2.0        # radius of shack hartmann sensor
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max= 10           # maximum fringe order

# Gather centroids and slope
x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh)

# Make Zernike matrix
# Chosen is not to include piston, Zernikes evaluated in flat position
#make test x and y
j_max = 10           #Maximum fringe number for zernike evaluation
kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
n, m = Zn.Zernike_j_2_nm(np.array(range(2, int(kmax)+1))) #find n and m pairs for maximum fringe number
Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
power_mat = Zn.Zernike_power_mat(Kmax+1, Janssen = True)
Z_mat = np.zeros((len(x_pos_flat), Kmax))
for jj in range(2, int(Kmax)+1):
    Z_mat[:, jj-2] = Zn.Zernike_xy(x_pos_flat, y_pos_flat, power_mat, jj) #begin at j = 2

#Invert and solve for beta
Z_mat_inv = np.linalg.pinv(Z_mat)
dW_plus = dWdx + 1j * dWdy
dW_min = dWdx - 1j * dWdy
beta_plus = np.dot(Z_mat_inv, dW_plus)
beta_min = np.dot(Z_mat_inv, dW_min)

a = np.zeros(j_max-1, dtype = np.complex_)
a_normalized = np.zeros(j_max-1, dtype = np.complex_)
for jj in range(2, j_max+1):
    n, m = Zn.Zernike_j_2_nm(jj)
    index1 = Zn.Zernike_nm_2_j(n - 1.0, m + 1.0)
    index2 = Zn.Zernike_nm_2_j(n - 1.0, m - 1.0)
    index3 = Zn.Zernike_nm_2_j(n + 1.0, m + 1.0)
    index4 = Zn.Zernike_nm_2_j(n + 1.0, m - 1.0)
    fact1 = 1.0 / ( n * ( 1 + (((n-abs(m))/2) > 0)))
    fact2 = 1.0 / (2 * (n+2) * ( 1 + (((n+2-abs(m))/2) > 0)))
    a[jj-2] = fact1 * (beta_plus[index1] + beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4]) 
    a_normalized[jj-2] = a[jj-2] * np.sqrt(n+1)

