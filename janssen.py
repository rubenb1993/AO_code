## Find the real Zernike coefficients using the derivative of the wavefront

## Part of the Adaptive Optics code by Ruben Biesheuvel (ruben.biesheuvel@gmail.com)

## Will use [1] A. J. E. M. Janssen, Zernike expansion of derivatives and Laplacians of the Zernike circle polynomials,
##                J. Opt. Soc. Am. A, Vol 31, No. 7, 2014

## Code will go through:
#Gathering Centroids
#Centroids -> dWx and dWy
#Find complex coefficients
#Convert complex coefficients to real

# Note that Janssen in his paper normalizes according to int(|C_n^m|^2) = pi/(n+1)
# This code makes use of the normalization int(|C_n^m|^2) = pi

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
import scipy.linalg as lin

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def avg_complex_zernike(x_pos_norm, y_pos_norm, j_max, r_int_px, spot_size = 35):
    """Given the meshgrids for x and y, and given the maximum fringe order, complex zernike retursn
    an (len(x), len(y), j_max) sized matrix with values of the complex Zernike polynomial at the given points"""
    half_len_box = spot_size/r_int_px
    box_px = 2*spot_size
    x_left = x_pos_norm - half_len_box
    x_right = x_pos_norm + half_len_box
    y_up = y_pos_norm + half_len_box
    y_down = y_pos_norm - half_len_box
    j_range = np.arange(1, j_max+2)
    Cnm_avg = np.zeros(len(x_pos_norm), len(j_range))
    for ii in range(len(x_pos_norm)):
        x, y = np.linspace(x_left[ii], x_right[ii], box_px), np.linspace(y_down[ii], y_up[ii], box_px)
        xx, yy = np.meshgrid(x, y)
        mask = [xx**2 + yy**2 >= 1]
        Cnm_xy = Zn.complex_zernike(j_max, xx, yy)
        tiled_mask = np.tile(mask, (Cnm_xy.shape[2],1,1)).T
        Cnm_mask = np.ma.array(Cnm_xy, mask = tiled_mask)
        Cnm_avg[ii,:] = np.sum(Cnm_mask, axis = (0,1))/box_px**2
    return Cnm_avg


def coeff(x_pos_zero, y_pos_zero, zero_image, dist_image, px_size, f, r_sh_m, j_max, wavelength):
    ## Given paramters for centroid gathering
    
    [ny,nx] = zero_image.shape
    r_sh_px = r_sh_m / px_size
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)

    centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px

##    sh.snapImage()
##    dist_image = sh.getImage().astype(float)

    # Gather centroids and slope
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m, wavelength)

    # Make Zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    Z_mat = Zn.complex_zernike(Kmax, x_pos_norm, y_pos_norm)

    #Invert and solve for beta
    dW_plus = dWdx + 1j * dWdy
    dW_min = dWdx - 1j * dWdy
    beta_plus = lin.lstsq(Z_mat, dW_plus)[0]
    beta_min = lin.lstsq(Z_mat, dW_min)[0]

    kmax = int(kmax)

    a = np.zeros(kmax, dtype = np.complex_)
    a_check = np.zeros(j_max, dtype = np.complex_)
    #a_avg = np.zeros(j_max, dtype = np.complex_)
    for jj in range(2, kmax+1):
        n, m = Zn.Zernike_j_2_nm(jj)
        index1 = int(Zn.Zernike_nm_2_j(n - 1.0, m + 1.0) - 1)
        index2 = int(Zn.Zernike_nm_2_j(n - 1.0, m - 1.0) - 1)
        index3 = int(Zn.Zernike_nm_2_j(n + 1.0, m + 1.0) - 1)
        index4 = int(Zn.Zernike_nm_2_j(n + 1.0, m - 1.0) - 1)
        fact1 = 1.0 / ( 2 * n * ( 1 + (n != abs(m))))
        fact2 = 1.0 / (2 * (n+2) * ( 1 + (((n+2) != abs(m)))))
        if m + 1.0 > n - 1.0:
            a[jj-1] = fact1 * (beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])
        elif np.abs(m - 1.0) > np.abs(n - 1.0):
            a[jj-1] = fact1 * (beta_plus[index1]) - fact2 * (beta_plus[index3] + beta_min[index4])
        else:
            a[jj-1] = fact1 * (beta_plus[index1] + beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])

    for jj in range(2, j_max+2):
        n, m = Zn.Zernike_j_2_nm(jj)
        if m > 0:
            j_min = int(Zn.Zernike_nm_2_j(n, -m))
            a_check[jj-2] = (1.0/np.sqrt(2*n+2))*(a[jj-1] + a[j_min-1])
        elif m < 0:
            j_plus = int(Zn.Zernike_nm_2_j(n, np.abs(m)))
            a_check[jj-2] = (1.0/np.sqrt(2*n+2)) * (a[j_plus - 1] - a[jj-1]) * 1j
        else:
            a_check[jj-2] = (1.0/np.sqrt(n+1)) * a[jj-1]
    return a_check  

def coeff_optimum(x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm, xx, yy, dist_image, image_control, px_size, f, r_sh_m, wavelength, j_max):
    # Gather centroids and slope
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m, wavelength)

    # Make Zernike matrix
    kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(1, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    Z_mat = Zn.complex_zernike(Kmax, x_pos_norm, y_pos_norm)

    #Invert and solve for beta
    dW_plus = dWdx + 1j * dWdy
    dW_min = dWdx - 1j * dWdy
    beta_plus = lin.lstsq(Z_mat, dW_plus)[0]
    beta_min = lin.lstsq(Z_mat, dW_min)[0]

    kmax = int(kmax)

    a = np.zeros(kmax, dtype = np.complex_)
    a_check = np.zeros(j_max, dtype = np.complex_)
    #a_avg = np.zeros(j_max, dtype = np.complex_)
    for jj in range(2, kmax+1):
        n, m = Zn.Zernike_j_2_nm(jj)
        index1 = int(Zn.Zernike_nm_2_j(n - 1.0, m + 1.0) - 1)
        index2 = int(Zn.Zernike_nm_2_j(n - 1.0, m - 1.0) - 1)
        index3 = int(Zn.Zernike_nm_2_j(n + 1.0, m + 1.0) - 1)
        index4 = int(Zn.Zernike_nm_2_j(n + 1.0, m - 1.0) - 1)
        fact1 = 1.0 / ( 2 * n * ( 1 + (n != abs(m))))
        fact2 = 1.0 / (2 * (n+2) * ( 1 + (((n+2) != abs(m)))))
        if m + 1.0 > n - 1.0:
            a[jj-1] = fact1 * (beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])
        elif np.abs(m - 1.0) > np.abs(n - 1.0):
            a[jj-1] = fact1 * (beta_plus[index1]) - fact2 * (beta_plus[index3] + beta_min[index4])
        else:
            a[jj-1] = fact1 * (beta_plus[index1] + beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])

    for jj in range(2, j_max+2):
        n, m = Zn.Zernike_j_2_nm(jj)
        if m > 0:
            j_min = int(Zn.Zernike_nm_2_j(n, -m))
            a_check[jj-2] = (1.0/np.sqrt(2*n+2))*(a[jj-1] + a[j_min-1])
        elif m < 0:
            j_plus = int(Zn.Zernike_nm_2_j(n, np.abs(m)))
            a_check[jj-2] = (1.0/np.sqrt(2*n+2)) * (a[j_plus - 1] - a[jj-1]) * 1j
        else:
            a_check[jj-2] = (1.0/np.sqrt(n+1)) * a[jj-1]
    return np.real(a_check)  
