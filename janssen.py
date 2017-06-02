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
##import mirror_control as mc
import edac40
##import MMCorePy
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

def avg_complex_zernike(x_pos_norm, y_pos_norm, n_max, r_sh_px, order, spot_size = 35, integration_spots = 70):
    """Given the meshgrids for x and y, and given the maximum fringe order, complex zernike retursn
    an (len(x), len(y), j_max) sized matrix with values of the complex Zernike polynomial at the given points"""
    half_len_box = spot_size/float(r_sh_px)
    x_left = x_pos_norm - half_len_box
    x_right = x_pos_norm + half_len_box
    y_up = y_pos_norm + half_len_box
    y_down = y_pos_norm - half_len_box
##    j = Zn.max_n_to_j(n_max, order= order)[str(n_max)]
    j = np.insert(Zn.max_n_to_j(n_max, order = order)[str(n_max)], 0, 1)
    Cnm_avg = np.zeros((len(x_pos_norm), len(j)), dtype = np.complex_)
    for ii in range(len(x_pos_norm)):
        x, y = np.linspace(x_left[ii], x_right[ii], integration_spots), np.linspace(y_down[ii], y_up[ii], integration_spots)
        xx, yy = np.meshgrid(x, y)
        mask = [xx**2 + yy**2 >= 1]
        Cnm_xy = Zn.complex_zernike(n_max, xx, yy, order)
        tiled_mask = np.tile(mask, (Cnm_xy.shape[2],1,1)).T
        Cnm_mask = np.ma.array(Cnm_xy, mask = tiled_mask)
        Cnm_avg[ii,:] = Cnm_mask.mean(axis = (0,1))
    return Cnm_avg

def coeff_from_dist(x_pos_flat_f, y_pos_flat_f, x_pos_dist_f, y_pos_dist_f, x_pos_norm_f, y_pos_norm_f, px_size_sh, f_sh, r_sh_m, wavelength, n_max, r_sh_px, box_len, order):
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist_f, y_pos_dist_f, x_pos_flat_f, y_pos_flat_f, px_size_sh, f_sh, r_sh_m, wavelength)

    # Make Zernike matrix
    j = Zn.max_n_to_j(n_max, order = order)[str(n_max)]
    j_fit_janss = Zn.max_n_to_j(n_max-1, order = order)[str(n_max-1)]
    compl_Z_mat = avg_complex_zernike(x_pos_norm_f, y_pos_norm_f, n_max, r_sh_px, order, spot_size = box_len)

    #Invert and solve for beta
    dW_plus = dWdx + 1j * dWdy
    dW_min = dWdx - 1j * dWdy
    beta_plus = np.linalg.lstsq(compl_Z_mat, dW_plus)[0]
    beta_min = np.linalg.lstsq(compl_Z_mat, dW_min)[0]

##    beta_plus = np.insert(beta_plus, 0, 0.0+ 1j*0)
##    beta_min = np.insert(beta_min, 0, 0.0 + 1j*0)
    
    a_compl_janss = np.zeros(len(j), dtype = np.complex_)
    a_real_janss = np.zeros(len(j_fit_janss), dtype = np.complex_)

    for jj in range(2, len(j_fit_janss)+2):
        n, m = Zn.Zernike_j_2_nm(jj, ordering = order)
        index1 = int(Zn.Zernike_nm_2_j(n - 1.0, m + 1.0, ordering = order) - 1)
        index2 = int(Zn.Zernike_nm_2_j(n - 1.0, m - 1.0, ordering = order) - 1)
        index3 = int(Zn.Zernike_nm_2_j(n + 1.0, m + 1.0, ordering = order) - 1)
        index4 = int(Zn.Zernike_nm_2_j(n + 1.0, m - 1.0, ordering = order) - 1)
        fact1 = 1.0 / ( 2 * n * ( 1 + (n != abs(m))))
        fact2 = 1.0 / (2 * (n+2) * ( 1 + (((n+2) != abs(m)))))
        if m + 1.0 > n - 1.0:
            a_compl_janss[jj-2] = fact1 * (beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])
        elif np.abs(m - 1.0) > np.abs(n - 1.0):
            a_compl_janss[jj-2] = fact1 * (beta_plus[index1]) - fact2 * (beta_plus[index3] + beta_min[index4])
        else:
            a_compl_janss[jj-2] = fact1 * (beta_plus[index1] + beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])

    for jj in range(2, len(j_fit_janss)+2):
        n, m = Zn.Zernike_j_2_nm(jj, ordering = order)
        if m > 0:
            j_min = int(Zn.Zernike_nm_2_j(n, -m, ordering = order))
            a_real_janss[jj-2] = (1.0/np.sqrt(2*n+2))*(a_compl_janss[jj-2] + a_compl_janss[j_min-2])
        elif m < 0:
            j_plus = int(Zn.Zernike_nm_2_j(n, np.abs(m), ordering = order))
            a_real_janss[jj-2] = (1.0/np.sqrt(2*n+2)) * (a_compl_janss[j_plus - 2] - a_compl_janss[jj-2]) * 1j
        else:
            a_real_janss[jj-2] = (1.0/np.sqrt(n+1)) * a_compl_janss[jj-2]
    a_real_janss = np.real(a_real_janss)
    return a_real_janss
