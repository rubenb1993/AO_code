# -*- coding: utf-8 -*-
## Find the real Zernike coefficients using the derivative of the wavefront

## Part of the Adaptive Optics code by Ruben Biesheuvel (ruben.biesheuvel@gmail.com)

## Code will go through:
#Gathering Centroids
#Centroids -> dWx and dWy
#Build "geometry matrix" (x and y derivative of Zernike pol. in all centroid positions)
#solve system: s = G a using pseudo-inverse


## Note, this program normalizes accoring to int(|Z_j|^2) = pi

import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import PIL.Image
import numpy as np
import Hartmann as Hm
import Zernike as Zn
import edac40
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata

def matrix_avg_gradient(x_pos_norm, y_pos_norm , j, r_sh_px, power_mat, spot_size, integration_px = 70):
    """Compute the average gradient in each sub_aperture based, using the fact that the average distance between spots is 70 px.
    For each point (x,y) normalised, the alogirthm creates a box around it, """
    #power_mat = Zn.Zernike_power_mat(j_max+2)
    half_len_box = spot_size/r_sh_px
    x_left = x_pos_norm - half_len_box
    x_right = x_pos_norm + half_len_box
    y_up = y_pos_norm + half_len_box
    y_down = y_pos_norm - half_len_box
    #j_range = np.arange(2, j_max+2)
    G = np.zeros((2*len(x_pos_norm), len(j)))
    for ii in range(len(x_pos_norm)):
        x = np.linspace(x_left[ii], x_right[ii], integration_px)
        y = np.linspace(y_down[ii], y_up[ii], integration_px)
        xx, yy = np.meshgrid(x, y)
        mask = [xx**2 + yy**2 >= 1]
        Zx, Zy = Zn.xder_brug(xx, yy, power_mat, j), Zn.yder_brug(xx, yy, power_mat, j)
        tiled_mask = np.tile(mask, (Zx.shape[2],1,1)).T
        Zxn, Zyn = np.ma.array(Zx, mask=tiled_mask), np.ma.array(Zy, mask=tiled_mask)
        G[ii, :] = Zxn.mean(axis = (0,1))#np.sum(Zxn, axis = (0,1))/integration_px**2
        G[ii+len(x_pos_norm), :] = Zyn.mean(axis = (0,1))#np.sum(Zyn, axis = (0,1))/integration_px**2
    return G
    
def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
    
##def LSQ_coeff(x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, G, zero_image, dist_image, px_size, r_sh_px, f, j_max, wavelength):   
##    ## Given paramters for centroid gathering
##    r_sh_m = px_size * r_sh_px
##    [ny,nx] = zero_image.shape
##    x = np.linspace(1, nx, nx)
##    y = np.linspace(1, ny, ny)
##    xx, yy = np.meshgrid(x, y)
##
##    #### Gather 'real' centroid positions
##    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
##    centre= Hm.centroid_centre(x_pos_flat, y_pos_flat)
##    
##    ### Normalize x, y
##    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
##    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
##
##    # Gather centroids and slope
##    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
##    s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m, wavelength))
##    a = np.linalg.lstsq(G, s)[0]
##    return a





