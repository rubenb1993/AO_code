# -*- coding: utf-8 -*-
## Find the real Zernike coefficients using the derivative of the wavefront

## Part of the Adaptive Optics code by Ruben Biesheuvel (ruben.biesheuvel@gmail.com)

## Code will go through:
#Gathering Centroids
#Centroids -> dWx and dWy
#Build "geometry matrix" (x and y derivative of Zernike pol. in all centroid positions)
#solve system: s = G a using pseudo-inverse


## Note, this program normalizes accoring to ... $$$ fill in $$$

#import dmctr
import sys
if "C:\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Micro-Manager-1.4")
#import MMCorePy
import PIL.Image
from scipy import special
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
import Hartmann as Hm
import Zernike as Zn

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

##### Create the geometry matrix #### 
def geometry_matrix(x, y, j_max):
    """Creates the geometry matrix of the SH sensor according to the lecture slides of Imaging Physics (2015-2016)"""
    power_mat = Zn.Zernike_power_mat(j_max)
    B_brug = np.zeros((2*len(x), j_max))
    for jj in range(2, j_max+1):
        B_brug[:len(x), jj-2] = Zn.xder_brug(x,y, power_mat, jj)
        B_brug[len(x):, jj-2] = Zn.yder_brug(x, y, power_mat, jj)
    return B_brug

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
    
##### Make list of maxima given "flat" wavefront ####
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
j_max= 10           # maximum fringe order

# Gather centroids and slope
#x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy)
#dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh)

#G = geometry_matrix(x_pos_flat, y_pos_flat, j_max)
#s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh))
#Binv = np.linalg.pinv(geometry_matrix(x_pos_flat, y_pos_flat, j_max))
#a = np.dot(Binv, s)

    
    