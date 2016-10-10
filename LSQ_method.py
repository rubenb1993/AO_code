# -*- coding: utf-8 -*-
## Find the real Zernike coefficients using the derivative of the wavefront

## Part of the Adaptive Optics code by Ruben Biesheuvel (ruben.biesheuvel@gmail.com)

## Code will go through:
#Gathering Centroids
#Centroids -> dWx and dWy
#Build "geometry matrix" (x and y derivative of Zernike pol. in all centroid positions)
#solve system: s = G a using pseudo-inverse


## Note, this program normalizes accoring to int(|Z_j|^2) = pi

#import dmctr
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
import Zernike as Zn
import edac40
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata



### Define font for figures
##rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
##rc('text', usetex=True)

##### Create the geometry matrix #### 
def geometry_matrix(x, y, j_max):
    """Creates the geometry matrix of the SH sensor according to the lecture slides of Imaging Physics (2015-2016)"""
    power_mat = Zn.Zernike_power_mat(j_max)
    B_brug = np.zeros((2*len(x), j_max))
    for jj in range(2, j_max+1):
        B_brug[:len(x), jj-2] = Zn.xder_brug(x, y, power_mat, jj)
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
    
###### Set up cameras
####cam1=MMCorePy.CMMCore()
####
####cam1.loadDevice("cam","IDS_uEye","IDS uEye")
####cam1.initializeDevice("cam")
####cam1.setCameraDevice("cam")
####cam1.setProperty("cam","Pixel Clock",43)
####cam1.setProperty("cam","Exposure",0.0668)
##
##cam1=MMCorePy.CMMCore()
##sh = cam1
##
##cam1.loadDevice("cam","IDS_uEye","IDS uEye")
##cam1.initializeDevice("cam")
##cam1.setCameraDevice("cam")
##cam1.setProperty("cam","Pixel Clock",150)
##cam1.setProperty("cam", "PixelType", '8bit mono')
##cam1.setProperty("cam","Exposure",0.0434)
##
##
####cam2=MMCorePy.CMMCore()
####sh = cam2
####
####cam2.loadDevice("cam","IDS_uEye","IDS uEye")
####cam2.initializeDevice("cam")
####cam2.setCameraDevice("cam")
####cam2.setProperty("cam","Pixel Clock", 150)
####cam2.setProperty("cam","PixelType", '8bit mono')
####cam2.setProperty("cam","Exposure", 0.0434)
##
##
##cam2=MMCorePy.CMMCore()
##
##cam2.loadDevice("cam","IDS_uEye","IDS uEye")
##cam2.initializeDevice("cam")
##cam2.setCameraDevice("cam")
##cam2.setProperty("cam","Pixel Clock", 43)
###cam2.setProperty("cam","PixelType", '8bit mono')
##cam2.setProperty("cam","Exposure", 0.0668)
##
##global mirror
##mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

#reference image
#impath_zero = os.path.abspath("flat_def_mirror_reference.tif")
#impath_dist = os.path.abspath("20160928_defocus_test/defocus_10.0_sh.tif")
#zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
#dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)
##plt.imshow(zero_image, cmap = 'bone')
##fig = plt.figure()
##plt.imshow(dist_image, cmap = 'bone')
##plt.show()
def LSQ_coeff(x_pos_zero, y_pos_zero, zero_image, sh, px_size, f, j_max):   
    ##### Make list of maxima given "flat" wavefront ####
    #x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image) #initial guess of positions

    ## Given paramters for centroid gathering
    [ny,nx] = zero_image.shape
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)

    #### Gather 'real' centroid positions
    #zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float) #reload image due to image corruption
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)

    #zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float) #reload image due to image corruption
    centre, r_sh_px, r_sh_m = Hm.centroid_centre(x_pos_flat, y_pos_flat, zero_image, xx, yy, px_size)

    ### Normalize x, y
    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px

    ##### Plot to see that really r is between 0 and 1
    ##plt.hist([x_pos_norm**2 + y_pos_norm**2])
    ##plt.show()
    ##plt.scatter(x_pos_norm, y_pos_norm)
    ##plt.show()

    # Gather centroids and slope
    sh.snapImage()
    dist_image = sh.getImage().astype(float)
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
    G = geometry_matrix(x_pos_norm, y_pos_norm, j_max)
    ##zi = griddata((x_pos_norm, y_pos_norm), G[:len(x_pos_norm),2], (xi, yi), method='linear')
    ##plt.imshow(zi, vmin=G[:len(x_pos_norm),2].min(), vmax=G[:len(x_pos_norm),2].max(), origin='lower',
    ##           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
    ##plt.colorbar()
    ##plt.show()

    s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m))
    G_inv = np.linalg.pinv(G)
    a = np.dot(G_inv, s)
    return a
##plot_zernike(j_max, a)
##




