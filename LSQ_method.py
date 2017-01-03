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
def geometry_matrix(x, y, j_max, r_sh_px, spot_size = 35):
    """Creates the geometry matrix of the SH sensor according to the lecture slides of Imaging Physics (2015-2016)"""
    power_mat = Zn.Zernike_power_mat(j_max+2)
    displacement = spot_size / r_sh_px
    B_brug = np.zeros((2*len(x), j_max))
    B_brug_top = np.zeros((len(x), j_max))
    B_brug_bot = np.zeros((len(x), j_max))
    B_brug_left = np.zeros((len(x), j_max))
    B_brug_right = np.zeros((len(x), j_max))
    j_range = np.arange(2, j_max+2)
    B_brug_right = Zn.xder_brug(x + displacement, y, power_mat, j_range)
    B_brug_left = Zn.xder_brug(x - displacement, y, power_mat, j_range)
    B_brug_bot = Zn.yder_brug(x, y - displacement, power_mat, j_range)
    B_brug_top= Zn.yder_brug(x, y + displacement, power_mat, j_range)
    B_brug[:len(x),:] = (B_brug_left + B_brug_right) / 2.0
    B_brug[len(x):, :] = (B_brug_top + B_brug_bot) / 2.0
    return B_brug

def geometry_matrix_2(x, y, j_max, r_sh_px, spot_size = 35.0):
    "using semi-gauss"
    power_mat = Zn.Zernike_power_mat(j_max+2)
    displacement = spot_size / r_sh_px
    G_top = np.zeros((len(x), j_max))
    G_bot = np.zeros((len(x), j_max))
    G_left = np.zeros((len(x), j_max))
    G_right = np.zeros((len(x), j_max))
    G = np.zeros((2*len(x), j_max))
    j_range = np.arange(2, j_max+2)
    G_left= Zn.Zernike_xy(x - displacement, y, power_mat, j_range)
    G_right = Zn.Zernike_xy(x + displacement, y, power_mat, j_range)
    G_top = Zn.Zernike_xy(x, y + displacement, power_mat, j_range)
    G_bot = Zn.Zernike_xy(x, y - displacement, power_mat, j_range)
    h = 2 * spot_size / r_sh_px
    G[:len(x), :] = (G_right - G_left) / h
    G[len(x):, :] = (G_top - G_bot) / h
    return G

def matrix_avg_gradient(x_pos_norm, y_pos_norm , j_max, r_int_px, power_mat, spot_size = 35):
    """Compute the average gradient in each sub_aperture based, using the fact that the average distance between spots is 70 px.
    For each point (x,y) normalised, the alogirthm creates a box around it, """
    #power_mat = Zn.Zernike_power_mat(j_max+2)
    half_len_box = spot_size/r_int_px
    box_px = 2*spot_size
    x_left = x_pos_norm - half_len_box
    x_right = x_pos_norm + half_len_box
    y_up = y_pos_norm + half_len_box
    y_down = y_pos_norm - half_len_box
    j_range = np.arange(2, j_max+2)
    G = np.zeros((2*len(x_pos_norm), len(j_range)))
    for ii in range(len(x_pos_norm)):
        x = np.linspace(x_left[ii], x_right[ii], box_px)
        y = np.linspace(y_down[ii], y_up[ii], box_px)
        xx, yy = np.meshgrid(x, y)
        mask = [xx**2 + yy**2 >= 1]
        Zx, Zy = Zn.xder_brug(xx, yy, power_mat, j_range), Zn.yder_brug(xx, yy, power_mat, j_range)
        tiled_mask = np.tile(mask, (Zx.shape[2],1,1)).T
        Zxn, Zyn = np.ma.array(Zx, mask=tiled_mask), np.ma.array(Zy, mask=tiled_mask)
        G[ii, :] = np.sum(Zxn, axis = (0,1))/box_px**2
        G[ii+len(x_pos_norm), :] = np.sum(Zyn, axis = (0,1))/box_px**2
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

#### old version
##def LSQ_coeff(x_pos_zero, y_pos_zero, G, zero_image, sh, px_size, r_sh_px, f, j_max):   
##    ##### Make list of maxima given "flat" wavefront ####
##    #x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image) #initial guess of positions
##
##    ## Given paramters for centroid gathering
##    r_sh_m = px_size * r_sh_px
##    [ny,nx] = zero_image.shape
##    x = np.linspace(1, nx, nx)
##    y = np.linspace(1, ny, ny)
##    xx, yy = np.meshgrid(x, y)
##
##    #### Gather 'real' centroid positions
##    #zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float) #reload image due to image corruption
##    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
##
##    #zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float) #reload image due to image corruption
##    centre= Hm.centroid_centre(x_pos_flat, y_pos_flat)
##    
##    ### Normalize x, y
##    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
##    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
##
##    ##### Plot to see that really r is between 0 and 1
##    ##plt.hist([x_pos_norm**2 + y_pos_norm**2])
##    ##plt.show()
##    ##plt.scatter(x_pos_norm, y_pos_norm)
##    ##plt.show()
##
##    # Gather centroids and slope
##    sh.snapImage()
##    dist_image = sh.getImage().astype(float)
##    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
##    #G = geometry_matrix_2(x_pos_norm, y_pos_norm, j_max, r_sh_px)
##    ##zi = griddata((x_pos_norm, y_pos_norm), G[:len(x_pos_norm),2], (xi, yi), method='linear')
##    ##plt.imshow(zi, vmin=G[:len(x_pos_norm),2].min(), vmax=G[:len(x_pos_norm),2].max(), origin='lower',
##    ##           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
##    ##plt.colorbar()
##    ##plt.show()
##
##    s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m))
##    #a = np.linalg.lstsq(G, s)[0]
##    #G_inv = np.linalg.pinv(G)
##    #a = np.dot(G_inv, s)
##    a = np.linalg.lstsq(G, s)[0]
##    return a

def LSQ_coeff(x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, G, zero_image, dist_image, px_size, r_sh_px, f, j_max, wavelength):   
    ## Given paramters for centroid gathering
    r_sh_m = px_size * r_sh_px
    [ny,nx] = zero_image.shape
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)

    #### Gather 'real' centroid positions
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
    centre= Hm.centroid_centre(x_pos_flat, y_pos_flat)
    
    ### Normalize x, y
    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px

    # Gather centroids and slope
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
    s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m, wavelength))
    a = np.linalg.lstsq(G, s)[0]
    return a
##plot_zernike(j_max, a)
##




