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

##u_dm = np.zeros(19)
##mc.set_displacement(u_dm, mirror)
##time.sleep(0.2)
##sh.snapImage()
##zero_image = sh.getImage().astype(float)


### Test figures
#impath_zero = os.path.abspath("dm_sh.tif")
#impath_dist = os.path.abspath("reference_sh.tif")
#zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
#dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)

#Get 0 measurement saved as image
def coeff(x_pos_zero, y_pos_zero, zero_image, sh, px_size, f, r_sh_m, j_max,):
#x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)



    ## Given paramters for centroid gathering
    [ny,nx] = zero_image.shape
    #px_size = 5.2e-6     # width of pixels 
    #f = 17.6e-3            # focal length
    #r_sh = 1.924e-3        # radius of shack hartmann sensor
    r_sh_px = r_sh_m / px_size
    x = np.linspace(1, nx, nx)
    y = np.linspace(1, ny, ny)
    xx, yy = np.meshgrid(x, y)
    #j_max= 10           # maximum fringe order
    #zero_image = sh.getImage().astype(float)
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)

    centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
    x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
    y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
###normalize x and y

###Move mirror and snap new image
##u_dm = np.ones(19)
##u_dm[4] = 0
##u_dm[7] = 0
##mc.set_displacement(u_dm, mirror)
##time.sleep(0.2)
    sh.snapImage()
    dist_image = sh.getImage().astype(float)

    # Gather centroids and slope
    x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
    dWdx, dWdy = Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m)

    # Make Zernike matrix
    # Chosen is not to include piston, Zernikes evaluated in flat position
    #make test x and y
    #j_max = 10           #Maximum fringe number for zernike evaluation
    kmax = np.power(np.ceil(np.sqrt(j_max)),2) #estimation of maximum fringe number
    n, m = Zn.Zernike_j_2_nm(np.array(range(2, int(kmax)+1))) #find n and m pairs for maximum fringe number
    Kmax = np.max(Zn.Zernike_nm_2_j(n+1, np.abs(m)+1)) #find highest order of j for which beta is needed
    power_mat = Zn.Zernike_power_mat(Kmax+1)
    Z_mat = Zn.complex_zernike(Kmax+1, x_pos_norm, y_pos_norm)
##    for jj in range(2, int(Kmax)+1):
##        n, m = Zn.Zernike_j_2_nm(jj)
##        if m > 0:
##            j_min = Zn.Zernike_nm_2_j(n, -m)
##            Z_mat[:, jj-2] =(Zn.Zernike_xy(x_pos_norm, y_pos_norm, power_mat, jj) - 1j * Zn.Zernike_xy(x_pos_norm, y_pos_norm, power_mat, j_min))
##        elif m < 0:
##            j_plus = Zn.Zernike_nm_2_j(n, np.abs(m))
##            Z_mat[:, jj-2] = (Zn.Zernike_xy(x_pos_norm, y_pos_norm, power_mat, j_plus) + 1j * Zn.Zernike_xy(x_pos_norm, y_pos_norm, power_mat, jj))
##        else:
##            Z_mat[:, jj-2] = Zn.Zernike_xy(x_pos_norm, y_pos_norm, power_mat, jj) #begin at j = 2

    #Invert and solve for beta
    Z_mat_inv = np.linalg.pinv(Z_mat)
    dW_plus = dWdx + 1j * dWdy
    dW_min = dWdx - 1j * dWdy
    beta_plus = np.dot(Z_mat_inv, dW_plus)
    beta_min = np.dot(Z_mat_inv, dW_min)

    a = np.zeros(kmax, dtype = np.complex_)
    a_real = np.zeros(j_max)
    a_check = np.zeros(j_max, dtype = np.complex_)
    kmax = int(kmax)
    for jj in range(2, kmax):
        n, m = Zn.Zernike_j_2_nm(jj)
        index1 = Zn.Zernike_nm_2_j(n - 1.0, m + 1.0)
        index2 = Zn.Zernike_nm_2_j(n - 1.0, m - 1.0)
        index3 = Zn.Zernike_nm_2_j(n + 1.0, m + 1.0)
        index4 = Zn.Zernike_nm_2_j(n + 1.0, m - 1.0)
        fact1 = 1.0 / ( 2 * n * ( 1 + (((n-abs(m))/2) > 0)))
        fact2 = 1.0 / (2 * (n+2) * ( 1 + (((n+2-abs(m))/2) > 0)))
        a[jj-2] = fact1 * (beta_plus[index1] + beta_min[index2]) - fact2 * (beta_plus[index3] + beta_min[index4])

    for jj in range(2, j_max+2):
        n, m = Zn.Zernike_j_2_nm(jj)
        if m > 0:
            a_real[jj-2] = np.sqrt(2*n+2) * np.real(a[jj-2])
            j_min = Zn.Zernike_nm_2_j(n, -m)
            a_check[jj-2] = np.sqrt(2*n+2)*(a[jj-2] + a[j_min-2])/2.0
        elif m < 0:
            a_real[jj-2] = np.sqrt(2*n+2) * np.imag(a[jj-2])
            j_plus = Zn.Zernike_nm_2_j(n, np.abs(m))
            a_check[jj-2] = np.sqrt(2*n+2) * (a[j_plus - 2] - a[jj-2])/(2.0 * 1j)
        else:
            a_real[jj-2] = np.sqrt(n+1) * np.real(a[jj-2])
            a_check[jj-2] = np.sqrt(n+1) * a[jj-2] + (abs(np.imag(a[jj-2])) <= 1e-21) * 1j
        #a_normalized[jj-2] = a[jj-2] * np.sqrt(n+1)
    return a_real, a_check  
#print a_real, a_check
