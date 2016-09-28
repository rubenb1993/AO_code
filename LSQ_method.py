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
    
def abberations2voltages(G, V2D, a, f, r_sh):
    V2D_inv = np.linalg.pinv(V2D)
    v = (f/r_sh) * np.dot(V2D_inv, np.dot(G, a))
    return v

def cart2pol(x, y):
    "returns polar coordinates given x and y"
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
    
    
#### Set up mirrors
#mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
#n_act = 19
#half_volt = 6.0
#voltages = half_volt * np.ones(n_act)  # V = 0 to 12V
#mirror.set(voltages)

#### Set up cameras
##cam1=MMCorePy.CMMCore()
##
##cam1.loadDevice("cam","IDS_uEye","IDS uEye")
##cam1.initializeDevice("cam")
##cam1.setCameraDevice("cam")
##cam1.setProperty("cam","Pixel Clock",43)
##cam1.setProperty("cam","Exposure",0.0668)
##cam1.snapImage()
##PIL.Image.fromarray(cam1.getImage().astype("float")).save("camera1.tif")
##
##cam2=MMCorePy.CMMCore()
##
##cam2.loadDevice("cam","IDS_uEye","IDS uEye")
##cam2.initializeDevice("cam")
##cam2.setCameraDevice("cam")
##cam2.setProperty("cam","Pixel Clock", 150)
##cam2.setProperty("cam","PixelType", '8bit mono')
##cam2.setProperty("cam","Exposure", 0.0434)
##cam2.snapImage()
##PIL.Image.fromarray(cam2.getImage().astype("float")).save("camera2.tif")
##
###
###cam2=MMCorePy.CMMCore()
###
###cam2.loadDevice("cam","ThorlabsUSBCamera","ThorCam")
###cam2.initializeDevice("cam")
###cam2.setCameraDevice("cam")
###cam2.setProperty("cam","PixelClockMHz",30)
###cam2.setProperty("cam","Exposure",0.1)
###
###cam1.snapImage()
###cam2.snapImage()
###
###cam1.snapImage()
###cam2.snapImage()
###
###PIL.Image.fromarray(cam1.getImage()).save("camera1.tif")
###PIL.Image.fromarray(cam2.getImage()).save("camera2.tif")
##
#### Get shref
###PIL.Image.fromarray(cam2.getImage()).save("shref.tif")
##
#reference image
impath_zero = os.path.abspath("20160928_defocus_test/zero_image.tif")
impath_dist = os.path.abspath("20160928_defocus_test/defocus_10.0_sh.tif")
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)
##plt.imshow(zero_image, cmap = 'bone')
##fig = plt.figure()
##plt.imshow(dist_image, cmap = 'bone')
##plt.show()


#zero_image_copy = zero_image
#x_pos_flat, y_pos_flat = Hm.zero_positions(zero_image_copy)



##### Make list of maxima given "flat" wavefront ####
x_pos_flat, y_pos_flat = Hm.zero_positions(zero_image) #initial guess of positions

## Given paramters for centroid gathering
[ny,nx] = zero_image.shape
px_size = 5.2e-6     # width of pixels 
f = 17.6e-3            # focal length
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max= 10           # maximum fringe order

#### Gather 'real' centroid positions
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float) #reload image due to image corruption
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_flat, y_pos_flat, zero_image, xx, yy)

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
x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)

G = geometry_matrix(x_pos_norm, y_pos_norm, j_max)
xi, yi = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x_pos_norm, y_pos_norm), G[:len(x_pos_norm),2], (xi, yi), method='linear')
##plt.imshow(zi, vmin=G[:len(x_pos_norm),2].min(), vmax=G[:len(x_pos_norm),2].max(), origin='lower',
##           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
##plt.colorbar()
##plt.show()

s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m))
G_inv = np.linalg.pinv(G)
a = np.dot(G_inv, s)
print a

### plot zernikes according to coefficients
power_mat = Zn.Zernike_power_mat(j_max+1)
Z = np.zeros(xi.shape)
for jj in range(len(a)):
    Z += a[jj]*Zn.Zernike_xy(xi, yi, power_mat, jj+2)

Z /= 632.8e-9
plt.contourf(xi, yi, Z, rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth = 0)
cbar = plt.colorbar()
plt.title("Defocus 10")
cbar.ax.set_ylabel('lambda')
plt.savefig('reconstructed_defocus_10.png', bbox_inches='tight')


### make Voltages 2 displacement matrix
##mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
##voltage_avg = 6.0
##stroke_50 = 3.0
##actuators = 19
##voltages = voltage_avg * np.ones(actuators)  # V = 0 to 12V, actuator 4 and 7 are tip and tilt
##
##V2D = np.zeros(shape = (2 * len(x_pos_flat), len(voltages))) #matrix containing the displacement of the spots due to 50% stroke 
##centroid_0 = np.hstack((x_pos_flat, y_pos_flat))
##
##for i in range(len(voltages)):
##    voltages = np.zeros(actuators)
##    voltages = voltage_avg * np.ones(actuators)
##    voltages[i] += stroke_50
##    mirror.set(voltages)
##    time.sleep(0.005)
##    cam1.snapImage()
##    image = cam1.getImage().astype(float)
##    centroid_i = np.hstack(Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy))
##    displacement = centroid_0 - centroid_i
##    V2D[:,i] = displacement / stroke_50 ## normalize with stroke voltages in order to get real displacement with voltages (assume linear relation between stroke and displacement)
##
##
##
