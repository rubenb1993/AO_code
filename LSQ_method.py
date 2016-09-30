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

def plot_zernike(j_max, a, savefigure = False, title = 'zernike_plot'):
    ### plot zernikes according to coefficients
    xi, yi = np.linspace(-1, 1, 300), np.linspace(-1, 1, 300)
    xi, yi = np.meshgrid(xi, yi)
    power_mat = Zn.Zernike_power_mat(j_max+1)
    Z = np.zeros(xi.shape)
    for jj in range(len(a)):
        Z += a[jj]*Zn.Zernike_xy(xi, yi, power_mat, jj+2)

    Z /= 632.8e-9
    plt.contourf(xi, yi, Z, rstride=1, cstride=1, cmap=cm.YlGnBu_r, linewidth = 0)
    cbar = plt.colorbar()
    #plt.title("Defocus 10")
    cbar.ax.set_ylabel('lambda')
    if savefigure:
        plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()
    
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

cam1=MMCorePy.CMMCore()

cam1.loadDevice("cam","IDS_uEye","IDS uEye")
cam1.initializeDevice("cam")
cam1.setCameraDevice("cam")
cam1.setProperty("cam","Pixel Clock",150)
cam1.setProperty("cam", "PixelType", '8bit mono')
cam1.setProperty("cam","Exposure",0.0434)


##cam2=MMCorePy.CMMCore()
##
##cam2.loadDevice("cam","IDS_uEye","IDS uEye")
##cam2.initializeDevice("cam")
##cam2.setCameraDevice("cam")
##cam2.setProperty("cam","Pixel Clock", 150)
##cam2.setProperty("cam","PixelType", '8bit mono')
##cam2.setProperty("cam","Exposure", 0.0434)

cam2=MMCorePy.CMMCore()

cam2.loadDevice("cam","IDS_uEye","IDS uEye")
cam2.initializeDevice("cam")
cam2.setCameraDevice("cam")
cam2.setProperty("cam","Pixel Clock", 43)
#cam2.setProperty("cam","PixelType", '8bit mono')
cam2.setProperty("cam","Exposure", 0.0668)




#reference image
impath_zero = os.path.abspath("flat_def_mirror_reference.tif")
impath_dist = os.path.abspath("20160928_defocus_test/defocus_10.0_sh.tif")
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)
##plt.imshow(zero_image, cmap = 'bone')
##fig = plt.figure()
##plt.imshow(dist_image, cmap = 'bone')
##plt.show()

##### Make list of maxima given "flat" wavefront ####
x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image) #initial guess of positions

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
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)

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
#x_pos_dist, y_pos_dist = Hm.centroid_positions(x_pos_flat, y_pos_flat, dist_image, xx, yy)
#G = geometry_matrix(x_pos_norm, y_pos_norm, j_max)
##zi = griddata((x_pos_norm, y_pos_norm), G[:len(x_pos_norm),2], (xi, yi), method='linear')
##plt.imshow(zi, vmin=G[:len(x_pos_norm),2].min(), vmax=G[:len(x_pos_norm),2].max(), origin='lower',
##           extent=[x_pos_norm.min(), x_pos_norm.max(), y_pos_norm.min(), y_pos_norm.max()])
##plt.colorbar()
##plt.show()

#s = np.hstack(Hm.centroid2slope(x_pos_dist, y_pos_dist, x_pos_flat, y_pos_flat, px_size, f, r_sh_m))
#G_inv = np.linalg.pinv(G)
#a = np.dot(G_inv, s)

##plot_zernike(j_max, a)

###make voltage 2 distance matrix
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

actuators = 19
voltage_0 = 3.0
stroke = 6.0

voltages = voltage_0 * np.ones(actuators)
mirror.set(voltages)
time.sleep(0.2)
cam1.snapImage()
zero_image = cam1.getImage().astype(float)

x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
V2D = np.zeros(shape = (2 * len(x_pos_zero), actuators))
cam1.snapImage()
zero_image = cam1.getImage().astype(float)

centroid_0 = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy))
for i in range(actuators):
    print(i)
    voltages = voltage_0 * np.ones(actuators)
    voltages[i] += stroke
    mirror.set(voltages)
    time.sleep(0.2)
    cam1.snapImage()
    image = cam1.getImage().astype(float)
    centroid_i = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, image, xx, yy))
    if i == 4 or i == 7:
        displ = centroid_0 - centroid_i
        V2D[:, i] = displ / (stroke**2)
    else:
        displ = centroid_0 - centroid_i
        V2D[:, i] = displ / (stroke**2)

### plot results, see if they compare
sim_range = np.arange(-5.9, 6.1, 0.1)
displacement_sim = np.outer(V2D[0,:], sim_range)
##for jj in range(actuators):
##    rela, = axarr[(plot_index[0][jj], plot_index[1][jj])].plot(np.arange(-6, 6.1, 0.1), displacement_sim[jj,:])

voltage_avg = 6.0
stroke_50 = np.arange(-5.5, 6.5, 1)
actuators = 19
voltages = 6.0 * np.ones(actuators)  # V = 0 to 12V, actuator 4 and 7 are tip and tilt
voltages[4] = 6.0
voltages[7] = 6.0
mirror.set(voltages)
time.sleep(0.2)
cam1.snapImage()
cam1.snapImage()
zero_image = cam1.getImage().astype(float)

## Given paramters for centroid gathering
[ny,nx] = zero_image.shape
px_size = 5.2e-6     # width of pixels 
f = 17.6e-3            # focal length
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max= 10           # maximum fringe order


x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
cam1.snapImage()
zero_image = cam1.getImage().astype(float) #re load image due to corruption
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
centroid_0 = np.hstack((x_pos_flat, y_pos_flat))


abs_displ = np.zeros((len(stroke_50), actuators))
displ_per_volt = np.zeros((len(stroke_50), actuators))
f, axarr = plt.subplots(5, 4, sharex = True)
f2, axarr2 = plt.subplots(5, 4, sharex = True)
f.suptitle('x-displacement of centroid 0 due to actuators')
plot_index = np.unravel_index(np.array(range(actuators)), (5, 4))

for jj in range(actuators):
    print jj
    for ii in range(len(stroke_50)):
        voltage = stroke_50[ii]
        voltages = voltage_avg * np.ones(actuators)
        voltages[jj] += voltage
        #voltages[4] = 6.0
        #voltages[7] = 6.0
        mirror.set(voltages)
        time.sleep(0.2)
        cam1.snapImage()
        image = cam1.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy))
        displacement = centroid_0 - centroid_i
        abs_displ[ii, jj] = displacement[0]
        displ_per_volt[ii, jj] = abs_displ[ii,jj] / (voltage + 6.0)**2
    abso_real, abso_sim= axarr[(plot_index[0][jj], plot_index[1][jj])].plot(stroke_50, abs_displ[:,jj], sim_range, displacement_sim[jj, :] * (sim_range + 6.0))
    rela_real,  rela_sim= axarr2[(plot_index[0][jj], plot_index[1][jj])].plot(stroke_50, displ_per_volt[:,jj], sim_range, displacement_sim[jj,:] / (sim_range + 6.0))
    f.legend((abso_real, abso_sim), ('real', 'sim'), 'lower right', fontsize = 9)
    f2.legend((rela_real, rela_sim), ('real', 'sim'), 'lower right', fontsize = 9)
    axarr[(plot_index[0][jj], plot_index[1][jj])].set_title(str(jj))
axarr[(plot_index[0][16], plot_index[1][16])].set_xlabel('V')
axarr[(plot_index[0][16], plot_index[1][16])].set_ylabel('x-displacement w.r.t. 0 [px]')
axarr2[(plot_index[0][16], plot_index[1][16])].set_xlabel('V')
axarr2[(plot_index[0][16], plot_index[1][16])].set_ylabel('displacement / V [px]')

plt.show()



# make Voltages 2 displacement matrix
##mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
##voltage_avg = 6.0
##stroke_50 = [-4.0, 4.0]
##actuators = 19
##voltages = 6.0 * np.ones(actuators)  # V = 0 to 12V, actuator 4 and 7 are tip and tilt
##mirror.set(voltages)
##time.sleep(0.1)
##cam2.snapImage()
##cam2.snapImage()
##zero_image = cam2.getImage().astype(float)
##
##centroid_0 = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy))
##f, axarr = plt.subplots(len(stroke_50), 1, sharex = True, sharey = True)
##avg_disp = np.zeros(len(stroke_50))
##for jj in range(len(stroke_50)):
##    voltage = stroke_50[jj]
##    V2D = np.zeros(shape = (2 * len(x_pos_flat), len(voltages))) #matrix containing the displacement of the spots due to 50% stroke 
##    for i in range(actuators):
##        if i == 4 or i == 7:
##            continue
##        #voltages = np.zeros(actuators)
##        print(i)
##        voltages = voltage_avg * np.ones(actuators)
##        voltages[i] += voltage
##        mirror.set(voltages)
##        time.sleep(0.1)
##        cam2.snapImage()
##        cam2.snapImage()
##        image = cam2.getImage().astype(float)
##        #PIL.Image.fromarray(image).save("actuator_volt" + str(voltage) + str(i) + ".tif")
##        centroid_i = np.hstack(Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy))
##        displacement = centroid_0 - centroid_i
##        ## normalize with stroke voltages in order to get real displacement with voltages (assume linear relation between stroke and displacement
##        V2D[:,i] = displacement
##    avg_disp[jj] = np.average(V2D)
##    axarr[jj].hist(V2D)
##    axarr[jj].set_title('V = ' + str(voltage))
##
##f2 = plt.figure()
##ax = f2.add_subplot(111)
##ax.plot(stroke_50, avg_disp)
##plt.show()


##    if voltage == 10.0:
##        print(voltage)
##        V2D_10 = V2D
##    else:
##        print(voltage)
##        V2D_5 = V2D



