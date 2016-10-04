import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
from matplotlib import rc
import mirror_control as mc
import Hartmann as Hm
import displacement_matrix as Dm
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=False)
    
#### Set up cameras
##cam1=MMCorePy.CMMCore()
##
##cam1.loadDevice("cam","IDS_uEye","IDS uEye")
##cam1.initializeDevice("cam")
##cam1.setCameraDevice("cam")
##cam1.setProperty("cam","Pixel Clock",43)
##cam1.setProperty("cam","Exposure",0.0668)

cam1=MMCorePy.CMMCore()
sh = cam1

cam1.loadDevice("cam","IDS_uEye","IDS uEye")
cam1.initializeDevice("cam")
cam1.setCameraDevice("cam")
cam1.setProperty("cam","Pixel Clock",150)
cam1.setProperty("cam", "PixelType", '8bit mono')
cam1.setProperty("cam","Exposure",0.0434)


##cam2=MMCorePy.CMMCore()
##sh = cam2
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

global mirror
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

actuators = 19
u_dm_0 = np.zeros(actuators)
u_dm_test = np.linspace(-0.9, 0.9, 11)
mc.set_displacement(u_dm_0, mirror)
time.sleep(0.2)
sh.snapImage()
zero_image = sh.getImage().astype(float)

## Given paramters for centroid gathering
[ny,nx] = zero_image.shape
px_size = 5.2e-6     # width of pixels 
f = 17.6e-3            # focal length
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max= 10           # maximum fringe order

### test linearity of non-linear actuators with new set_displacement function
x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
sh.snapImage()
zero_image = sh.getImage().astype(float) #re load image due to corruption
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
centroid_0 = np.hstack((x_pos_flat, y_pos_flat))

abs_displ = np.zeros((len(u_dm_test), actuators))
f, axarr = plt.subplots(10, 2, sharex = True, figsize = (5.42, 8.22) )
#f.suptitle('RMS displacement of centroids')
plot_index = np.unravel_index(np.array(range(actuators+1)), (10, 2))

actnum=np.arange(0,19,1)
linacts=np.where(np.logical_or(actnum==4,actnum==7))
others=np.where(np.logical_and(actnum!=4,actnum!=7))

V2D = Dm.gather_displacement_matrix(mirror, sh)
rms_sim = np.sqrt(np.sum(np.square(V2D), axis = 0) / float(len(V2D[:, 0])))
x_sim = np.array([-1.0, 0, 1.0])
y_sim = np.array([rms_sim, np.zeros(len(rms_sim)), rms_sim])

### Re-make zero image in order to account for hysteresis in the system CHECK IF THIS IS OK
x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
mc.set_displacement(np.zeros(actuators), mirror)
time.sleep(0.2)
sh.snapImage()
zero_image = sh.getImage().astype(float) #re load image due to corruption
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
centroid_0 = np.hstack((x_pos_flat, y_pos_flat))


label_size = 6
### First the non-linear actuators
for jj in np.nditer(others):
    print(jj)
    for ii in range(len(u_dm_test)):
        voltage = u_dm_test[ii]
        voltages = np.zeros(actuators)
        voltages[jj] += voltage
        mc.set_displacement(voltages, mirror)
        time.sleep(0.2)
        sh.snapImage()
        image = sh.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy))
        displ = centroid_0 - centroid_i
        abs_displ[ii, jj] = np.sqrt(np.mean(np.square(displ)))
    abso, sim = axarr[(plot_index[0][jj], plot_index[1][jj])].plot(u_dm_test, abs_displ[:,jj], x_sim, y_sim[:, jj])
    axarr[(plot_index[0][jj], plot_index[1][jj])].yaxis.set_ticks(np.linspace(0, 1.5, 2))
    axarr[(plot_index[0][jj], plot_index[1][jj])].xaxis.set_ticks(np.arange(-1, 2, 1.0))
    
    axarr[(plot_index[0][jj], plot_index[1][jj])].set_ylabel('act ' + str(jj), fontsize = 9)
### Then the linear actuators (due to hysteresis)
for jj in np.nditer(linacts):    
    for ii in range(len(u_dm_test)):
        voltage = u_dm_test[ii]
        voltages = np.zeros(actuators)
        voltages[jj] += voltage
        set_displacement(voltages, mirror)
        time.sleep(0.2)
        sh.snapImage()
        image = sh.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy))
        displ = centroid_0 - centroid_i
        abs_displ[ii, jj] = np.sqrt(np.mean(np.square(displ)))
    abso, sim= axarr[(plot_index[0][jj], plot_index[1][jj])].plot(u_dm_test, abs_displ[:,jj], x_sim, y_sim[:, jj])
    axarr[(plot_index[0][jj], plot_index[1][jj])].xaxis.set_ticks(np.arange(-1, 2, 1.0))
    axarr[(plot_index[0][jj], plot_index[1][jj])].yaxis.set_ticks(np.linspace(0, 6.0, 3))
    plt.rcParams['xtick.labelsize'] = label_size 
    plt.rcParams['ytick.labelsize'] = label_size 
    axarr[(plot_index[0][jj], plot_index[1][jj])].set_ylabel('act ' + str(jj), fontsize = 9)
axarr[(plot_index[0][19], plot_index[1][19])].set_xlabel('Input signal (u_dm)', fontsize = 9)
axarr[(plot_index[0][19], plot_index[1][19])].set_ylabel('dr[px]', fontsize = 9)
axarr[(plot_index[0][19], plot_index[1][19])].xaxis.set_ticks(np.arange(-1, 2, 1.0))
axarr[(plot_index[0][19], plot_index[1][19])].yaxis.set_ticks(np.linspace(0, 6.0, 3))
f.legend((abso, sim), ('real displacement', 'simulated displacment'), loc = (0.6,0.07), fontsize = 9)
f.tight_layout()
plt.show()
        
