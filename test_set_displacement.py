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
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40

def set_displacement(u_dm):
    u_dm = u_dm * 72.0
    u_l = np.zeros(u_dm.shape)
    u_l = np.maximum(u_dm, -72.0 * np.ones(u_l.shape))
    u_l = np.minimum(u_dm, 72.0 * np.ones(u_l.shape))    
    actnum=np.arange(0,19,1)
    linacts=np.where(np.logical_or(actnum==4,actnum==7))
    others=np.where(np.logical_and(actnum!=4,actnum!=7))
    u_l[linacts]=(u_dm[linacts]+72.0)/12
    u_l[others]=np.sqrt(u_dm[others]+72.0)
    
    mirror.set(u_l)
    
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

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

actuators = 19
u_dm_0 = np.zeros(actuators)
u_dm_test = np.linspace(-0.9, 0.9, 10)
set_displacement(u_dm_0)
time.sleep(0.1)
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
f, axarr = plt.subplots(5, 4, sharex = True)
f.suptitle('x-displacement of centroid 0 due to actuators')
plot_index = np.unravel_index(np.array(range(actuators)), (5, 4))

for jj in range(actuators):
    if jj == 4 or jj == 7:
        continue
    print(jj)
    for ii in range(len(u_dm_test)):

        voltage = u_dm_test[ii]
        voltages = np.zeros(actuators)
        voltages[jj] += voltage
        set_displacement(voltages)
        time.sleep(0.2)
        sh.snapImage()
        image = sh.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy))
        displ = centroid_0 - centroid_i
        abs_displ[ii, jj] = np.sqrt(np.mean(np.square(displ)))
    abso, = axarr[(plot_index[0][jj], plot_index[1][jj])].plot(u_dm_test, abs_displ[:,jj])
    f.legend((abso, ), ('|d|'), 'lower right', fontsize = 9)
    axarr[(plot_index[0][jj], plot_index[1][jj])].set_title(str(jj))
plt.show()

        
