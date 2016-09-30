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


###reference image
##impath_zero = os.path.abspath("flat_def_mirror_reference.tif
##zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
voltage_avg = 6.0
stroke_50 = np.arange(-5.5, 135.5, 20.0)
stroke_tt = np.linspace(-5.5, 6.0, num = len(stroke_50))
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
#f2, axarr2 = plt.subplots(5, 4, sharex = True)
f.suptitle('x-displacement of centroid 0 due to actuators')
plot_index = np.unravel_index(np.array(range(actuators)), (5, 4))

for jj in range(actuators):
    print jj
    for ii in range(len(stroke_50)):
        voltages = voltage_avg * np.ones(actuators)
        
        if jj == 4 or jj == 7:
            voltages = voltage_avg * np.ones(actuators)
            voltage = stroke_tt[ii]
            voltages[jj] += voltage
        else:
            voltages = voltage_avg * np.ones(actuators)
            voltage = stroke_50[ii]
            voltages[jj] += voltage
            voltages = np.sqrt(voltages)
            print(voltages[jj])
        #voltages[4] = 6.0
        #voltages[7] = 6.0
        mirror.set(voltages)
        time.sleep(0.2)
        cam1.snapImage()
        image = cam1.getImage().astype(float)
        centroid_i = np.hstack(Hm.centroid_positions(x_pos_flat, y_pos_flat, image, xx, yy))
        displacement = centroid_0 - centroid_i
        abs_displ[ii, jj] = displacement[0]
        #displ_per_volt[ii, jj] = abs_displ[ii,jj] / (voltage + 6.0)
    abso, = axarr[(plot_index[0][jj], plot_index[1][jj])].plot(stroke_50, abs_displ[:,jj])
    #rela,  = axarr2[(plot_index[0][jj], plot_index[1][jj])].plot(stroke_50, displ_per_volt[:,jj])
    f.legend((abso, ), ('|d|'), 'lower right', fontsize = 9)
    #f2.legend((rela, ), ('d|/V^2'), 'lower right', fontsize = 9)
    axarr[(plot_index[0][jj], plot_index[1][jj])].set_title(str(jj))
axarr[(plot_index[0][16], plot_index[1][16])].set_xlabel('D_inf')
axarr[(plot_index[0][16], plot_index[1][16])].set_ylabel('x-displacement w.r.t. mid-stroke [px]')
#axarr2[(plot_index[0][16], plot_index[1][16])].set_xlabel('V')
#axarr2[(plot_index[0][16], plot_index[1][16])].set_ylabel('displacement / V [px]')

plt.show()
