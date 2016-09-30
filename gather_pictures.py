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
import edac40

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
voltages = np.ones(19) * 6.0
mirror.set(voltages)


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
cam2.snapImage()
cam1.snapImage()
PIL.Image.fromarray(cam1.getImage().astype("float")).save("cam1_test.tif")
cam2.snapImage()
PIL.Image.fromarray(cam2.getImage().astype("float")).save("cam2_test.tif")

##for j in range(10):
##    i = j*10.0
##    voltage = np.sin(i%100*2*np.pi/100.0)*6.0
##    voltages = np.ones(19) * voltage + 6.0
##    voltages[4] = 6.0
##    voltages[7] = 6.0
##
##    mirror.set(voltages)
##    time.sleep(0.005)
##    cam2.snapImage()
##    #PIL.Image.fromarray(cam1.getImage().astype("float")).save("defocus_" + str(i) + "_inter.tif")
##    PIL.Image.fromarray(cam2.getImage().astype("float")).save("defocus_" + str(i) + "_sh.tif")
