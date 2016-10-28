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
import mirror_control as mc

mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
#u_dm = np.zeros(19)
#mc.set_displacement(u_dm, mirror)

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

u_dm = np.ones(19) * 0.3
u_dm[4] = 0
u_dm[7] = 0
mc.set_displacement(u_dm, mirror)
time.sleep(0.3)

cam2.snapImage()
cam1.snapImage()
PIL.Image.fromarray(cam1.getImage().astype("float")).save("cam1_test.tif")
cam2.snapImage()
PIL.Image.fromarray(cam2.getImage().astype("float")).save("int_test_i0.tif")

u_dm[4] = 0.3
u_dm[7] = 0.2
mc.set_displacement(u_dm, mirror)
time.sleep(0.3)
cam2.snapImage()
PIL.Image.fromarray(cam2.getImage().astype("float")).save("int_test_i1.tif")

u_dm[4] = 0.6
u_dm[7] = -0.3
mc.set_displacement(u_dm, mirror)
time.sleep(0.3)
cam2.snapImage()
PIL.Image.fromarray(cam2.getImage().astype("float")).save("int_test_i2.tif")
