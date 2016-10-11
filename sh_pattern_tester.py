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
import displacement_matrix as Dm
import mirror_control as mc
import LSQ_method as LSQ
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker

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

u_dm = np.zeros(19)
mc.set_displacement(u_dm, mirror)
time.sleep(0.05)
sh.snapImage()
image_zeros = sh.getImage().astype(float)
dm_zeros = Hm.zero_positions(image_zeros)
##
##impath_dm = os.path.abspath("dm_sh.tif")
##sh_dm = np.asarray(PIL.Image.open(impath_dm)).astype(float)
##
impath_ref = os.path.abspath("reference_sh.tif")
sh_ref = np.asarray(PIL.Image.open(impath_ref)).astype(float)
##
##dm_zeros = Hm.zero_positions(sh_dm)
ref_zeros = Hm.zero_positions(sh_ref)

plt.scatter(dm_zeros[0][:], dm_zeros[1][:], color = 'r')
plt.scatter(ref_zeros[0][:], ref_zeros[1][ :], color = 'g')
#plt.show()
u_dm = np.ones(19)
u_dm[4] = 0
u_dm[7] = 0

mc.set_displacement(u_dm, mirror)
time.sleep(0.05)
sh.snapImage()
image_ones = sh.getImage().astype(float)
PIL.Image.fromarray(image_ones).save("sh_max_deflection.tif")

dm_ones = Hm.zero_positions(image_ones)
plt.scatter(dm_ones[0][:], dm_ones[1][:], color = 'b')
plt.show()
