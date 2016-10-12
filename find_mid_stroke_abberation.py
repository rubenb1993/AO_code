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


### Test figures
impath_zero = os.path.abspath("dm_sh.tif")
impath_dist = os.path.abspath("ref_sh.tif")
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
dist_image = np.asarray(PIL.Image.open(impath_dist)).astype(float)

## Given paramters for centroid gathering
[ny,nx] = zero_image.shape
px_size_sh = 5.2e-6     # width of pixels 
f_sh = 17.6e-3            # focal length
r_sh_px = 360
r_sh_m = r_sh_px * px_size_sh
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max= 10           # maximum fringe order
wavelength = 632e-9


x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
zero_image = np.asarray(PIL.Image.open(impath_zero)).astype(float)
a = LSQ.LSQ_coeff(x_pos_zero, y_pos_zero, zero_image, sh, px_size_sh, r_sh_px, f_sh, j_max)
print a/wavelength


