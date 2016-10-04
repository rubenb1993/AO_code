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
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker

### Define font for figures
##rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
##rc('text', usetex=False)

    
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
u_dm = np.zeros(actuators)
#u_dm_test = np.linspace(-0.9, 0.9, 11)
mc.set_displacement(u_dm, mirror)
time.sleep(0.2)

raw_input("block reference mirror!")

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

### Gather centroids of current picture (u_dm 0) and voltage 2 distance (v2d) matrix
x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
sh.snapImage()
zero_image = sh.getImage().astype(float) #re load image due to corruption
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
centroid_0 = np.hstack((x_pos_flat, y_pos_flat))

actnum=np.arange(0,19,1)
linacts=np.where(np.logical_or(actnum==4,actnum==7))
others=np.where(np.logical_and(actnum!=4,actnum!=7))

V2D = Dm.gather_displacement_matrix(mirror, sh, x_pos_zero, y_pos_zero)
V2D_inv = np.linalg.pinv(V2D)
#rms_sim = np.sqrt(np.sum(np.square(V2D), axis = 0) / float(len(V2D[:, 0])))

#### Gather control wavefront and correct towards this wf
impath_control = os.path.abspath("flat_mirror_reference.tif")
image_control = np.asarray(PIL.Image.open(impath_control)).astype(float)

centroid_control = np.hstack(Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy))
d = centroid_control - centroid_0
u_dm_diff = np.dot(V2D_inv, d)
u_dm -= u_dm_diff
if np.any(np.abs(u_dm) > 1.0):
    print("maximum deflection of mirror reached")
    print(u_dm)
mc.set_displacement(u_dm, mirror)

raw_input('remove black paper!')
cam2.snapImage()
PIL.Image.fromarray(cam2.getImage().astype("float")).save("interference_pattern_after_inversion.tif")
plt.imshow(cam2.getImage().astype('float'), cmap = 'bone')
plt.show()
