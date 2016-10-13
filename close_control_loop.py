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
import Zernike as Zn
import mirror_control as mc
import LSQ_method as LSQ
import matplotlib.pyplot as plt
from matplotlib import cm
import edac40
import matplotlib.ticker as ticker
import janssen

def abberations2voltages(G, V2D_inv, a, f, r_sh, px_size):
    """Given the geometry matrix of the SH sensor, the voltage2displacement matrix inverse, the wanted abberations a, focal length and physical size of the sh sensor
    it return the voltages v for which to control the mirror.
    input G: (2*sh_spots, zernike modes) matrix containing the x and y derivative of all zernike modes at the given sh spots
    V2D_inv: pseudo inverse of (2*sh_spots, actuators) matrix containing the displacement of each spot when the actuator is moved
    a: (zernike modes,) vector containing the coefficients for the desired abberation (coefficients will be small due to normalization & definition)
    f: scalar focal length [m]
    r_sh: scalar physical size of SH sensor [m]"""
    v = (f/(r_sh * px_size)) * np.dot(V2D_inv, np.dot(G, a))
    return v


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

raw_input("Did you calibrate the reference mirror? Block DM")
sh.snapImage()
image_control = sh.getImage().astype(float)

raw_input("block reference mirror!")
sh.snapImage()
zero_image = sh.getImage().astype(float)

## Given paramters for centroid gathering
[ny,nx] = zero_image.shape
px_size_sh = 5.2e-6     # width of pixels 
f_sh = 17.6e-3            # focal length
r_sh_m = 1.924e-3
r_sh_px = r_sh_m / px_size_sh
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max= 10           # maximum fringe order

### Gather centroids of current picture (u_dm 0) and voltage 2 distance (v2d) matrix
x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)


actnum=np.arange(0,19,1)
linacts=np.where(np.logical_or(actnum==4,actnum==7))
others=np.where(np.logical_and(actnum!=4,actnum!=7))

V2D = Dm.gather_displacement_matrix(mirror, sh, x_pos_zero, y_pos_zero)
V2D_inv = np.linalg.pinv(V2D)
#rms_sim = np.sqrt(np.sum(np.square(V2D), axis = 0) / float(len(V2D[:, 0])))

#### Gather control wavefront and correct towards this wf
#impath_control = os.path.abspath("flat_mirror_reference.tif")
#image_control = np.asarray(PIL.Image.open(impath_control)).astype(float)

scaling = 0.75
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
centroid_control = np.hstack((x_pos_flat, y_pos_flat))
for i in range(20):
    print(i)
    sh.snapImage()
    zero_image = sh.getImage().astype(float)
    x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, zero_image, xx, yy)
    centroid_0 = np.hstack((x_pos_flat, y_pos_flat))
    d = centroid_control - centroid_0
    u_dm_diff = np.dot(V2D_inv, d)
    u_dm -= scaling * u_dm_diff
    if np.any(np.abs(u_dm) > 1.0):
        print("maximum deflection of mirror reached")
        print(u_dm)
    mc.set_displacement(u_dm, mirror)
    time.sleep(0.2)
print np.max(u_dm), np.min(u_dm)
plt.hist(u_dm)
plt.show()
print(u_dm)

raw_input('remove black paper!')
cam2.snapImage()
PIL.Image.fromarray(cam2.getImage().astype("float")).save("interference_pattern_after_inversion.tif")
plt.imshow(cam2.getImage().astype('float'), cmap = 'bone')
plt.show()

##### Introduce abberations on purpose
### Normalize x, y
wavelength = 632e-9
r_sh_m = 1.924e-3 #physical radius
r_sh_px = r_sh_m / px_size_sh #radius is px
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
if np.any(np.sqrt(x_pos_norm **2 + y_pos_norm**2) > 1):
    print "somethings gone wrong in normalization"

G = LSQ.geometry_matrix_2(x_pos_norm, y_pos_norm, j_max, r_sh_px)
a = np.zeros(j_max)
a[2] = 0.5 * wavelength
v_abb = abberations2voltages(G, V2D_inv, a, f_sh, r_sh_m, px_size_sh)
u_dm -= v_abb
if np.any(np.abs(u_dm) > 1.0):
    print("maximum deflection of mirror reached")
    print(u_dm)
mc.set_displacement(u_dm, mirror)
time.sleep(0.3)
cam2.snapImage()
#PIL.Image.fromarray(cam2.getImage().astype("float")).save("interference_pattern_after_inversion.tif")
#plt.imshow(cam2.getImage().astype('float'), cmap = 'bone')
#plt.show()
raw_input("re-cover the reference mirror")
a_measured = LSQ.LSQ_coeff(x_pos_zero, y_pos_zero, image_control, sh, px_size_sh, r_sh_px, f_sh, j_max)

plt.imshow(cam2.getImage().astype('float'), cmap = 'bone')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(plt.figaspect(0.5)))
ax1.imshow(cam2.getImage().astype('float'), cmap = 'bone')
ax1.set_title('measured interferogram')
Zn.plot_interferogram(j_max, a_measured, ax2)
ax2.set_title('interferogram simulated from measured coefficients')
#a_janss_real, a_janss_check = janssen.coeff(x_pos_zero, y_pos_zero, image_control, sh, px_size_sh, f_sh, r_sh_m, j_max)
