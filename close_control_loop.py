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

def filter_positions(inside, *args):
    new_positions = []
    for arg in args:
        new_positions.append(np.array(arg)[inside])
    return new_positions

def vec2str(vec):
    i = len(vec)
    if i != 1:
        return str(vec[0]) + '_' + vec2str(vec[1:])
    else:
        return str(vec[0])

### Define font for figures
##rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
##rc('text', usetex=False)

    
#### Set up cameras
cam1=MMCorePy.CMMCore()
cam1.loadDevice("cam","IDS_uEye","IDS uEye")
cam1.initializeDevice("cam")
cam1.setCameraDevice("cam")
pixel_clock = cam1.getPropertyUpperLimit("cam", "Pixel Clock")
if pixel_clock == 150.0:
    cam1.setProperty("cam","Pixel Clock",150)
    cam1.setProperty("cam", "PixelType", '8bit mono')
    cam1.setProperty("cam","Exposure",0.0434)
    sh = cam1

    cam2=MMCorePy.CMMCore()
    cam2.loadDevice("cam","IDS_uEye","IDS uEye")
    cam2.initializeDevice("cam")
    cam2.setCameraDevice("cam")
    cam2.setProperty("cam","Pixel Clock", 43)
    cam2.setProperty("cam","Exposure", 0.0668)
    int_cam = cam2
else:
    cam1.setProperty("cam","Pixel Clock", 43)
    cam1.setProperty("cam","Exposure", 0.0668)
    int_cam = cam1

    cam2=MMCorePy.CMMCore()
    cam2.loadDevice("cam","IDS_uEye","IDS uEye")
    cam2.initializeDevice("cam")
    cam2.setCameraDevice("cam")
    cam2.setProperty("cam","Pixel Clock", 150)
    cam2.setProperty("cam","PixelType", '8bit mono')
    cam2.setProperty("cam","Exposure", 0.0434)
    sh = cam2


global mirror
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here

actuators = 19
u_dm = np.zeros(actuators)
#u_dm_test = np.linspace(-0.9, 0.9, 11)
mc.set_displacement(u_dm, mirror)
time.sleep(0.2)
##raw_input('making 0 measurement')
##int_cam.snapImage()
##control_show = int_cam.getImage().astype(float)
##control_show =np.flipud(control_show)
##control_show = np.fliplr(control_show)

raw_input("Did you calibrate the reference mirror? Block DM")
sh.snapImage()
image_control = sh.getImage().astype(float)

raw_input("block reference mirror!")
sh.snapImage()
zero_image = sh.getImage().astype(float)

## Given paramters for centroid gathering
[ny,nx] = zero_image.shape
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 410
r_sh_m = r_int_px * px_size_int
r_sh_px = r_sh_m / px_size_sh
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)
j_max= 20          # maximum fringe order
### Normalize x, y
wavelength = 632e-9
#r_sh_m = 1.924e-3 #physical radius
#r_sh_px = r_sh_m / px_size_sh #radius is px

### Gather centroids of current picture (u_dm 0) and voltage 2 distance (v2d) matrix
x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image)
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (35.0/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)

##f3 = plt.figure(figsize = plt.figaspect(1.))
##ax3 = f3.add_subplot(1,1,1)
##ax3.scatter(x_pos_norm, y_pos_norm, color='b')
##ax3.scatter(x_pos_norm_f, y_pos_norm_f, color ='r')
##ax3.set_xlim([-1, 1])
##ax3.set_ylim([-1, 1])
###plt.scatter(centre[0],centre[1], color = 'k')
##circle1 = plt.Circle([0,0] , 1, color = 'k', fill=False)
##ax3.add_artist(circle1)
##plt.show()
actnum=np.arange(0,19,1)
linacts=np.where(np.logical_or(actnum==4,actnum==7))
others=np.where(np.logical_and(actnum!=4,actnum!=7))

V2D = Dm.gather_displacement_matrix(mirror, sh, x_pos_zero_f, y_pos_zero_f)
np.save('test_V2D_2', V2D)
V2D_inv = np.linalg.pinv(V2D)
#rms_sim = np.sqrt(np.sum(np.square(V2D), axis = 0) / float(len(V2D[:, 0])))

#### Gather control wavefront and correct towards this wf
#impath_control = os.path.abspath("flat_mirror_reference.tif")
#image_control = np.asarray(PIL.Image.open(impath_control)).astype(float)

scaling = 0.75
##x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_control, xx, yy)
centroid_control = np.hstack((x_pos_flat_f, y_pos_flat_f))
for i in range(30):
    print(i)
    sh.snapImage()
    zero_image = sh.getImage().astype(float)
    x_pos_flat_f, y_pos_flat_f = Hm.centroid_positions(x_pos_zero_f, y_pos_zero_f, zero_image, xx, yy)
    centroid_0 = np.hstack((x_pos_flat_f, y_pos_flat_f))
    d = centroid_control - centroid_0
    u_dm_diff = np.dot(V2D_inv, d)
    u_dm -= scaling * u_dm_diff
    mc.set_displacement(u_dm, mirror)
    time.sleep(0.05)
#plt.hist(u_dm)
#plt.show()
if np.any(np.abs(u_dm) > 1.0):
    print("maximum deflection of mirror reached")
print(u_dm)
##
raw_input('remove black paper!')
int_cam.snapImage()
flat_wf = int_cam.getImage().astype('float')
flat_wf = np.flipud(flat_wf)
flat_wf = np.fliplr(flat_wf)

if np.any(np.sqrt(x_pos_norm_f **2 + y_pos_norm_f**2) > (1+35/r_sh_px)):
    print "somethings gone wrong in normalization"


G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
a = np.zeros(j_max)

ind = np.array([1, 3, 5, 7])
a[ind] = 0.15 * wavelength
a[1] = 2 * wavelength
##
##n_ind, m_ind = Zn.Zernike_j_2_nm(ind+2)
##if m_ind + 1 > n_ind - 1:
##    print('this was edge case 1')
##elif np.abs(m_ind - 1.0) > np.abs(n_ind - 1.0):
##    print('this was edge case 2')
##else:
##    print('this was no edge case')



u_dm_flat = u_dm
v_abb = abberations2voltages(G, V2D_inv, a, f_sh, r_sh_m, px_size_sh)
u_dm -= v_abb
##u_dm_minus = u_dm - v_abb
##u_dm_plus = u_dm + v_abb
##if np.abs(np.average(u_dm_plus)) > np.abs(np.average(u_dm_minus)):
##    u_dm = u_dm_minus
##else:
##    u_dm = u_dm_plus

if np.any(np.abs(u_dm) > 1.0):
    print("maximum deflection of mirror reached")
    print(u_dm)
mc.set_displacement(u_dm, mirror)
time.sleep(0.3)
int_cam.snapImage()
interferogram = int_cam.getImage().astype('float')
interferogram = np.flipud(interferogram)
interferogram = np.fliplr(interferogram)
#PIL.Image.fromarray(cam2.getImage().astype("float")).save("interference_pattern_after_inversion.tif")
#plt.imshow(cam2.getImage().astype('float'), cmap = 'bone')
#plt.show()
raw_input("re-cover the reference mirror")
a_measured_new = LSQ.LSQ_coeff(x_pos_zero_f, y_pos_zero_f, G, image_control, sh, px_size_sh, r_sh_px, f_sh, j_max)
a_janss_real, a_janss_check = janssen.coeff(x_pos_zero, y_pos_zero, image_control, sh, px_size_sh, f_sh, r_sh_m, j_max)

##
#plt.imshow(cam2.getImage().astype('float'), cmap = 'bone')
circle1 = plt.Circle((375,375), 375, color = 'white', fill=False, linewidth = 2)
f, axarr = plt.subplots(2, 3, figsize=(9.31,5.91))

##axarr[0,0].set_title('Initial wavefront', fontsize = 9)
##axarr[0,0].imshow(control_show[164:914, 363:1113], cmap = 'bone')
axarr[0,0].set_title('flat wavefront', fontsize = 9)
axarr[0,0].imshow(flat_wf[164:914, 363:1113], cmap = 'bone')
axarr[0,1].set_title('interferogram of wanted abberation', fontsize = 9)
Zn.plot_interferogram(j_max, a, axarr[0,1])
axarr[1, 0].imshow(interferogram[164:914, 363:1113], cmap = 'bone')
axarr[1, 0].add_artist(circle1)
axarr[1, 0].set_title('measured interferogram', fontsize = 9)
Zn.plot_interferogram(j_max, a_measured_new, axarr[1,1])
axarr[1, 1].set_title('interferogram simulated from LSQ', fontsize = 9)
Zn.plot_interferogram(j_max, np.real(a_janss_check), axarr[1,2])
axarr[1,2].set_title('interferogram simulated from Janssen', fontsize = 9)
ax = axarr.reshape(-1)
for i in range(len(ax)):
    if i != 2:
        ax[i].get_xaxis().set_ticks([])
        ax[i].get_yaxis().set_ticks([])
    else:
        ax[i].tick_params(labelsize=6)


indexs = np.arange(2, j_max+2, 1)
axarr[0,2].plot(indexs, a/wavelength, 'ro', label = 'intended')
axarr[0,2].plot(indexs, a_measured_new/(wavelength), 'bo', label = 'LSQ solution')
#ax.plot(indexs, np.real(a_janss_check)/(wavelength), 'go', label = 'measured_janss_check')
axarr[0,2].plot(indexs, np.real(a_janss_check)/wavelength, 'ko', label = 'Janssen solution')
axarr[0,2].set_xlim([0, j_max+2])
axarr[0,2].legend(loc = 'best', fontsize = 6)
axarr[0,2].set_xlabel('Coeffcient', fontsize = 7, labelpad = -0.75)
axarr[0,2].set_ylabel('a [\lambda]', fontsize = 7, labelpad = -2)
axarr[0,2].set_title('measured coefficients', fontsize = 9)
#axarr[0,2].tick_params(axis='both', pad=-1)
plt.savefig('try_2_single_aberrations_j_' + vec2str(ind+2) +'.png', bbox_inches='tight')

plt.show()

#a_janss_real, a_janss_check = janssen.coeff(x_pos_zero, y_pos_zero, image_control, sh, px_size_sh, f_sh, r_sh_m, j_max)
