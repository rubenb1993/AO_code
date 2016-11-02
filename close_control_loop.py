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

def vec2str(vec):
    i = len(vec)
    if i != 1:
        return str(vec[0]) + '_' + vec2str(vec[1:])
    else:
        return str(vec[0])

### Define font for figures
##rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
##rc('text', usetex=False)

#### Set up cameras and mirror
global mirror
mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
sh, int_cam = mc.set_up_cameras()

actuators = 19
u_dm = np.zeros(actuators)
mc.set_displacement(u_dm, mirror)
time.sleep(0.2)

raw_input("Did you calibrate the reference mirror? Block DM")
sh.snapImage()
image_control = sh.getImage().astype(float)

raw_input("block reference mirror!")
sh.snapImage()
zero_image = sh.getImage().astype(float)

## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 340
r_sh_px = 370
r_sh_m = r_sh_px * px_size_int
j_max= 20          # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px

### make flat wavefront
u_dm, x_pos_norm_f, y_pos_norm_f, x_pos_zero_f, y_pos_zero_f, V2D = mc.flat_wavefront(u_dm, zero_image, image_control, r_sh_px, r_int_px, sh, mirror, show_accepted_spots = True)

##
raw_input('remove black paper!')
int_cam.snapImage()
flat_wf = int_cam.getImage().astype('float')
flat_wf = np.flipud(flat_wf)
flat_wf = np.fliplr(flat_wf)


G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
a = np.zeros(j_max)

ind = np.array([0])
#a[ind] = 0.15 * wavelength
a[ind] = 3 * wavelength

u_dm_flat = u_dm
V2D_inv = np.linalg.pinv(V2D)
v_abb = (f_sh/(r_sh_m * px_size_sh)) * np.dot(V2D_inv, np.dot(G, a))
u_dm -= v_abb

if np.any(np.abs(u_dm) > 1.0):
    print("maximum deflection of mirror reached")
    print(u_dm)
    
mc.set_displacement(u_dm, mirror)

raw_input("dont touch anything now")
time.sleep(0.5)
int_cam.snapImage()
interferogram = int_cam.getImage().astype('float')
interferogram = np.flipud(interferogram)
interferogram = np.fliplr(interferogram)

raw_input("re-cover the reference mirror")
a_measured_new = LSQ.LSQ_coeff(x_pos_zero_f, y_pos_zero_f, G, image_control, sh, px_size_sh, r_sh_px, f_sh, j_max)
a_janss_check = janssen.coeff(x_pos_zero_f, y_pos_zero_f, image_control, sh, px_size_sh, f_sh, r_sh_m, j_max)

circle1 = plt.Circle((375,375), 375, color = 'white', fill=False, linewidth = 2)
f, axarr = plt.subplots(2, 3, figsize=(9.31,5.91))
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
