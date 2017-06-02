import sys
import os
import time
if "C:\Program Files\Micro-Manager-1.4" not in sys.path:
    sys.path.append("C:\Program Files\Micro-Manager-1.4")
import MMCorePy
import PIL.Image
import numpy as np
from matplotlib import rc
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

# Define font for figures
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size' : 7})
rc('text', usetex=False)


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

## image making parameters
dpi_num = 600
int_im_size = (4.98, 3.07)
int_im_size_23 = (0.66 * 4.98, 3.07)
int_im_size_13 = (0.33 * 4.98, 3.07)

#### Set up cameras and mirror
new_yn = raw_input("do you want to take new images? y/n")
folder_name = "20161213_new_inters/"
if new_yn == 'y':
    global mirror
    mirror = edac40.OKOMirror("169.254.158.203") # Enter real IP in here
    sh, int_cam = mc.set_up_cameras()

    actuators = 19
    u_dm = np.zeros(actuators)
    mc.set_displacement(u_dm, mirror)
    time.sleep(0.2)

    raw_input("Did you calibrate the reference mirror? Block DM")
    sh.snapImage()
    image_ref_mirror = sh.getImage().astype(float)
    PIL.Image.fromarray(image_ref_mirror).save(folder_name + "image_ref_mirror.tif")


    raw_input("block reference mirror!")
    sh.snapImage()
    zero_pos_dm = sh.getImage().astype(float)
    PIL.Image.fromarray(zero_pos_dm).save(folder_name + "zero_pos_dm.tif")

    ### make flat wavefront
    u_dm, x_pos_norm_f, y_pos_norm_f, x_pos_zero_f, y_pos_zero_f, V2D = mc.flat_wavefront(u_dm, zero_pos_dm, image_ref_mirror, r_sh_px, r_int_px, sh, mirror, show_accepted_spots = True)

    ##
##    raw_input('remove black paper!')
##    int_cam.snapImage()
##    flat_wf = int_cam.getImage().astype('float')
##    flat_wf = np.flipud(flat_wf)
##    flat_wf = np.fliplr(flat_wf)


    G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)
    a = np.zeros(j_max)

    ind = np.array([0])
    #a[ind] = 0.15 * wavelength
    a[2] = 3 * wavelength

    u_dm_flat = u_dm
    #V2D_inv = np.linalg.pinv(V2D)
    v_abb = (f_sh/(r_sh_m * px_size_sh)) * np.linalg.lstsq(V2D, np.dot(G, a))[0]#np.dot(V2D_inv, np.dot(G, a))
    u_dm -= v_abb

    if np.any(np.abs(u_dm) > 1.0):
        print("maximum deflection of mirror reached")
        print(u_dm)
        
    mc.set_displacement(u_dm, mirror)

    raw_input("keep it covered!")
    sh.snapImage()
    dist_image = sh.getImage().astype('float')
    PIL.Image.fromarray(dist_image).save(folder_name + "dist_image.tif")

    raw_input("remove piece of paper")
    time.sleep(0.2)
    int_cam.snapImage()
    image_i0 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i0).save(folder_name + "interferogram_0.tif")

    raw_input("tip and tilt 1")
    time.sleep(1)
    int_cam.snapImage()
    image_i1 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i1).save(folder_name + "interferogram_1.tif")

    raw_input("tip and tilt 2")
    time.sleep(1)
    int_cam.snapImage()
    image_i2 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i2).save(folder_name + "interferogram_2.tif")

    raw_input("tip and tilt 3")
    time.sleep(1)
    int_cam.snapImage()
    image_i3 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i3).save(folder_name + "interferogram_3.tif")

    raw_input("tip and tilt 4")
    time.sleep(1)
    int_cam.snapImage()
    image_i4 = int_cam.getImage().astype(float)
    PIL.Image.fromarray(image_i4).save(folder_name + "interferogram_4.tif")
else:
    zero_image = np.array(PIL.Image.open(folder_name + "zero_pos_dm.tif"))
    zero_image_zeros = np.copy(zero_image)
    dist_image = np.array(PIL.Image.open(folder_name + "dist_image.tif"))
    image_ref_mirror = np.array(PIL.Image.open(folder_name + "image_ref_mirror.tif"))
    image_i0 = np.array(PIL.Image.open(folder_name + "interferogram_0.tif"))
    image_i1 = np.array(PIL.Image.open(folder_name + "interferogram_1.tif"))
    image_i2 = np.array(PIL.Image.open(folder_name + "interferogram_2.tif"))
    image_i3 = np.array(PIL.Image.open(folder_name + "interferogram_3.tif"))
    image_i4 = np.array(PIL.Image.open(folder_name + "interferogram_4.tif"))

## Given paramters for centroid gathering
px_size_sh = 5.2e-6     # width of pixels
px_size_int = 5.2e-6
f_sh = 17.6e-3            # focal length
r_int_px = 340
r_sh_px = 340 
r_sh_m = r_sh_px * px_size_int 
j_max= 30         # maximum fringe order
wavelength = 632e-9 #[m]
box_len = 35.0 #half width of subaperture box in px

### things we need
[ny,nx] = zero_image.shape
x = np.linspace(1, nx, nx)
y = np.linspace(1, ny, ny)
xx, yy = np.meshgrid(x, y)

x_pos_zero, y_pos_zero = Hm.zero_positions(zero_image_zeros)
x_pos_flat, y_pos_flat = Hm.centroid_positions(x_pos_zero, y_pos_zero, image_ref_mirror, xx, yy)
centre = Hm.centroid_centre(x_pos_flat, y_pos_flat)
x_pos_norm = ((x_pos_flat - centre[0]))/r_sh_px
y_pos_norm = ((y_pos_flat - centre[1]))/r_sh_px
inside = np.where(np.sqrt(x_pos_norm**2 + y_pos_norm**2) <= (1 + (box_len/r_sh_px))) #35 is the half the width of the pixel box aroudn a centroid and r_sh_px is the scaling factor
x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, x_pos_norm_f, y_pos_norm_f = mc.filter_positions(inside, x_pos_zero, y_pos_zero, x_pos_flat, y_pos_flat, x_pos_norm, y_pos_norm)
G = LSQ.matrix_avg_gradient(x_pos_norm_f, y_pos_norm_f, j_max, r_sh_px)

## Shack Hartmann methods
a_measured_new = LSQ.LSQ_coeff(x_pos_zero_f, y_pos_zero_f, x_pos_flat_f, y_pos_flat_f, G, image_ref_mirror, dist_image, px_size_sh, r_sh_px, f_sh, j_max) 
a_janss_check = janssen.coeff(x_pos_zero_f, y_pos_zero_f, image_ref_mirror, dist_image, px_size_sh, f_sh, r_sh_m, j_max)


## interferogram analysis
## centre and radius of interferogam. Done by eye, with the help of define_radius.py
x0 = 550
y0 = 484
radius = 375
    
circle1 = plt.Circle((375,375), 375, color = 'white', fill=False, linewidth = 2)
f, axarr = plt.subplots(2, 3, figsize=(9.31,5.91))
axarr[0,0].set_title('flat wavefront', fontsize = 9)
#axarr[0,0].imshow(flat_wf[164:914, 363:1113], cmap = 'bone')
Zn.plot_interferogram(j_max, a_measured_new, ax = axarr[0,0], want_phi_old = True)
axarr[0,1].set_title('interferogram of wanted abberation', fontsize = 9)
#Zn.plot_interferogram(j_max, a, axarr[0,1])
axarr[1, 0].imshow(image_i0[y0-radius:y0+radius, x0-radius:x0+radius], cmap = 'bone')
axarr[1, 0].add_artist(circle1)
axarr[1, 0].set_title('measured interferogram', fontsize = 9)
Zn.plot_interferogram(j_max, a_measured_new, ax = axarr[1,1])
axarr[1, 1].set_title('interferogram simulated from LSQ', fontsize = 9)
Zn.plot_interferogram(j_max, np.real(a_janss_check), ax = axarr[1,2])
axarr[1,2].set_title('interferogram simulated from Janssen', fontsize = 9)

ax = axarr.reshape(-1)
for i in range(len(ax)):
    if i != 2:
        ax[i].get_xaxis().set_ticks([])
        ax[i].get_yaxis().set_ticks([])
    else:
        ax[i].tick_params(labelsize=6)


indexs = np.arange(2, j_max+2, 1)
#axarr[0,2].plot(indexs, a/wavelength, 'ro', label = 'intended')
axarr[0,2].plot(indexs, a_measured_new/(wavelength), 'bo', label = 'LSQ solution')
#ax.plot(indexs, np.real(a_janss_check)/(wavelength), 'go', label = 'measured_janss_check')
axarr[0,2].plot(indexs, np.real(a_janss_check)/wavelength, 'ko', label = 'Janssen solution')
axarr[0,2].set_xlim([0, j_max+2])
axarr[0,2].legend(loc = 'best', fontsize = 6)
axarr[0,2].set_xlabel('Coeffcient', fontsize = 7, labelpad = -0.75)
axarr[0,2].set_ylabel('a [\lambda]', fontsize = 7, labelpad = -2)
axarr[0,2].set_title('measured coefficients', fontsize = 9)
#axarr[0,2].tick_params(axis='both', pad=-1)
#lt.savefig('try_2_single_aberrations_j_' + vec2str(ind+2) +'.png', bbox_inches='tight')

plt.show()

#a_janss_real, a_janss_check = janssen.coeff(x_pos_zero, y_pos_zero, image_control, sh, px_size_sh, f_sh, r_sh_m, j_max)
